import asyncio

from app.core.config import Settings
from app.models.schemas import ChatRequest
from app.services.answer_deduper import AnswerDeduper
from app.services.chat_service import ChatService
from app.services.fallback_service import FallbackService
from app.services.github_models_client import GitHubModelsClient
from app.services.knowledge_service import KnowledgeBaseService
from app.services.safety_service import SafetyService
from app.services.session_store import SessionStore
from app.services.vector_store import VectorStore


def build_chat_service(**setting_overrides) -> ChatService:
    defaults = {
        "ENABLE_GITHUB_CHAT_GENERATION": False,
        "GITHUB_TOKEN": None,
        "TARGET_AUDIENCE": "female_only",
        "RESPONSE_STYLE": "feminine",
    }
    defaults.update(setting_overrides)
    settings = Settings(
        **defaults,
    )
    kb_service = KnowledgeBaseService(settings)
    kb_service.load()
    github_client = GitHubModelsClient(settings)
    vector_store = VectorStore(settings, github_client, kb_service)
    safety_service = SafetyService()
    session_store = SessionStore(max_turns=settings.HISTORY_TURNS_TO_KEEP)
    fallback_service = FallbackService()
    return ChatService(
        settings=settings,
        kb_service=kb_service,
        vector_store=vector_store,
        github_client=github_client,
        safety_service=safety_service,
        session_store=session_store,
        fallback_service=fallback_service,
    )


def run_chat(service: ChatService, **payload):
    request = ChatRequest(debug=True, **payload)
    return asyncio.run(service.chat(request))


def test_short_emotional_signal_leads_to_followup_question():
    service = build_chat_service()

    reply = run_chat(
        service,
        conversation_id="short-sad",
        mbti_type="INFJ",
        user_message="sad",
        user_gender="female",
    )

    assert reply.debug and reply.debug.detected_intent == "SHORT_EMOTIONAL_SIGNAL"
    assert reply.debug.response_mode == "CHAT_MODE"
    assert reply.response.follow_up_question
    assert "سبب" in reply.response.follow_up_question
    assert not reply.response.practical_steps
    assert not reply.recommended_videos


def test_vague_distress_leads_to_empathic_clarification():
    service = build_chat_service()

    reply = run_chat(
        service,
        conversation_id="vague-distress",
        user_message="مش كويسة",
        user_gender="female",
    )

    assert reply.debug and reply.debug.detected_intent == "VAGUE_DISTRESS"
    assert reply.response.grounded_answer in {"أنا معاكي.", "أقدر أساعدك، بس محتاجة أفهم الصورة أقرب الأول."}
    assert reply.response.follow_up_question
    assert "حزن" in (reply.response.follow_up_question + (reply.response.choice_prompt or ""))


def test_direct_issue_uses_grounded_support_mode():
    service = build_chat_service()

    reply = run_chat(
        service,
        conversation_id="direct-issue",
        mbti_type="INFJ",
        user_message="أنا بتعب من الناس وبشيل همهم",
        user_gender="female",
    )

    assert reply.debug and reply.debug.response_mode == "GROUNDED_SUPPORT_MODE"
    assert "الناس" in reply.response.understanding or "الناس" in reply.response.grounded_answer
    assert len(reply.response.practical_steps) <= 4
    assert reply.recommendations.unlocked is False


def test_recommendations_delayed_until_threshold():
    service = build_chat_service(RECOMMENDATIONS_AFTER_TURN=3)

    reply = run_chat(
        service,
        conversation_id="delay-recommendations",
        mbti_type="INFJ",
        user_message="أنا بتعب من الناس وبشيل همهم",
        user_gender="female",
    )

    assert reply.turn_number == 1
    assert reply.recommendations.unlocked is False
    assert not reply.recommended_videos
    assert not reply.recommended_books
    assert not reply.recommended_podcasts


def test_explicit_resource_request_can_unlock_recommendations_early():
    service = build_chat_service(RECOMMENDATIONS_AFTER_TURN=3)
    conversation_id = "explicit-resource"

    run_chat(
        service,
        conversation_id=conversation_id,
        mbti_type="INFJ",
        user_message="أنا بتعب من الناس وبشيل همهم",
        user_gender="female",
    )
    reply = run_chat(
        service,
        conversation_id=conversation_id,
        mbti_type="INFJ",
        user_message="رشحيلي فيديوهين",
        user_gender="female",
        max_videos=2,
    )

    assert reply.debug and reply.debug.detected_intent == "RESOURCE_REQUEST"
    assert reply.recommendations.unlocked is True
    assert reply.recommended_videos
    assert len(reply.recommended_videos) <= 2


def test_links_only_filters_out_null_link_recommendations():
    service = build_chat_service(RECOMMENDATIONS_AFTER_TURN=3)
    conversation_id = "links-only"

    run_chat(
        service,
        conversation_id=conversation_id,
        mbti_type="INFJ",
        user_message="أنا بتعب من الناس وبشيل همهم",
        user_gender="female",
    )
    reply = run_chat(
        service,
        conversation_id=conversation_id,
        mbti_type="INFJ",
        user_message="رشحيلي فيديوهات وكتب عن ده",
        user_gender="female",
        recommendation_links_only=True,
    )

    all_items = reply.recommended_videos + reply.recommended_books + reply.recommended_podcasts
    assert all_items
    assert all(item.url for item in all_items)


def test_mbti_language_stays_nondiagnostic():
    service = build_chat_service()

    reply = run_chat(
        service,
        conversation_id="mbti-soft",
        mbti_type="INFJ",
        user_message="أنا بتعب من الناس وبشيل همهم",
        user_gender="female",
    )

    assert "مش كسبب مؤكد" in reply.response.mbti_connection or "مش تفسير نهائي" in reply.response.mbti_connection
    assert "تشخيص" in reply.response.mbti_connection or "تشخيص" in reply.response.support_note


def test_gender_style_follows_neutral_config():
    service = build_chat_service(TARGET_AUDIENCE="all", RESPONSE_STYLE="neutral")

    reply = run_chat(
        service,
        conversation_id="neutral-style",
        user_message="sad",
        user_gender="male",
    )

    combined = " ".join(
        filter(
            None,
            [
                reply.response.understanding,
                reply.response.grounded_answer,
                reply.response.follow_up_question or "",
            ],
        )
    )
    assert "حاسة" not in combined
    assert "معاكي" not in combined


def test_answer_deduper_removes_repeated_meanings():
    payload = AnswerDeduper().dedupe_structured(
        {
            "understanding": "واضح إن فيه ضغط. واضح إن فيه ضغط.",
            "mbti_connection": "",
            "grounded_answer": "فيه استنزاف من الناس. فيه استنزاف من الناس.",
            "practical_steps": [
                "حددي أكثر موقف يستنزفك.",
                "حددي أكثر موقف يستنزفك.",
                "اكتبي موقف واحد فقط.",
            ],
        },
        max_steps=4,
    )

    assert payload["understanding"].count("واضح") == 1
    assert payload["grounded_answer"].count("استنزاف") == 1
    assert len(payload["practical_steps"]) == 2


def test_crisis_input_triggers_immediate_crisis_mode():
    service = build_chat_service()

    reply = run_chat(
        service,
        conversation_id="crisis",
        user_message="مش عايزة أكمل",
        user_gender="female",
    )

    assert reply.safety.is_crisis is True
    assert reply.debug and reply.debug.response_mode == "CRISIS_MODE"
    assert "خطر" in reply.response.understanding or "الأمان" in reply.response.understanding
    assert not reply.recommended_videos
