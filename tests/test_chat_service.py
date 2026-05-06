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
    service = build_chat_service(RECOMMENDATIONS_AFTER_TURN=4)

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


def test_explicit_resource_request_before_turn_four_does_not_unlock_recommendations():
    service = build_chat_service(RECOMMENDATIONS_AFTER_TURN=4)
    conversation_id = "explicit-resource-early"

    run_chat(
        service,
        conversation_id=conversation_id,
        user_message="sad",
        user_gender="female",
    )
    reply = run_chat(
        service,
        conversation_id=conversation_id,
        user_message="رشحيلي فيديوهين",
        user_gender="female",
        max_videos=2,
    )

    assert reply.debug and reply.debug.detected_intent == "RESOURCE_REQUEST"
    assert reply.turn_number == 2
    assert reply.recommendations.unlocked is False
    assert not reply.recommended_videos


def test_explicit_resource_request_unlocks_recommendations_on_turn_four():
    service = build_chat_service(RECOMMENDATIONS_AFTER_TURN=4)
    conversation_id = "explicit-resource-turn4"

    run_chat(
        service,
        conversation_id=conversation_id,
        mbti_type="INFJ",
        user_message="sad",
        user_gender="female",
    )
    run_chat(
        service,
        conversation_id=conversation_id,
        mbti_type="INFJ",
        user_message="مش عارفة ليه",
        user_gender="female",
    )
    run_chat(
        service,
        conversation_id=conversation_id,
        mbti_type="INFJ",
        user_message="حاسّة إنه بقاله فترة",
        user_gender="female",
    )
    reply = run_chat(
        service,
        conversation_id=conversation_id,
        mbti_type="INFJ",
        user_message="طب ابعتيلي فيديو يفيدني",
        user_gender="female",
        max_videos=2,
    )

    assert reply.debug and reply.debug.detected_intent == "RESOURCE_REQUEST"
    assert reply.turn_number == 4
    assert reply.recommendations.unlocked is True
    assert reply.recommended_videos
    assert len(reply.recommended_videos) <= 2


def test_links_only_filters_out_null_link_recommendations():
    service = build_chat_service(RECOMMENDATIONS_AFTER_TURN=4)
    conversation_id = "links-only"

    run_chat(
        service,
        conversation_id=conversation_id,
        mbti_type="INFJ",
        user_message="sad",
        user_gender="female",
    )
    run_chat(
        service,
        conversation_id=conversation_id,
        mbti_type="INFJ",
        user_message="مش عارفة ليه",
        user_gender="female",
    )
    run_chat(
        service,
        conversation_id=conversation_id,
        mbti_type="INFJ",
        user_message="حاسّة إنه بقاله فترة",
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


def test_turn_four_with_enough_context_moves_forward_and_can_include_links():
    service = build_chat_service(RECOMMENDATIONS_AFTER_TURN=4)
    conversation_id = "turn-four-forward"

    run_chat(
        service,
        conversation_id=conversation_id,
        mbti_type="INFJ",
        user_message="sad",
        user_gender="female",
    )
    run_chat(
        service,
        conversation_id=conversation_id,
        mbti_type="INFJ",
        user_message="مش عارفة ليه",
        user_gender="female",
    )
    run_chat(
        service,
        conversation_id=conversation_id,
        mbti_type="INFJ",
        user_message="حاسّة إنه بقاله فترة",
        user_gender="female",
    )
    reply = run_chat(
        service,
        conversation_id=conversation_id,
        mbti_type="INFJ",
        user_message="أعمل إيه؟",
        user_gender="female",
        max_videos=2,
    )

    assert reply.turn_number == 4
    assert reply.debug and reply.debug.response_mode in {"GROUNDED_SUPPORT_MODE", "RESOURCE_MODE"}
    assert reply.response.practical_steps
    assert reply.response.follow_up_question
    assert reply.response.follow_up_question.count("؟") <= 1


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


def test_exact_repeated_short_message_is_detected():
    service = build_chat_service()
    conversation_id = "repeat-detected"

    run_chat(service, conversation_id=conversation_id, user_message="sad", user_gender="female")
    reply = run_chat(service, conversation_id=conversation_id, user_message="sad", user_gender="female")

    assert reply.debug
    assert reply.debug.exact_repetition_count == 2
    assert reply.debug.repeated_meaning_count == 2
    assert reply.safety.is_crisis is False


def test_second_repeated_message_does_not_get_same_reply():
    service = build_chat_service()
    conversation_id = "repeat-varied-reply"

    first = run_chat(service, conversation_id=conversation_id, user_message="sad", user_gender="female")
    second = run_chat(service, conversation_id=conversation_id, user_message="sad", user_gender="female")

    assert first.response.follow_up_question != second.response.follow_up_question
    assert first.response.understanding != second.response.understanding
    assert second.debug and second.debug.repeated_meaning_count == 2


def test_third_repetition_shifts_to_guided_choice_question():
    service = build_chat_service()
    conversation_id = "repeat-guided-choice"

    run_chat(service, conversation_id=conversation_id, user_message="sad", user_gender="female")
    run_chat(service, conversation_id=conversation_id, user_message="sad", user_gender="female")
    third = run_chat(service, conversation_id=conversation_id, user_message="sad", user_gender="female")

    combined = " ".join(filter(None, [third.response.follow_up_question, third.response.choice_prompt]))
    assert third.debug and third.debug.repeated_meaning_count >= 3
    assert "موقف" in combined
    assert "بدون سبب" in combined or "ضغط" in combined


def test_loop_detection_changes_strategy_after_repeated_user_input():
    service = build_chat_service()
    conversation_id = "repeat-loop"

    run_chat(service, conversation_id=conversation_id, user_message="sad", user_gender="female")
    run_chat(service, conversation_id=conversation_id, user_message="sad", user_gender="female")
    run_chat(service, conversation_id=conversation_id, user_message="sad", user_gender="female")
    fourth = run_chat(service, conversation_id=conversation_id, user_message="sad", user_gender="female")

    assert fourth.debug and fourth.debug.loop_detected is True
    assert fourth.debug.response_mode == "GROUNDED_SUPPORT_MODE"
    assert fourth.response.practical_steps
    assert "خطوة خطوة" in (fourth.response.follow_up_question or "")


def test_repeated_sad_does_not_trigger_crisis_mode():
    service = build_chat_service()
    conversation_id = "repeat-not-crisis"

    for _ in range(4):
        reply = run_chat(service, conversation_id=conversation_id, user_message="sad", user_gender="female")

    assert reply.safety.is_crisis is False
    assert reply.debug and reply.debug.response_mode != "CRISIS_MODE"


def test_repetition_memory_resets_when_user_adds_meaningful_detail():
    service = build_chat_service()
    conversation_id = "repeat-reset"

    run_chat(service, conversation_id=conversation_id, user_message="sad", user_gender="female")
    run_chat(service, conversation_id=conversation_id, user_message="sad", user_gender="female")
    reply = run_chat(
        service,
        conversation_id=conversation_id,
        user_message="sad because my breakup still hurts",
        user_gender="female",
    )

    assert reply.debug
    assert reply.debug.repeated_meaning_count == 1
    assert reply.debug.exact_repetition_count == 1
    assert reply.debug.loop_detected is False


def test_near_duplicate_inputs_count_as_repeated_meaning():
    service = build_chat_service()
    conversation_id = "repeat-near-duplicate"

    run_chat(service, conversation_id=conversation_id, user_message="sad", user_gender="female")
    reply = run_chat(service, conversation_id=conversation_id, user_message="I feel sad", user_gender="female")

    assert reply.debug
    assert reply.debug.repeated_meaning_count == 2
    assert reply.debug.semantic_repetition_count == 2
