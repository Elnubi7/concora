from app.core.config import get_settings
from app.services.fallback_service import FallbackService
from app.services.knowledge_service import KnowledgeBaseService
from app.services.session_store import SessionStore


def test_rebuild_loads_both_sources():
    settings = get_settings()
    service = KnowledgeBaseService(settings)
    service.rebuild_from_sources()

    assert len(service.knowledge["mbti"]) == 16
    assert service.knowledge["emotion"]
    assert "INFJ" in service.knowledge["mbti"]


def test_chunks_created_from_both_sources():
    settings = get_settings()
    service = KnowledgeBaseService(settings)
    service.rebuild_from_sources()

    assert len(service.chunks) > 100
    assert any(chunk["domain"] == "mbti" for chunk in service.chunks)
    assert any(chunk["domain"] == "emotion" for chunk in service.chunks)


def test_resources_keep_title_and_url_when_present():
    settings = get_settings()
    service = KnowledgeBaseService(settings)
    service.rebuild_from_sources()

    videos = service.recommend_resources(
        "INFJ",
        "استنزاف وعلاقات وحدود",
        limit=20,
        category="video",
        issue_titles=["امتصاص مشاعر الآخرين + استنزاف"],
        topic_titles=["INFJ"],
        require_url=True,
    )
    assert any(item.get("url") and item["category"] == "video" and item["title"] for item in videos)
    assert all(item.get("source_domain") for item in videos)


def test_book_recommendations_exclude_video_summaries():
    settings = get_settings()
    service = KnowledgeBaseService(settings)
    service.rebuild_from_sources()

    books = service.recommend_resources(
        "INFP",
        "النقد والحساسية",
        limit=20,
        category="book",
        issue_titles=["تضخيم المشاعر الذاتية (emotional flooding)"],
        topic_titles=["INFP"],
        require_url=False,
    )
    assert books
    assert all("youtube" not in (item.get("source_domain") or "") for item in books)
    assert all("شرح كتاب" not in item["title"] and "تلخيص كتاب" not in item["title"] for item in books)


def test_fallback_softens_mbti_claims_and_supports_neutral_style():
    payload = FallbackService().compose_structured(
        mbti_type="INFJ",
        mbti_issue="امتصاص مشاعر الآخرين + استنزاف",
        mbti_core_problems=["امتصاص مشاعر الآخرين", "المثالية", "كبت المشاعر"],
        generic_topic="الحزن",
        generic_question="أنا ليه بحس بالحزن من غير سبب واضح؟",
        generic_anchor="في البيانات العامة ظهر محور التراكم الشعوري.",
        advice_steps=["حددي الإحساس الأساسي", "اكتبي اللي حاصل"],
        response_style="neutral",
    )
    assert "قد يساعد" in payload["mbti_connection"]
    assert "ليس تفسيرًا قطعيًا" in payload["mbti_connection"]
    assert payload["practical_steps"][0].startswith("تحديد")


def test_emotion_topics_parsed():
    settings = get_settings()
    service = KnowledgeBaseService(settings)
    service.rebuild_from_sources()

    titles = list(service.knowledge["emotion"].keys())
    assert any("الحزن" in title for title in titles)


def test_session_store_generates_conversation_and_counts_turns():
    store = SessionStore(max_turns=4)
    conv = store.ensure_conversation(None)
    store.append_user_message(conv, "رسالة أولى")
    turn = store.append_assistant_message(conv, "رد أول")
    assert conv.startswith("conv_")
    assert turn == 1
