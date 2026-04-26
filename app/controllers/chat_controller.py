from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.controllers.deps import get_chat_service, get_kb_service, get_vector_store
from app.models.schemas import ChatRequest, ChatResponse, IndexStatus, MBTIOverview, TopicOverview

router = APIRouter(tags=["chatbot"])


class ResetConversationRequest(BaseModel):
    conversation_id: str


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="رد production-ready يعتمد على MBTI + الداتا العامة، مع فتح الترشيحات بعد عدد ردود محدد",
)
async def chat(payload: ChatRequest, chat_service=Depends(get_chat_service)):
    return await chat_service.chat(payload)


@router.post("/conversations/reset", summary="مسح ذاكرة Conversation واحدة")
async def reset_conversation(payload: ResetConversationRequest, chat_service=Depends(get_chat_service)):
    chat_service.session_store.clear(payload.conversation_id)
    return {"status": "cleared", "conversation_id": payload.conversation_id}


@router.get("/mbti", response_model=list[MBTIOverview], summary="عرض الأنماط والمشكلات الجوهرية والنتائج")
async def list_mbti(kb_service=Depends(get_kb_service)):
    return [
        MBTIOverview(
            mbti_type=value["mbti_type"],
            core_problems=value.get("core_problems", []),
            consequences=value.get("consequences", []),
        )
        for value in kb_service.knowledge.get("mbti", {}).values()
    ]


@router.get("/topics", response_model=list[TopicOverview], summary="عرض التوبكس المستخرجة من ملف المشاعر العام")
async def list_topics(kb_service=Depends(get_kb_service)):
    return [
        TopicOverview(
            topic_title=value["topic_title"],
            questions_count=len(value.get("questions", [])),
            resources_count=len(value.get("resources", [])),
        )
        for value in kb_service.knowledge.get("emotion", {}).values()
    ]


@router.get("/resources/{mbti_type}", summary="كل الموارد المستخرجة لنمط محدد مع موارد عامة أيضًا")
async def mbti_resources(mbti_type: str, kb_service=Depends(get_kb_service)):
    mbti_type = mbti_type.upper().strip()
    return {
        "mbti_type": mbti_type,
        "videos": kb_service.recommend_resources(mbti_type, query="", limit=50, category="video", require_url=True),
        "books": kb_service.recommend_resources(mbti_type, query="", limit=50, category="book", require_url=False),
        "podcasts": kb_service.recommend_resources(mbti_type, query="", limit=50, category="podcast", require_url=True),
        "all_resources": kb_service.recommend_resources(mbti_type, query="", limit=200),
    }


@router.post("/index/rebuild", response_model=IndexStatus, summary="إعادة parsing للمصدرين وبناء الـ embeddings index")
async def rebuild_index(vector_store=Depends(get_vector_store), kb_service=Depends(get_kb_service)):
    kb_service.rebuild_from_sources()
    await vector_store.ensure_index(rebuild=True)
    return IndexStatus(
        status="rebuilt",
        chunks_count=len(kb_service.chunks),
        embeddings_count=len(vector_store.vectors),
        used_provider="github_models" if vector_store.vectors else "lexical_only",
    )


@router.get("/retrieve/debug", summary="فحص نتائج الاسترجاع داخليًا")
async def retrieval_debug(query: str, mbti_type: str, top_k: int = 8, vector_store=Depends(get_vector_store)):
    mbti_type = mbti_type.upper().strip()
    return await vector_store.search(query=query, mbti_type=mbti_type, top_k=top_k)
