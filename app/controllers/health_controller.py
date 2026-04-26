from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request):
    container = request.app.state.container
    kb = container.kb_service
    return {
        "status": "ok",
        "mbti_types": len(kb.knowledge.get("mbti", {})),
        "emotion_topics": len(kb.knowledge.get("emotion", {})),
        "chunks": len(kb.chunks),
        "github_embeddings_enabled": bool(container.github_client.enabled),
    }
