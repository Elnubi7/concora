from __future__ import annotations

from app.services.chat_service import ChatService
from app.services.conversation_policy import ConversationPolicy
from app.services.answer_deduper import AnswerDeduper
from app.services.fallback_service import FallbackService
from app.services.github_models_client import GitHubModelsClient
from app.services.knowledge_service import KnowledgeBaseService
from app.services.safety_service import SafetyService
from app.services.session_store import SessionStore
from app.services.vector_store import VectorStore


class AppContainer:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.kb_service = KnowledgeBaseService(settings)
        self.github_client = GitHubModelsClient(settings)
        self.vector_store = VectorStore(settings, self.github_client, self.kb_service)
        self.safety_service = SafetyService()
        self.session_store = SessionStore(max_turns=settings.HISTORY_TURNS_TO_KEEP)
        self.fallback_service = FallbackService()
        self.conversation_policy = ConversationPolicy(settings)
        self.answer_deduper = AnswerDeduper()
        self.chat_service = ChatService(
            settings=settings,
            kb_service=self.kb_service,
            vector_store=self.vector_store,
            github_client=self.github_client,
            safety_service=self.safety_service,
            session_store=self.session_store,
            fallback_service=self.fallback_service,
            conversation_policy=self.conversation_policy,
            answer_deduper=self.answer_deduper,
        )

    async def initialize(self) -> None:
        self.kb_service.load()
        if self.settings.AUTO_BUILD_EMBEDDINGS_ON_STARTUP and self.github_client.enabled:
            try:
                await self.vector_store.ensure_index(rebuild=False)
            except Exception:
                # Do not fail app startup if embedding prebuild fails.
                pass
