from __future__ import annotations

import json
from app.utils.text_utils import cosine_similarity, lexical_overlap_score


class VectorStore:
    def __init__(self, settings, github_client, kb_service) -> None:
        self.settings = settings
        self.github_client = github_client
        self.kb_service = kb_service
        self.vectors: dict[str, list[float]] = {}

    def _load_saved_index(self) -> None:
        if not self.settings.VECTOR_INDEX_PATH.exists():
            return
        payload = json.loads(self.settings.VECTOR_INDEX_PATH.read_text(encoding="utf-8"))
        self.vectors = payload.get("vectors", {})

    async def ensure_index(self, rebuild: bool = False) -> None:
        if self.vectors and not rebuild:
            return

        if self.settings.VECTOR_INDEX_PATH.exists() and not rebuild:
            self._load_saved_index()
            if self.vectors:
                return

        if not self.github_client.enabled:
            self.vectors = {}
            return

        chunks = self.kb_service.chunks
        if not chunks:
            return

        all_vectors: dict[str, list[float]] = {}
        batch_size = self.settings.EMBEDDING_BATCH_SIZE
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start:start + batch_size]
            batch_texts = [chunk["text"] for chunk in batch_chunks]
            batch_vectors = await self.github_client.embed_texts(batch_texts)
            for chunk, vector in zip(batch_chunks, batch_vectors):
                all_vectors[chunk["id"]] = vector

        self.vectors = all_vectors
        self.settings.VECTOR_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.settings.VECTOR_INDEX_PATH.write_text(
            json.dumps({"vectors": self.vectors}, ensure_ascii=False),
            encoding="utf-8",
        )

    async def search(self, query: str, mbti_type: str | None, top_k: int = 8) -> list[dict]:
        if not self.vectors and self.settings.VECTOR_INDEX_PATH.exists():
            self._load_saved_index()

        query_vector: list[float] | None = None
        if self.github_client.enabled:
            try:
                await self.ensure_index(rebuild=False)
                embedded = await self.github_client.embed_texts([query])
                query_vector = embedded[0]
            except Exception:
                query_vector = None

        scored: list[tuple[float, dict]] = []
        for chunk in self.kb_service.chunks:
            score = 0.0
            if chunk.get("domain") == "mbti":
                if mbti_type and chunk.get("mbti_type") == mbti_type:
                    score += 0.75
                elif mbti_type and chunk.get("mbti_type") != mbti_type:
                    score -= 0.08
            elif chunk.get("domain") == "emotion":
                score += 0.22

            if chunk.get("chunk_type") in {"issue", "emotion_question"}:
                score += 0.15
            elif chunk.get("chunk_type") in {"overview", "emotion_topic"}:
                score += 0.06

            score += lexical_overlap_score(query, chunk["text"]) * 1.8
            score += lexical_overlap_score(query, chunk.get("title", "")) * 0.9

            if query_vector is not None and chunk["id"] in self.vectors:
                score += cosine_similarity(query_vector, self.vectors[chunk["id"]]) * 2.2

            scored.append((score, chunk))

        scored.sort(key=lambda row: row[0], reverse=True)

        results: list[dict] = []
        for score, chunk in scored[:top_k]:
            results.append({
                "chunk_id": chunk["id"],
                "title": chunk["title"],
                "chunk_type": chunk["chunk_type"],
                "domain": chunk.get("domain", "unknown"),
                "mbti_type": chunk.get("mbti_type"),
                "topic_title": chunk.get("topic_title"),
                "score": round(score, 4),
                "text": chunk["text"],
            })
        return results
