from __future__ import annotations

import re

from app.utils.text_utils import lexical_overlap_score, normalize_text


class AnswerDeduper:
    def dedupe_structured(self, payload: dict, *, max_steps: int = 4) -> dict:
        payload = dict(payload)
        payload["grounded_answer"] = self._dedupe_sentences(payload.get("grounded_answer", ""))
        payload["understanding"] = self._dedupe_sentences(payload.get("understanding", ""))
        payload["mbti_connection"] = self._dedupe_sentences(payload.get("mbti_connection", ""))
        payload["practical_steps"] = self._dedupe_items(payload.get("practical_steps") or [], max_items=max_steps)
        return payload

    def _dedupe_sentences(self, text: str) -> str:
        if not text:
            return ""
        parts = [part.strip() for part in re.split(r"(?<=[.!؟])\s+|\n+", text) if part.strip()]
        kept: list[str] = []
        for part in parts:
            normalized = normalize_text(part)
            if not normalized:
                continue
            if any(lexical_overlap_score(normalized, normalize_text(existing)) >= 0.72 for existing in kept):
                continue
            kept.append(part)
        return " ".join(kept).strip()

    def _dedupe_items(self, items: list[str], *, max_items: int) -> list[str]:
        kept: list[str] = []
        for item in items:
            clean = item.strip()
            normalized = normalize_text(clean)
            if not clean or not normalized:
                continue
            if any(lexical_overlap_score(normalized, normalize_text(existing)) >= 0.72 for existing in kept):
                continue
            kept.append(clean)
            if len(kept) >= max_items:
                break
        return kept
