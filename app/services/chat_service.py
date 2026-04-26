from __future__ import annotations

import json
import re
from typing import Any

from app.models.schemas import (
    ChatResponse,
    RecommendationsMeta,
    ResourceItem,
    RetrievedChunk,
    StructuredAnswer,
)


class ChatService:
    def __init__(
        self,
        settings,
        kb_service,
        vector_store,
        github_client,
        safety_service,
        session_store,
        fallback_service,
    ) -> None:
        self.settings = settings
        self.kb_service = kb_service
        self.vector_store = vector_store
        self.github_client = github_client
        self.safety_service = safety_service
        self.session_store = session_store
        self.fallback_service = fallback_service

    async def chat(self, request) -> ChatResponse:
        response_style = self._resolve_response_style(request.response_style)
        conversation_id = self.session_store.ensure_conversation(request.conversation_id)
        self.session_store.append_user_message(conversation_id, request.user_message)
        unlock_turn = request.recommendations_after_turn or self.settings.RECOMMENDATIONS_AFTER_TURN

        safety = self.safety_service.inspect(
            message=request.user_message,
            user_gender=request.user_gender,
            audience_mode=self.settings.TARGET_AUDIENCE,
        )

        if safety.blocked and safety.is_crisis:
            crisis = StructuredAnswer(
                understanding="الرسالة فيها مؤشرات خطر مباشر، فمش مناسب أكمل نصيحة عادية هنا.",
                mbti_connection="في الحالة دي الأولوية للأمان الفوري، وليس لتحليل MBTI.",
                grounded_answer="الأولوية الآن هي عدم البقاء وحيدًا وطلب مساعدة فورية من شخص موثوق أو مختص/خدمة طوارئ.",
                practical_steps=self._style_steps([
                    "اتواصلي فورًا مع شخص قريب وموثوق يكون موجودًا الآن.",
                    "اطلبي مساعدة عاجلة من مختص/خدمة طوارئ في بلدك حالًا.",
                    "لو في شيء ممكن استخدامه في إيذاء النفس، أبعديه فورًا واطلبي من شخص آخر الاحتفاظ به.",
                ], response_style),
                support_note="دي استجابة أمان عاجلة وليست تشخيصًا أو علاجًا.",
            )
            turn_number = self.session_store.append_assistant_message(conversation_id, crisis.grounded_answer)
            return ChatResponse(
                source="safety_guard",
                conversation_id=conversation_id,
                turn_number=turn_number,
                mbti_type=request.mbti_type,
                safety=safety,
                response=crisis,
                recommendations=RecommendationsMeta(
                    unlocked=False,
                    current_turn=turn_number,
                    unlock_turn=unlock_turn,
                    note="تم إيقاف الترشيحات لأن الحالة تتطلب أمانًا فوريًا.",
                ),
            )

        if safety.blocked and request.user_gender != "female":
            blocked = StructuredAnswer(
                understanding=self.safety_service.gender_restriction_response(),
                mbti_connection="الخدمة الحالية مهيأة لفئة محددة حسب الإعدادات.",
                grounded_answer="تم إيقاف الرد بسبب إعدادات الجمهور المستهدف الحالية في المشروع.",
                practical_steps=[],
                support_note="يمكن تعديل TARGET_AUDIENCE و RESPONSE_STYLE من الإعدادات.",
            )
            turn_number = self.session_store.append_assistant_message(conversation_id, blocked.grounded_answer)
            return ChatResponse(
                source="safety_guard",
                conversation_id=conversation_id,
                turn_number=turn_number,
                mbti_type=request.mbti_type,
                safety=safety,
                response=blocked,
                recommendations=RecommendationsMeta(
                    unlocked=False,
                    current_turn=turn_number,
                    unlock_turn=unlock_turn,
                    note="الترشيحات متوقفة بسبب إعدادات الجمهور المستهدف الحالية.",
                ),
            )

        overview = self.kb_service.get_mbti_overview(request.mbti_type)
        history_context = self.session_store.get_recent_user_messages(conversation_id, limit=2)
        retrieval_query = self._build_retrieval_query(request.user_message, history_context[:-1], request.mbti_type)
        retrieved = await self.vector_store.search(
            query=retrieval_query,
            mbti_type=request.mbti_type,
            top_k=request.top_k,
        )

        matched_issue_titles = self._extract_titles(retrieved, domain="mbti", chunk_types={"issue"})
        matched_topic_titles = self._extract_titles(retrieved, domain="emotion", chunk_types={"emotion_question", "emotion_topic"}, field="topic_title")
        matched_question_titles = self._extract_titles(retrieved, domain="emotion", chunk_types={"emotion_question"})

        primary_issue = matched_issue_titles[0] if matched_issue_titles else None
        primary_topic = matched_topic_titles[0] if matched_topic_titles else None
        primary_question = matched_question_titles[0] if matched_question_titles else None
        emotion_entry = self.kb_service.get_emotion_entry(topic_title=primary_topic, question_title=primary_question)

        advice_steps = self.kb_service.merge_advice(
            mbti_type=request.mbti_type,
            issue_title=primary_issue,
            topic_title=primary_topic,
            question_title=primary_question,
        )

        structured_dict = None
        if self.github_client.enabled and self.settings.ENABLE_GITHUB_CHAT_GENERATION:
            structured_dict = await self._try_llm_structured_answer(
                request_message=request.user_message,
                mbti_type=request.mbti_type,
                overview=overview,
                retrieved=retrieved,
                advice_steps=advice_steps,
                conversation_history=self.session_store.get_history(conversation_id),
                response_style=response_style,
            )

        if not structured_dict:
            structured_dict = self.fallback_service.compose_structured(
                mbti_type=request.mbti_type,
                mbti_issue=primary_issue,
                mbti_core_problems=overview.get("core_problems", []),
                generic_topic=primary_topic,
                generic_question=(emotion_entry or {}).get("question") if emotion_entry else primary_question,
                generic_anchor=(emotion_entry or {}).get("answer_intro") if emotion_entry else None,
                advice_steps=advice_steps,
                response_style=response_style,
            )

        structured = StructuredAnswer(**structured_dict)
        turn_number = self.session_store.append_assistant_message(conversation_id, structured.grounded_answer)
        recommendations_unlocked = bool(request.include_recommendations and turn_number >= unlock_turn)

        recommended_videos: list[dict] = []
        recommended_books: list[dict] = []
        recommended_podcasts: list[dict] = []
        if recommendations_unlocked:
            recommended_videos = self.kb_service.recommend_resources(
                mbti_type=request.mbti_type,
                query=request.user_message,
                limit=request.max_videos,
                category="video",
                issue_titles=matched_issue_titles,
                topic_titles=matched_topic_titles,
                question_titles=matched_question_titles,
                require_url=True if request.recommendation_links_only else True,
            )
            recommended_books = self.kb_service.recommend_resources(
                mbti_type=request.mbti_type,
                query=request.user_message,
                limit=request.max_books,
                category="book",
                issue_titles=matched_issue_titles,
                topic_titles=matched_topic_titles,
                question_titles=matched_question_titles,
                require_url=request.recommendation_links_only,
            )
            recommended_podcasts = self.kb_service.recommend_resources(
                mbti_type=request.mbti_type,
                query=request.user_message,
                limit=request.max_podcasts,
                category="podcast",
                issue_titles=matched_issue_titles,
                topic_titles=matched_topic_titles,
                question_titles=matched_question_titles,
                require_url=True if request.recommendation_links_only else True,
            )

        recommendations_note = (
            f"الترشيحات مفتوحة من الرد رقم {unlock_turn}."
            if recommendations_unlocked
            else f"الترشيحات ستظهر بداية من الرد رقم {unlock_turn} حتى تركز الردود الأولى على الفهم والتنظيم قبل اقتراح الموارد."
        )
        source = (
            "retrieval_augmented_generation"
            if self.github_client.enabled and self.settings.ENABLE_GITHUB_CHAT_GENERATION
            else "retrieval_augmented_fallback"
        )

        return ChatResponse(
            source=source,
            conversation_id=conversation_id,
            turn_number=turn_number,
            mbti_type=request.mbti_type,
            matched_issue_titles=matched_issue_titles,
            matched_topic_titles=matched_topic_titles,
            response=structured,
            recommendations=RecommendationsMeta(
                unlocked=recommendations_unlocked,
                current_turn=turn_number,
                unlock_turn=unlock_turn,
                note=recommendations_note,
            ),
            recommended_videos=[ResourceItem(**item) for item in recommended_videos],
            recommended_books=[ResourceItem(**item) for item in recommended_books],
            recommended_podcasts=[ResourceItem(**item) for item in recommended_podcasts],
            safety=safety,
            retrieved_chunks=[RetrievedChunk(**chunk) for chunk in retrieved],
        )

    def _resolve_response_style(self, request_style: str) -> str:
        if request_style and request_style != "config":
            return request_style
        style = (self.settings.RESPONSE_STYLE or "feminine").strip().lower()
        return style if style in {"feminine", "neutral"} else "feminine"

    def _build_retrieval_query(self, user_message: str, recent_messages: list[str], mbti_type: str | None) -> str:
        parts = [user_message]
        if mbti_type:
            parts.insert(0, f"MBTI {mbti_type}")
        for msg in recent_messages[-2:]:
            if msg and msg != user_message:
                parts.append(msg)
        return " | ".join(parts)

    def _extract_titles(self, retrieved: list[dict], *, domain: str, chunk_types: set[str], field: str = "title") -> list[str]:
        seen: set[str] = set()
        items: list[str] = []
        for chunk in retrieved:
            if chunk.get("domain") != domain or chunk.get("chunk_type") not in chunk_types:
                continue
            value = (chunk.get(field) or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            items.append(value)
        return items

    async def _try_llm_structured_answer(
        self,
        *,
        request_message: str,
        mbti_type: str | None,
        overview: dict[str, Any],
        retrieved: list[dict[str, Any]],
        advice_steps: list[str],
        conversation_history: list[dict[str, str]],
        response_style: str,
    ) -> dict[str, Any] | None:
        history_lines = []
        for item in conversation_history[-4:]:
            role = "المستخدم" if item.get("role") == "user" else "المساعد"
            history_lines.append(f"{role}: {item.get('content', '')}")

        retrieved_context = []
        for chunk in retrieved[:8]:
            retrieved_context.append(
                {
                    "domain": chunk.get("domain"),
                    "title": chunk.get("title"),
                    "topic_title": chunk.get("topic_title"),
                    "mbti_type": chunk.get("mbti_type"),
                    "text": chunk.get("text"),
                }
            )

        style_note = "neutral Arabic" if response_style == "neutral" else "Arabic Egyptian, feminine voice"
        system = (
            "أنت مساعد دعم نفسي معلوماتي يعمل بالاسترجاع من بيانات داخلية. "
            "أرجع JSON فقط بدون أي نص إضافي. "
            "التزم بالسياق المسترجع فقط. لا تشخّص. "
            "أي ربط بـ MBTI يجب أن يكون احتماليًا ولتفسير ميل عام فقط، وليس سببًا مؤكدًا."
        )
        user_prompt = {
            "task": "generate_retrieval_based_json_answer",
            "output_keys": ["understanding", "mbti_connection", "grounded_answer", "practical_steps", "support_note"],
            "constraints": {
                "language": style_note,
                "max_steps": 4,
                "no_diagnosis": True,
                "no_overclaim_grounding": True,
                "mbti_is_tentative": True,
                "avoid_repetition": True,
            },
            "user_message": request_message,
            "mbti_type": mbti_type,
            "mbti_overview": {
                "core_problems": overview.get("core_problems", []),
                "consequences": overview.get("consequences", []),
            },
            "history": history_lines,
            "retrieved_context": retrieved_context,
            "suggested_steps_pool": advice_steps[:6],
        }

        try:
            content = await self.github_client.chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
                ],
                temperature=0.1,
                max_tokens=self.settings.MAX_RESPONSE_TOKENS,
            )
            payload = self._extract_json(content)
            if not isinstance(payload, dict):
                return None
            steps = payload.get("practical_steps") or []
            if not isinstance(steps, list):
                steps = []
            return {
                "understanding": str(payload.get("understanding", "")).strip(),
                "mbti_connection": str(payload.get("mbti_connection", "")).strip(),
                "grounded_answer": str(payload.get("grounded_answer", "")).strip(),
                "practical_steps": self._style_steps([str(item).strip() for item in steps if str(item).strip()][:4], response_style),
                "support_note": str(payload.get("support_note", "")).strip(),
            }
        except Exception:
            return None

    def _extract_json(self, text: str) -> Any:
        text = text.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
        candidate = fence_match.group(1) if fence_match else text
        try:
            return json.loads(candidate)
        except Exception:
            match = re.search(r"(\{.*\})", candidate, flags=re.S)
            if match:
                return json.loads(match.group(1))
            raise

    def _style_steps(self, steps: list[str], response_style: str) -> list[str]:
        if response_style != "neutral":
            return steps
        return [self.fallback_service._adapt_step(step, "neutral") for step in steps]
