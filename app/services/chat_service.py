from __future__ import annotations

import json
import re
from typing import Any

from app.models.schemas import (
    ChatResponse,
    DebugMeta,
    RecommendationsMeta,
    ResourceItem,
    RetrievedChunk,
    StructuredAnswer,
)
from app.services.answer_deduper import AnswerDeduper
from app.services.conversation_policy import (
    INTENT_RESOURCE_REQUEST,
    MODE_CHAT,
    MODE_CRISIS,
    MODE_GROUNDED,
    MODE_RESOURCE,
    ConversationPolicy,
)
from app.utils.text_utils import lexical_overlap_score, normalize_text


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
        conversation_policy: ConversationPolicy | None = None,
        answer_deduper: AnswerDeduper | None = None,
    ) -> None:
        self.settings = settings
        self.kb_service = kb_service
        self.vector_store = vector_store
        self.github_client = github_client
        self.safety_service = safety_service
        self.session_store = session_store
        self.fallback_service = fallback_service
        self.conversation_policy = conversation_policy or ConversationPolicy(settings)
        self.answer_deduper = answer_deduper or AnswerDeduper()

    async def chat(self, request) -> ChatResponse:
        response_style = self._resolve_response_style(request.response_style, request.user_gender)
        conversation_id = self.session_store.ensure_conversation(request.conversation_id)
        self.session_store.append_user_message(conversation_id, request.user_message)
        conversation_history = self.session_store.get_history(conversation_id)
        prior_turns = self.session_store.get_turn_number(conversation_id)
        next_turn_number = prior_turns + 1
        unlock_turn = request.recommendations_after_turn or self.settings.RECOMMENDATIONS_AFTER_TURN

        safety = self.safety_service.inspect(
            message=request.user_message,
            user_gender=request.user_gender,
            audience_mode=self.settings.TARGET_AUDIENCE,
        )
        policy = self.conversation_policy.decide(
            message=request.user_message,
            conversation_history=conversation_history,
            assistant_turns=prior_turns,
            response_style=response_style,
        )

        if safety.blocked and safety.is_crisis:
            structured = StructuredAnswer(**self._compose_crisis_response(response_style))
            turn_number = self.session_store.append_assistant_message(conversation_id, self._history_text(structured))
            return ChatResponse(
                source="safety_guard",
                conversation_id=conversation_id,
                turn_number=turn_number,
                mbti_type=request.mbti_type,
                safety=safety,
                response=structured,
                recommendations=RecommendationsMeta(
                    unlocked=False,
                    current_turn=turn_number,
                    unlock_turn=unlock_turn,
                    note="تم إيقاف الترشيحات لأن الحالة تتطلب أمانًا فوريًا.",
                ),
                debug=self._build_debug_meta(
                    enabled=request.debug or self.settings.DEBUG_CHAT_METADATA,
                    detected_intent="CRISIS_SIGNAL",
                    response_mode=MODE_CRISIS,
                    followup_question_reason=None,
                    issue_match_scores={},
                    topic_match_scores={},
                    recommendation_triggered=False,
                ),
            )

        if safety.blocked and request.user_gender != "female":
            structured = StructuredAnswer(
                understanding=self.safety_service.gender_restriction_response(),
                grounded_answer="تم إيقاف الرد بسبب إعدادات الجمهور المستهدف الحالية في المشروع.",
                practical_steps=[],
                support_note="يمكن تعديل TARGET_AUDIENCE و RESPONSE_STYLE من الإعدادات.",
            )
            turn_number = self.session_store.append_assistant_message(conversation_id, self._history_text(structured))
            return ChatResponse(
                source="safety_guard",
                conversation_id=conversation_id,
                turn_number=turn_number,
                mbti_type=request.mbti_type,
                safety=safety,
                response=structured,
                recommendations=RecommendationsMeta(
                    unlocked=False,
                    current_turn=turn_number,
                    unlock_turn=unlock_turn,
                    note="الترشيحات متوقفة بسبب إعدادات الجمهور المستهدف الحالية.",
                ),
                debug=self._build_debug_meta(
                    enabled=request.debug or self.settings.DEBUG_CHAT_METADATA,
                    detected_intent=policy.detected_intent,
                    response_mode=policy.response_mode,
                    followup_question_reason=policy.followup_question_reason,
                    issue_match_scores={},
                    topic_match_scores={},
                    recommendation_triggered=False,
                ),
            )

        if policy.response_mode == MODE_CHAT:
            structured_dict = self._compose_chat_mode_response(
                message=request.user_message,
                response_style=response_style,
                primary_emotion=policy.primary_emotion,
                detected_intent=policy.detected_intent,
                follow_up_question=policy.follow_up_question,
                choice_prompt=policy.choice_prompt,
            )
            structured = StructuredAnswer(**structured_dict)
            turn_number = self.session_store.append_assistant_message(conversation_id, self._history_text(structured))
            return ChatResponse(
                source="conversation_policy_chat",
                conversation_id=conversation_id,
                turn_number=turn_number,
                mbti_type=request.mbti_type,
                safety=safety,
                response=structured,
                recommendations=RecommendationsMeta(
                    unlocked=False,
                    current_turn=turn_number,
                    unlock_turn=unlock_turn,
                    note=f"الترشيحات ستظهر بداية من الرد رقم {unlock_turn} أو عند طلب موارد محددة بعد اتضاح الموضوع.",
                ),
                debug=self._build_debug_meta(
                    enabled=request.debug or self.settings.DEBUG_CHAT_METADATA,
                    detected_intent=policy.detected_intent,
                    response_mode=policy.response_mode,
                    followup_question_reason=policy.followup_question_reason,
                    issue_match_scores={},
                    topic_match_scores={},
                    recommendation_triggered=False,
                ),
            )

        overview = self.kb_service.get_mbti_overview(request.mbti_type)
        history_context = self.session_store.get_recent_user_messages(conversation_id, limit=3)
        retrieval_query = self._build_retrieval_query(
            user_message=request.user_message,
            recent_messages=history_context[:-1],
            mbti_type=request.mbti_type,
            primary_emotion=policy.primary_emotion,
            response_mode=policy.response_mode,
        )
        retrieved = await self.vector_store.search(
            query=retrieval_query,
            mbti_type=request.mbti_type,
            top_k=request.top_k,
            history_context=history_context[:-1],
            intent=policy.detected_intent,
            primary_emotion=policy.primary_emotion,
        )

        matched_issue_titles = self._extract_titles(retrieved, domain="mbti", chunk_types={"issue"})
        matched_topic_titles = self._extract_titles(
            retrieved,
            domain="emotion",
            chunk_types={"emotion_question", "emotion_topic"},
            field="topic_title",
        )
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

        recommendations_unlocked = bool(
            request.include_recommendations
            and (
                next_turn_number >= unlock_turn
                or (policy.detected_intent == INTENT_RESOURCE_REQUEST and policy.allow_immediate_resources)
            )
        )
        recommendation_query = self._build_resource_query(
            user_message=request.user_message,
            recent_messages=history_context[:-1],
            primary_emotion=policy.primary_emotion,
        )
        recommended_videos: list[dict] = []
        recommended_books: list[dict] = []
        recommended_podcasts: list[dict] = []
        if recommendations_unlocked:
            recommended_videos = self.kb_service.recommend_resources(
                mbti_type=request.mbti_type,
                query=recommendation_query,
                limit=request.max_videos,
                category="video",
                issue_titles=matched_issue_titles,
                topic_titles=matched_topic_titles,
                question_titles=matched_question_titles,
                require_url=True if request.recommendation_links_only else True,
            )
            recommended_books = self.kb_service.recommend_resources(
                mbti_type=request.mbti_type,
                query=recommendation_query,
                limit=request.max_books,
                category="book",
                issue_titles=matched_issue_titles,
                topic_titles=matched_topic_titles,
                question_titles=matched_question_titles,
                require_url=request.recommendation_links_only,
            )
            recommended_podcasts = self.kb_service.recommend_resources(
                mbti_type=request.mbti_type,
                query=recommendation_query,
                limit=request.max_podcasts,
                category="podcast",
                issue_titles=matched_issue_titles,
                topic_titles=matched_topic_titles,
                question_titles=matched_question_titles,
                require_url=True if request.recommendation_links_only else True,
            )

        if policy.response_mode == MODE_RESOURCE:
            structured_dict = self._compose_resource_mode_response(
                response_style=response_style,
                issue_title=primary_issue,
                topic_title=primary_topic,
                videos_count=len(recommended_videos),
                books_count=len(recommended_books),
                podcasts_count=len(recommended_podcasts),
                resources_unlocked=recommendations_unlocked,
            )
            source = "conversation_policy_resource"
        else:
            structured_dict = None
            if self.github_client.enabled and self.settings.ENABLE_GITHUB_CHAT_GENERATION:
                structured_dict = await self._try_llm_structured_answer(
                    request_message=request.user_message,
                    mbti_type=request.mbti_type,
                    overview=overview,
                    retrieved=retrieved,
                    advice_steps=advice_steps,
                    conversation_history=conversation_history,
                    response_style=response_style,
                    detected_intent=policy.detected_intent,
                    primary_emotion=policy.primary_emotion,
                    next_turn_number=next_turn_number,
                    unlock_turn=unlock_turn,
                )
            if not structured_dict:
                structured_dict = self._compose_grounded_support_response(
                    message=request.user_message,
                    response_style=response_style,
                    mbti_type=request.mbti_type,
                    overview=overview,
                    primary_issue=primary_issue,
                    primary_topic=primary_topic,
                    primary_question=primary_question,
                    emotion_entry=emotion_entry,
                    advice_steps=advice_steps,
                    primary_emotion=policy.primary_emotion,
                    next_turn_number=next_turn_number,
                    unlock_turn=unlock_turn,
                )
            source = (
                "retrieval_augmented_generation"
                if self.github_client.enabled and self.settings.ENABLE_GITHUB_CHAT_GENERATION
                else "retrieval_augmented_fallback"
            )

        max_steps = 2 if next_turn_number <= 2 else 4
        structured = StructuredAnswer(
            **self.answer_deduper.dedupe_structured(structured_dict, max_steps=max_steps)
        )
        turn_number = self.session_store.append_assistant_message(conversation_id, self._history_text(structured))
        recommendations_note = self._build_recommendations_note(
            recommendations_unlocked=recommendations_unlocked,
            unlock_turn=unlock_turn,
            resource_request=policy.detected_intent == INTENT_RESOURCE_REQUEST and policy.allow_immediate_resources,
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
            debug=self._build_debug_meta(
                enabled=request.debug or self.settings.DEBUG_CHAT_METADATA,
                detected_intent=policy.detected_intent,
                response_mode=policy.response_mode,
                followup_question_reason=policy.followup_question_reason,
                issue_match_scores=self._build_match_scores(retrieved, domain="mbti", chunk_types={"issue"}),
                topic_match_scores=self._build_match_scores(
                    retrieved,
                    domain="emotion",
                    chunk_types={"emotion_question", "emotion_topic"},
                    field="topic_title",
                ),
                recommendation_triggered=bool(recommended_videos or recommended_books or recommended_podcasts),
            ),
        )

    def _resolve_response_style(self, request_style: str, user_gender: str) -> str:
        if request_style and request_style != "config":
            return request_style
        style = (self.settings.RESPONSE_STYLE or "feminine").strip().lower()
        if style not in {"feminine", "neutral"}:
            style = "feminine"
        if self.settings.TARGET_AUDIENCE != "female_only" and user_gender != "female":
            return "neutral"
        return style

    def _compose_chat_mode_response(
        self,
        *,
        message: str,
        response_style: str,
        primary_emotion: str | None,
        detected_intent: str,
        follow_up_question: str | None,
        choice_prompt: str | None,
    ) -> dict[str, Any]:
        understanding = self._chat_acknowledgment(primary_emotion, response_style)
        if detected_intent == "ADVICE_REQUEST":
            grounded_answer = self._style_line(
                feminine="أقدر أساعدك، بس محتاجة أفهم الصورة أقرب الأول.",
                neutral="أقدر أساعدك، بس محتاج أفهم الصورة أقرب الأول.",
                response_style=response_style,
            )
        elif detected_intent == "SMALL_TALK_OR_GREETING":
            grounded_answer = self._style_line(
                feminine="أنا هنا معاكي.",
                neutral="أنا هنا معك.",
                response_style=response_style,
            )
        else:
            grounded_answer = self._style_line(
                feminine="أنا معاكي.",
                neutral="أنا هنا معك.",
                response_style=response_style,
            )
        return {
            "understanding": understanding,
            "mbti_connection": "",
            "grounded_answer": grounded_answer,
            "practical_steps": [],
            "follow_up_question": follow_up_question,
            "choice_prompt": choice_prompt,
            "support_note": "لو الإحساس وصل لخطر على سلامتك، الأولوية لطلب مساعدة فورية الآن.",
        }

    def _compose_grounded_support_response(
        self,
        *,
        message: str,
        response_style: str,
        mbti_type: str | None,
        overview: dict[str, Any],
        primary_issue: str | None,
        primary_topic: str | None,
        primary_question: str | None,
        emotion_entry: dict[str, Any] | None,
        advice_steps: list[str],
        primary_emotion: str | None,
        next_turn_number: int,
        unlock_turn: int,
    ) -> dict[str, Any]:
        understanding = self._compose_understanding(
            message=message,
            response_style=response_style,
            primary_issue=primary_issue,
            primary_emotion=primary_emotion,
        )
        grounded_answer = self._compose_grounded_answer(
            message=message,
            response_style=response_style,
            primary_issue=primary_issue,
            primary_topic=primary_topic,
            primary_question=primary_question,
            emotion_entry=emotion_entry,
            primary_emotion=primary_emotion,
        )
        mbti_connection = self._compose_mbti_connection(
            response_style=response_style,
            mbti_type=mbti_type,
            primary_issue=primary_issue,
            mbti_core_problems=overview.get("core_problems", []),
            message=message,
        )

        steps = self._select_steps(
            advice_steps=advice_steps,
            response_style=response_style,
            primary_emotion=primary_emotion,
            max_steps=2 if next_turn_number <= 2 else 4,
        )
        follow_up_question = self._compose_transition_question(
            response_style=response_style,
            allow_resources=next_turn_number >= unlock_turn,
        )
        return {
            "understanding": understanding,
            "mbti_connection": mbti_connection,
            "grounded_answer": grounded_answer,
            "practical_steps": steps,
            "follow_up_question": follow_up_question,
            "choice_prompt": None,
            "support_note": "ده دعم معلوماتي ومساندة أولية، مش تشخيص ولا بديل عن مختص لو الاحتياج أكبر.",
        }

    def _compose_resource_mode_response(
        self,
        *,
        response_style: str,
        issue_title: str | None,
        topic_title: str | None,
        videos_count: int,
        books_count: int,
        podcasts_count: int,
        resources_unlocked: bool,
    ) -> dict[str, Any]:
        if not resources_unlocked:
            return {
                "understanding": self._style_line(
                    feminine="أقدر أرشح لك موارد، لكن محتاجة أولًا أعرف الموضوع الأقرب.",
                    neutral="أقدر أرشح لك موارد، لكن محتاج أولًا أعرف الموضوع الأقرب.",
                    response_style=response_style,
                ),
                "mbti_connection": "",
                "grounded_answer": "لو تحدد/ي هل الموضوع حزن ولا قلق ولا استنزاف من الناس، الترشيح هيطلع أدق.",
                "practical_steps": [],
                "follow_up_question": "تحب/ي نحدد نوع الموضوع الأول؟",
                "choice_prompt": None,
                "support_note": "",
            }

        focus = issue_title or topic_title or "الموضوع اللي اتكلمنا عنه"
        counts = []
        if videos_count:
            counts.append(f"{videos_count} فيديو")
        if books_count:
            counts.append(f"{books_count} كتاب")
        if podcasts_count:
            counts.append(f"{podcasts_count} بودكاست")
        counts_text = "، ".join(counts) if counts else "مجموعة موارد"
        return {
            "understanding": self._style_line(
                feminine=f"رشحت لك موارد مرتبطة أكثر بـ {focus}.",
                neutral=f"رشحت لك موارد مرتبطة أكثر بـ {focus}.",
                response_style=response_style,
            ),
            "mbti_connection": "",
            "grounded_answer": self._style_line(
                feminine=f"اخترت لك {counts_text} لأنهم الأقرب للمشكلة الحالية، مش ترشيحات عامة وخلاص.",
                neutral=f"اخترت لك {counts_text} لأنهم الأقرب للمشكلة الحالية، مش ترشيحات عامة وخلاص.",
                response_style=response_style,
            ),
            "practical_steps": [],
            "follow_up_question": self._style_line(
                feminine="لو تحبي، بعد ما تشوفيهم أقدر أرتب لك الأهم منهم أو أشرح ليه اخترتهم.",
                neutral="لو تحب، بعد ما تشوفهم أقدر أرتب لك الأهم منهم أو أشرح ليه اخترتهم.",
                response_style=response_style,
            ),
            "choice_prompt": None,
            "support_note": "",
        }

    def _compose_crisis_response(self, response_style: str) -> dict[str, Any]:
        return {
            "understanding": self._style_line(
                feminine="واضح من كلامك إن فيه خطر مباشر، فالأولوية الآن للأمان.",
                neutral="واضح من كلامك إن فيه خطر مباشر، فالأولوية الآن للأمان.",
                response_style=response_style,
            ),
            "mbti_connection": "",
            "grounded_answer": self._style_line(
                feminine="محتاجاكي ما تفضليش لوحدك دلوقتي وتتواصلي فورًا مع شخص موثوق أو خدمة طوارئ/دعم أزمة في بلدك.",
                neutral="محتاجك ما تفضلش لوحدك دلوقتي وتتواصل فورًا مع شخص موثوق أو خدمة طوارئ/دعم أزمة في بلدك.",
                response_style=response_style,
            ),
            "practical_steps": self._style_steps(
                [
                    self._style_line(
                        feminine="اتواصلي الآن مع شخص قريب واطلبي منه يفضل معاكي أو يكلمك فورًا.",
                        neutral="تواصل الآن مع شخص قريب واطلب منه يفضل معك أو يكلمك فورًا.",
                        response_style=response_style,
                    ),
                    self._style_line(
                        feminine="لو فيه أي وسيلة ممكن تؤذي نفسك بيها، ابعديها وخلّي شخص تاني يحتفظ بها.",
                        neutral="لو فيه أي وسيلة ممكن تؤذي نفسك بها، ابعدها وخلي شخص تاني يحتفظ بها.",
                        response_style=response_style,
                    ),
                    self._style_line(
                        feminine="اطلبي مساعدة عاجلة الآن من طوارئ أو خط دعم أزمة أو مختص قريب.",
                        neutral="اطلب مساعدة عاجلة الآن من طوارئ أو خط دعم أزمة أو مختص قريب.",
                        response_style=response_style,
                    ),
                ],
                response_style,
            ),
            "follow_up_question": None,
            "choice_prompt": None,
            "support_note": "دي استجابة أمان عاجلة وليست جلسة دعم عادية.",
        }

    def _compose_understanding(
        self,
        *,
        message: str,
        response_style: str,
        primary_issue: str | None,
        primary_emotion: str | None,
    ) -> str:
        normalized = normalize_text(message)
        if "الناس" in normalized and any(word in normalized for word in ["هم", "مشاعر", "بتعب", "استنزاف", "شايل", "شايله", "بشيل"]):
            return self._style_line(
                feminine="فهمت. واضح إنك شايلة حمل مشاعر الناس وده مستنزفك.",
                neutral="فهمت. واضح إنك شايل حمل مشاعر الناس وده مستنزفك.",
                response_style=response_style,
            )
        if primary_emotion == "sadness" and any(word in normalized for word in ["من غير سبب", "بدون سبب", "مش عارفه السبب", "مش عارف السبب"]):
            return "فهمت. الحزن هنا شكله حاضر من غير سبب واحد مباشر وواضح."
        if primary_emotion == "anxiety":
            return "فهمت. اللي ظاهر إن القلق واخد مساحة كبيرة عندك."
        if primary_emotion == "burnout":
            return "فهمت. اللي واضح إن فيه استنزاف وضغط متراكمين عليك."
        if primary_issue:
            return f"فهمت. أقرب لب المشكلة هنا هو: {primary_issue}."
        return self._style_line(
            feminine="فهمت. واضح إن الموضوع تقيل عليكي ومش مجرد إحساس عابر.",
            neutral="فهمت. واضح إن الموضوع تقيل عليك ومش مجرد إحساس عابر.",
            response_style=response_style,
        )

    def _compose_grounded_answer(
        self,
        *,
        message: str,
        response_style: str,
        primary_issue: str | None,
        primary_topic: str | None,
        primary_question: str | None,
        emotion_entry: dict[str, Any] | None,
        primary_emotion: str | None,
    ) -> str:
        normalized = normalize_text(message)
        if "الناس" in normalized and any(word in normalized for word in ["هم", "بشيل", "شايل", "شايله", "استنزاف"]):
            return self._style_line(
                feminine="لما تفضلي شايلة هم الناس طول الوقت، حدودك بتتعب وبتحسي إنك مستنزفة حتى لو ما فيش موقف واحد واضح.",
                neutral="لما تفضل شايل هم الناس طول الوقت، حدودك بتتعب وبتحس إنك مستنزف حتى لو ما فيش موقف واحد واضح.",
                response_style=response_style,
            )
        if primary_emotion == "sadness" and any(word in normalized for word in ["من غير سبب", "بدون سبب", "مش عارفه السبب", "مش عارف السبب"]):
            return "أحيانًا الحزن اللي من غير سبب واضح بيكون تراكم ضغط أو مشاعر متأجلة، مش لازم يكون له سبب واحد مباشر عشان يكون حقيقي."
        if primary_emotion == "anxiety":
            return "القلق لما يفضل شغال في الخلفية بيخلّي الجسم والعقل في حالة استنفار، حتى لو السبب مش واضح طول الوقت."
        if primary_emotion == "burnout":
            return "الإرهاق النفسي غالبًا ما يكون من تراكم حمل داخلي أو مسؤوليات أو احتكاك مستمر بالناس، مش من ضعف فيك."
        anchor = ""
        if emotion_entry:
            anchor = (emotion_entry.get("answer_intro") or "").strip()
            if not anchor:
                details = emotion_entry.get("details") or []
                anchor = next((item.strip() for item in details if item.strip()), "")
        if anchor:
            return self._trim_sentence(anchor)
        if primary_question:
            return f"أقرب سؤال مشابه في المعرفة كان: {primary_question}، وده بيدعم إن الموضوع محتاج فهم أهدأ وخطوات أصغر."
        if primary_topic:
            return f"المحور الأقرب هنا هو {primary_topic}، لذلك الرد موجه للفهم العملي للمشكلة نفسها قبل أي تعميم."
        if primary_issue:
            return f"المحور القريب هنا هو {primary_issue}، والأهم إننا نتعامل معه كعبء نفسي قابل للفهم والتنظيم، مش كحكم ثابت عليك."
        return "اللي عندك يستحق التعامل معه بهدوء وبشكل مباشر، من غير تهويل ومن غير تجاهل."

    def _compose_mbti_connection(
        self,
        *,
        response_style: str,
        mbti_type: str | None,
        primary_issue: str | None,
        mbti_core_problems: list[str],
        message: str,
    ) -> str:
        if not mbti_type:
            return ""
        if primary_issue and lexical_overlap_score(message, primary_issue) >= 0.05:
            return self._style_line(
                feminine=f"ولو نمط {mbti_type} قريب منك، فممكن يفسر ميلًا عامًا لهذا النوع من الضغط، لكن مش كسبب مؤكد ولا تشخيص.",
                neutral=f"ولو نمط {mbti_type} قريب منك، فممكن يفسر ميلًا عامًا لهذا النوع من الضغط، لكن مش كسبب مؤكد ولا تشخيص.",
                response_style=response_style,
            )
        if mbti_core_problems:
            return self._style_line(
                feminine=f"{mbti_type} هنا مجرد خلفية مساندة لا أكثر؛ مثلاً قد يظهر معه ميل لمحاور مثل {', '.join(mbti_core_problems[:2])}، لكن ده مش تفسير نهائي ولا تشخيص، والاعتماد الأساسي على كلامك الحالي.",
                neutral=f"{mbti_type} هنا مجرد خلفية مساندة لا أكثر؛ مثلاً قد يظهر معه ميل لمحاور مثل {', '.join(mbti_core_problems[:2])}، لكن ده مش تفسير نهائي ولا تشخيص، والاعتماد الأساسي على كلامك الحالي.",
                response_style=response_style,
            )
        return ""

    def _compose_transition_question(self, *, response_style: str, allow_resources: bool) -> str:
        if allow_resources:
            return self._style_line(
                feminine="تحبي نكمل بخطوات عملية، نفهم ليه ده بيتكرر، ولا أرشح لك موارد مناسبة؟",
                neutral="تحب نكمل بخطوات عملية، نفهم ليه ده بيتكرر، ولا أرشح لك موارد مناسبة؟",
                response_style=response_style,
            )
        return self._style_line(
            feminine="تحبي نكمل بخطوتين عمليتين، ولا نفهم الأول ليه ده بيتكرر معاكي؟",
            neutral="تحب نكمل بخطوتين عمليتين، ولا نفهم الأول ليه ده بيتكرر معك؟",
            response_style=response_style,
        )

    def _chat_acknowledgment(self, primary_emotion: str | None, response_style: str) -> str:
        mapping = {
            "sadness": self._style_line(
                feminine="واضح إنك حاسة بحزن.",
                neutral="واضح إن في حزن حاضر بقوة.",
                response_style=response_style,
            ),
            "anxiety": self._style_line(
                feminine="واضح إن القلق عالي عندك.",
                neutral="واضح إن القلق عالي.",
                response_style=response_style,
            ),
            "burnout": self._style_line(
                feminine="واضح إنك مستنزفة وتعبانة.",
                neutral="واضح إن فيه استنزاف وتعب.",
                response_style=response_style,
            ),
            "self_blame": self._style_line(
                feminine="واضح إنك قاسية على نفسك شوية.",
                neutral="واضح إن فيه قسوة على النفس هنا.",
                response_style=response_style,
            ),
            "relationship_distress": self._style_line(
                feminine="واضح إن في وجع جاي من علاقة أو تعامل مع ناس.",
                neutral="واضح إن فيه وجع جاي من علاقة أو تعامل مع ناس.",
                response_style=response_style,
            ),
            "confusion": self._style_line(
                feminine="واضح إنك متلخبطة ومشاعرِك مش مستقرة دلوقتي.",
                neutral="واضح إن فيه لخبطة ومشاعر غير مستقرة دلوقتي.",
                response_style=response_style,
            ),
        }
        return mapping.get(
            primary_emotion,
            self._style_line(
                feminine="واضح إنك مش مرتاحة ومضغوطة.",
                neutral="واضح إن فيه ضيق وتعب.",
                response_style=response_style,
            ),
        )

    def _select_steps(
        self,
        *,
        advice_steps: list[str],
        response_style: str,
        primary_emotion: str | None,
        max_steps: int,
    ) -> list[str]:
        unique_steps = self._style_steps(advice_steps, response_style)[:max_steps]
        if unique_steps:
            return unique_steps
        defaults_map = {
            "sadness": [
                self._style_line(
                    feminine="سمّي الإحساس زي ما هو من غير ما تضغطي على نفسك تفسريه كله مرة واحدة.",
                    neutral="سمّ الإحساس زي ما هو من غير ما تضغط على نفسك تفسره كله مرة واحدة.",
                    response_style=response_style,
                ),
                self._style_line(
                    feminine="اكتبي أو قولي لنفسك: إيه أكتر وقت الحزن بيعلى فيه؟",
                    neutral="اكتب أو قل لنفسك: إيه أكتر وقت الحزن بيعلى فيه؟",
                    response_style=response_style,
                ),
            ],
            "anxiety": [
                "حددي هل القلق مربوط بموضوع واحد ولا منتشر في اليوم كله.",
                "خدي دقيقة تنفس أبطأ قبل ما تدخلي في دوامة التوقعات.",
            ],
            "burnout": [
                "قسمي الحمل الحالي إلى جزء عاجل وجزء ممكن يتأجل.",
                "اسألي نفسك: مين أو إيه أكثر شيء يستنزفك الآن؟",
            ],
        }
        return self._style_steps(defaults_map.get(primary_emotion, ["حددي أكثر نقطة ضاغطة الآن."]), response_style)[:max_steps]

    def _build_recommendations_note(self, *, recommendations_unlocked: bool, unlock_turn: int, resource_request: bool) -> str:
        if resource_request and recommendations_unlocked:
            return "تم فتح الترشيحات الآن لأنك طلبت موارد بشكل صريح وبعد توفر سياق كافٍ."
        if recommendations_unlocked:
            return f"الترشيحات مفتوحة من الرد رقم {unlock_turn}."
        return f"الترشيحات ستظهر بداية من الرد رقم {unlock_turn} حتى تركز الردود الأولى على الفهم والتنظيم قبل اقتراح الموارد."

    def _build_retrieval_query(
        self,
        *,
        user_message: str,
        recent_messages: list[str],
        mbti_type: str | None,
        primary_emotion: str | None,
        response_mode: str,
    ) -> str:
        parts = [user_message]
        if primary_emotion:
            parts.insert(0, primary_emotion)
        for msg in recent_messages[-2:]:
            if msg and msg != user_message:
                parts.append(msg)
        if mbti_type and response_mode != MODE_CHAT:
            parts.append(f"MBTI {mbti_type}")
        return " | ".join(part for part in parts if part)

    def _build_resource_query(self, *, user_message: str, recent_messages: list[str], primary_emotion: str | None) -> str:
        parts = [msg for msg in recent_messages[-2:] if msg]
        if primary_emotion:
            parts.insert(0, primary_emotion)
        parts.append(user_message)
        return " | ".join(dict.fromkeys(part for part in parts if part))

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

    def _build_match_scores(
        self,
        retrieved: list[dict],
        *,
        domain: str,
        chunk_types: set[str],
        field: str = "title",
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for chunk in retrieved:
            if chunk.get("domain") != domain or chunk.get("chunk_type") not in chunk_types:
                continue
            title = (chunk.get(field) or "").strip()
            if not title:
                continue
            scores[title] = max(scores.get(title, 0.0), float(chunk.get("score") or 0.0))
        return scores

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
        detected_intent: str,
        primary_emotion: str | None,
        next_turn_number: int,
        unlock_turn: int,
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

        style_note = "neutral conversational Arabic" if response_style == "neutral" else "warm Egyptian Arabic, feminine voice"
        system = (
            "أنت مساعد دعم نفسي معلوماتي يعمل بالاسترجاع من بيانات داخلية. "
            "أرجع JSON فقط بدون أي نص إضافي. "
            "تكلم بشكل طبيعي ومختصر، بدون نبرة علاجية جامدة. "
            "أي ربط بـ MBTI يجب أن يكون احتماليًا وقصيرًا أو يُترك فارغًا لو غير لازم."
        )
        user_prompt = {
            "task": "generate_retrieval_based_json_answer",
            "output_keys": [
                "understanding",
                "mbti_connection",
                "grounded_answer",
                "practical_steps",
                "follow_up_question",
                "choice_prompt",
                "support_note",
            ],
            "constraints": {
                "language": style_note,
                "intent": detected_intent,
                "primary_emotion": primary_emotion,
                "max_steps": 2 if next_turn_number <= 2 else 4,
                "no_diagnosis": True,
                "no_overclaim_grounding": True,
                "mbti_is_tentative": True,
                "avoid_repetition": True,
                "be_short_if_early_turn": next_turn_number <= 2,
                "offer_resources_only_if_allowed": next_turn_number >= unlock_turn,
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
                "practical_steps": self._style_steps([str(item).strip() for item in steps if str(item).strip()], response_style),
                "follow_up_question": str(payload.get("follow_up_question", "")).strip() or None,
                "choice_prompt": str(payload.get("choice_prompt", "")).strip() or None,
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
            return [step.strip() for step in steps if step.strip()]
        return [self.fallback_service._adapt_step(step.strip(), "neutral") for step in steps if step.strip()]

    def _style_line(self, *, feminine: str, neutral: str, response_style: str) -> str:
        return feminine if response_style == "feminine" else neutral

    def _trim_sentence(self, text: str, max_length: int = 220) -> str:
        clean = re.sub(r"\s+", " ", text).strip()
        if len(clean) <= max_length:
            return clean
        sentence = re.split(r"[.!؟]", clean, maxsplit=1)[0].strip()
        if 20 <= len(sentence) <= max_length:
            return sentence
        return clean[: max_length - 3].rstrip() + "..."

    def _history_text(self, structured: StructuredAnswer) -> str:
        parts = [
            structured.understanding,
            structured.grounded_answer,
            " ".join(structured.practical_steps[:2]) if structured.practical_steps else "",
            structured.follow_up_question or "",
        ]
        return " ".join(part.strip() for part in parts if part and part.strip())

    def _build_debug_meta(
        self,
        *,
        enabled: bool,
        detected_intent: str,
        response_mode: str,
        followup_question_reason: str | None,
        issue_match_scores: dict[str, float],
        topic_match_scores: dict[str, float],
        recommendation_triggered: bool,
    ) -> DebugMeta | None:
        if not enabled:
            return None
        return DebugMeta(
            detected_intent=detected_intent,
            response_mode=response_mode,
            followup_question_reason=followup_question_reason,
            issue_match_scores=issue_match_scores,
            topic_match_scores=topic_match_scores,
            recommendation_triggered=recommendation_triggered,
        )
