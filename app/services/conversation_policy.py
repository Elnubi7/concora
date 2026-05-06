from __future__ import annotations

from dataclasses import dataclass

from app.services.repetition_detector import RepetitionState
from app.utils.text_utils import normalize_text, tokenize


INTENT_SHORT_EMOTIONAL_SIGNAL = "SHORT_EMOTIONAL_SIGNAL"
INTENT_VAGUE_DISTRESS = "VAGUE_DISTRESS"
INTENT_CLEAR_PROBLEM_STATEMENT = "CLEAR_PROBLEM_STATEMENT"
INTENT_ADVICE_REQUEST = "ADVICE_REQUEST"
INTENT_RESOURCE_REQUEST = "RESOURCE_REQUEST"
INTENT_CRISIS_SIGNAL = "CRISIS_SIGNAL"
INTENT_SMALL_TALK_OR_GREETING = "SMALL_TALK_OR_GREETING"

MODE_CHAT = "CHAT_MODE"
MODE_GROUNDED = "GROUNDED_SUPPORT_MODE"
MODE_RESOURCE = "RESOURCE_MODE"
MODE_CRISIS = "CRISIS_MODE"


@dataclass
class PolicyDecision:
    detected_intent: str
    response_mode: str
    primary_emotion: str | None = None
    enough_context: bool = False
    follow_up_question: str | None = None
    followup_question_reason: str | None = None
    choice_prompt: str | None = None
    allow_immediate_resources: bool = False
    is_followup_answer: bool = False
    repeat_aware_strategy: str = "default"
    repetition_state: RepetitionState | None = None


class ConversationPolicy:
    SHORT_SIGNAL_WORDS = {
        "sad": "sadness",
        "anxious": "anxiety",
        "depressed": "sadness",
        "confused": "confusion",
        "tired": "burnout",
        "exhausted": "burnout",
        "قلقان": "anxiety",
        "قلقانه": "anxiety",
        "قلقانة": "anxiety",
        "خايف": "anxiety",
        "خايفة": "anxiety",
        "حزين": "sadness",
        "حزينه": "sadness",
        "حزينة": "sadness",
        "زعلان": "sadness",
        "زعلانه": "sadness",
        "زعلانة": "sadness",
        "مخنوق": "burnout",
        "مخنوقه": "burnout",
        "مخنوقة": "burnout",
        "تعبان": "burnout",
        "تعبانه": "burnout",
        "تعبانة": "burnout",
        "مرهق": "burnout",
        "مرهقه": "burnout",
        "مرهقة": "burnout",
        "تايه": "confusion",
        "تايهه": "confusion",
        "تايهة": "confusion",
        "مش كويس": "distress",
        "مش كويسه": "distress",
        "مش كويسة": "distress",
        "مش بخير": "distress",
    }
    VAGUE_DISTRESS_PHRASES = {
        "مش كويس",
        "مش كويسه",
        "مش كويسة",
        "مش بخير",
        "مش تمام",
        "حاسه بتعب",
        "حاسه بضيق",
        "حاسس بتعب",
        "حاسس بضيق",
        "متلغبطه",
        "متلخبطه",
        "متلخبطة",
        "ملخبط",
        "ملخبطه",
        "تايه",
        "تايهه",
        "تايهة",
    }
    RESOURCE_MARKERS = {
        "رشح", "رشحي", "رشحلي", "رشحيلي", "فيديو", "فيديوهات", "كتاب", "كتب", "بودكاست",
        "بودكاستات", "لينك", "لينكات", "روابط", "رابط", "مصادر", "resource", "resources",
        "podcast", "video", "videos", "book", "link", "links",
    }
    ADVICE_MARKERS = {
        "اعمل ايه", "أعمل ايه", "أعمل إيه", "اعمل إيه", "ايه الحل", "إيه الحل", "الحل ايه",
        "الحل إيه", "تنصح", "تنصحني", "اساعدني", "ساعدني", "محتاج نصيحة", "نصيحه", "نصيحة",
        "what should i do", "what do i do",
    }
    GREETING_MARKERS = {
        "اهلا", "أهلا", "هاي", "hi", "hello", "السلام عليكم", "صباح الخير", "مساء الخير",
    }
    CRISIS_MARKERS = {
        "مش عايزة اكمل", "مش عايزه اكمل", "مش عايز اكمل", "مش قادر اكمل", "مش قادره اكمل",
        "مش عايزة اعيش", "مش عايزه اعيش", "مش عايز اعيش", "نفسي اموت", "نفسي اختفي",
        "kill myself", "suicide", "self harm",
    }
    SADNESS_MARKERS = {"حزن", "زعل", "حزين", "زعلان", "sad", "depressed"}
    ANXIETY_MARKERS = {"قلق", "توتر", "خوف", "anxious", "anxiety", "panic"}
    BURNOUT_MARKERS = {"تعب", "ارهاق", "استنزاف", "مرهق", "burnout", "exhausted", "tired", "مخنوق", "ضغط"}
    SELF_BLAME_MARKERS = {"ذنب", "جلد", "لوم", "مقصر", "غلط", "فشلت", "self blame", "guilt"}
    RELATIONSHIP_MARKERS = {"الناس", "علاقه", "علاقة", "حدود", "رفض", "تجاهل", "ضغط", "misunderstanding"}
    CONFUSION_MARKERS = {"تايه", "لخبطه", "لخبطه", "ملخبط", "مش فاهم", "confused"}
    PROBLEM_CONNECTORS = {
        "بسبب", "علشان", "عشان", "من", "مع", "بعد", "لما", "لأن", "لان", "دايمًا",
        "دايما", "كل ما", "حاسس ان", "حاسه ان", "حاسس إنه", "حاسه إنه",
    }

    def __init__(self, settings) -> None:
        self.settings = settings

    def decide(
        self,
        *,
        message: str,
        conversation_history: list[dict[str, str]],
        assistant_turns: int,
        response_style: str,
        repetition_state: RepetitionState | None = None,
    ) -> PolicyDecision:
        repetition_state = repetition_state or RepetitionState()
        normalized = normalize_text(message)
        tokens = tokenize(message)
        last_assistant = self._last_message(conversation_history, role="assistant")
        prior_user_messages = [item["content"] for item in conversation_history if item.get("role") == "user"][:-1]
        followup_answer = bool(
            last_assistant
            and "؟" in last_assistant
            and prior_user_messages
            and not repetition_state.failed_to_answer_previous_followup
        )
        primary_emotion = self._detect_primary_emotion(normalized, tokens, prior_user_messages)
        enough_context = (
            followup_answer
            or self._has_specific_context(message, normalized, tokens)
            or repetition_state.repeated_meaning_count >= 3
        )
        repeat_strategy = self._choose_repeat_strategy(repetition_state)

        if self._contains_any(normalized, self.CRISIS_MARKERS):
            return PolicyDecision(
                detected_intent=INTENT_CRISIS_SIGNAL,
                response_mode=MODE_CRISIS,
                primary_emotion=primary_emotion,
                enough_context=True,
                repetition_state=repetition_state,
            )

        if self._is_small_talk(normalized, tokens):
            return PolicyDecision(
                detected_intent=INTENT_SMALL_TALK_OR_GREETING,
                response_mode=MODE_CHAT,
                primary_emotion=primary_emotion,
                enough_context=False,
                follow_up_question=self._style_line(
                    feminine="تحبي نبدأ بحاجة مضايقاكي دلوقتي، ولا مجرد دردشة خفيفة؟",
                    neutral="تحب نبدأ بحاجة مضايقاك دلوقتي، ولا مجرد دردشة خفيفة؟",
                    response_style=response_style,
                ),
                followup_question_reason="small_talk_opening",
                repetition_state=repetition_state,
            )

        if self._is_resource_request(normalized):
            return PolicyDecision(
                detected_intent=INTENT_RESOURCE_REQUEST,
                response_mode=MODE_RESOURCE if (assistant_turns + 1) >= self.settings.RECOMMENDATIONS_AFTER_TURN else MODE_CHAT,
                primary_emotion=primary_emotion,
                enough_context=enough_context,
                allow_immediate_resources=(assistant_turns + 1) >= self.settings.RECOMMENDATIONS_AFTER_TURN,
                follow_up_question=None if (assistant_turns + 1) >= self.settings.RECOMMENDATIONS_AFTER_TURN else self._style_line(
                    feminine="أكيد. تحبي موارد عن إيه بالضبط: حزن، قلق، ضغط من الناس، ولا حاجة تانية؟",
                    neutral="أكيد. تحب موارد عن إيه بالضبط: حزن، قلق، ضغط من الناس، ولا حاجة تانية؟",
                    response_style=response_style,
                ),
                followup_question_reason=None if (assistant_turns + 1) >= self.settings.RECOMMENDATIONS_AFTER_TURN else "resource_request_needs_topic",
                choice_prompt=None if (assistant_turns + 1) >= self.settings.RECOMMENDATIONS_AFTER_TURN else "ممكن نحدد الموضوع الأول عشان الترشيح يطلع فعلاً مناسب.",
                repetition_state=repetition_state,
            )

        if self._is_advice_request(normalized):
            if enough_context:
                return PolicyDecision(
                    detected_intent=INTENT_ADVICE_REQUEST,
                    response_mode=MODE_GROUNDED,
                    primary_emotion=primary_emotion,
                    enough_context=True,
                    is_followup_answer=followup_answer,
                    repeat_aware_strategy=repeat_strategy,
                    repetition_state=repetition_state,
                )
            question, reason, choice = self._build_follow_up(
                primary_emotion,
                response_style,
                vague=True,
                repetition_state=repetition_state,
            )
            return PolicyDecision(
                detected_intent=INTENT_ADVICE_REQUEST,
                response_mode=MODE_CHAT,
                primary_emotion=primary_emotion,
                enough_context=False,
                follow_up_question=question,
                followup_question_reason=reason,
                choice_prompt=choice,
                repeat_aware_strategy=repeat_strategy,
                repetition_state=repetition_state,
            )

        if self._is_vague_distress(normalized, tokens) and not followup_answer and assistant_turns < self.settings.CHAT_FOLLOWUP_TURNS:
            question, reason, choice = self._build_follow_up(
                primary_emotion,
                response_style,
                vague=True,
                repetition_state=repetition_state,
            )
            return PolicyDecision(
                detected_intent=INTENT_VAGUE_DISTRESS,
                response_mode=MODE_CHAT,
                primary_emotion=primary_emotion,
                enough_context=False,
                follow_up_question=question,
                followup_question_reason=reason,
                choice_prompt=choice,
                repeat_aware_strategy=repeat_strategy,
                repetition_state=repetition_state,
            )

        if self._is_short_emotional_signal(message, normalized, tokens):
            if repetition_state.loop_detected or repetition_state.repeated_meaning_count >= 4:
                return PolicyDecision(
                    detected_intent=INTENT_SHORT_EMOTIONAL_SIGNAL,
                    response_mode=MODE_GROUNDED,
                    primary_emotion=primary_emotion,
                    enough_context=True,
                    is_followup_answer=False,
                    repeat_aware_strategy="micro_support",
                    repetition_state=repetition_state,
                )
            if repetition_state.repeated_meaning_count >= 3:
                question, reason, choice = self._build_follow_up(
                    primary_emotion,
                    response_style,
                    vague=False,
                    repetition_state=repetition_state,
                )
                return PolicyDecision(
                    detected_intent=INTENT_SHORT_EMOTIONAL_SIGNAL,
                    response_mode=MODE_CHAT,
                    primary_emotion=primary_emotion,
                    enough_context=False,
                    follow_up_question=question,
                    followup_question_reason=reason,
                    choice_prompt=choice,
                    repeat_aware_strategy=repeat_strategy,
                    repetition_state=repetition_state,
                )
            if enough_context or assistant_turns >= self.settings.CHAT_FOLLOWUP_TURNS:
                return PolicyDecision(
                    detected_intent=INTENT_CLEAR_PROBLEM_STATEMENT,
                    response_mode=MODE_GROUNDED,
                    primary_emotion=primary_emotion,
                    enough_context=True,
                    is_followup_answer=followup_answer,
                    repeat_aware_strategy=repeat_strategy,
                    repetition_state=repetition_state,
                )
            question, reason, choice = self._build_follow_up(
                primary_emotion,
                response_style,
                vague=False,
                repetition_state=repetition_state,
            )
            return PolicyDecision(
                detected_intent=INTENT_SHORT_EMOTIONAL_SIGNAL,
                response_mode=MODE_CHAT,
                primary_emotion=primary_emotion,
                enough_context=False,
                follow_up_question=question,
                followup_question_reason=reason,
                choice_prompt=choice,
                repeat_aware_strategy=repeat_strategy,
                repetition_state=repetition_state,
            )

        return PolicyDecision(
            detected_intent=INTENT_CLEAR_PROBLEM_STATEMENT,
            response_mode=MODE_GROUNDED,
            primary_emotion=primary_emotion,
            enough_context=True,
            is_followup_answer=followup_answer,
            repeat_aware_strategy=repeat_strategy,
            repetition_state=repetition_state,
        )

    def _is_short_emotional_signal(self, message: str, normalized: str, tokens: list[str]) -> bool:
        if len(message.strip()) <= 16 and normalized in self.SHORT_SIGNAL_WORDS:
            return True
        return len(tokens) <= 3 and any(token in self.SHORT_SIGNAL_WORDS for token in tokens) and not self._has_specific_context(message, normalized, tokens)

    def _is_vague_distress(self, normalized: str, tokens: list[str]) -> bool:
        if normalized in self.VAGUE_DISTRESS_PHRASES:
            return True
        return len(tokens) <= 4 and any(phrase in normalized for phrase in self.VAGUE_DISTRESS_PHRASES)

    def _is_resource_request(self, normalized: str) -> bool:
        return self._contains_any(normalized, self.RESOURCE_MARKERS)

    def _is_advice_request(self, normalized: str) -> bool:
        return self._contains_any(normalized, self.ADVICE_MARKERS) or normalized.endswith("اعمل ايه") or normalized.endswith("اعمل اي")

    def _is_small_talk(self, normalized: str, tokens: list[str]) -> bool:
        if normalized in {normalize_text(item) for item in self.GREETING_MARKERS}:
            return True
        return len(tokens) <= 3 and self._contains_any(normalized, self.GREETING_MARKERS)

    def _has_specific_context(self, message: str, normalized: str, tokens: list[str]) -> bool:
        if len(tokens) >= 5:
            return True
        if any(marker in normalized for marker in self.PROBLEM_CONNECTORS):
            return True
        return any(word in normalized for word in ["الناس", "الدراسة", "الشغل", "العيلة", "العائله", "مسؤوليات", "حدود", "غلطه", "موقف"])

    def _detect_primary_emotion(self, normalized: str, tokens: list[str], prior_user_messages: list[str]) -> str | None:
        token_set = set(tokens)
        context = " ".join(prior_user_messages[-2:])
        combined = f"{normalized} {normalize_text(context)}".strip()

        def has_markers(markers: set[str]) -> bool:
            return any(marker in combined for marker in markers) or bool(token_set & markers)

        if has_markers(self.SADNESS_MARKERS):
            return "sadness"
        if has_markers(self.ANXIETY_MARKERS):
            return "anxiety"
        if has_markers(self.SELF_BLAME_MARKERS):
            return "self_blame"
        if has_markers(self.RELATIONSHIP_MARKERS):
            return "relationship_distress"
        if has_markers(self.BURNOUT_MARKERS):
            return "burnout"
        if has_markers(self.CONFUSION_MARKERS):
            return "confusion"
        return "distress" if normalized else None

    def _build_follow_up(
        self,
        primary_emotion: str | None,
        response_style: str,
        *,
        vague: bool,
        repetition_state: RepetitionState,
    ) -> tuple[str, str, str | None]:
        if repetition_state.loop_detected or repetition_state.repeated_meaning_count >= 3:
            return self._build_guided_choice(primary_emotion, response_style)
        if repetition_state.repeated_meaning_count == 2:
            return self._build_repeat_follow_up(primary_emotion, response_style)
        if primary_emotion == "sadness":
            return (
                self._style_line(
                    feminine="الحزن ده مرتبط بحاجة حصلت، ولا موجود من غير سبب واضح؟",
                    neutral="الحزن ده مرتبط بحاجة حصلت، ولا موجود من غير سبب واضح؟",
                    response_style=response_style,
                ),
                "sadness_trigger_vs_general",
                None,
            )
        if primary_emotion == "anxiety":
            return (
                self._style_line(
                    feminine="القلق ده من موضوع معين، ولا حاسة إنه منتشر في كذا حاجة؟",
                    neutral="القلق ده من موضوع معين، ولا حاسس إنه منتشر في كذا حاجة؟",
                    response_style=response_style,
                ),
                "anxiety_specific_vs_general",
                None,
            )
        if primary_emotion == "burnout":
            return (
                self._style_line(
                    feminine="الإرهاق ده جاي أكتر من ضغط نفسي، ناس، دراسة، ولا مسؤوليات؟",
                    neutral="الإرهاق ده جاي أكتر من ضغط نفسي، ناس، دراسة، ولا مسؤوليات؟",
                    response_style=response_style,
                ),
                "burnout_source_selection",
                None,
            )
        if primary_emotion == "self_blame":
            return (
                self._style_line(
                    feminine="الإحساس ده بدأ بعد غلط أو نقد أو مقارنة، ولا ده أسلوبك مع نفسك من فترة؟",
                    neutral="الإحساس ده بدأ بعد غلط أو نقد أو مقارنة، ولا ده أسلوبك مع نفسك من فترة؟",
                    response_style=response_style,
                ),
                "self_blame_origin",
                None,
            )
        if primary_emotion == "relationship_distress":
            return (
                self._style_line(
                    feminine="الوجع ده جاي من رفض، ضغط، حدود، ولا سوء فهم مع حد؟",
                    neutral="الوجع ده جاي من رفض، ضغط، حدود، ولا سوء فهم مع حد؟",
                    response_style=response_style,
                ),
                "relationship_distress_source",
                None,
            )
        if primary_emotion == "confusion":
            return (
                self._style_line(
                    feminine="حاسّة إنك تايهة بين اختيارات، ولا مش قادرة تفهمي اللي جواكي نفسه؟",
                    neutral="حاسس إنك تايه بين اختيارات، ولا مش قادر تفهم اللي جواك نفسه؟",
                    response_style=response_style,
                ),
                "confusion_type",
                None,
            )

        question = self._style_line(
            feminine="تحبي نحدد الأول: اللي غالب أكتر حزن، قلق، ضغط، ولا لخبطة مشاعر؟",
            neutral="تحب نحدد الأول: اللي غالب أكتر حزن، قلق، ضغط، ولا لخبطة مشاعر؟",
            response_style=response_style,
        )
        choice = None
        if vague:
            choice = "لو أسهل، اختار/ي أقرب وصف دلوقتي وأنا أبني عليه."
        return question, "generic_emotion_clarification", choice

    def _build_repeat_follow_up(self, primary_emotion: str | None, response_style: str) -> tuple[str, str, str | None]:
        if primary_emotion == "sadness":
            return (
                self._style_line(
                    feminine="حاسّة إن الإحساس نفسه لسه تقيل عليكي. ده زعل من موقف، ولا حزن عام بقاله شوية؟",
                    neutral="حاسس إن الإحساس نفسه لسه تقيل عليك. ده زعل من موقف، ولا حزن عام بقاله شوية؟",
                    response_style=response_style,
                ),
                "repeat_sadness_simplified",
                None,
            )
        if primary_emotion == "burnout":
            return (
                self._style_line(
                    feminine="واضح إن الخنقة لسه مستمرة. أقرب سبب ليها ناس، مسؤوليات، ولا ضغط داخلي؟",
                    neutral="واضح إن الخنقة لسه مستمرة. أقرب سبب ليها ناس، مسؤوليات، ولا ضغط داخلي؟",
                    response_style=response_style,
                ),
                "repeat_burnout_simplified",
                None,
            )
        if primary_emotion == "anxiety":
            return (
                self._style_line(
                    feminine="واضح إن القلق لسه ماسكك. هو من موضوع محدد، ولا منتشر في اليوم كله؟",
                    neutral="واضح إن القلق لسه ماسكك. هو من موضوع محدد، ولا منتشر في اليوم كله؟",
                    response_style=response_style,
                ),
                "repeat_anxiety_simplified",
                None,
            )
        return (
            self._style_line(
                feminine="حاسّة إن نفس الإحساس لسه موجود. تحبي نختار أقرب وصف ليه بدل ما نشرحه؟",
                neutral="حاسس إن نفس الإحساس لسه موجود. تحب نختار أقرب وصف ليه بدل ما نشرحه؟",
                response_style=response_style,
            ),
            "repeat_generic_simplified",
            "ممكن تختار/ي كلمة أقرب: حزن / قلق / ضغط.",
        )

    def _build_guided_choice(self, primary_emotion: str | None, response_style: str) -> tuple[str, str, str | None]:
        if primary_emotion == "sadness":
            return (
                self._style_line(
                    feminine="تمام، مش لازم تشرحي كتير دلوقتي. اختاري كلمة واحدة بس: موقف / بدون سبب / ضغط",
                    neutral="تمام، مش لازم تشرح كتير دلوقتي. اختر كلمة واحدة بس: موقف / بدون سبب / ضغط",
                    response_style=response_style,
                ),
                "repeat_sadness_guided_choice",
                None,
            )
        if primary_emotion == "burnout":
            return (
                self._style_line(
                    feminine="خلينا نبسّطها جدًا: ناس / مسؤوليات / بدون سبب واضح. اختاري الأقرب.",
                    neutral="خلينا نبسّطها جدًا: ناس / مسؤوليات / بدون سبب واضح. اختر الأقرب.",
                    response_style=response_style,
                ),
                "repeat_burnout_guided_choice",
                None,
            )
        return (
            self._style_line(
                feminine="خلينا نمشي بطريقة أبسط: 1) موقف 2) بدون سبب 3) ضغط. اختاري الأقرب.",
                neutral="خلينا نمشي بطريقة أبسط: 1) موقف 2) بدون سبب 3) ضغط. اختر الأقرب.",
                response_style=response_style,
            ),
            "repeat_generic_guided_choice",
            None,
        )

    def _choose_repeat_strategy(self, repetition_state: RepetitionState) -> str:
        if repetition_state.loop_detected or repetition_state.repeated_meaning_count >= 4:
            return "micro_support"
        if repetition_state.repeated_meaning_count >= 3:
            return "guided_choice"
        if repetition_state.repeated_meaning_count == 2:
            return "simplified_followup"
        return "default"

    def _style_line(self, *, feminine: str, neutral: str, response_style: str) -> str:
        return feminine if response_style == "feminine" else neutral

    def _last_message(self, history: list[dict[str, str]], role: str) -> str | None:
        for item in reversed(history):
            if item.get("role") == role:
                return item.get("content", "")
        return None

    def _contains_any(self, normalized: str, markers: set[str]) -> bool:
        return any(normalize_text(marker) in normalized for marker in markers)
