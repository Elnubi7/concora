from __future__ import annotations

from dataclasses import asdict, dataclass, field

from app.utils.text_utils import lexical_overlap_score, normalize_text, tokenize


@dataclass
class RepetitionState:
    last_user_message_normalized: str = ""
    last_meaning_key: str = ""
    repeated_message_count: int = 1
    repeated_meaning_count: int = 1
    exact_repetition_count: int = 1
    semantic_repetition_count: int = 1
    loop_detected: bool = False
    last_followup_question: str | None = None
    last_detected_emotion: str | None = None
    failed_to_answer_previous_followup: bool = False
    recent_assistant_followups: list[str] = field(default_factory=list)
    recent_assistant_openings: list[str] = field(default_factory=list)
    recent_assistant_modes: list[str] = field(default_factory=list)
    same_assistant_mode_count: int = 0
    last_assistant_mode: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "RepetitionState":
        if not payload:
            return cls()
        return cls(**payload)


class RepetitionDetector:
    EMOTION_BUCKETS = {
        "sadness": {
            "sad", "depressed", "down", "حزين", "حزينه", "حزينة", "زعلان", "زعلانه", "زعلانة", "حزن", "زعل",
        },
        "anxiety": {
            "anxious", "anxiety", "قلقان", "قلقانه", "قلقانة", "قلق", "توتر", "خوف",
        },
        "burnout": {
            "tired", "exhausted", "burnout", "تعبان", "تعبانه", "تعبانة", "مخنوق", "مخنوقه", "مخنوقة", "مرهق", "مرهقة",
        },
        "distress": {
            "مش كويس", "مش كويسه", "مش كويسة", "مش بخير", "مش تمام", "تايه", "تايهة", "متلخبطة",
        },
    }

    def normalize_message(self, text: str) -> str:
        return normalize_text(text)

    def detect(
        self,
        *,
        message: str,
        previous_state: RepetitionState | None,
        recent_user_messages: list[str],
    ) -> RepetitionState:
        state = previous_state or RepetitionState()
        normalized = self.normalize_message(message)
        tokens = tokenize(message)
        meaning_key = self._meaning_key(normalized, tokens)
        previous_user_messages = recent_user_messages[:-1]
        previous_message = self.normalize_message(previous_user_messages[-1]) if previous_user_messages else ""
        previous_meaning = state.last_meaning_key or (self._meaning_key(previous_message, tokenize(previous_user_messages[-1])) if previous_user_messages else "")
        adds_detail = self._adds_meaningful_detail(normalized, previous_message)

        exact_repeat = bool(previous_message and normalized == previous_message)
        semantic_repeat = self._is_semantic_repeat(
            current_normalized=normalized,
            current_tokens=tokens,
            current_meaning=meaning_key,
            previous_normalized=previous_message,
            previous_meaning=previous_meaning,
        )
        if semantic_repeat and adds_detail:
            semantic_repeat = False

        if exact_repeat:
            exact_count = state.exact_repetition_count + 1
            message_count = state.repeated_message_count + 1
        else:
            exact_count = 1
            message_count = 1

        if semantic_repeat:
            semantic_count = state.semantic_repetition_count + 1
            meaning_count = state.repeated_meaning_count + 1
        else:
            semantic_count = 1
            meaning_count = 1

        failed_followup = bool(
            state.last_followup_question
            and semantic_repeat
            and not adds_detail
        )

        same_mode_count = state.same_assistant_mode_count
        if state.last_assistant_mode == "CHAT_MODE":
            same_mode_count = max(same_mode_count, 1)

        loop_detected = bool(
            meaning_count >= 4
            and failed_followup
            and (same_mode_count >= 2 or len(state.recent_assistant_followups) >= 2)
        )

        return RepetitionState(
            last_user_message_normalized=normalized,
            last_meaning_key=meaning_key,
            repeated_message_count=message_count,
            repeated_meaning_count=meaning_count,
            exact_repetition_count=exact_count,
            semantic_repetition_count=semantic_count,
            loop_detected=loop_detected,
            last_followup_question=state.last_followup_question,
            last_detected_emotion=meaning_key or state.last_detected_emotion,
            failed_to_answer_previous_followup=failed_followup,
            recent_assistant_followups=list(state.recent_assistant_followups),
            recent_assistant_openings=list(state.recent_assistant_openings),
            recent_assistant_modes=list(state.recent_assistant_modes),
            same_assistant_mode_count=same_mode_count,
            last_assistant_mode=state.last_assistant_mode,
        )

    def register_assistant_response(
        self,
        state: RepetitionState,
        *,
        response_mode: str,
        follow_up_question: str | None,
        opening_phrase: str | None,
    ) -> RepetitionState:
        recent_followups = list(state.recent_assistant_followups)
        recent_openings = list(state.recent_assistant_openings)
        recent_modes = list(state.recent_assistant_modes)

        if follow_up_question:
            recent_followups.append(follow_up_question.strip())
            recent_followups = recent_followups[-3:]

        if opening_phrase:
            recent_openings.append(opening_phrase.strip())
            recent_openings = recent_openings[-3:]

        recent_modes.append(response_mode)
        recent_modes = recent_modes[-4:]

        same_mode_count = 1
        if state.last_assistant_mode == response_mode:
            same_mode_count = state.same_assistant_mode_count + 1

        return RepetitionState(
            last_user_message_normalized=state.last_user_message_normalized,
            last_meaning_key=state.last_meaning_key,
            repeated_message_count=state.repeated_message_count,
            repeated_meaning_count=state.repeated_meaning_count,
            exact_repetition_count=state.exact_repetition_count,
            semantic_repetition_count=state.semantic_repetition_count,
            loop_detected=state.loop_detected,
            last_followup_question=follow_up_question or state.last_followup_question,
            last_detected_emotion=state.last_detected_emotion,
            failed_to_answer_previous_followup=state.failed_to_answer_previous_followup,
            recent_assistant_followups=recent_followups,
            recent_assistant_openings=recent_openings,
            recent_assistant_modes=recent_modes,
            same_assistant_mode_count=same_mode_count,
            last_assistant_mode=response_mode,
        )

    def _meaning_key(self, normalized: str, tokens: list[str]) -> str:
        if not normalized:
            return ""
        token_set = set(tokens)
        for bucket, markers in self.EMOTION_BUCKETS.items():
            normalized_markers = {normalize_text(marker) for marker in markers}
            if normalized in normalized_markers or token_set & normalized_markers:
                return bucket
            if any(marker in normalized for marker in normalized_markers):
                return bucket
        return normalized

    def _is_semantic_repeat(
        self,
        *,
        current_normalized: str,
        current_tokens: list[str],
        current_meaning: str,
        previous_normalized: str,
        previous_meaning: str,
    ) -> bool:
        if not current_normalized or not previous_normalized:
            return False
        if current_normalized == previous_normalized:
            return True
        if current_meaning and previous_meaning and current_meaning == previous_meaning:
            return True
        overlap = lexical_overlap_score(current_normalized, previous_normalized)
        if overlap >= 0.58:
            return True
        previous_tokens = set(tokenize(previous_normalized))
        if previous_tokens and set(current_tokens) and len(set(current_tokens) & previous_tokens) >= 2:
            return True
        return False

    def _adds_meaningful_detail(self, current_normalized: str, previous_normalized: str) -> bool:
        if not previous_normalized:
            return True
        current_tokens = set(tokenize(current_normalized))
        previous_tokens = set(tokenize(previous_normalized))
        return len(current_tokens - previous_tokens) >= 2
