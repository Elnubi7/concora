from app.core.constants import CRISIS_KEYWORDS
from app.models.schemas import SafetyMeta
from app.utils.text_utils import normalize_text


class SafetyService:
    def inspect(self, message: str, user_gender: str, audience_mode: str) -> SafetyMeta:
        normalized = normalize_text(message)
        normalized_keywords = {normalize_text(keyword) for keyword in CRISIS_KEYWORDS}
        if any(keyword in normalized for keyword in normalized_keywords):
            return SafetyMeta(
                is_crisis=True,
                blocked=True,
                reason="Crisis keywords detected in user message.",
            )

        if audience_mode == "female_only" and user_gender != "female":
            return SafetyMeta(
                is_crisis=False,
                blocked=True,
                reason="Service is currently configured for girls/women only.",
            )

        return SafetyMeta(is_crisis=False, blocked=False, reason=None)

    def gender_restriction_response(self) -> str:
        return "الخدمة الحالية مهيأة للبنات/النساء فقط حسب الإعدادات الحالية للمشروع."
