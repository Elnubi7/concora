from __future__ import annotations

import re

from app.utils.text_utils import normalize_text


class FallbackService:
    NEUTRAL_STEP_MAP = {
        "حددي": "تحديد",
        "اكتبي": "كتابة",
        "لاحظي": "ملاحظة",
        "اسألي": "سؤال النفس",
        "اتواصلي": "التواصل",
        "اطلبي": "طلب",
        "حاولي": "محاولة",
        "ابدئي": "البدء",
        "امشي": "المشي",
        "شاركي": "مشاركة",
        "قولي": "قول",
        "راجعي": "مراجعة",
        "سيبيه": "ترك",
        "تذكري": "تذكّر",
        "قللي": "تقليل",
        "قسمي": "تقسيم",
        "قسّمي": "تقسيم",
        "تجنبي": "تجنب",
        "اختاري": "اختيار",
        "اعملي": "تنفيذ",
        "خدي": "أخذ",
    }

    def compose_structured(
        self,
        *,
        mbti_type: str | None,
        mbti_issue: str | None,
        mbti_core_problems: list[str],
        generic_topic: str | None,
        generic_question: str | None,
        generic_anchor: str | None,
        advice_steps: list[str],
        response_style: str = "feminine",
    ) -> dict:
        if generic_question:
            understanding = f"أقرب مطابقة واضحة في البيانات العامة كانت السؤال: {generic_question}."
        elif generic_anchor:
            understanding = f"أقرب محور ظاهر في البيانات العامة هو: {generic_anchor}."
        elif mbti_issue:
            understanding = f"أقرب محور ظاهر في بيانات MBTI هو: {mbti_issue}."
        else:
            understanding = "في الرسالة ضغط نفسي واضح، والرد مبني على أقرب مقاطع مسترجعة من البيانات المتاحة."

        if mbti_issue and mbti_type:
            mbti_connection = (
                f"قد يساعد نمط {mbti_type} في فهم ميل عام هنا، خصوصًا لأن أقرب محور في بياناته هو: {mbti_issue}. "
                "لكن ده ليس تفسيرًا قطعيًا للحالة."
            )
        elif mbti_core_problems and mbti_type:
            mbti_connection = (
                f"قد يفيد {mbti_type} فقط كخلفية عامة؛ في البيانات المرتبطة به تظهر محاور مثل: "
                f"{', '.join(mbti_core_problems[:3])}. لكن الاعتماد الأساسي هنا كان على السؤال نفسه."
            )
        else:
            mbti_connection = "لا يوجد ربط قوي كفاية بـ MBTI هنا، لذلك الاعتماد الأساسي كان على البيانات العامة المرتبطة بالمشاعر والسلوك."

        grounded_bits: list[str] = []
        if generic_topic:
            grounded_bits.append(f"الموضوع الأقرب في البيانات العامة: {generic_topic}.")
        if generic_question:
            grounded_bits.append(f"وأقرب سؤال مطابق: {generic_question}.")
        if generic_anchor:
            grounded_bits.append(generic_anchor)
        if mbti_issue:
            grounded_bits.append(f"وفي بيانات MBTI ظهر محور قريب هو: {mbti_issue}.")
        grounded_answer = " ".join(bit for bit in grounded_bits if bit).strip()
        if not grounded_answer:
            grounded_answer = "الجواب هنا مبني على الاسترجاع من ملف MBTI وملف المشاعر، بدون تشخيص وبدون التعامل مع MBTI كسبب مؤكد."

        unique_steps: list[str] = []
        seen: set[str] = set()
        for step in advice_steps:
            clean = step.strip()
            key = normalize_text(clean)
            if not clean or not key or key in seen:
                continue
            seen.add(key)
            unique_steps.append(self._adapt_step(clean, response_style))
            if len(unique_steps) >= 4:
                break

        if not unique_steps:
            defaults = [
                "تحديد الإحساس الأساسي بدل تركه عامًا.",
                "اختيار خطوة صغيرة يمكن تنفيذها اليوم.",
                "مراجعة أثر الخطوة بهدوء ومن غير جلد ذات.",
            ]
            unique_steps = [self._adapt_step(item, response_style) for item in defaults]

        support_note = (
            "الرد مبني على البيانات المرفوعة فقط، وليس تشخيصًا أو بديلًا عن مختص/مختصة. "
            "إذا وُجد خطر مباشر أو أفكار مؤذية، فالأولوية لطلب مساعدة فورية."
        )
        return {
            "understanding": understanding,
            "mbti_connection": mbti_connection,
            "grounded_answer": grounded_answer,
            "practical_steps": unique_steps,
            "support_note": support_note,
        }

    def _adapt_step(self, step: str, style: str) -> str:
        if style != "neutral":
            return step
        clean = step.strip()
        for feminine, neutral in self.NEUTRAL_STEP_MAP.items():
            if clean.startswith(feminine + " ") or clean == feminine:
                rest = clean[len(feminine):].strip()
                return f"{neutral} {rest}".strip()
        clean = re.sub(r"\bجربي\b", "تجربة", clean)
        clean = re.sub(r"\bخليكي\b", "الالتزام بأن يكون", clean)
        return clean
