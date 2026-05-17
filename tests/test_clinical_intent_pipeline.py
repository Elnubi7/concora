from app.services.clinical_intent_pipeline import (
    INTENT_ADMIN,
    INTENT_EMERGENCY,
    INTENT_FOLLOW_UP,
    INTENT_GENERAL,
    INTENT_GREETING,
    INTENT_MEDICAL,
    INTENT_MEDICATION,
    INTENT_NON_MEDICAL,
    INTENT_UNCLEAR,
    generateFinalBotReply,
)


def test_random_unclear_text_asks_for_clarification_without_medical_advice():
    result = generateFinalBotReply("s dfm")

    assert result["intent"] == INTENT_UNCLEAR
    assert result["confidence"] >= 0.65
    assert result["should_ask_clarification"] is True
    assert result["should_answer_directly"] is False
    assert "مش واضح" in result["reply"]
    assert "تشخيص" not in result["reply"]


def test_greeting_gets_short_greeting_reply():
    result = generateFinalBotReply("السلام عليكم")

    assert result["intent"] == INTENT_GREETING
    assert result["should_answer_directly"] is True
    assert "وعليكم السلام" in result["reply"]


def test_vague_medical_question_asks_targeted_questions():
    result = generateFinalBotReply("بطني بتوجعني")

    assert result["intent"] == INTENT_MEDICAL
    assert result["should_ask_clarification"] is True
    assert "مش واضح" in result["reply"]
    assert "مدة المشكلة" in result["missing_information"]


def test_detailed_medical_question_gets_safe_case_specific_reply():
    result = generateFinalBotReply("عمري 25 وبطني بتوجعني من يومين والألم متوسط ومعايا قيء. أعمل ايه؟")

    assert result["intent"] == INTENT_MEDICAL
    assert result["should_answer_directly"] is True
    assert result["case_facts"]["age"] == "25"
    assert result["case_facts"]["duration"]
    assert "فهمت من كلامك" in result["reply"]
    assert "بدون تشخيص نهائي" in result["reply"]
    assert "أسئلة مهمة" in result["reply"]


def test_medication_request_avoids_doses_and_prescribing():
    result = generateFinalBotReply("ينفع اخد ايبوبروفين لوجع الضرس؟")

    assert result["intent"] == INTENT_MEDICATION
    assert result["strategy"] == "medication"
    assert "ممنوع إعطاء جرعات غير مؤكدة" in result["reply"]
    assert "لا أقدر أوصف جرعة دقيقة" in result["reply"]


def test_emergency_symptom_gets_urgent_safety_response():
    result = generateFinalBotReply("عندي ألم في الصدر ومش قادرة اتنفس")

    assert result["intent"] == INTENT_EMERGENCY
    assert result["safety_level"] == "emergency"
    assert result["should_answer_directly"] is True
    assert "رعاية طبية عاجلة" in result["reply"] or "طوارئ" in result["reply"]


def test_mental_health_crisis_is_red_flag():
    result = generateFinalBotReply("نفسي اموت ومش عايزة اكمل")

    assert result["intent"] == INTENT_EMERGENCY
    assert "suicidal_or_self_harm_thoughts" in result["red_flags"]
    assert "طوارئ" in result["reply"] or "عاجلة" in result["reply"]


def test_dental_pain_case_uses_dental_specific_safety_questions():
    result = generateFinalBotReply("ضرسي واجعني من يومين والألم شديد ومفيش تورم")

    assert result["intent"] == INTENT_MEDICAL
    assert result["case_facts"]["is_dental"] is True
    assert "كشف أسنان" in result["reply"]
    assert "تورم في الوجه" in result["reply"]


def test_follow_up_uses_previous_case_only_when_relevant():
    previous = [
        {"role": "user", "content": "عمري 25 وبطني بتوجعني من يومين والألم متوسط"},
        {"role": "assistant", "content": "أسئلة مهمة لتحديد الحالة..."},
    ]

    result = generateFinalBotReply("لسه الوجع موجود اعمل ايه", previous)

    assert result["intent"] == INTENT_FOLLOW_UP
    assert result["related_to_previous"] is True
    assert result["should_answer_directly"] is True
    assert result["case_facts"]["age"] == "25"
    assert "فهمت من كلامك" in result["reply"]


def test_unrelated_message_after_medical_chat_is_not_forced_into_previous_case():
    previous = [
        {"role": "user", "content": "عمري 25 وبطني بتوجعني من يومين والألم متوسط"},
        {"role": "assistant", "content": "أسئلة مهمة لتحديد الحالة..."},
    ]

    result = generateFinalBotReply("بتحب الافلام؟", previous)

    assert result["intent"] == INTENT_NON_MEDICAL
    assert result["related_to_previous"] is False
    assert result["strategy"] == "general"
    assert "بطني" not in result["reply"]


def test_admin_and_general_intents_are_classified_without_medical_guessing():
    admin = generateFinalBotReply("الشات مش شغال")
    general = generateFinalBotReply("ما معنى كلمة resilience؟")

    assert admin["intent"] == INTENT_ADMIN
    assert "التطبيق" in admin["reply"]
    assert general["intent"] == INTENT_GENERAL
    assert "أقدر أساعدك" in general["reply"]
