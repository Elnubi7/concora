from __future__ import annotations

import re
from typing import Any

from app.utils.text_utils import lexical_overlap_score, normalize_text, tokenize


INTENT_UNCLEAR = "unclear_or_gibberish"
INTENT_GREETING = "greeting"
INTENT_GENERAL = "general_question"
INTENT_MEDICAL = "medical_question"
INTENT_MENTAL_HEALTH = "mental_health_question"
INTENT_MEDICATION = "medication_question"
INTENT_EMERGENCY = "emergency_or_red_flag"
INTENT_FOLLOW_UP = "follow_up_to_previous_case"
INTENT_ADMIN = "admin_or_app_usage"
INTENT_NON_MEDICAL = "non_medical_chat"

SAFETY_LOW = "low"
SAFETY_MODERATE = "moderate"
SAFETY_HIGH = "high"
SAFETY_EMERGENCY = "emergency"

DIRECT_REPLY_INTENTS = {
    INTENT_UNCLEAR,
    INTENT_GREETING,
    INTENT_GENERAL,
    INTENT_MEDICAL,
    INTENT_MEDICATION,
    INTENT_EMERGENCY,
    INTENT_FOLLOW_UP,
    INTENT_ADMIN,
    INTENT_NON_MEDICAL,
}


def _normalized_markers(markers: set[str]) -> set[str]:
    return {normalize_text(marker) for marker in markers}


GREETING_MARKERS = _normalized_markers({
    "السلام عليكم", "وعليكم السلام", "اهلا", "أهلا", "هاي", "مرحبا", "صباح الخير", "مساء الخير",
    "hi", "hello", "hey",
})

ADMIN_MARKERS = _normalized_markers({
    "التطبيق", "الابلكيشن", "الشات", "الحساب", "تسجيل الدخول", "كلمة السر", "الاشعارات",
    "app", "login", "password", "account", "notification", "chat history",
})

MEDICATION_MARKERS = _normalized_markers({
    "دواء", "دوا", "علاج", "برشام", "حبوب", "مضاد حيوي", "مسكن", "جرعة", "جرعه", "ينفع اخد",
    "اخد ايه", "آخد ايه", "باراسيتامول", "ايبوبروفين", "اسبرين", "مضاد", "antibiotic", "dose",
    "dosage", "medicine", "medication", "pill", "drug", "paracetamol", "ibuprofen",
})

MEDICAL_MARKERS = _normalized_markers({
    "الم", "وجع", "صداع", "سخونية", "حرارة", "كحة", "قيء", "ترجيع", "اسهال", "امساك", "دوخة",
    "دوار", "نزيف", "تورم", "انتفاخ", "حساسية", "طفح", "حمل", "حامل", "طفل", "سن", "معدة",
    "بطني", "البطن", "صدر", "تنفس", "اسنان", "ضرس", "لثة", "طبيب", "دكتور", "تحليل", "سكر",
    "ضغط", "infection", "pain", "fever", "vomiting", "diarrhea", "dizzy", "bleeding", "swelling",
    "pregnant", "child", "tooth", "dental", "doctor", "clinic",
})

DENTAL_MARKERS = _normalized_markers({
    "اسنان", "سنان", "ضرس", "لثة", "خراج", "حشو", "dental", "tooth", "gum", "abscess",
})

MENTAL_HEALTH_MARKERS = _normalized_markers({
    "قلق", "توتر", "اكتئاب", "حزن", "مخنوق", "مخنوقة", "خوف", "هلع", "نوبة هلع", "وسواس",
    "مشاعري", "نفسيتي", "تعبان نفسيا", "مضغوط", "anxiety", "depression", "panic", "sad",
    "stressed", "mental", "therapy", "مش كويس", "مش كويسة", "مش كويسه", "تعبانه نفسيا", "تعبان نفسيا",
    "بتعب من الناس", "بشيل همهم", "حاسة", "حاسه", "حاسس", "مش عارفة", "مش عارفه", "مش عارف",
})

CRISIS_MARKERS = _normalized_markers({
    "انتحار", "انتحر", "اقتل نفسي", "اؤذي نفسي", "اذي نفسي", "مش عايز اعيش", "مش عايزة اعيش",
    "نفسي اموت", "مش عايز اكمل", "مش عايزة اكمل", "suicide", "kill myself", "hurt myself",
    "self harm", "suicidal thoughts", "dont want to live", "don't want to live",
})

RED_FLAG_MARKERS = {
    "chest_pain": _normalized_markers({"الم في الصدر", "وجع في الصدر", "chest pain"}),
    "severe_breathlessness": _normalized_markers({
        "ضيق تنفس شديد", "مش قادر اتنفس", "مش قادرة اتنفس", "نهجان شديد", "severe shortness of breath",
        "can't breathe", "cant breathe",
    }),
    "loss_of_consciousness": _normalized_markers({"اغماء", "فقدان وعي", "loss of consciousness", "fainted"}),
    "severe_allergy": _normalized_markers({
        "حساسية شديدة", "تورم الشفايف", "تورم اللسان", "anaphylaxis", "severe allergic reaction",
    }),
    "uncontrolled_bleeding": _normalized_markers({"نزيف شديد", "نزيف مش بيقف", "uncontrolled bleeding"}),
    "neurological_weakness": _normalized_markers({
        "ضعف مفاجئ", "تنميل نصف الجسم", "اعوجاج الفم", "جلطة", "stroke", "neurological weakness",
    }),
    "dental_infection": _normalized_markers({
        "تورم في الوش", "تورم في الوجه", "تورم الوجه", "وشي وارم", "وجهي وارم",
        "خراج مع حرارة", "ضرس مع سخونية", "facial swelling fever",
    }),
    "pregnancy_medication": _normalized_markers({"حامل واخد دوا", "حامل ودواء", "pregnant medication"}),
    "child_medication": _normalized_markers({"طفل ودواء", "جرعة طفل", "child medication", "child dose"}),
}

DURATION_MARKERS = _normalized_markers({
    "من امبارح", "من يوم", "من يومين", "من اسبوع", "بقاله", "بقالي", "منذ", "ساعه", "ساعة",
    "يوم", "يومين", "اسبوع", "شهر", "hour", "day", "days", "week", "month",
})

SEVERITY_MARKERS = _normalized_markers({
    "خفيف", "متوسط", "شديد", "قوي", "لا يحتمل", "مش مستحمل", "مش مستحمله", "mild", "moderate",
    "severe", "worst",
})

QUESTION_MARKERS = _normalized_markers({
    "ايه", "إيه", "هل", "ازاي", "اعمل ايه", "ماذا", "ليه", "ينفع", "ممكن", "what", "how", "why",
    "can i", "should i", "ما معنى", "معنى", "meaning",
})

NON_MEDICAL_CHAT_MARKERS = _normalized_markers({
    "فيلم", "افلام", "مزيكا", "اغنيه", "رياضه", "قهوه", "اكل", "سفر", "هزار", "بتحب", "بحب",
    "movie", "music", "sport", "coffee", "travel", "joke",
})


def normalizeUserMessage(message: str) -> str:
    return re.sub(r"\s+", " ", (message or "").strip())


def detectLanguage(message: str) -> str:
    if re.search(r"[\u0600-\u06FF]", message or ""):
        return "ar"
    return "en"


def detectRedFlags(message: str) -> dict[str, Any]:
    normalized = normalize_text(message)
    found: list[str] = []

    if any(marker in normalized for marker in CRISIS_MARKERS):
        found.append("suicidal_or_self_harm_thoughts")

    for flag_name, markers in RED_FLAG_MARKERS.items():
        if any(marker in normalized for marker in markers):
            found.append(flag_name)

    if "تورم" in normalized and "حراره" in normalized and any(marker in normalized for marker in DENTAL_MARKERS):
        found.append("severe_dental_infection_with_swelling_or_fever")
    if "حامل" in normalized and any(marker in normalized for marker in MEDICATION_MARKERS):
        found.append("pregnancy_medication_uncertainty")
    if "طفل" in normalized and any(marker in normalized for marker in MEDICATION_MARKERS | {"جرعه"}):
        found.append("child_medication_uncertainty")

    safety_level = SAFETY_EMERGENCY if found else SAFETY_LOW
    return {"found": bool(found), "red_flags": sorted(set(found)), "safety_level": safety_level}


def detectIntent(message: str, previousContext: list[dict[str, str]] | None = None) -> dict[str, Any]:
    clean = normalizeUserMessage(message)
    normalized = normalize_text(clean)
    tokens = tokenize(clean)
    previousContext = previousContext or []
    red_flags = detectRedFlags(clean)

    if red_flags["found"]:
        return _intent(INTENT_EMERGENCY, 0.98, related=False)
    if _is_unclear(clean, normalized, tokens):
        return _intent(INTENT_UNCLEAR, 0.94, related=False)
    if _contains(normalized, GREETING_MARKERS) and len(tokens) <= 4:
        return _intent(INTENT_GREETING, 0.96, related=False)
    if _looks_like_follow_up(clean, previousContext):
        return _intent(INTENT_FOLLOW_UP, 0.78, related=True)
    if _contains(normalized, ADMIN_MARKERS):
        return _intent(INTENT_ADMIN, 0.82, related=False)
    if _contains(normalized, MEDICATION_MARKERS):
        return _intent(INTENT_MEDICATION, 0.9, related=_is_related_to_previous(clean, previousContext))
    if _contains(normalized, MEDICAL_MARKERS):
        return _intent(INTENT_MEDICAL, 0.84, related=_is_related_to_previous(clean, previousContext))
    if _contains(normalized, MENTAL_HEALTH_MARKERS):
        return _intent(INTENT_MENTAL_HEALTH, 0.82, related=_is_related_to_previous(clean, previousContext))
    if _contains(normalized, NON_MEDICAL_CHAT_MARKERS):
        return _intent(INTENT_NON_MEDICAL, 0.82, related=False)
    if _contains(normalized, QUESTION_MARKERS):
        return _intent(INTENT_GENERAL, 0.72, related=_is_related_to_previous(clean, previousContext))
    return _intent(INTENT_NON_MEDICAL, 0.7 if len(tokens) >= 2 else 0.55, related=False)


def checkMessageCompleteness(message: str, intent: str) -> dict[str, Any]:
    clean = normalizeUserMessage(message)
    normalized = normalize_text(clean)
    tokens = tokenize(clean)
    missing: list[str] = []
    understandable = intent != INTENT_UNCLEAR and bool(tokens)
    enough = understandable

    if intent == INTENT_UNCLEAR:
        missing = ["السؤال أو الأعراض المقصودة"]
        enough = False
    elif intent in {INTENT_MEDICAL, INTENT_FOLLOW_UP}:
        if not _contains(normalized, DURATION_MARKERS):
            missing.append("مدة المشكلة")
        if not _contains(normalized, SEVERITY_MARKERS):
            missing.append("شدة الأعراض")
        if not re.search(r"\b\d{1,3}\b", normalized):
            missing.append("السن إذا كان مهمًا")
        if len(tokens) < 5:
            missing.append("تفاصيل الأعراض المصاحبة")
        enough = len(missing) <= 2 and len(tokens) >= 5
    elif intent == INTENT_MEDICATION:
        for item in ["سبب طلب الدواء", "السن", "حمل/رضاعة إن وجد", "أدوية حالية أو حساسية"]:
            if item not in missing:
                missing.append(item)
        if len(tokens) >= 8 and re.search(r"\b\d{1,3}\b", normalized):
            missing = [item for item in missing if item != "السن"]
        enough = len(tokens) >= 6
    elif intent == INTENT_MENTAL_HEALTH:
        enough = True
    elif intent in {INTENT_GREETING, INTENT_ADMIN, INTENT_GENERAL, INTENT_NON_MEDICAL}:
        enough = True

    return {
        "is_understandable": understandable,
        "has_enough_context": enough,
        "missing_information": missing,
        "requires_clarification": not enough,
        "related_to_previous": False,
        "safety_level": SAFETY_LOW,
    }


def extractCaseFacts(message: str, previousContext: list[dict[str, str]] | None = None) -> dict[str, Any]:
    clean = normalizeUserMessage(message)
    normalized = normalize_text(clean)
    previousContext = previousContext or []
    red_flags = detectRedFlags(clean)

    complaint = _first_matching_phrase(normalized, MEDICAL_MARKERS | MENTAL_HEALTH_MARKERS | DENTAL_MARKERS)
    medication = _first_matching_phrase(normalized, MEDICATION_MARKERS)
    related_previous_summary = _previous_user_summary(previousContext) if _is_related_to_previous(clean, previousContext) else ""
    previous_facts = _extract_previous_case_facts(related_previous_summary) if related_previous_summary else {}
    symptoms = _extract_symptoms(normalized)
    if previous_facts.get("symptoms"):
        symptoms = sorted(set(symptoms + previous_facts["symptoms"]), key=len, reverse=True)[:6]

    facts = {
        "language": detectLanguage(clean),
        "user_complaint": complaint or previous_facts.get("user_complaint", ""),
        "duration": _extract_duration(clean) or previous_facts.get("duration", ""),
        "age": _extract_age(clean) or previous_facts.get("age", ""),
        "symptoms": symptoms,
        "severity": _extract_severity(normalized) or previous_facts.get("severity", ""),
        "relevant_history": _extract_history(normalized) or previous_facts.get("relevant_history", ""),
        "current_medication": medication or previous_facts.get("current_medication", ""),
        "red_flags": red_flags["red_flags"],
        "what_user_is_asking": _extract_user_ask(clean),
        "is_dental": _contains(normalized, DENTAL_MARKERS) or bool(previous_facts.get("is_dental")),
        "related_previous_summary": related_previous_summary,
    }
    return facts


def decideResponseStrategy(
    intent: str,
    confidence: float,
    completeness: dict[str, Any],
    redFlags: dict[str, Any],
) -> dict[str, Any]:
    if redFlags.get("found") or intent == INTENT_EMERGENCY:
        strategy = "emergency"
        answer_directly = True
        ask_clarification = False
    elif confidence < 0.65 or intent == INTENT_UNCLEAR:
        strategy = "clarification"
        answer_directly = False
        ask_clarification = True
    elif completeness.get("requires_clarification") and intent in {INTENT_MEDICAL, INTENT_FOLLOW_UP, INTENT_MENTAL_HEALTH}:
        strategy = "clarification"
        answer_directly = False
        ask_clarification = True
    elif intent == INTENT_MEDICATION:
        strategy = "medication"
        answer_directly = True
        ask_clarification = False
    elif intent in {INTENT_MEDICAL, INTENT_FOLLOW_UP}:
        strategy = "medical"
        answer_directly = True
        ask_clarification = False
    elif intent in {INTENT_GREETING, INTENT_GENERAL, INTENT_ADMIN, INTENT_NON_MEDICAL}:
        strategy = "general"
        answer_directly = True
        ask_clarification = False
    else:
        strategy = "pass_to_existing_mental_health"
        answer_directly = False
        ask_clarification = False

    return {
        "strategy": strategy,
        "should_answer_directly": answer_directly,
        "should_ask_clarification": ask_clarification,
    }


def buildClarificationReply(missingInfo: list[str], language: str = "ar") -> str:
    if not missingInfo:
        return "مش واضح قصدك. ممكن تكتب السؤال تاني بشكل أوضح أو تقول المشكلة/الأعراض اللي بتسأل عنها؟"
    needed = "، ".join(dict.fromkeys(missingInfo[:5]))
    return f"مش واضح قصدك من الرسالة أو المعلومات ناقصة. اكتب السؤال أو الأعراض بتفصيل بسيط، خصوصًا: {needed}."


def buildSafeMedicalReply(caseFacts: dict[str, Any], safetyLevel: str = SAFETY_LOW) -> str:
    if safetyLevel == SAFETY_EMERGENCY or caseFacts.get("red_flags"):
        flags = "، ".join(caseFacts.get("red_flags") or ["علامات خطر"])
        return (
            f"فيه علامة خطر محتملة: {flags}.\n"
            "الأفضل طلب رعاية طبية عاجلة/طوارئ الآن، خصوصًا لو الأعراض شديدة أو بتزيد. "
            "ما تعتمدش على رد الشات في الحالة دي، وخلي شخص قريب يساعدك في الوصول للطبيب."
        )

    if caseFacts.get("current_medication"):
        med = caseFacts["current_medication"]
        complaint = caseFacts.get("user_complaint") or "الأعراض الحالية"
        return (
            f"هل الدواء مطلوب أصلًا؟\n"
            f"فهمت إن السؤال عن {med} مع {complaint}. ما ينفعش أأكد احتياج الدواء أو أدي جرعة من غير تقييم السن، الحالة، الحساسية، والأدوية الحالية.\n\n"
            "الفئة العلاجية المناسبة إن وجدت\n"
            "اختيار نوع العلاج يعتمد على السبب: ألم، التهاب، حساسية، عدوى، أو سبب آخر. المضاد الحيوي مثلًا لا يستخدم إلا عند وجود سبب واضح يحدده طبيب/صيدلي.\n\n"
            "المخاطر والتحذيرات\n"
            "تجنب خلط الأدوية أو تكرار نفس المادة الفعالة، وراجع مختصًا لو في حمل/رضاعة، طفل، مرض كبد/كلى، قرحة، سيولة، أو حساسية دوائية.\n\n"
            "متى يجب الرجوع للطبيب/الصيدلي\n"
            "لو الأعراض شديدة، مستمرة، معها حرارة/تورم/ضيق نفس/نزيف، أو الدواء لطفل أو أثناء الحمل.\n\n"
            "ممنوع إعطاء جرعات غير مؤكدة\n"
            "لا أقدر أوصف جرعة دقيقة هنا بدون بيانات سريرية مؤكدة. اكتب السن، الوزن لو طفل، سبب الدواء، والأدوية الحالية."
        )

    complaint = caseFacts.get("user_complaint") or "مشكلة صحية"
    symptoms = "، ".join(caseFacts.get("symptoms") or [])
    duration = caseFacts.get("duration") or "غير مذكورة"
    severity = caseFacts.get("severity") or "غير مذكورة"
    questions = _targeted_questions(caseFacts)
    red_flag_line = _red_flag_line(caseFacts)
    safe_steps = _safe_steps(caseFacts)

    return (
        f"فهمت من كلامك أن عندك {complaint}"
        f"{' مع ' + symptoms if symptoms else ''}. المدة: {duration}، والشدة: {severity}.\n\n"
        "الاحتمالات العامة بدون تشخيص نهائي\n"
        "الأعراض ممكن يكون لها أكثر من سبب، والتحديد يعتمد على المكان الدقيق، المدة، الشدة، والأعراض المصاحبة. لا أقدر أشخص نهائيًا من الرسالة فقط.\n\n"
        "علامات الخطر التي تستدعي طبيب/طوارئ\n"
        f"{red_flag_line}\n\n"
        "ماذا تفعل الآن بشكل آمن\n"
        f"{safe_steps}\n\n"
        "أسئلة مهمة لتحديد الحالة\n"
        f"{questions}"
    )


def buildGeneralReply(intent: str, caseFacts: dict[str, Any]) -> str:
    if intent == INTENT_GREETING:
        return "وعليكم السلام. أهلا بيك، اكتب سؤالك أو الأعراض اللي حابب تسأل عنها."
    if intent == INTENT_ADMIN:
        return "لو المشكلة في استخدام التطبيق، اكتب لي الجزء اللي مش شغال: تسجيل الدخول، الشات، الإشعارات، أو حفظ المحادثات."
    if intent == INTENT_GENERAL:
        return "أقدر أساعدك. اكتب السؤال بتفاصيل كافية عشان أرد بشكل أدق."
    return "تمام. أقدر أرد على الكلام العام، ولو عندك سؤال صحي أو نفسي اكتب التفاصيل بوضوح."


def generateFinalBotReply(message: str, previousContext: list[dict[str, str]] | None = None) -> dict[str, Any]:
    normalized = normalizeUserMessage(message)
    language = detectLanguage(normalized)
    intent_result = detectIntent(normalized, previousContext)
    intent = intent_result["intent"]
    confidence = intent_result["confidence"]
    completeness = checkMessageCompleteness(normalized, intent)
    completeness["related_to_previous"] = intent_result.get("related_to_previous", False)
    red_flags = detectRedFlags(normalized)
    case_facts = extractCaseFacts(normalized, previousContext)
    if intent == INTENT_FOLLOW_UP and completeness["related_to_previous"]:
        context_signals = sum(
            1
            for value in [
                case_facts.get("user_complaint"),
                case_facts.get("duration"),
                case_facts.get("severity"),
                case_facts.get("age"),
                case_facts.get("symptoms"),
            ]
            if value
        )
        if context_signals >= 2:
            completeness["has_enough_context"] = True
            completeness["requires_clarification"] = False
            completeness["missing_information"] = []
    strategy = decideResponseStrategy(intent, confidence, completeness, red_flags)
    if intent == INTENT_FOLLOW_UP and completeness["related_to_previous"] and not _case_has_medical_surface(case_facts):
        strategy = {
            "strategy": "pass_to_existing_mental_health",
            "should_answer_directly": False,
            "should_ask_clarification": False,
        }
    elif intent == INTENT_GENERAL and completeness["related_to_previous"]:
        strategy = {
            "strategy": "pass_to_existing_mental_health",
            "should_answer_directly": False,
            "should_ask_clarification": False,
        }

    if strategy["strategy"] == "clarification":
        reply = buildClarificationReply(completeness["missing_information"], language)
    elif strategy["strategy"] == "emergency":
        reply = buildSafeMedicalReply(case_facts, SAFETY_EMERGENCY)
    elif strategy["strategy"] in {"medical", "medication"}:
        reply = buildSafeMedicalReply(case_facts, red_flags["safety_level"])
    elif strategy["strategy"] == "general":
        reply = buildGeneralReply(intent, case_facts)
    else:
        reply = ""

    return {
        "normalized_message": normalized,
        "language": language,
        "intent": intent,
        "confidence": confidence,
        "missing_information": completeness["missing_information"],
        "safety_level": red_flags["safety_level"],
        "red_flags": red_flags["red_flags"],
        "should_answer_directly": strategy["should_answer_directly"],
        "should_ask_clarification": strategy["should_ask_clarification"],
        "requires_clarification": completeness["requires_clarification"],
        "is_understandable": completeness["is_understandable"],
        "has_enough_context": completeness["has_enough_context"],
        "related_to_previous": completeness["related_to_previous"],
        "case_facts": case_facts,
        "strategy": strategy["strategy"],
        "reply": reply,
    }


def _intent(intent: str, confidence: float, *, related: bool) -> dict[str, Any]:
    return {"intent": intent, "confidence": confidence, "related_to_previous": related}


def _contains(normalized: str, markers: set[str]) -> bool:
    return any(marker and marker in normalized for marker in markers)


def _is_unclear(clean: str, normalized: str, tokens: list[str]) -> bool:
    if not normalized:
        return True
    if len(normalized) <= 2:
        return True
    if _contains(normalized, GREETING_MARKERS | MENTAL_HEALTH_MARKERS | MEDICAL_MARKERS | MEDICATION_MARKERS):
        return False
    if len(tokens) <= 2 and not re.search(r"[\u0600-\u06FF]", clean):
        letters = re.sub(r"[^a-zA-Z]", "", clean)
        if letters and not re.search(r"[aeiouAEIOU]", letters):
            return True
    if len(tokens) <= 2 and re.fullmatch(r"[a-zA-Z\s]{1,8}", clean):
        return True
    return False


def _looks_like_follow_up(clean: str, previousContext: list[dict[str, str]]) -> bool:
    normalized = normalize_text(clean)
    if not previousContext:
        return False
    follow_markers = _normalized_markers({
        "طيب اعمل ايه", "اعمل ايه", "طب ايه", "وكمان", "لسه", "نفس الوجع", "من ساعتها",
        "what should i do", "still", "same pain", "رشح", "رشحيلي", "ابعت", "ابعتيلي", "فيديو", "كتاب",
    })
    if not _contains(normalized, follow_markers):
        return False
    return _is_related_to_previous(clean, previousContext)


def _is_related_to_previous(clean: str, previousContext: list[dict[str, str]]) -> bool:
    previous_user = [item.get("content", "") for item in previousContext if item.get("role") == "user"]
    if not previous_user:
        return False
    recent = " ".join(previous_user[-2:])
    recent_normalized = normalize_text(recent)
    current_normalized = normalize_text(clean)
    if _contains(recent_normalized, MEDICAL_MARKERS | MENTAL_HEALTH_MARKERS | MEDICATION_MARKERS):
        if lexical_overlap_score(current_normalized, recent_normalized) >= 0.05:
            return True
        return _contains(current_normalized, _normalized_markers({
            "اعمل ايه", "لسه", "نفس", "still", "same", "رشح", "رشحيلي", "ابعت", "ابعتيلي", "فيديو", "كتاب",
        }))
    return False


def _case_has_medical_surface(caseFacts: dict[str, Any]) -> bool:
    complaint = normalize_text(caseFacts.get("user_complaint") or "")
    symptoms = {normalize_text(symptom) for symptom in caseFacts.get("symptoms") or []}
    medical_markers = MEDICAL_MARKERS | DENTAL_MARKERS
    return bool(
        caseFacts.get("is_dental")
        or caseFacts.get("current_medication")
        or complaint in medical_markers
        or any(symptom in medical_markers for symptom in symptoms)
    )


def _first_matching_phrase(normalized: str, markers: set[str]) -> str:
    matches = [marker for marker in markers if marker and marker in normalized]
    if not matches:
        return ""
    return max(matches, key=len)


def _extract_duration(clean: str) -> str:
    match = re.search(r"(منذ|من|بقالي|بقاله)\s+([\w\u0600-\u06FF\s]{1,20})", clean, flags=re.I)
    if match:
        return match.group(0).strip()
    match = re.search(r"\b\d+\s*(ساعه|ساعة|يوم|ايام|أيام|اسبوع|أسبوع|شهر|hour|hours|day|days|week|month)\b", clean, flags=re.I)
    return match.group(0).strip() if match else ""


def _extract_age(clean: str) -> str:
    patterns = [
        r"(?:سني|عمري|السن)\s*(\d{1,3})",
        r"(\d{1,3})\s*(?:سنة|سنه|عام|years old|yo)",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean, flags=re.I)
        if match:
            return match.group(1)
    return ""


def _extract_symptoms(normalized: str) -> list[str]:
    symptom_markers = MEDICAL_MARKERS | MENTAL_HEALTH_MARKERS | DENTAL_MARKERS
    symptoms = [marker for marker in symptom_markers if marker and marker in normalized]
    return sorted(set(symptoms), key=len, reverse=True)[:6]


def _extract_severity(normalized: str) -> str:
    return _first_matching_phrase(normalized, SEVERITY_MARKERS)


def _extract_history(normalized: str) -> str:
    history_markers = _normalized_markers({"سكر", "ضغط", "حساسيه", "كبد", "كلي", "قرحه", "حامل", "رضاعه", "diabetes", "hypertension"})
    return _first_matching_phrase(normalized, history_markers)


def _extract_user_ask(clean: str) -> str:
    if "?" in clean or "؟" in clean:
        return clean
    normalized = normalize_text(clean)
    if _contains(normalized, QUESTION_MARKERS | MEDICATION_MARKERS):
        return clean
    return "طلب نصيحة أو تفسير عام"


def _extract_previous_case_facts(previous_message: str) -> dict[str, Any]:
    if not previous_message:
        return {}
    normalized = normalize_text(previous_message)
    return {
        "user_complaint": _first_matching_phrase(normalized, MEDICAL_MARKERS | MENTAL_HEALTH_MARKERS | DENTAL_MARKERS),
        "duration": _extract_duration(previous_message),
        "age": _extract_age(previous_message),
        "symptoms": _extract_symptoms(normalized),
        "severity": _extract_severity(normalized),
        "relevant_history": _extract_history(normalized),
        "current_medication": _first_matching_phrase(normalized, MEDICATION_MARKERS),
        "is_dental": _contains(normalized, DENTAL_MARKERS),
    }


def _previous_user_summary(previousContext: list[dict[str, str]]) -> str:
    previous_user = [item.get("content", "") for item in previousContext if item.get("role") == "user"]
    return previous_user[-1] if previous_user else ""


def _targeted_questions(caseFacts: dict[str, Any]) -> str:
    questions = []
    if not caseFacts.get("age"):
        questions.append("السن كام؟")
    if not caseFacts.get("duration"):
        questions.append("الأعراض بقالها قد إيه؟")
    if not caseFacts.get("severity"):
        questions.append("الشدة خفيفة ولا متوسطة ولا شديدة؟")
    if caseFacts.get("is_dental"):
        questions.append("فيه تورم في الوجه أو حرارة أو صعوبة بلع؟")
    else:
        questions.append("فيه حرارة، قيء، نزيف، ضيق نفس، أو ألم شديد؟")
    return " ".join(questions)


def _red_flag_line(caseFacts: dict[str, Any]) -> str:
    if caseFacts.get("is_dental"):
        return "تورم الوجه، حرارة، صعوبة فتح الفم/البلع، أو ألم شديد متزايد يحتاج كشف عاجل."
    return "ألم صدر، ضيق نفس شديد، إغماء، ضعف/تنميل مفاجئ، نزيف لا يتوقف، حساسية شديدة، أو ألم شديد متزايد."


def _safe_steps(caseFacts: dict[str, Any]) -> str:
    if caseFacts.get("is_dental"):
        return "احجز كشف أسنان قريبًا، وتجنب وضع مسكنات مباشرة على اللثة. لو في تورم وجه أو حرارة، اعتبرها حالة عاجلة."
    if caseFacts.get("user_complaint") in {"بطني", "البطن", "معده", "اسهال", "قيء"}:
        return "اشرب سوائل على دفعات، وتجنب الأكل التقيل مؤقتًا. اطلب كشفًا لو الألم شديد، مستمر، أو معه حرارة/قيء متكرر/دم."
    return "راقب الأعراض، ارتح، وتجنب أي دواء غير موصوف لو عندك حساسية أو أمراض مزمنة. راجع طبيبًا لو الأعراض شديدة أو مستمرة."
