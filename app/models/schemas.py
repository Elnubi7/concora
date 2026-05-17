from typing import Literal
from pydantic import BaseModel, Field, field_validator
from app.core.constants import MBTI_TYPES


class ResourceItem(BaseModel):
    title: str = Field(..., description="اسم المورد بعد التنظيف")
    category: str = Field(..., description="video أو book أو podcast")
    url: str | None = Field(default=None, description="الرابط الأصلي إن وجد")
    source_collection: str = Field(..., description="mbti أو emotion")
    source_domain: str | None = Field(default=None, description="hostname الحقيقي للرابط مثل youtube.com إن وجد")
    source_document: str = Field(..., description="اسم الملف الذي جاء منه المورد")
    issue_title: str | None = Field(default=None, description="المحور الأقرب المرتبط بالمورد")
    topic_title: str | None = Field(default=None, description="التوبك العام المرتبط بالمورد")
    score: float | None = Field(default=None, description="درجة الملاءمة")
    why_recommended: str | None = Field(default=None, description="سبب ظهور المورد")


class RetrievedChunk(BaseModel):
    chunk_id: str
    title: str
    chunk_type: str
    domain: str
    mbti_type: str | None = None
    topic_title: str | None = None
    score: float
    text: str


class SafetyMeta(BaseModel):
    is_crisis: bool = False
    reason: str | None = None
    blocked: bool = False


class RecommendationsMeta(BaseModel):
    unlocked: bool
    current_turn: int
    unlock_turn: int
    note: str


class DebugMeta(BaseModel):
    detected_intent: str
    response_mode: str
    followup_question_reason: str | None = None
    intent_confidence: float | None = None
    missing_information: list[str] = Field(default_factory=list)
    safety_level: str | None = None
    should_answer_directly: bool | None = None
    should_ask_clarification: bool | None = None
    is_understandable: bool | None = None
    has_enough_context: bool | None = None
    related_to_previous: bool | None = None
    red_flags: list[str] = Field(default_factory=list)
    issue_match_scores: dict[str, float] = Field(default_factory=dict)
    topic_match_scores: dict[str, float] = Field(default_factory=dict)
    recommendation_triggered: bool = False
    repeated_message_count: int = 1
    repeated_meaning_count: int = 1
    exact_repetition_count: int = 1
    semantic_repetition_count: int = 1
    loop_detected: bool = False


class StructuredAnswer(BaseModel):
    understanding: str = Field(..., description="فهم مختصر للسؤال")
    mbti_connection: str = Field(default="", description="ربط احتمالي غير قطعي مع MBTI")
    grounded_answer: str = Field(..., description="إجابة retrieval-based بدون ادعاء مبالغ")
    practical_steps: list[str] = Field(default_factory=list, description="خطوات عملية قصيرة")
    follow_up_question: str | None = Field(default=None, description="سؤال متابعة قصير عند الحاجة")
    choice_prompt: str | None = Field(default=None, description="اختيار لطيف يساعد المستخدم يوضح أكثر")
    support_note: str = Field(default="", description="تنبيه الدعم الآمن")


class ChatRequest(BaseModel):
    conversation_id: str | None = Field(default=None, description="لو فارغ سيرجع الخادم معرفًا جديدًا")
    user_message: str = Field(..., min_length=2, description="رسالة المستخدم")
    mbti_type: str | None = Field(default=None, description="أحد أنماط MBTI أو اتركيه فارغًا")
    user_gender: Literal["female", "male", "other"] = Field(default="female")
    response_style: Literal["config", "feminine", "neutral"] = Field(default="config")
    top_k: int = Field(default=8, ge=1, le=15)
    max_videos: int = Field(default=2, ge=0, le=5)
    max_books: int = Field(default=3, ge=0, le=6)
    max_podcasts: int = Field(default=2, ge=0, le=5)
    include_recommendations: bool = Field(default=True)
    recommendations_after_turn: int = Field(default=4, ge=1, le=10)
    recommendation_links_only: bool = Field(default=True, description="لو true لا تظهر أي توصية بدون رابط")
    debug: bool = Field(default=False, description="إرجاع بيانات تشخيصية آمنة للتطوير والاختبارات فقط")

    @field_validator("mbti_type")
    @classmethod
    def validate_mbti(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.upper().strip()
        if not normalized:
            return None
        if normalized not in MBTI_TYPES:
            raise ValueError(f"mbti_type must be one of: {', '.join(MBTI_TYPES)}")
        return normalized


class ChatResponse(BaseModel):
    source: str
    conversation_id: str
    turn_number: int
    mbti_type: str | None = None
    matched_issue_titles: list[str] = Field(default_factory=list)
    matched_topic_titles: list[str] = Field(default_factory=list)
    response: StructuredAnswer
    recommendations: RecommendationsMeta
    recommended_videos: list[ResourceItem] = Field(default_factory=list)
    recommended_books: list[ResourceItem] = Field(default_factory=list)
    recommended_podcasts: list[ResourceItem] = Field(default_factory=list)
    safety: SafetyMeta
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    debug: DebugMeta | None = None


class MBTIOverview(BaseModel):
    mbti_type: str
    core_problems: list[str]
    consequences: list[str]


class TopicOverview(BaseModel):
    topic_title: str
    questions_count: int
    resources_count: int


class IndexStatus(BaseModel):
    status: str
    chunks_count: int
    embeddings_count: int
    used_provider: str
