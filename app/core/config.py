from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    APP_NAME: str = "Mental Health MBTI RAG Bot"
    APP_VERSION: str = "3.1.0"
    API_PREFIX: str = "/api/v1"

    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    DATA_DIR: Path = BASE_DIR / "data"
    SOURCE_DIR: Path = DATA_DIR / "source"
    MBTI_DOCX_PATH: Path = SOURCE_DIR / "mbti_mental_health_girls.docx"
    EMOTION_DOCX_PATH: Path = SOURCE_DIR / "rufi_emotional_knowledge.docx"
    KB_JSON_PATH: Path = DATA_DIR / "processed" / "knowledge_base.json"
    CHUNKS_JSON_PATH: Path = DATA_DIR / "processed" / "chunks.json"
    VECTOR_INDEX_PATH: Path = DATA_DIR / "processed" / "vector_index.json"

    GITHUB_TOKEN: str | None = None
    GITHUB_MODELS_BASE_URL: str = "https://models.github.ai"
    GITHUB_MODELS_ORG: str | None = None
    GITHUB_API_VERSION: str = "2026-03-10"
    GITHUB_CHAT_MODEL: str = "openai/gpt-4.1"
    GITHUB_EMBEDDING_MODEL: str = "openai/text-embedding-3-small"
    ENABLE_GITHUB_CHAT_GENERATION: bool = True

    DEFAULT_TOP_K: int = 8
    HISTORY_TURNS_TO_KEEP: int = 8
    EMBEDDING_BATCH_SIZE: int = 32
    AUTO_BUILD_EMBEDDINGS_ON_STARTUP: bool = False
    ENABLE_RULE_BASED_FALLBACK: bool = True

    MAX_RESPONSE_TOKENS: int = 1100
    DEFAULT_TEMPERATURE: float = 0.15
    REQUEST_TIMEOUT_SECONDS: int = 60

    TARGET_AUDIENCE: str = "female_only"  # female_only | all
    RESPONSE_STYLE: str = "feminine"  # feminine | neutral
    RECOMMENDATIONS_AFTER_TURN: int = 4
    DEFAULT_MAX_VIDEOS: int = 2
    DEFAULT_MAX_BOOKS: int = 3
    DEFAULT_MAX_PODCASTS: int = 2
    CHAT_FOLLOWUP_TURNS: int = 2
    DEBUG_CHAT_METADATA: bool = False

    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])


@lru_cache
def get_settings() -> Settings:
    return Settings()
