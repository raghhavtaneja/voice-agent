"""Central settings — the only place env vars are read."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Anthropic (production LLM per CLAUDE.md; optional while on the Gemini backend)
    anthropic_api_key: str | None = None
    claude_model: str = "claude-sonnet-4-5"

    # Gemini — earlier interim backend; kept for fallback (free tier too small).
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash-lite"

    # Groq — current interim LLM backend: generous free tier + fastest inference.
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    # Pinecone — integrated index handles embeddings server-side
    pinecone_api_key: str
    pinecone_index: str = "voice-agent"
    pinecone_embed_model: str = "llama-text-embed-v2"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Supabase
    supabase_url: str
    supabase_service_key: str
    supabase_storage_bucket: str = "documents"

    # Vapi (optional until phone numbers are wired up)
    vapi_api_key: str | None = None
    vapi_webhook_secret: str | None = None

    # RAG
    # Smaller chunks keep each piece semantically focused (e.g. ~1-2 FAQ Q&As)
    # so embeddings stay sharp; 512 produced page-sized, topic-mixed chunks.
    chunk_tokens: int = 200
    chunk_overlap_tokens: int = 40
    top_k: int = 4
    # Cosine-similarity floor for retrieval. Matches below this are treated as
    # "no relevant context" — the LLM is told the docs don't cover the question.
    # Calibrated against llama-text-embed-v2 (on-topic ~0.45-0.70, off-topic ~0.20).
    min_relevance_score: float = 0.30

    # App
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings = get_settings()
