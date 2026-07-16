from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str
    jwt_secret: str
    jwt_expire_minutes: int = 480
    admin_email: str
    admin_password: str

    cors_origins: str = "http://localhost:5173"

    database_url: str = "sqlite:///./data/app.db"
    chroma_persist_dir: str = "./data/chroma"

    max_pages: int = 500
    max_crawl_depth: int = 5
    crawler_concurrency: int = 2
    crawler_delay_seconds: float = 1.0

    # Threshold applied to the post-rerank relevance score (0-1, sigmoid of the
    # cross-encoder's raw logit) — not a raw cosine similarity, since reranking replaced
    # that as the final relevance judgment. See app/llm/retrieval.py.
    relevance_threshold: float = 0.01
    top_k_chunks: int = 25
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    openai_model: str = "gpt-4o-mini"

    response_cache_ttl_seconds: int = 600

    daily_sync_hour: int = 3
    log_level: str = "INFO"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
