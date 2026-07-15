"""Application configuration.

Identical in spirit to the manual project: all settings load once from ``.env``
into a typed, cached ``Settings`` object. LangChain doesn't change how *we*
manage configuration — it only changes how the RAG components are built.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- OpenAI ---
    openai_api_key: str = Field(..., description="OpenAI API key.")
    openai_model: str = Field("gpt-4.1", description="Chat completion model.")
    embedding_model: str = Field(
        "text-embedding-3-small", description="Embedding model."
    )

    # --- Vector store ---
    chroma_db: Path = Field(Path("./chroma_db"), description="Chroma data dir.")
    chroma_collection: str = Field("photonx_docs_lc", description="Collection.")

    # --- Documents ---
    docs_dir: Path = Field(
        Path("./docs"), description="Source folder for PDF/DOCX documents."
    )

    # --- Chunking ---
    chunk_size: int = Field(800, ge=100, description="Chunk size in characters.")
    chunk_overlap: int = Field(150, ge=0, description="Overlap in characters.")

    # --- Retrieval ---
    top_k: int = Field(5, ge=1, le=50, description="Chunks retrieved per query.")

    # --- Generation ---
    llm_temperature: float = Field(0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(800, ge=1)

    # --- Logging ---
    log_level: str = Field("INFO", description="Root log level.")

    # --- CORS ---
    cors_origins: str = Field(
        "http://localhost:5173,http://127.0.0.1:5173",
        description="Comma-separated allowed browser origins.",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse ``cors_origins`` into a clean list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def model_post_init(self, __context: object) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE "
                f"(got overlap={self.chunk_overlap}, size={self.chunk_size})."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide singleton ``Settings`` instance."""
    return Settings()  # type: ignore[call-arg]
