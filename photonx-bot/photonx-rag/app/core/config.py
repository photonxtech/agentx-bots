"""Application configuration.

All runtime settings are loaded from environment variables (and a local
``.env`` file) exactly once and exposed as a cached, immutable ``Settings``
object. Centralizing configuration here means no other module ever reads
``os.environ`` directly — they depend on this typed object instead, which keeps
the rest of the codebase testable and decoupled from the environment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated application settings.

    Values are read from environment variables first, then from a ``.env``
    file. Field names are matched case-insensitively to env var names.
    """

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
    chroma_collection: str = Field("photonx_docs", description="Collection name.")

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
    # Comma-separated list of origins allowed to call the API from a browser
    # (the React frontend). Defaults cover the local Vite dev server.
    cors_origins: str = Field(
        "http://localhost:5173,http://127.0.0.1:5173",
        description="Comma-separated allowed browser origins.",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse ``cors_origins`` into a clean list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def model_post_init(self, __context: object) -> None:
        """Validate cross-field invariants after the model is built."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE "
                f"(got overlap={self.chunk_overlap}, size={self.chunk_size})."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide singleton ``Settings`` instance.

    Cached so the ``.env`` file and environment are parsed only once. This
    function is the dependency-injection seam used across the app (including
    FastAPI ``Depends``), which makes overriding config in tests trivial.
    """
    return Settings()  # type: ignore[call-arg]
