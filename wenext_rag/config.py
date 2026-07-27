import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # API Keys
    GEMINI_API_KEY: str

    # Paths
    PROJECT_ROOT: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    DB_PATH: Path = BASE_DIR / "chroma_db"
    FRONTEND_DIR: Path = BASE_DIR / "frontend"
    INGEST_MANIFEST_PATH: Path = BASE_DIR / "chroma_db" / "ingest_manifest.json"

    # RAG Settings
    RELEVANCE_THRESHOLD: float = 0.85
    TOP_K: int = 8
    EMBEDDING_MODEL: str = "gemini-embedding-2-preview"
    CHAT_MODEL: str = "gemini-2.5-flash"
    COLLECTION_NAME: str = "rag_documents"
    PIPELINE_VERSION: str = "langchain-v1"

    # Text Splitting Settings
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: 150

    # Server Settings
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True
    MAX_SESSION_HISTORY_TURNS: int = 20

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
