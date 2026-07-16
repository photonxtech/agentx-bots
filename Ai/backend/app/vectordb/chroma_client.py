from functools import lru_cache
from pathlib import Path

import chromadb

from app.config import settings


@lru_cache
def get_chroma_client() -> chromadb.ClientAPI:
    # `data/` is gitignored — a fresh checkout/deploy never has it, so create it up front
    # rather than relying on ChromaDB to do so (same issue that hit the SQLite connection).
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)
