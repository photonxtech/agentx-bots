from functools import lru_cache

import chromadb

from app.config import settings


@lru_cache
def get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)
