from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache
def get_embedder() -> HuggingFaceEmbeddings:
    logger.info("Loading embedding model %s", settings.embedding_model)
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedder().embed_documents(texts)


@lru_cache(maxsize=2048)
def _embed_query_cached(text: str) -> tuple[float, ...]:
    # An embedding is a pure function of (text, model) — no TTL/invalidation needed,
    # unlike retrieval/response caches whose correctness depends on the site's content.
    return tuple(get_embedder().embed_query(text))


def embed_query(text: str) -> list[float]:
    return list(_embed_query_cached(text))
