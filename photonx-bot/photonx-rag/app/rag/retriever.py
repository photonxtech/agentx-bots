"""Retriever (STEP 5).

Turns a natural-language question into the ``top_k`` most relevant chunks by
embedding the query and running a cosine-similarity search over the vector
store. Composition (embedder + vector store) rather than inheritance keeps each
piece independently testable and swappable — this is where a re-ranker or
hybrid (keyword + vector) search would slot in later.
"""

from __future__ import annotations

from app.core.logger import get_logger
from app.rag.embedding import Embedder
from app.rag.vector_store import VectorStore
from app.schemas.documents import RetrievedChunk

logger = get_logger(__name__)


class Retriever:
    """Embed a query and fetch the most similar chunks."""

    def __init__(
        self, embedder: Embedder, vector_store: VectorStore, top_k: int
    ) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._top_k = top_k

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Return chunks most relevant to ``question`` (highest score first)."""
        logger.info("Retrieving context for query: %r", question)
        query_vector = self._embedder.embed_query(question)
        results = self._store.query(query_vector, top_k=self._top_k)
        results.sort(key=lambda r: r.score, reverse=True)
        return results
