"""Vector store (STEP 4).

Thin wrapper over a persistent ChromaDB collection. We supply our own
embeddings (from :class:`~app.rag.embedding.Embedder`), so Chroma is used purely
as an ANN index + metadata store. The collection is configured for **cosine**
distance to match the retrieval requirement.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.exceptions import VectorStoreError
from app.core.logger import get_logger
from app.schemas.documents import Chunk, RetrievedChunk

logger = get_logger(__name__)


class VectorStore:
    """Persistent ChromaDB-backed store keyed on chunk embeddings."""

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        try:
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:  # chromadb raises broad exceptions
            raise VectorStoreError(f"Failed to open ChromaDB: {exc}") from exc

        self._collection_name = collection_name
        logger.info(
            "Vector store ready (collection=%s, dir=%s)",
            collection_name,
            persist_dir,
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert (or upsert) chunks and their vectors.

        Args:
            chunks: Chunks to store; ``chunk.id`` is used as the vector id.
            embeddings: Vectors aligned 1:1 with ``chunks``.

        Raises:
            VectorStoreError: On length mismatch or a Chroma failure.
        """
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "length mismatch"
            )
        if not chunks:
            return

        try:
            self._collection.upsert(
                ids=[c.id for c in chunks],
                documents=[c.text for c in chunks],
                embeddings=embeddings,
                metadatas=[c.metadata() for c in chunks],
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to insert vectors: {exc}") from exc

        logger.info("Upserted %d vectors into '%s'", len(chunks), self._collection_name)

    def query(
        self, embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks for a query embedding.

        Chroma returns a cosine *distance* in ``[0, 2]``; we convert it to a
        similarity *score* in ``[0, 1]`` via ``score = 1 - distance / 2``.
        """
        try:
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Query failed: {exc}") from exc

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for text, meta, distance in zip(documents, metadatas, distances):
            retrieved.append(
                RetrievedChunk(
                    text=text,
                    document=str(meta.get("document", "unknown")),
                    page=int(meta.get("page", 0) or 0),
                    section=str(meta.get("section", "General")),
                    score=self._distance_to_score(float(distance)),
                )
            )

        logger.info("Retrieved %d chunks (top_k=%d)", len(retrieved), top_k)
        return retrieved

    def reset(self) -> None:
        """Delete and recreate the collection (drops all vectors).

        Used before a full re-ingest so stale chunks from a previous run don't
        linger alongside freshly generated ones.
        """
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to reset collection: {exc}") from exc
        logger.info("Reset collection '%s'", self._collection_name)

    def count(self) -> int:
        """Return the number of vectors stored."""
        try:
            return self._collection.count()
        except Exception as exc:
            raise VectorStoreError(f"Count failed: {exc}") from exc

    def distinct_documents(self) -> int:
        """Return the number of distinct source documents indexed."""
        try:
            metadatas = self._collection.get(include=["metadatas"]).get(
                "metadatas", []
            )
        except Exception as exc:
            raise VectorStoreError(f"Metadata scan failed: {exc}") from exc
        return len({m.get("document") for m in metadatas if m})

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        """Map cosine distance [0, 2] → similarity score [0, 1]."""
        score = 1.0 - (distance / 2.0)
        return max(0.0, min(1.0, score))
