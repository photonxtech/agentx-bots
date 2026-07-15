"""Ingestion service (orchestrates STEPs 1–4).

Coordinates the load → chunk → embed → store pipeline for the docs folder. It
holds no infrastructure logic itself; it only sequences the injected
components, which keeps the pipeline readable and each stage independently
testable.
"""

from __future__ import annotations

from pathlib import Path

from app.core.exceptions import DocumentError
from app.core.logger import get_logger
from app.rag.chunker import Chunker
from app.rag.document_loader import CompositeLoader
from app.rag.embedding import Embedder
from app.rag.vector_store import VectorStore
from app.schemas.documents import PageContent

logger = get_logger(__name__)


class IngestionService:
    """Load, chunk, embed, and store all documents in the docs folder."""

    def __init__(
        self,
        loader: CompositeLoader,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        docs_dir: Path,
    ) -> None:
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._store = vector_store
        self._docs_dir = docs_dir

    def ingest(self, reset: bool = True) -> tuple[int, int]:
        """Run the full ingestion pipeline.

        Args:
            reset: If ``True`` (default), clear the collection first so the
                index reflects exactly the current docs folder. Set ``False``
                to append to an existing index.

        Returns:
            ``(num_documents, num_chunks)`` that were embedded and stored.

        Raises:
            DocumentError: If there are no readable PDFs.
        """
        logger.info("Starting ingestion from %s (reset=%s)", self._docs_dir, reset)

        pages: list[PageContent] = list(self._loader.load_dir(self._docs_dir))
        if not pages:
            raise DocumentError("No readable content found in any PDF.")

        num_documents = len({p.document for p in pages})

        chunks = self._chunker.chunk_pages(pages)
        if not chunks:
            raise DocumentError("Documents produced no chunks.")

        if reset:
            self._store.reset()

        embeddings = self._embedder.embed_texts([c.text for c in chunks])
        self._store.add(chunks, embeddings)

        logger.info(
            "Ingestion complete: %d documents, %d chunks.",
            num_documents,
            len(chunks),
        )
        return num_documents, len(chunks)
