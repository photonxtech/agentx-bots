"""Composition root / dependency-injection container.

All concrete objects are constructed *here*, once, and wired together. Every
other module receives its collaborators via constructor arguments and depends on
abstractions, not on ``get_settings()`` or the OpenAI SDK. This is the single
place to change when swapping an implementation (e.g. a different vector store),
and it makes the whole graph trivial to fake in tests.
"""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from app.core.config import Settings, get_settings
from app.core.logger import get_logger
from app.rag.chunker import Chunker
from app.rag.document_loader import CompositeLoader
from app.rag.embedding import Embedder
from app.rag.llm import LLMClient
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.services.ingestion_service import IngestionService
from app.services.qa_service import QAService

logger = get_logger(__name__)


class Container:
    """Holds and wires the application's long-lived components."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        openai_client = OpenAI(api_key=settings.openai_api_key)

        # RAG components. The composite loader indexes both PDF and DOCX files.
        self.loader = CompositeLoader()
        self.chunker = Chunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.embedder = Embedder(
            client=openai_client, model=settings.embedding_model
        )
        self.vector_store = VectorStore(
            persist_dir=str(settings.chroma_db),
            collection_name=settings.chroma_collection,
        )
        self.retriever = Retriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=settings.top_k,
        )
        self.llm = LLMClient(
            client=openai_client,
            model=settings.openai_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

        # Services (orchestration).
        self.ingestion_service = IngestionService(
            loader=self.loader,
            chunker=self.chunker,
            embedder=self.embedder,
            vector_store=self.vector_store,
            docs_dir=settings.docs_dir,
        )
        self.qa_service = QAService(
            retriever=self.retriever, llm=self.llm
        )

        logger.info("Application container initialized.")


@lru_cache(maxsize=1)
def get_container() -> Container:
    """Return the process-wide container singleton (built lazily)."""
    return Container(get_settings())
