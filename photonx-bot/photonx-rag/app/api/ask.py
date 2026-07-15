"""Q&A and operational endpoints (STEP 8: POST /ask, GET /health, GET /stats)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.exceptions import EmbeddingError, LLMError, VectorStoreError
from app.core.logger import get_logger
from app.schemas.api import (
    AskRequest,
    AskResponse,
    HealthResponse,
    StatsResponse,
)
from app.services.container import Container, get_container

logger = get_logger(__name__)
router = APIRouter(tags=["qa"])


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question answered only from PhotonX docs",
)
def ask(
    request: AskRequest, container: Container = Depends(get_container)
) -> AskResponse:
    """Answer a question strictly from the indexed documentation."""
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question must not be empty.",
        )

    try:
        answer, sources, kind = container.qa_service.ask(question)
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Embedding failure: {exc}",
        ) from exc
    except LLMError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=f"LLM failure: {exc}"
        ) from exc
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector store failure: {exc}",
        ) from exc

    return AskResponse(answer=answer, sources=sources, kind=kind)


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health() -> HealthResponse:
    """Return a static liveness indicator."""
    return HealthResponse(status="ok")


@router.get("/stats", response_model=StatsResponse, summary="Index statistics")
def stats(container: Container = Depends(get_container)) -> StatsResponse:
    """Report what is currently indexed and which models are configured."""
    try:
        chunks = container.vector_store.count()
        documents = container.vector_store.distinct_documents()
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector store failure: {exc}",
        ) from exc

    settings = container.settings
    return StatsResponse(
        collection=settings.chroma_collection,
        documents=documents,
        chunks=chunks,
        embedding_model=settings.embedding_model,
        llm_model=settings.openai_model,
    )
