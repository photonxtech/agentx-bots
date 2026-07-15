"""Q&A and operational endpoints (POST /ask, GET /health, GET /stats)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.exceptions import LLMError, VectorStoreError
from app.schemas.api import (
    AskRequest,
    AskResponse,
    HealthResponse,
    Source,
    StatsResponse,
)
from app.services.container import Container, get_container

router = APIRouter(tags=["qa"])


@router.post("/ask", response_model=AskResponse)
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
        answer, sources, kind = container.pipeline.ask(question)
    except (LLMError, VectorStoreError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc

    return AskResponse(
        answer=answer, sources=[Source(**s) for s in sources], kind=kind
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a static liveness indicator."""
    return HealthResponse(status="ok")


@router.get("/stats", response_model=StatsResponse)
def stats(container: Container = Depends(get_container)) -> StatsResponse:
    """Report what is currently indexed and which models are configured."""
    try:
        chunks = container.pipeline.count()
        documents = container.pipeline.distinct_documents()
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    settings = container.settings
    return StatsResponse(
        collection=settings.chroma_collection,
        documents=documents,
        chunks=chunks,
        embedding_model=settings.embedding_model,
        llm_model=settings.openai_model,
    )
