"""Ingestion endpoint (STEP 8: POST /ingest)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.exceptions import (
    DocumentError,
    EmbeddingError,
    VectorStoreError,
)
from app.core.logger import get_logger
from app.schemas.api import IngestResponse
from app.services.container import Container, get_container

logger = get_logger(__name__)
router = APIRouter(tags=["ingestion"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Index all PDFs in the docs folder",
)
def ingest(container: Container = Depends(get_container)) -> IngestResponse:
    """Load, chunk, embed, and store every PDF under the configured docs dir."""
    try:
        documents, chunks = container.ingestion_service.ingest()
    except DocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Embedding failure: {exc}",
        ) from exc
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector store failure: {exc}",
        ) from exc

    return IngestResponse(status="success", documents=documents, chunks=chunks)
