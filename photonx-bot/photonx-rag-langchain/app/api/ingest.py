"""Ingestion endpoint (POST /ingest)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.exceptions import DocumentError, VectorStoreError
from app.schemas.api import IngestResponse
from app.services.container import Container, get_container

router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(container: Container = Depends(get_container)) -> IngestResponse:
    """Load, split, embed, and store every PDF under the docs dir."""
    try:
        documents, chunks = container.pipeline.ingest()
    except DocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc
    return IngestResponse(status="success", documents=documents, chunks=chunks)
