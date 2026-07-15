"""Internal domain models used across the RAG pipeline.

These are the objects that flow between components (loader → chunker →
embedder → vector store → retriever). Keeping them as Pydantic models gives us
validation, clear contracts, and free (de)serialization.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PageContent(BaseModel):
    """A single extracted page of a PDF."""

    document: str = Field(..., description="Source file name.")
    page: int = Field(..., ge=1, description="1-based page number.")
    text: str = Field(..., description="Raw extracted text for the page.")


class Chunk(BaseModel):
    """A retrievable unit of text plus its provenance metadata."""

    id: str = Field(..., description="Stable, unique chunk identifier.")
    text: str = Field(..., description="Chunk text.")
    document: str = Field(..., description="Source file name.")
    page: int = Field(..., ge=1, description="1-based page number.")
    section: str = Field("General", description="Detected heading / section.")

    def metadata(self) -> dict[str, str | int]:
        """Return the flat metadata dict persisted alongside the vector."""
        return {
            "document": self.document,
            "page": self.page,
            "section": self.section,
            "chunk_id": self.id,
        }


class RetrievedChunk(BaseModel):
    """A chunk returned from a similarity search, with its score."""

    text: str
    document: str
    page: int
    section: str
    score: float = Field(..., description="Cosine similarity in [0, 1].")
