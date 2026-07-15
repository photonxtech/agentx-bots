"""Request/response models for the public REST API.

These are deliberately separate from the internal domain models
(``app.schemas.documents``): the API contract can evolve independently of
internal representations, and we never leak internal fields we don't intend to
expose.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# --- /ingest ---------------------------------------------------------------
class IngestResponse(BaseModel):
    """Result of indexing the docs folder."""

    status: str = Field(..., examples=["success"])
    documents: int = Field(..., description="Number of PDFs processed.")
    chunks: int = Field(..., description="Total chunks embedded and stored.")


# --- /ask ------------------------------------------------------------------
class AskRequest(BaseModel):
    """A user question."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        examples=["What services does PhotonX provide?"],
    )


class Source(BaseModel):
    """A citation pointing back to the source documentation."""

    document: str
    page: int
    section: str


class AskResponse(BaseModel):
    """An answer plus its supporting citations.

    ``kind`` tells the client how to present the reply:
      - ``answer``  — a grounded answer from the docs (has ``sources``).
      - ``chat``    — a conversational reply (greeting, identity, thanks, …).
      - ``redirect``— the question wasn't in the docs; a warm "can't answer".
    """

    answer: str
    sources: list[Source] = Field(default_factory=list)
    kind: str = Field(
        "answer",
        description="Reply type: 'answer', 'chat', or 'redirect'.",
        examples=["answer"],
    )


# --- /health & /stats ------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field("ok", examples=["ok"])


class StatsResponse(BaseModel):
    collection: str
    documents: int = Field(..., description="Distinct source documents indexed.")
    chunks: int = Field(..., description="Total vectors stored.")
    embedding_model: str
    llm_model: str


class ErrorResponse(BaseModel):
    """Uniform error envelope returned for handled failures."""

    detail: str
