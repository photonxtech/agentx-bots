"""Pydantic request/response models for the Multi-RAG API.

These define the JSON contract the frontend (or any client) talks to. They are
deliberately thin mirrors of the data structures already produced by the RAG
pipeline and `chat_store.py`, so nothing in the core logic has to change.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Chats
# --------------------------------------------------------------------------- #
class CreateChatRequest(BaseModel):
    title: str = Field(default="New Chat", description="Display title for the chat.")


class Message(BaseModel):
    id: str | None = Field(
        default=None,
        description="Stable id for this message. Assistant messages use it to "
        "target POST /chats/{chat_id}/messages/{message_id}/metrics.",
    )
    role: str
    content: str
    sources: list["SourceChunk"] = Field(default_factory=list)
    metrics: "Metrics | None" = None


class ChatSummary(BaseModel):
    """Lightweight view of a chat for the list/sidebar."""
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    file: str | None = Field(
        default=None,
        description="The first indexed filename for this chat, if any (backward-compatible "
                    "single-file view — see `files` for the full list).",
    )
    files: list[str] = Field(
        default_factory=list,
        description="All indexed filenames for this chat (a chat may hold up to "
                    "config.MAX_DOCUMENTS_PER_CHAT distinct documents).",
    )
    chunk_count: int = Field(
        default=0, description="How many chunks are indexed for this chat."
    )


class ChatDetail(ChatSummary):
    """Full chat, including the message history."""
    messages: list[Message] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Upload / ingestion
# --------------------------------------------------------------------------- #
class UploadResult(BaseModel):
    chat_id: str
    file: str
    chunks_indexed: int = Field(description="Chunks added for this chat by this upload.")
    backend: str
    message: str


# --------------------------------------------------------------------------- #
# Ask / answer
# --------------------------------------------------------------------------- #
class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="The user's question for this chat's document.")


class SourceChunk(BaseModel):
    """One retrieved chunk that grounded the answer."""
    index: int
    source: str
    kind: str
    score: float
    text: str = Field(description="A snippet of the chunk text (first ~400 chars).")
    images: list[str] = Field(default_factory=list, description="Paths to extracted images, if any.")


class Metrics(BaseModel):
    rewrite_ms: int
    search_ms: int
    ttft_ms: int
    generation_ms: int
    confidence_pct: int
    # DeepEval scores in [0, 1]; absent/None when eval is disabled, skipped
    # (small talk / "I don't know"), not attempted (context_precision/
    # context_recall/answer_correctness need a golden-set match), or a judge
    # call failed for that specific metric.
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_relevancy: float | None = None
    context_recall: float | None = None
    answer_correctness: float | None = None
    # Per-metric failure reason (metric name -> short exception summary), for
    # any metric that returned None because its judge call failed rather than
    # because it wasn't attempted.
    errors: dict[str, str] | None = None


class CalculateMetricsRequest(BaseModel):
    """Body for POST .../metrics — which LangSmith dataset(s), if any, to
    check this answer's ground truth against (see rag.golden_set). Omit or
    leave empty for the three reference-free metrics only."""
    dataset_names: list[str] = Field(default_factory=list)


class GoldenDatasetList(BaseModel):
    datasets: list[str] = Field(
        description="Every LangSmith dataset available to pick from in the "
                    "\"Calculate Metrics\" dropdown. Empty if LangSmith isn't "
                    "configured or is unreachable."
    )


class AskResponse(BaseModel):
    chat_id: str
    question: str
    search_query: str = Field(description="The (possibly rewritten) query used for retrieval.")
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    is_smalltalk: bool = Field(default=False, description="True if answered as a greeting, not via RAG.")
    metrics: Metrics | None = None
    message_id: str | None = Field(
        default=None,
        description="Target for POST /chats/{chat_id}/messages/{message_id}/metrics "
        "to compute DeepEval scores on demand. None for smalltalk replies.",
    )


# --------------------------------------------------------------------------- #
# Q&A + metrics log (Postgres, see db.py)
# --------------------------------------------------------------------------- #
class QaLogEntry(BaseModel):
    """One logged, answered turn with its 6 RAGAS-style metrics."""
    id: int
    chat_id: str
    question: str
    answer: str
    sources: list[dict] = Field(default_factory=list)
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_relevancy: float | None = None
    context_recall: float | None = None
    answer_correctness: float | None = None
    created_at: str


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    status: str
    backend: str
    total_chunks: int
    chats: int


class ModelsResponse(BaseModel):
    models: list[str]
    default: str


class ResetResponse(BaseModel):
    status: str
    message: str


# Resolve forward references (Message references SourceChunk / Metrics).
Message.model_rebuild()
