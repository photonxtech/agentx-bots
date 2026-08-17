"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime
from uuid import UUID


# ──────────────────── Session ────────────────────

class SessionCreate(BaseModel):
    name: Optional[str] = "Untitled Session"


class SessionSummary(BaseModel):
    id: UUID
    name: str
    created_at: datetime
    document_filename: Optional[str] = None
    document_count: int = 0
    has_qa: bool = False
    # Runs in this session that nobody has rated yet.
    unverified_runs: int = 0

    class Config:
        from_attributes = True


class SessionDetail(BaseModel):
    id: UUID
    name: str
    created_at: datetime
    document_filename: Optional[str] = None
    # All uploaded documents: [{filename, path, size}]
    documents: Optional[Any] = None
    qa_json: Optional[Any] = None
    qa_meta: Optional[Any] = None
    run_configs: List["RunConfigWithResults"] = []

    class Config:
        from_attributes = True


# ──────────────────── QA ────────────────────

class QAPair(BaseModel):
    question: str
    answer: str


class QAGenerateResponse(BaseModel):
    total: int
    preview: List[QAPair]
    # Which generator produced the set (deepeval vs fallback), plus any caveats.
    meta: Optional[Any] = None


# ──────────────────── Run Config ────────────────────

class EvaluateRequest(BaseModel):
    chat_model: str = Field(..., description="Groq model id used to generate answers, e.g. openai/gpt-oss-120b")
    embedding_model: str = Field(..., description="e.g. bge-small, bge-base, bge-large")
    # Informational only — retrieval always runs on Chroma with its default index.
    vectordb: str = Field(..., description="Informational only — retrieval always uses ChromaDB")
    chunk_size: int = Field(512, ge=100, le=10000)
    chunk_overlap: int = Field(50, ge=0, le=5000)
    reranker_model: Optional[str] = Field(None, description="e.g. cohere-rerank-v3")
    search_type: str = Field("semantic", pattern="^(keyword|semantic|hybrid)$")
    # Informational only — Chroma's own index settings are not driven by this.
    index_type: Optional[str] = Field(None, pattern="^(hnsw|ivf|pq)$", description="Informational only")
    top_k: int = Field(5, ge=1, le=100)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class RunResultOut(BaseModel):
    id: UUID
    question: str
    generated_answer: Optional[str] = None
    expected_answer: Optional[str] = None
    metrics: Optional[Any] = None

    class Config:
        from_attributes = True


class RunConfigOut(BaseModel):
    id: UUID
    created_at: datetime
    chat_model: str
    embedding_model: str
    vectordb: str
    chunk_size: int
    chunk_overlap: int
    reranker_model: Optional[str] = None
    search_type: str
    index_type: Optional[str] = None
    top_k: int
    temperature: float
    langsmith_experiment_url: Optional[str] = None
    # Batch-level metric scores for the whole test set (+ "_meta").
    metrics: Optional[Any] = None
    results: List[RunResultOut] = []
    # Human ratings on this run, newest first. Forward-referenced because
    # FeedbackOut is declared below; resolved by model_rebuild() at the bottom.
    feedback: List["FeedbackOut"] = []

    class Config:
        from_attributes = True


class RunConfigWithResults(RunConfigOut):
    pass


# ──────────────────── Human feedback ────────────────────

# Aspect tags map to pipeline stages, so a thumbs-down points at something
# fixable rather than expressing undirected dissatisfaction.
FEEDBACK_ASPECTS = [
    "scores_disagree",      # the metrics don't match my own read of the answers
    "bad_answers",          # generated answers are wrong or unhelpful
    "bad_questions",        # the synthetic test questions are poor
    "bad_ground_truth",     # the expected answers are wrong
    "bad_retrieval",        # retrieved context was irrelevant
    "too_slow",             # the run took too long / cost too much
]


class FeedbackCreate(BaseModel):
    rating: str = Field(..., pattern="^(up|down)$")
    comment: Optional[str] = Field(None, max_length=5000)
    aspects: Optional[List[str]] = None


class FeedbackOut(BaseModel):
    id: UUID
    run_config_id: UUID
    created_at: datetime
    rating: str
    comment: Optional[str] = None
    aspects: Optional[Any] = None
    synced_to_langsmith: bool = False

    class Config:
        from_attributes = True


class FeedbackSummary(BaseModel):
    total: int
    up: int
    down: int
    # Counts per aspect tag, most frequent first — this is the "what should we
    # fix next" view.
    aspect_counts: dict
    recent_comments: List[Any] = []


# Rebuild forward refs
RunConfigOut.model_rebuild()
RunConfigWithResults.model_rebuild()
SessionDetail.model_rebuild()
