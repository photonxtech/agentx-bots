"""
SQLAlchemy ORM models for sessions, run_configs, and run_results.
"""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text, Boolean, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from backend.database import Base


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, default="Untitled Session")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    # Legacy single-document fields. Kept in sync with documents[0] so sessions
    # created before multi-file upload keep working unchanged.
    document_filename = Column(String(500), nullable=True)
    document_path = Column(String(1000), nullable=True)
    # All uploaded documents: [{"filename": str, "path": str, "size": int}]
    documents = Column(JSON, nullable=True)
    qa_json = Column(JSON, nullable=True)  # Full generated QA test cases
    # How the test set was produced (generator, model, notes) — recorded so a
    # framework test set is never confused with a fallback one.
    qa_meta = Column(JSON, nullable=True)
    # User-uploaded, manually-created Q&A pairs (seed test set). Stored
    # separately from qa_json so that re-generating Q&A never wipes the user's
    # own pairs — only the system-generated set is replaced on re-generation.
    seed_qa_json = Column(JSON, nullable=True)

    # Relationships
    run_configs = relationship("RunConfig", back_populates="session", cascade="all, delete-orphan", order_by="RunConfig.created_at")


class RunConfig(Base):
    __tablename__ = "run_configs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # RAG configuration fields
    chat_model = Column(String(255), nullable=False)
    embedding_model = Column(String(255), nullable=False)
    vectordb = Column(String(255), nullable=False)
    chunk_size = Column(Integer, nullable=False, default=512)
    chunk_overlap = Column(Integer, nullable=False, default=50)
    reranker_model = Column(String(255), nullable=True)
    search_type = Column(String(50), nullable=False, default="semantic")  # keyword | semantic | hybrid
    index_type = Column(String(50), nullable=True)  # hnsw | ivf | pq (only for semantic/hybrid)
    top_k = Column(Integer, nullable=False, default=5)
    temperature = Column(Float, nullable=False, default=0.7)
    langsmith_experiment_url = Column(String(1000), nullable=True)  # LangSmith experiment link

    # Batch-level metric scores for the whole test set, plus a "_meta" block
    # recording which models produced them and how many cases actually scored.
    metrics = Column(JSON, nullable=True)

    # LangSmith run id of the batch-summary run. Stored so human feedback can be
    # attached to the same run that carries the automated scores.
    langsmith_summary_run_id = Column(String(64), nullable=True)

    # Relationships
    session = relationship("Session", back_populates="run_configs")
    results = relationship("RunResult", back_populates="run_config", cascade="all, delete-orphan", order_by="RunResult.id")
    feedback = relationship("Feedback", back_populates="run_config", cascade="all, delete-orphan", order_by="Feedback.created_at.desc()")


class Feedback(Base):
    """Human judgement on a completed evaluation run.

    Postgres is the source of truth; LangSmith gets a mirror so the rating sits
    beside the automated scores on the same trace. The local copy is what makes
    the dashboard possible — the questions that matter are relational:

      * "do runs humans marked down actually score lower?"  (join to metrics)
      * "which aspect is reported most, and is it trending?" (group by + time)

    Neither is expressible against LangSmith's feedback API, which filters only
    by run id and key.

    `aspects` deliberately holds pipeline-stage tags rather than free sentiment —
    a thumbs-down saying "retrieval was irrelevant" points at a fixable stage.
    """
    __tablename__ = "feedback"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_config_id = Column(String(36), ForeignKey("run_configs.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    rating = Column(String(8), nullable=False)          # "up" | "down"
    comment = Column(Text, nullable=True)
    aspects = Column(JSON, nullable=True)              # list of stage tags

    # The scores as they stood when the human judged them. Frozen so a later
    # re-run of the same config cannot rewrite what was actually being rated —
    # without this, the human-vs-machine comparison drifts silently.
    metrics_snapshot = Column(JSON, nullable=True)

    synced_to_langsmith = Column(Boolean, nullable=False, default=False)

    run_config = relationship("RunConfig", back_populates="feedback")


class RunResult(Base):
    __tablename__ = "run_results"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_config_id = Column(String(36), ForeignKey("run_configs.id", ondelete="CASCADE"), nullable=False)

    question = Column(Text, nullable=False)
    generated_answer = Column(Text, nullable=True)
    expected_answer = Column(Text, nullable=True)
    metrics = Column(JSON, nullable=True)  # e.g. {"faithfulness": 0.9, "relevancy": 0.85, ...}

    # Relationships
    run_config = relationship("RunConfig", back_populates="results")
