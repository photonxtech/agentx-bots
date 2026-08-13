"""Postgres logging for chat Q&A turns + the 6 RAGAS-style metrics.

This runs ALONGSIDE the existing JSON chat_store.py, not instead of it:
chat_store.py stays the source of truth for chat/message state (used to
render the UI). This module is a separate, purely additive write-path that
logs every answered turn — question, answer, sources, and the 6 metrics
(faithfulness, answer_relevancy, context_precision, context_relevancy,
context_recall, answer_correctness) — for history/analytics.

Every function here is defensive: if DATABASE_URL isn't set, Postgres is
unreachable, or a query fails, we log a warning and continue. Q&A logging
must never break a chat answer.
"""

from __future__ import annotations

import logging

import psycopg2
import psycopg2.pool
from psycopg2.extras import Json, RealDictCursor

import config

logger = logging.getLogger(__name__)

_pool: psycopg2.pool.SimpleConnectionPool | None = None

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS qa_logs (
    id BIGSERIAL PRIMARY KEY,
    chat_id TEXT NOT NULL,
    message_id TEXT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources JSONB,
    faithfulness DOUBLE PRECISION,
    answer_relevancy DOUBLE PRECISION,
    context_precision DOUBLE PRECISION,
    context_relevancy DOUBLE PRECISION,
    context_recall DOUBLE PRECISION,
    answer_correctness DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE qa_logs ADD COLUMN IF NOT EXISTS message_id TEXT;
CREATE INDEX IF NOT EXISTS idx_qa_logs_chat_id ON qa_logs (chat_id);
CREATE INDEX IF NOT EXISTS idx_qa_logs_created_at ON qa_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_qa_logs_message_id ON qa_logs (message_id);
"""


def init() -> None:
    """Open the connection pool and create qa_logs if needed.

    Call once at app startup (see api/main.py's lifespan). Safe to call even
    when DATABASE_URL is unset or Postgres is unreachable: leaves logging
    disabled instead of crashing the app.
    """
    global _pool
    if not config.DATABASE_URL:
        logger.warning("DATABASE_URL not set — Q&A/metrics logging to Postgres is disabled.")
        return
    try:
        _pool = psycopg2.pool.SimpleConnectionPool(1, 10, dsn=config.DATABASE_URL)
        conn = _pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(_SCHEMA_SQL)
            conn.commit()
        finally:
            _pool.putconn(conn)
        logger.info("Postgres Q&A/metrics logging enabled.")
    except Exception:
        logger.exception("Postgres init failed — Q&A/metrics logging is disabled.")
        _pool = None


def close() -> None:
    """Close all pooled connections (call at app shutdown)."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


def log_qa(
    chat_id: str,
    question: str,
    answer: str,
    sources: list[dict],
    metrics: dict | None,
    message_id: str | None = None,
) -> None:
    """Insert one answered turn. Never raises.

    DeepEval/RAGAS scores are no longer computed at answer time (see
    api.service.RagService.evaluate_message) — they're all NULL here and
    filled in later by update_qa_metrics(), keyed on message_id, whenever the
    user clicks "Calculate Metrics" (or stay NULL forever if they never do).
    """
    if _pool is None:
        return
    metrics = metrics or {}
    try:
        conn = _pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO qa_logs (
                        chat_id, message_id, question, answer, sources,
                        faithfulness, answer_relevancy, context_precision,
                        context_relevancy, context_recall, answer_correctness
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        chat_id, message_id, question, answer, Json(sources or []),
                        metrics.get("faithfulness"),
                        metrics.get("answer_relevancy"),
                        metrics.get("context_precision"),
                        metrics.get("context_relevancy"),
                        metrics.get("context_recall"),
                        metrics.get("answer_correctness"),
                    ),
                )
            conn.commit()
        finally:
            _pool.putconn(conn)
    except Exception:
        logger.exception("Failed to log Q&A turn to Postgres (chat_id=%s)", chat_id)


def update_qa_metrics(message_id: str, metrics: dict | None) -> None:
    """Fill in the 6 RAGAS/DeepEval scores for an already-logged turn, once
    the user clicks "Calculate Metrics". No-op if message_id wasn't logged
    (e.g. Postgres was down at answer time) — never raises."""
    if _pool is None or not message_id:
        return
    metrics = metrics or {}
    try:
        conn = _pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE qa_logs SET
                        faithfulness = %s, answer_relevancy = %s,
                        context_precision = %s, context_relevancy = %s,
                        context_recall = %s, answer_correctness = %s
                    WHERE message_id = %s
                    """,
                    (
                        metrics.get("faithfulness"),
                        metrics.get("answer_relevancy"),
                        metrics.get("context_precision"),
                        metrics.get("context_relevancy"),
                        metrics.get("context_recall"),
                        metrics.get("answer_correctness"),
                        message_id,
                    ),
                )
            conn.commit()
        finally:
            _pool.putconn(conn)
    except Exception:
        logger.exception("Failed to update Q&A metrics in Postgres (message_id=%s)", message_id)


def fetch_qa_logs(chat_id: str | None = None, limit: int = 100) -> list[dict]:
    """Most recent logged turns, newest first; filtered to one chat if given."""
    if _pool is None:
        return []
    try:
        conn = _pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if chat_id:
                    cur.execute(
                        "SELECT * FROM qa_logs WHERE chat_id = %s ORDER BY created_at DESC LIMIT %s",
                        (chat_id, limit),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM qa_logs ORDER BY created_at DESC LIMIT %s",
                        (limit,),
                    )
                rows = cur.fetchall()
        finally:
            _pool.putconn(conn)
        return [dict(r) for r in rows]
    except Exception:
        logger.exception("Failed to fetch Q&A logs from Postgres")
        return []
