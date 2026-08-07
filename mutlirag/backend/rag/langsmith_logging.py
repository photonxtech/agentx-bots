"""Live LangSmith logging for chat turns.

This is what makes a question asked in the frontend show up in LangSmith
immediately — one run per chat turn (question/answer/context as input/output),
with each RAGAS metric api.service.RagService.evaluate_answer() already
computed attached to that run as feedback. It's the live-traffic counterpart
to scripts/run_langsmith_eval.py, which instead replays the curated golden-set
dataset as a single batch "experiment" — the two write to the same LangSmith
project but are otherwise independent; this module never touches a dataset.

No-op (not an error) when LANGSMITH_TRACING/LANGSMITH_API_KEY aren't set, same
as db.py's Postgres logging. Never raises — a LangSmith outage must not break
a chat answer.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_RAGAS_KEYS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_relevancy",
    "context_recall",
    "answer_correctness",
)

_client = None
_enabled: bool | None = None


def _get_client():
    global _client, _enabled
    if _enabled is None:
        _enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(
            os.getenv("LANGSMITH_API_KEY")
        )
        if _enabled:
            from langsmith import Client

            _client = Client()
    return _client


def log_chat_turn(
    chat_id: str, question: str, answer: str, contexts: list[str], metrics: dict | None
) -> None:
    """Log one live chat turn to LangSmith as a run, with its RAGAS scores as feedback."""
    client = _get_client()
    if client is None:
        return
    try:
        run_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        client.create_run(
            name="chat_turn",
            run_type="chain",
            id=run_id,
            inputs={"chat_id": chat_id, "question": question},
            outputs={"answer": answer, "context": contexts},
            start_time=now,
            end_time=now,
        )
        for key in _RAGAS_KEYS:
            score = (metrics or {}).get(key)
            if score is not None:
                client.create_feedback(run_id=run_id, key=key, score=score)
    except Exception:
        logger.warning("LangSmith chat-turn logging failed", exc_info=True)
