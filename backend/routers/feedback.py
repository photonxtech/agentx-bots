"""
Human feedback on completed evaluation runs.

Why this exists: every number upstream comes from an LLM judging an LLM against
LLM-generated ground truth. Nothing in that loop can tell you whether the scores
match human judgement. This is the only place a person enters, so it is the
anchor for validating everything else.

Two stores, each doing a job the other cannot:

  * **Postgres** is the source of truth. Feedback is written here first and the
    write cannot fail because a third party is down. It is also what makes the
    dashboard possible — "do runs humans marked down actually score lower?" and
    "which aspect is trending?" are relational questions, and LangSmith's
    feedback API filters only by run id and key.
  * **LangSmith** receives a mirror, so the human rating sits beside the
    automated scores on the same trace. Best-effort and always after the commit:
    a LangSmith outage costs the mirror, never the feedback.
"""
import os
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from backend.database import get_db
from backend.models import Feedback, RunConfig
from backend.schemas import (
    FeedbackCreate, FeedbackOut, FeedbackSummary, FEEDBACK_ASPECTS,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["feedback"])

FEEDBACK_KEY = "human_rating"

def _mirror_to_langsmith(run_config: RunConfig, fb: Feedback, db: DBSession) -> bool:
    """Copy the rating onto the LangSmith batch-summary run.

    Best-effort. Runs after the Postgres commit, so failure costs only the mirror.
    """
    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return False

    summary_run_id = run_config.langsmith_summary_run_id
    if not summary_run_id:
        # Run was evaluated while LangSmith was unreachable, so it has no run to
        # hang feedback on. Reconstruct one from what Postgres already holds.
        from backend.services.rag_evaluator import backfill_summary_run
        summary_run_id = backfill_summary_run(run_config)
        if not summary_run_id:
            return False
        run_config.langsmith_summary_run_id = summary_run_id
        db.commit()

    try:
        from langsmith import Client
        Client(api_key=api_key).create_feedback(
            run_id=summary_run_id,
            key=FEEDBACK_KEY,
            # 1/0 so it aggregates alongside the automated 0-1 metric scores.
            score=1.0 if fb.rating == "up" else 0.0,
            comment=fb.comment or None,
            # Joined, not a list: LangSmith coerces `value` to a scalar.
            value=", ".join(fb.aspects) if fb.aspects else None,
        )
        return True
    except Exception as e:
        logger.warning("LangSmith feedback mirror failed: %s", e)
        return False


@router.get("/feedback/aspects")
def list_aspects():
    """Aspect tags the UI offers. Served from the backend so the two cannot drift."""
    return {"aspects": FEEDBACK_ASPECTS}


@router.post("/sessions/{session_id}/runs/{run_config_id}/feedback",
             response_model=FeedbackOut, status_code=201)
def submit_feedback(
    session_id: UUID,
    run_config_id: UUID,
    payload: FeedbackCreate,
    db: DBSession = Depends(get_db),
):
    run_config = (
        db.query(RunConfig)
        .filter(RunConfig.id == run_config_id, RunConfig.session_id == session_id)
        .first()
    )
    if not run_config:
        raise HTTPException(status_code=404, detail="Run not found for this session")

    unknown = [a for a in (payload.aspects or []) if a not in FEEDBACK_ASPECTS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown aspect(s): {unknown}")

    fb = Feedback(
        run_config_id=run_config_id,
        session_id=session_id,
        rating=payload.rating,
        comment=(payload.comment or "").strip() or None,
        aspects=payload.aspects or None,
        metrics_snapshot=run_config.metrics,
    )
    db.add(fb)
    db.commit()          # committed before the mirror is attempted
    db.refresh(fb)

    if _mirror_to_langsmith(run_config, fb, db):
        fb.synced_to_langsmith = True
        db.commit()
        db.refresh(fb)

    return fb


@router.get("/sessions/{session_id}/runs/{run_config_id}/feedback",
            response_model=list[FeedbackOut])
def get_run_feedback(session_id: UUID, run_config_id: UUID, db: DBSession = Depends(get_db)):
    return (
        db.query(Feedback)
        .filter(Feedback.run_config_id == run_config_id, Feedback.session_id == session_id)
        .order_by(Feedback.created_at.desc())
        .all()
    )


@router.get("/feedback/summary", response_model=FeedbackSummary)
def feedback_summary(limit: int = 20, db: DBSession = Depends(get_db)):
    rows = db.query(Feedback).order_by(Feedback.created_at.desc()).all()

    aspect_counts: dict[str, int] = {}
    for r in rows:
        for a in (r.aspects or []):
            aspect_counts[a] = aspect_counts.get(a, 0) + 1

    recent = [
        {
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "rating": r.rating,
            "comment": r.comment,
            "aspects": r.aspects or [],
            "run_config_id": str(r.run_config_id),
        }
        for r in rows if r.comment
    ][:limit]

    return FeedbackSummary(
        total=len(rows),
        up=sum(1 for r in rows if r.rating == "up"),
        down=sum(1 for r in rows if r.rating == "down"),
        aspect_counts=dict(sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)),
        recent_comments=recent,
    )
