"""
Session CRUD routes.
"""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession, joinedload
from backend.database import get_db
from backend.models import Session, RunConfig, Feedback
from backend.schemas import SessionCreate, SessionSummary, SessionDetail

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _unrated_counts(db: DBSession) -> dict:
    """How many runs per session still have no human rating.

    One grouped query for the whole sidebar rather than a count per session —
    the list view renders on every navigation, so N+1 here would be felt.
    """
    from sqlalchemy import func
    rows = (
        db.query(RunConfig.session_id, func.count(RunConfig.id))
        .outerjoin(Feedback, Feedback.run_config_id == RunConfig.id)
        .filter(Feedback.id.is_(None))
        .group_by(RunConfig.session_id)
        .all()
    )
    return {sid: n for sid, n in rows}


@router.post("", response_model=SessionSummary, status_code=201)
@router.post("/", response_model=SessionSummary, status_code=201, include_in_schema=False)
def create_session(payload: SessionCreate, db: DBSession = Depends(get_db)):
    """Create a new evaluation session."""
    session = Session(name=payload.name or "Untitled Session")
    db.add(session)
    db.commit()
    db.refresh(session)
    # Don't call _unrated_counts for just one session - new session has 0 runs anyway
    return SessionSummary(
        id=session.id,
        name=session.name,
        created_at=session.created_at,
        document_filename=session.document_filename,
        document_count=len(session.documents or ([session.document_path] if session.document_path else [])),
        has_qa=session.qa_json is not None,
        has_seed_qa=bool(session.seed_qa_json),
        unverified_runs=0,  # New session always has 0 unrated runs
    )


@router.get("", response_model=list[SessionSummary])
@router.get("/", response_model=list[SessionSummary], include_in_schema=False)
def list_sessions(db: DBSession = Depends(get_db)):
    """List all sessions (for sidebar)."""
    sessions = db.query(Session).order_by(Session.created_at.desc()).all()
    unrated = _unrated_counts(db)
    return [
        SessionSummary(
            id=s.id,
            name=s.name,
            created_at=s.created_at,
            document_filename=s.document_filename,
            document_count=len(s.documents or ([s.document_path] if s.document_path else [])),
            has_qa=s.qa_json is not None,
            has_seed_qa=bool(s.seed_qa_json),
            unverified_runs=unrated.get(s.id, 0),
        )
        for s in sessions
    ]


@router.get("/{session_id}", response_model=SessionDetail)
def get_session(session_id: UUID, db: DBSession = Depends(get_db)):
    """Get full session detail including all run configs and results."""
    session = (
        db.query(Session)
        .options(joinedload(Session.run_configs).joinedload(RunConfig.feedback))
        .filter(Session.id == session_id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: UUID, db: DBSession = Depends(get_db)):
    """Delete a session and all associated data."""
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete(session)
    db.commit()
    return None
