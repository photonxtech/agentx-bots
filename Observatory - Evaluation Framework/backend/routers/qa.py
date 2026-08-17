"""
Document upload, QA generation, and QA download routes.
"""
import os
import json
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session as DBSession
from backend.database import get_db
from backend.models import Session
from backend.services.document_reader import document_paths
from backend.schemas import QAGenerateResponse, QAPair
from backend.services.deepeval_testset_generator import generate_testset
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/api/sessions/{session_id}", tags=["qa"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")


ALLOWED_EXT = {".pdf", ".txt", ".docx", ".md", ".csv"}


def _session_documents(session) -> list[dict]:
    """Current document list, migrating a legacy single upload on first touch."""
    docs = list(session.documents or [])
    if not docs and session.document_path:
        docs = [{
            "filename": session.document_filename or os.path.basename(session.document_path),
            "path": session.document_path,
            "size": None,
        }]
    return docs


def _sync_legacy_fields(session, docs: list[dict]) -> None:
    """Keep the old single-document columns pointing at the first file, so
    anything still reading them (older rows, external queries) stays coherent."""
    if docs:
        session.document_filename = docs[0]["filename"]
        session.document_path = docs[0]["path"]
    else:
        session.document_filename = None
        session.document_path = None


@router.post("/upload")
async def upload_documents(
    session_id: UUID,
    files: list[UploadFile] = File(...),
    db: DBSession = Depends(get_db),
):
    """Upload one or more documents. Repeated calls append rather than replace,
    so files can be added to a session incrementally."""
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    docs = _session_documents(session)
    existing = {d["filename"] for d in docs}
    saved, skipped = [], []

    for file in files:
        name = os.path.basename(file.filename or "")
        if not name:
            continue
        if os.path.splitext(name)[1].lower() not in ALLOWED_EXT:
            skipped.append(f"{name} (unsupported type)")
            continue
        if name in existing:
            skipped.append(f"{name} (already uploaded)")
            continue

        file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{name}")
        content = await file.read()
        if not content:
            skipped.append(f"{name} (empty)")
            continue
        with open(file_path, "wb") as f:
            f.write(content)

        docs.append({"filename": name, "path": file_path, "size": len(content)})
        existing.add(name)
        saved.append(name)

    if not saved and skipped:
        raise HTTPException(status_code=400, detail="Nothing uploaded: " + "; ".join(skipped))

    session.documents = docs
    _sync_legacy_fields(session, docs)
    # The test set no longer matches the corpus once the corpus changes.
    if saved and session.qa_json:
        session.qa_json = None
        session.qa_meta = None
    db.commit()

    return {
        "message": f"Uploaded {len(saved)} file(s)",
        "saved": saved,
        "skipped": skipped,
        "documents": docs,
    }


@router.delete("/documents/{filename}")
def delete_document(session_id: UUID, filename: str, db: DBSession = Depends(get_db)):
    """Remove one document from the session."""
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    docs = _session_documents(session)
    remaining = [d for d in docs if d["filename"] != filename]
    if len(remaining) == len(docs):
        raise HTTPException(status_code=404, detail="Document not on this session")

    removed = next(d for d in docs if d["filename"] == filename)
    try:
        if removed.get("path") and os.path.exists(removed["path"]):
            os.remove(removed["path"])
    except OSError:
        pass  # the DB row is what matters; a stale file on disk is harmless

    session.documents = remaining
    _sync_legacy_fields(session, remaining)
    # Same reasoning as upload: the corpus changed, so the test set is stale.
    session.qa_json = None
    session.qa_meta = None
    db.commit()
    return {"message": f"Removed {filename}", "documents": remaining}


@router.post("/generate-qa", response_model=QAGenerateResponse)
async def generate_qa(
    session_id: UUID,
    num_questions: int = 6,
    db: DBSession = Depends(get_db),
):
    """Generate a synthetic test set from the uploaded document.


    Both are expensive, hence the small default. A large request will take
    many minutes and exhaust a rate-limited tier.
    """
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    paths = document_paths(session)
    if not paths:
        raise HTTPException(status_code=400, detail="No document uploaded yet")

    try:
        qa_pairs, meta = await generate_testset(paths, testset_size=num_questions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Save the full test set, provenance fields included
    session.qa_json = qa_pairs
    session.qa_meta = meta
    db.commit()

    return QAGenerateResponse(
        total=len(qa_pairs),
        preview=[QAPair(question=qp["question"], answer=qp["answer"]) for qp in session.qa_json],
        meta=meta,
    )


@router.get("/download-qa")
def download_qa(session_id: UUID, db: DBSession = Depends(get_db)):
    """Download the full QA JSON file."""
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.qa_json:
        raise HTTPException(status_code=400, detail="No QA data generated yet")

    return JSONResponse(
        content=session.qa_json,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=qa_testcases_{session_id}.json"},
    )
