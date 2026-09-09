"""
Document upload, QA generation, seed Q&A management, and QA download routes.
"""
import os
import io
import json
import csv
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


def _normalize_question(q: str) -> str:
    """Normalise a question string for duplicate detection:
    lower-case, strip, collapse runs of whitespace."""
    import re
    return re.sub(r"\s+", " ", (q or "").strip().lower())


def _parse_seed_qa(content: bytes, filename: str) -> list[dict]:
    """Parse the uploaded seed Q&A file (JSON or CSV) into a list of
    {question, answer} dicts.  Raises ValueError on invalid format."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".json":
        try:
            data = json.loads(content.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Could not parse JSON: {exc}") from exc
        if not isinstance(data, list):
            raise ValueError("JSON file must be an array of {question, answer} objects.")
        pairs = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} is not an object.")
            q = str(item.get("question") or "").strip()
            a = str(item.get("answer") or "").strip()
            if not q or not a:
                raise ValueError(f"Item {i} is missing a 'question' or 'answer' field.")
            pairs.append({"question": q, "answer": a, "source": "seed"})
        return pairs

    if ext == ".csv":
        try:
            text = content.decode("utf-8")
            reader = csv.DictReader(io.StringIO(text))
            pairs = []
            for i, row in enumerate(reader):
                q = str(row.get("question") or "").strip()
                a = str(row.get("answer") or "").strip()
                if not q or not a:
                    raise ValueError(f"Row {i + 1} is missing a 'question' or 'answer' column.")
                pairs.append({"question": q, "answer": a, "source": "seed"})
            if not pairs:
                raise ValueError("CSV file is empty or has no valid rows.")
            return pairs
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Could not parse CSV: {exc}") from exc

    raise ValueError(
        f"Unsupported file type '{ext}'. Upload a .json or .csv file."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Document upload / delete
# ─────────────────────────────────────────────────────────────────────────────

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
        if name in existing:
            skipped.append(f"{name} (already uploaded)")
            continue

        file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{name}")
        try:
            # Read the full file content via Starlette's async read, then write
            # to disk synchronously.  The previous approach — passing the
            # SpooledTemporaryFile to asyncio.to_thread + shutil.copyfileobj —
            # caused 1-2 min delays because the synchronous reads inside the
            # thread-pool worker contend with the event loop for the GIL.
            content = await file.read()
            file_size = len(content)

            if file_size == 0:
                skipped.append(f"{name} (empty)")
                continue

            with open(file_path, "wb") as out_f:
                out_f.write(content)

            doc = {"filename": name, "path": file_path, "size": file_size}
            docs.append(doc)
            existing.add(name)
            saved.append(name)
        except Exception as e:
            skipped.append(f"{name} (error: {str(e)})")

    if not saved and skipped:
        raise HTTPException(status_code=400, detail="Nothing uploaded: " + "; ".join(skipped))

    # Single DB transaction for all files
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


# ─────────────────────────────────────────────────────────────────────────────
# Seed Q&A upload / delete / download
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/upload-seed-qa")
async def upload_seed_qa(
    session_id: UUID,
    file: UploadFile = File(...),
    db: DBSession = Depends(get_db),
):
    """Upload a manually-created Q&A file (JSON or CSV) as the seed test set.

    The seed set is stored separately from the system-generated Q&A.
    Re-generating Q&A will never overwrite it — only the generated portion
    changes.  When evaluation runs, both seed and generated Q&A are used.

    Accepted formats:
      JSON — array of {"question": "...", "answer": "..."} objects
      CSV  — header row with 'question' and 'answer' columns
    """
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    name = os.path.basename(file.filename or "")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        pairs = _parse_seed_qa(content, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not pairs:
        raise HTTPException(status_code=400, detail="No valid Q&A pairs found in the file.")

    session.seed_qa_json = pairs
    db.commit()

    return {
        "message": f"Uploaded {len(pairs)} seed Q&A pair(s) from '{name}'.",
        "total": len(pairs),
        "preview": [{"question": p["question"], "answer": p["answer"]} for p in pairs[:5]],
    }


@router.delete("/seed-qa")
def delete_seed_qa(session_id: UUID, db: DBSession = Depends(get_db)):
    """Remove all uploaded seed Q&A pairs from this session."""
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.seed_qa_json = None
    db.commit()
    return {"message": "Seed Q&A cleared."}


@router.get("/download-seed-qa")
def download_seed_qa(session_id: UUID, db: DBSession = Depends(get_db)):
    """Download the uploaded seed Q&A as JSON."""
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.seed_qa_json:
        raise HTTPException(status_code=400, detail="No seed Q&A uploaded yet.")

    return JSONResponse(
        content=session.seed_qa_json,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=seed_qa_{session_id}.json"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Q&A generation
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/generate-qa", response_model=QAGenerateResponse)
async def generate_qa(
    session_id: UUID,
    num_questions: int = 6,
    db: DBSession = Depends(get_db),
):
    """Generate a synthetic test set from the uploaded document(s).

    If the session has user-uploaded seed Q&A pairs, the generator is given
    their questions so it can avoid producing duplicates.  The final generated
    set therefore consists only of *new* questions not already present in the
    seed set.

    Both are expensive, hence the small default. A large request will take
    many minutes and exhaust a rate-limited tier.
    """
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    paths = document_paths(session)
    if not paths:
        raise HTTPException(status_code=400, detail="No document uploaded yet")

    # Collect existing seed questions for deduplication inside the generator.
    seed_pairs: list[dict] = list(session.seed_qa_json or [])
    seed_questions: list[str] = [p["question"] for p in seed_pairs]

    try:
        qa_pairs, meta = await generate_testset(
            paths,
            testset_size=num_questions,
            seed_questions=seed_questions,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    def _sanitize_for_jsonb(obj):
        if isinstance(obj, str):
            return obj.replace("\x00", "").replace("\u0000", "")
        elif isinstance(obj, dict):
            return {k: _sanitize_for_jsonb(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_sanitize_for_jsonb(x) for x in obj]
        return obj

    # Tag generated pairs with their source so the UI and evaluator can
    # distinguish them from seed pairs.
    for pair in qa_pairs:
        pair.setdefault("source", "generated")

    # Save the generated set. Seed Q&A is stored separately and is not
    # affected by re-generation.
    session.qa_json = _sanitize_for_jsonb(qa_pairs)
    session.qa_meta = _sanitize_for_jsonb(meta)
    db.commit()

    # Return the FULL combined preview (seed first, then generated) so the UI
    # can show both sections in one call.
    combined_preview = [
        QAPair(question=p["question"], answer=p["answer"])
        for p in (seed_pairs + qa_pairs)
    ]

    # Augment meta with seed info.
    meta["seed_count"] = len(seed_pairs)
    meta["generated_count"] = len(qa_pairs)
    meta["total_combined"] = len(seed_pairs) + len(qa_pairs)

    return QAGenerateResponse(
        total=len(seed_pairs) + len(qa_pairs),
        preview=combined_preview,
        meta=meta,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Download generated Q&A
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/download-qa")
def download_qa(session_id: UUID, db: DBSession = Depends(get_db)):
    """Download the full Q&A JSON file (combined seed + generated)."""
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    seed_pairs = list(session.seed_qa_json or [])
    generated_pairs = list(session.qa_json or [])

    if not seed_pairs and not generated_pairs:
        raise HTTPException(status_code=400, detail="No Q&A data available yet")

    combined = seed_pairs + generated_pairs

    return JSONResponse(
        content=combined,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=qa_testcases_{session_id}.json"},
    )
