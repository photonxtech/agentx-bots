from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional
import hashlib
from pathlib import Path
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from rag_utils import (
    create_session,
    add_pdf_to_session,
    process_pdf,
    process_image,
    is_image_file,
    session_exists,
    get_pdf_names,
    get_content_hashes,
)

router = APIRouter()

DATA_DIR = Path("storage") / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}


@router.post("/upload")
@traceable(name="upload_turn")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
):
    """
    Upload one or more PDFs and/or photos.

    - If session_id is omitted or doesn't match an existing session, a NEW
      session is created (first upload of a chat).
    - If session_id IS provided and valid, the files are added to that
      session alongside whatever's already indexed there. This is what lets
      the browser attach a 2nd/3rd file to an in-progress chat instead of
      being locked to one file per session.
    - Duplicate detection is CONTENT-based (sha256 of the file's bytes), not
      just filename-based. Re-uploading the exact same file under a
      different name (e.g. a browser/OS auto-renamed "report-2.pdf" that's
      byte-for-byte identical to an already-attached "report.pdf") is
      correctly caught and skipped instead of silently duplicating its
      chunks in the vector store. Filename is still checked too, as a cheap
      first pass before reading the whole file.
    - Any file with an unsupported extension is rejected with a clean 400
      instead of crashing PyPDFLoader on something that isn't a PDF.
    """
    if not session_id or not session_exists(session_id):
        session_id = create_session()

    run_tree = get_current_run_tree()
    if run_tree:
        run_tree.metadata["session_id"] = session_id
        run_tree.tags = (run_tree.tags or []) + [f"session:{session_id}"]

    unsupported = [
        f.filename for f in files
        if Path(f.filename).suffix.lower() not in SUPPORTED_EXTENSIONS
    ]
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {', '.join(unsupported)}. "
                    f"Only PDF and image files (.pdf, .png, .jpg, .jpeg, .webp) are supported."
        )

    already_attached = set(get_pdf_names(session_id))
    already_hashes = get_content_hashes(session_id)

    session_folder = DATA_DIR / session_id
    session_folder.mkdir(parents=True, exist_ok=True)

    file_results = []
    skipped = []
    total_pages = 0
    total_chunks = 0

    for file in files:
        if file.filename in already_attached:
            skipped.append(file.filename)
            continue

        file_bytes = await file.read()
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        if content_hash in already_hashes:
            skipped.append(f"{file.filename} (same content as an already-attached file)")
            continue

        file_path = session_folder / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(file_bytes)

        # process_pdf / process_image both append to this session's existing
        # chunk/vector store rather than rebuilding it, so earlier files in
        # the session aren't re-embedded
        try:
            if is_image_file(file.filename):
                pages, chunks = process_image(str(file_path), session_id)
            else:
                pages, chunks = process_pdf(str(file_path), session_id)
        except Exception as e:
            # Most common cause: a PDF with no extractable text (e.g. a
            # scanned document with no text layer) — RecursiveCharacterTextSplitter
            # on an empty page list, or downstream BM25 on an empty corpus,
            # can throw. Surface a clean message instead of a raw 500.
            file_path.unlink(missing_ok=True)
            file_results.append({
                "filename": file.filename,
                "error": f"Couldn't process this file: {e}. If it's a scanned "
                         f"PDF with no selectable text, try uploading it as an image instead."
            })
            continue

        already_attached.add(file.filename)
        already_hashes.add(content_hash)
        add_pdf_to_session(session_id, file.filename, content_hash)

        total_pages += pages
        total_chunks += chunks
        file_results.append({
            "filename": file.filename,
            "pages": pages,
            "chunks": chunks,
        })

    return {
        "message": "File(s) uploaded successfully",
        "session_id": session_id,
        "files": file_results,
        "skipped_duplicates": skipped,
        "total_pages": total_pages,
        "total_chunks": total_chunks,
    }