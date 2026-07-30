from fastapi import APIRouter, HTTPException
from rag_utils import (
    list_sessions_summary,
    get_session_detail,
    create_session,
    delete_session,
    remove_file_from_session,
    session_exists,
    get_pdf_names,
)

router = APIRouter()


@router.get("/sessions")
async def get_sessions():
    """Lightweight list for the sidebar — no chat history, just enough to render each entry."""
    return list_sessions_summary()


@router.get("/sessions/{session_id}")
async def get_session_detail_route(session_id: str):
    """
    Full session detail, including chat_history. The frontend calls this
    whenever the user clicks a past chat, so the conversation actually
    reloads instead of just switching an id in memory.
    """
    detail = get_session_detail(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return detail


@router.post("/sessions/new")
async def new_session():
    """Explicitly create an empty session (used by the 'New Chat' button)."""
    session_id = create_session()
    return {"session_id": session_id}


@router.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    delete_session(session_id)
    return {"message": "Session deleted", "session_id": session_id}


@router.delete("/sessions/{session_id}/files/{filename}")
async def remove_file(session_id: str, filename: str):
    """Removes one attached PDF/photo from a session, leaving the chat and
    any other attached files untouched."""
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    if filename not in get_pdf_names(session_id):
        raise HTTPException(status_code=404, detail="File not found in this session")

    remove_file_from_session(session_id, filename)
    return {"message": "File deleted", "session_id": session_id, "filename": filename}