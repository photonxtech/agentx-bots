from fastapi import APIRouter
from pydantic import BaseModel

from rag_utils import (
    build_hybrid_retriever,
    get_answer,
    session_exists,
    get_pdf_names,
    get_chat_history,
    append_chat_turn,
)

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    question: str


@router.post("/chat")
async def chat(request: ChatRequest):
    if not session_exists(request.session_id):
        return {"error": "Invalid session ID"}

    if not get_pdf_names(request.session_id):
        return {"error": "No PDFs attached to this session yet"}

    retriever = build_hybrid_retriever(request.session_id)
    chat_history = get_chat_history(request.session_id)

    answer, sources, token_usage = get_answer(
        retriever,
        request.question,
        chat_history,
        session_id=request.session_id
    )

    append_chat_turn(
        request.session_id,
        request.question,
        answer,
        sources
    )

    return {
        "answer": answer,
        "sources": sources,
        "token_usage": token_usage
    }