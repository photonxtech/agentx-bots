from fastapi import APIRouter
from pydantic import BaseModel
from metrics import evaluate

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

    print("\n===== SOURCES RETURNED FROM get_answer =====")
    print("Number of sources:", len(sources))

    for i, s in enumerate(sources):
        print("\nSOURCE", i)
        print("Keys:", s.keys())
        print("Text length:", len(s.get("text", "")))
        print("Text preview:", s.get("text", "")[:300])

    print("===========================================\n")

    # Live evaluation — 4 reference-free RAGAS-style metrics, computed right
    # after the answer, using only the question/answer/retrieved context (no
    # ground truth needed, so this is safe on real user questions). Skipped
    # entirely if there's no real retrieved text (e.g. whole-document summary
    # answers, or the out-of-scope fallback) since there's nothing to score.
    contexts = [s["text"] for s in sources if s.get("text", "").strip()]
    metrics = evaluate(request.question, answer, contexts) if contexts else {}

    print("\n===== LIVE METRICS =====")
    print(metrics)
    print("=========================\n")

    append_chat_turn(
        request.session_id,
        request.question,
        answer,
        sources,
        metrics
    )

    print("\n===== RESPONSE SENT TO FRONTEND =====")
    print({
        "answer": answer,
        "sources": sources,
        "token_usage": token_usage,
        "metrics": metrics
    })
    print("=====================================\n")

    return {
        "answer": answer,
        "sources": sources,
        "token_usage": token_usage,
        "metrics": metrics
    }