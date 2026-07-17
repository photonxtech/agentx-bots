"""FastAPI application for Multi-RAG.

Exposes the RAG pipeline as a REST API. The heavy state (the ChromaDB-backed
vector store + the chat registry) is created ONCE in the lifespan handler and
shared across all requests via `RagService`.

Endpoints are declared with `def` (not `async def`) on purpose: ingestion,
embedding, OCR and Groq calls are blocking/CPU-bound, and FastAPI automatically
runs sync endpoints in a threadpool — so one slow upload never freezes the whole
server's event loop.

Run (from the mutlirag/ directory):

    uvicorn api.main:app --reload --port 8010

Then open http://localhost:8010/docs for interactive Swagger docs.
(Use any free --port if 8010 is taken.)
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
from rag import generator
from api import schemas
from api.service import ChatNotFoundError, NoDocumentError, RagService, smalltalk_reply

# The frontend lives OUTSIDE the backend, at <project root>/frontend. config.BASE_DIR
# is the project root (parent of backend/), so this resolves regardless of cwd.
FRONTEND_DIR = os.path.join(config.BASE_DIR, "frontend")

# Load secrets from <project root>/.env explicitly, so it works no matter which
# directory the server was launched from.
load_dotenv(os.path.join(config.BASE_DIR, ".env"))

# Populated in the lifespan handler and reused by every request.
service: RagService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vector store + chats once, before any request is served."""
    global service
    service = RagService()  # hydrates Chroma + chat history from disk
    yield
    # Nothing to tear down for the embedded Chroma backend. (If you switch to
    # Weaviate, close the client here: service.store.client.close().)


app = FastAPI(
    title="Multi-RAG API",
    description="REST backend for the multi-modal, per-chat RAG pipeline "
                "(ChromaDB + local embeddings + BM25 + Groq).",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow a separate frontend (React/Vue/etc.) to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def svc() -> RagService:
    """Return the initialized service or fail loudly if startup didn't run."""
    if service is None:  # pragma: no cover - only if lifespan was skipped
        raise HTTPException(status_code=503, detail="Service not initialized yet.")
    return service


# --------------------------------------------------------------------------- #
# Root & health
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def root():
    """Serve the browser frontend at the root URL."""
    return RedirectResponse(url="/ui/")


@app.get("/health", response_model=schemas.HealthResponse, tags=["meta"])
def health():
    return svc().health()


@app.get("/models", response_model=schemas.ModelsResponse, tags=["meta"])
def models():
    """Live list of Groq chat models (falls back to the curated config list)."""
    return {"models": generator.list_models(), "default": config.DEFAULT_MODEL}


# --------------------------------------------------------------------------- #
# Chats
# --------------------------------------------------------------------------- #
@app.post("/chats", response_model=schemas.ChatSummary, status_code=201, tags=["chats"])
def create_chat(body: schemas.CreateChatRequest):
    chat_id = svc().create_chat(title=body.title)
    return svc()._annotate(svc().get_chat(chat_id))


@app.get("/chats", response_model=list[schemas.ChatSummary], tags=["chats"])
def list_chats():
    return svc().list_chats()


@app.get("/chats/{chat_id}", response_model=schemas.ChatDetail, tags=["chats"])
def get_chat(chat_id: str):
    try:
        return svc().chat_detail(chat_id)
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found.")


@app.delete("/chats/{chat_id}", status_code=204, tags=["chats"])
def delete_chat(chat_id: str):
    try:
        svc().delete_chat(chat_id)
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found.")
    return None


# --------------------------------------------------------------------------- #
# File (one per chat)
# --------------------------------------------------------------------------- #
@app.post(
    "/chats/{chat_id}/upload",
    response_model=schemas.UploadResult,
    tags=["files"],
)
def upload_file(chat_id: str, file: UploadFile = File(...)):
    """Ingest + index ONE file into a chat (PDF, DOCX, PPTX, TXT/MD, image)."""
    try:
        svc().get_chat(chat_id)
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found.")

    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = svc().ingest_file(chat_id, file_bytes, file.filename or "upload")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read file: {e}")

    if result["chunks_indexed"] == 0:
        raise HTTPException(status_code=422, detail=result["message"])
    return result


@app.delete("/chats/{chat_id}/file", status_code=204, tags=["files"])
def remove_file(chat_id: str):
    """Remove this chat's indexed file so a different one can be uploaded."""
    try:
        svc().remove_file(chat_id)
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found.")
    except NoDocumentError:
        raise HTTPException(status_code=404, detail="This chat has no file to remove.")
    return None


# --------------------------------------------------------------------------- #
# Ask — non-streaming (single JSON response)
# --------------------------------------------------------------------------- #
@app.post("/chats/{chat_id}/ask", response_model=schemas.AskResponse, tags=["ask"])
def ask(chat_id: str, body: schemas.AskRequest):
    """Ask a question; return the full grounded answer + sources + metrics."""
    s = svc()
    try:
        chat = s.get_chat(chat_id)
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found.")

    history = list(chat.get("messages", []))  # history BEFORE this question

    # Greeting / small talk → friendly reply, no retrieval, no "I don't know".
    canned = smalltalk_reply(body.question, has_file=s.chat_file(chat_id) is not None)
    if canned:
        s.append_user_message(chat_id, body.question)
        s.append_assistant_message(chat_id, canned, [], None)
        return {
            "chat_id": chat_id,
            "question": body.question,
            "search_query": body.question,
            "answer": canned,
            "sources": [],
            "is_smalltalk": True,
            "metrics": None,
        }

    try:
        r = s.retrieve(chat_id, body.question, history)
    except NoDocumentError:
        raise HTTPException(
            status_code=400,
            detail="Upload a file for this chat first — each chat answers only "
                   "from its own document.",
        )

    s.append_user_message(chat_id, body.question)

    # Generate (collect the full stream server-side for the JSON response).
    t_gen = time.perf_counter()
    ttft = None
    parts: list[str] = []
    try:
        for delta in s.answer_stream(r["search_query"], r["hits"]):
            if ttft is None:
                ttft = time.perf_counter() - t_gen
            parts.append(delta)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Generation failed: {e}")

    answer_text = "".join(parts)
    generation_ms = int((time.perf_counter() - t_gen) * 1000)
    metrics = {
        "rewrite_ms": r["rewrite_ms"],
        "search_ms": r["search_ms"],
        "ttft_ms": int((ttft or 0) * 1000),
        "generation_ms": generation_ms,
        "confidence_pct": r["confidence_pct"],
    }

    s.append_assistant_message(chat_id, answer_text, r["sources"], metrics)

    return {
        "chat_id": chat_id,
        "question": body.question,
        "search_query": r["search_query"],
        "answer": answer_text,
        "sources": r["sources"],
        "metrics": metrics,
    }


# --------------------------------------------------------------------------- #
# Ask — streaming (Server-Sent Events), mirrors the live UI typing effect
# --------------------------------------------------------------------------- #
@app.post("/chats/{chat_id}/ask/stream", tags=["ask"])
def ask_stream(chat_id: str, body: schemas.AskRequest):
    """Stream the answer token-by-token as SSE.

    Event stream:
      data: {"type": "meta",  "search_query": ...}
      data: {"type": "token", "text": "..."}          (many)
      data: {"type": "done",  "metrics": {...}, "sources": [...]}
    """
    s = svc()
    try:
        chat = s.get_chat(chat_id)
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found.")

    history = list(chat.get("messages", []))

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    # Greeting / small talk → stream a friendly reply, no retrieval.
    canned = smalltalk_reply(body.question, has_file=s.chat_file(chat_id) is not None)
    if canned:
        s.append_user_message(chat_id, body.question)

        def smalltalk_stream():
            yield sse({"type": "meta", "search_query": body.question})
            for word in canned.split(" "):
                yield sse({"type": "token", "text": word + " "})
            s.append_assistant_message(chat_id, canned, [], None)
            yield sse({"type": "done", "metrics": None, "sources": []})

        return StreamingResponse(
            smalltalk_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        r = s.retrieve(chat_id, body.question, history)
    except NoDocumentError:
        raise HTTPException(
            status_code=400,
            detail="Upload a file for this chat first — each chat answers only "
                   "from its own document.",
        )

    s.append_user_message(chat_id, body.question)

    def event_stream():
        yield sse({"type": "meta", "search_query": r["search_query"]})

        t_gen = time.perf_counter()
        ttft = None
        parts: list[str] = []
        try:
            for delta in s.answer_stream(r["search_query"], r["hits"]):
                if ttft is None:
                    ttft = time.perf_counter() - t_gen
                parts.append(delta)
                yield sse({"type": "token", "text": delta})
        except Exception as e:
            yield sse({"type": "error", "message": f"Generation failed: {e}"})
            return

        answer_text = "".join(parts)
        metrics = {
            "rewrite_ms": r["rewrite_ms"],
            "search_ms": r["search_ms"],
            "ttft_ms": int((ttft or 0) * 1000),
            "generation_ms": int((time.perf_counter() - t_gen) * 1000),
            "confidence_pct": r["confidence_pct"],
        }
        # Persist the assistant turn only after the full answer is produced.
        s.append_assistant_message(chat_id, answer_text, r["sources"], metrics)
        yield sse({"type": "done", "metrics": metrics, "sources": r["sources"]})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
# Admin
# --------------------------------------------------------------------------- #
@app.post("/reset", response_model=schemas.ResetResponse, tags=["admin"])
def reset():
    """Delete ALL chats and ALL indexed documents. Irreversible."""
    svc().reset()
    return {"status": "ok", "message": "All chats and indexed documents deleted."}


# --------------------------------------------------------------------------- #
# Browser frontend — served same-origin so its fetch() calls hit THIS server.
# Mounted last so it never shadows the API routes above. Open http://<host>/ui/
# --------------------------------------------------------------------------- #
if os.path.isdir(FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")
