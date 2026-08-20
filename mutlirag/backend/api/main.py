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

# Load secrets from <project root>/.env BEFORE importing config (or anything
# that imports config) — config.py reads some env vars (e.g. DATABASE_URL) at
# *import* time, so .env must already be loaded into the process environment
# before that import runs, or those values freeze to None for the whole run.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

import config
import db
from rag import generator, langsmith_logging
from api import schemas
from api.service import (
    ChatNotFoundError,
    MessageNotFoundError,
    NoDocumentError,
    RagService,
    TooManyDocumentsError,
    smalltalk_reply,
)

# The frontend lives OUTSIDE the backend, at <project root>/frontend. config.BASE_DIR
# is the project root (parent of backend/), so this resolves regardless of cwd.
FRONTEND_DIR = os.path.join(config.BASE_DIR, "frontend")

# Populated in the lifespan handler and reused by every request.
service: RagService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vector store + chats once, before any request is served."""
    global service
    service = RagService()  # hydrates VectorStore (Weaviate / Chroma) + chat history
    db.init()  # Postgres Q&A/metrics log — no-op if DATABASE_URL is unset
    yield
    if service and hasattr(service.store, "client") and hasattr(service.store.client, "close"):
        try:
            service.store.client.close()
        except Exception:
            pass
    db.close()


app = FastAPI(
    title="Multi-RAG API",
    description="REST backend for the multi-modal, per-chat RAG pipeline "
                "(Weaviate + local embeddings + BM25 + Groq).",
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
    except TooManyDocumentsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read file: {e}")

    if result["chunks_indexed"] == 0:
        raise HTTPException(status_code=422, detail=result["message"])
    return result


@app.post(
    "/chats/{chat_id}/upload/stream",
    tags=["files"],
)
def upload_file_stream(chat_id: str, file: UploadFile = File(...)):
    """Ingest + index ONE file into a chat with real-time SSE progress updates."""
    s = svc()
    try:
        s.get_chat(chat_id)
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found.")

    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def progress_stream():
        try:
            for event in s.ingest_file_stream(chat_id, file_bytes, file.filename or "upload"):
                yield sse(event)
        except Exception as e:
            yield sse({"status": "error", "percent": 100, "message": f"Ingestion failed: {e}"})

    return StreamingResponse(
        progress_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/chats/{chat_id}/file", status_code=204, tags=["files"])
def remove_file(chat_id: str, filename: str | None = None):
    """Remove one of this chat's indexed files (a chat may hold more than
    one — see config.MAX_DOCUMENTS_PER_CHAT). `filename` selects which one;
    omit it only when the chat holds a single file."""
    try:
        svc().remove_file(chat_id, filename)
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

    ls_run_id = langsmith_logging.start_chat_turn(chat_id, body.question)
    try:
        r = s.retrieve(chat_id, body.question, history)
    except NoDocumentError:
        langsmith_logging.fail_chat_turn(ls_run_id, "NoDocumentError: chat has no uploaded file")
        raise HTTPException(
            status_code=400,
            detail="Upload a file for this chat first — each chat answers only "
                   "from its own document.",
        )
    contexts = [] if r.get("direct_answer") else generator.context_texts(r["hits"])
    raw_contexts = [doc.text for doc, _ in r["raw_hits"] if doc.text]
    langsmith_logging.log_retrieval(
        ls_run_id, r["search_query"], raw_contexts, r["search_ms"], contexts, r["rerank_ms"],
        diagnostics=r.get("retrieval_diagnostics"),
        neighbor_expansion_ms=r.get("neighbor_expansion_ms", 0),
        diversity_ms=r.get("diversity_ms", 0),
        retrieval_meta=r.get("retrieval_meta"),
    )

    s.append_user_message(chat_id, body.question)

    # TOC_QUERY/PAGE_QUERY with nothing matched (no TOC detected / page not
    # found) -> answer directly with a clear message, no LLM call, so an
    # empty context never gets a chance to be hallucinated over.
    if r.get("direct_answer"):
        answer_text = r["direct_answer"]
        metrics = {
            "rewrite_ms": r["rewrite_ms"],
            "search_ms": r["search_ms"],
            "neighbor_expansion_ms": r.get("neighbor_expansion_ms", 0),
            "diversity_ms": r.get("diversity_ms", 0),
            "ttft_ms": 0,
            "generation_ms": 0,
            "confidence_pct": r["confidence_pct"],
        }
        ls_run_id_str = str(ls_run_id.id) if ls_run_id else None
        message_id = s.append_assistant_message(
            chat_id, answer_text, r["sources"], metrics,
            question=body.question, contexts=contexts, langsmith_run_id=ls_run_id_str,
        )
        db.log_qa(chat_id, body.question, answer_text, r["sources"], metrics, message_id=message_id)
        langsmith_logging.end_chat_turn(ls_run_id, answer_text, contexts, metrics)
        return {
            "chat_id": chat_id,
            "question": body.question,
            "search_query": r["search_query"],
            "answer": answer_text,
            "sources": r["sources"],
            "metrics": metrics,
            "message_id": message_id,
        }

    # Generate (collect the full stream server-side for the JSON response).
    t_gen = time.perf_counter()
    ttft = None
    parts: list[str] = []
    usage: dict = {}
    try:
        for delta in s.answer_stream(
            r["search_query"], r["hits"], original_question=body.question,
            verified_context=r["query_type"] != "NORMAL_QUERY", usage=usage,
        ):
            if ttft is None:
                ttft = time.perf_counter() - t_gen
            parts.append(delta)
    except Exception as e:
        langsmith_logging.fail_chat_turn(ls_run_id, f"Generation failed: {e}")
        raise HTTPException(status_code=502, detail=f"Generation failed: {e}")

    answer_text = "".join(parts)
    generation_ms = int((time.perf_counter() - t_gen) * 1000)
    ttft_ms = int((ttft or 0) * 1000)
    metrics = {
        "rewrite_ms": r["rewrite_ms"],
        "search_ms": r["search_ms"],
        "neighbor_expansion_ms": r.get("neighbor_expansion_ms", 0),
        "diversity_ms": r.get("diversity_ms", 0),
        "ttft_ms": ttft_ms,
        "generation_ms": generation_ms,
        "confidence_pct": r["confidence_pct"],
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
    # DeepEval/RAGAS scores are NOT computed here — human-in-the-loop: the UI
    # shows a "Calculate Metrics" button after the answer, and only clicking
    # it hits POST /chats/{chat_id}/messages/{message_id}/metrics below.

    ls_run_id_str = str(ls_run_id.id) if ls_run_id else None
    message_id = s.append_assistant_message(
        chat_id, answer_text, r["sources"], metrics,
        question=body.question, contexts=contexts, langsmith_run_id=ls_run_id_str,
    )
    db.log_qa(chat_id, body.question, answer_text, r["sources"], metrics, message_id=message_id)
    langsmith_logging.log_generation(
        ls_run_id, config.DEFAULT_MODEL, r["search_query"], answer_text, ttft_ms, generation_ms,
        usage=usage,
    )
    langsmith_logging.end_chat_turn(ls_run_id, answer_text, contexts, metrics)

    return {
        "chat_id": chat_id,
        "question": body.question,
        "search_query": r["search_query"],
        "answer": answer_text,
        "sources": r["sources"],
        "metrics": metrics,
        "message_id": message_id,
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

    ls_run_id = langsmith_logging.start_chat_turn(chat_id, body.question)
    try:
        r = s.retrieve(chat_id, body.question, history)
    except NoDocumentError:
        langsmith_logging.fail_chat_turn(ls_run_id, "NoDocumentError: chat has no uploaded file")
        raise HTTPException(
            status_code=400,
            detail="Upload a file for this chat first — each chat answers only "
                   "from its own document.",
        )
    contexts = [] if r.get("direct_answer") else generator.context_texts(r["hits"])
    raw_contexts = [doc.text for doc, _ in r["raw_hits"] if doc.text]
    langsmith_logging.log_retrieval(
        ls_run_id, r["search_query"], raw_contexts, r["search_ms"], contexts, r["rerank_ms"],
        diagnostics=r.get("retrieval_diagnostics"),
        neighbor_expansion_ms=r.get("neighbor_expansion_ms", 0),
        diversity_ms=r.get("diversity_ms", 0),
        retrieval_meta=r.get("retrieval_meta"),
    )

    s.append_user_message(chat_id, body.question)

    # TOC_QUERY/PAGE_QUERY with nothing matched -> stream the clear "not
    # found" message directly, no LLM call, same reasoning as the non-
    # streaming endpoint above.
    if r.get("direct_answer"):
        answer_text = r["direct_answer"]

        def direct_stream():
            yield sse({"type": "meta", "search_query": r["search_query"]})
            for word in answer_text.split(" "):
                yield sse({"type": "token", "text": word + " "})
            metrics = {
                "rewrite_ms": r["rewrite_ms"],
                "search_ms": r["search_ms"],
                "neighbor_expansion_ms": r.get("neighbor_expansion_ms", 0),
                "diversity_ms": r.get("diversity_ms", 0),
                "ttft_ms": 0,
                "generation_ms": 0,
                "confidence_pct": r["confidence_pct"],
            }
            ls_run_id_str = str(ls_run_id.id) if ls_run_id else None
            message_id = s.append_assistant_message(
                chat_id, answer_text, r["sources"], metrics,
                question=body.question, contexts=contexts, langsmith_run_id=ls_run_id_str,
            )
            db.log_qa(chat_id, body.question, answer_text, r["sources"], metrics, message_id=message_id)
            langsmith_logging.end_chat_turn(ls_run_id, answer_text, contexts, metrics)
            yield sse({"type": "done", "metrics": metrics, "sources": r["sources"], "message_id": message_id})

        return StreamingResponse(
            direct_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def event_stream():
        yield sse({"type": "meta", "search_query": r["search_query"]})

        t_gen = time.perf_counter()
        ttft = None
        parts: list[str] = []
        usage: dict = {}
        try:
            for delta in s.answer_stream(
            r["search_query"], r["hits"], original_question=body.question,
            verified_context=r["query_type"] != "NORMAL_QUERY", usage=usage,
        ):
                if ttft is None:
                    ttft = time.perf_counter() - t_gen
                parts.append(delta)
                yield sse({"type": "token", "text": delta})
        except Exception as e:
            langsmith_logging.fail_chat_turn(ls_run_id, f"Generation failed: {e}")
            yield sse({"type": "error", "message": f"Generation failed: {e}"})
            return

        answer_text = "".join(parts)
        ttft_ms = int((ttft or 0) * 1000)
        generation_ms = int((time.perf_counter() - t_gen) * 1000)
        metrics = {
            "rewrite_ms": r["rewrite_ms"],
            "search_ms": r["search_ms"],
            "neighbor_expansion_ms": r.get("neighbor_expansion_ms", 0),
            "diversity_ms": r.get("diversity_ms", 0),
            "ttft_ms": ttft_ms,
            "generation_ms": generation_ms,
            "confidence_pct": r["confidence_pct"],
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        # DeepEval/RAGAS scores are NOT computed here — human-in-the-loop: the
        # UI shows a "Calculate Metrics" button after the answer, and only
        # clicking it hits POST /chats/{chat_id}/messages/{message_id}/metrics.
        # Persist the assistant turn only after the full answer is produced.
        ls_run_id_str = str(ls_run_id.id) if ls_run_id else None
        message_id = s.append_assistant_message(
            chat_id, answer_text, r["sources"], metrics,
            question=body.question, contexts=contexts, langsmith_run_id=ls_run_id_str,
        )
        db.log_qa(chat_id, body.question, answer_text, r["sources"], metrics, message_id=message_id)
        langsmith_logging.log_generation(
            ls_run_id, config.DEFAULT_MODEL, r["search_query"], answer_text, ttft_ms, generation_ms,
            usage=usage,
        )
        langsmith_logging.end_chat_turn(ls_run_id, answer_text, contexts, metrics)
        yield sse({"type": "done", "metrics": metrics, "sources": r["sources"], "message_id": message_id})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
# Metrics — computed on demand ("Calculate Metrics" button), not automatically
# at answer time. Scores against the same parent-chunk context the answer was
# actually generated from (persisted on the message, see RagService.
# append_assistant_message), and back-fills Postgres + LangSmith feedback.
#
# `dataset_names` (request body) — the LangSmith dataset(s) the user picked in
# the UI's dataset dropdown for THIS evaluation, populated from
# GET /langsmith/datasets below. Empty/omitted -> the three reference-free
# metrics only; no automatic global golden-set fallback (see rag.golden_set).
# --------------------------------------------------------------------------- #
@app.post(
    "/chats/{chat_id}/messages/{message_id}/metrics",
    response_model=schemas.Metrics,
    tags=["ask"],
)
def calculate_metrics(chat_id: str, message_id: str, body: schemas.CalculateMetricsRequest | None = None):
    dataset_names = body.dataset_names if body else []
    try:
        metrics = svc().evaluate_message(chat_id, message_id, dataset_names)
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found.")
    except MessageNotFoundError:
        raise HTTPException(status_code=404, detail=f"Message {message_id} not found in this chat.")
    return metrics


@app.get("/langsmith/datasets", response_model=schemas.GoldenDatasetList, tags=["ask"])
def langsmith_datasets():
    """Every LangSmith dataset available for the "Calculate Metrics" dataset
    picker (see rag.golden_set.list_available_datasets). Empty list, not an
    error, if LangSmith isn't configured or is unreachable."""
    return {"datasets": svc().list_golden_datasets()}


# --------------------------------------------------------------------------- #
# Q&A + metrics log (Postgres, see db.py) — read-back for history/analytics.
# --------------------------------------------------------------------------- #
@app.get("/qa-logs", response_model=list[schemas.QaLogEntry], tags=["qa-logs"])
def qa_logs(chat_id: str | None = None, limit: int = 100):
    """Recent logged Q&A turns + their 6 RAGAS-style metrics, newest first.

    Empty list (not an error) if Postgres logging isn't configured/reachable.
    """
    rows = db.fetch_qa_logs(chat_id=chat_id, limit=limit)
    for row in rows:
        row["created_at"] = row["created_at"].isoformat()
    return rows


@app.get("/chats/{chat_id}/qa-logs", response_model=list[schemas.QaLogEntry], tags=["qa-logs"])
def chat_qa_logs(chat_id: str, limit: int = 100):
    """Recent logged Q&A turns for one chat."""
    rows = db.fetch_qa_logs(chat_id=chat_id, limit=limit)
    for row in rows:
        row["created_at"] = row["created_at"].isoformat()
    return rows


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
