"""
PhotonX RAG - FastAPI Integration
=================================

Drop this file into the root of the PhotonXRAG repo (same folder as
rag_engine.py, ingest.py, app.py). It exposes the existing RAG pipeline
as a JSON API, no Streamlit required.

Run it:
    pip install fastapi "uvicorn[standard]"
    export GROQ_API_KEY=your_key_here
    python ingest.py            # if you haven't already built chroma_db/
    uvicorn main:app --reload --port 8000

Then hit it:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/ask \
         -H "Content-Type: application/json" \
         -d '{"query": "What services does PhotonX offer?"}'

Interactive docs: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import rag_engine

# ---------------------------------------------------------------------------
# Load heavy resources (embedder, reranker, chroma collection, bm25 index)
# exactly once at process startup, not per-request.
# ---------------------------------------------------------------------------
_resources: rag_engine.RagResources | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _resources
    try:
        _resources = rag_engine.load_resources()
    except RuntimeError as e:
        # Chroma collection empty / ingest.py not run yet. Let the app boot
        # anyway so /health can report the problem clearly instead of the
        # process crash-looping.
        print(f"[startup warning] {e}")
        _resources = None
    yield
    _resources = None


app = FastAPI(
    title="PhotonX RAG API",
    description="Grounded Q&A over PhotonX's internal documents.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow browser-based frontends (adjust origins for production).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")
    chat_history: list[ChatTurn] = Field(
        default_factory=list,
        description="Prior turns for follow-up context (last 6 are used).",
    )


class SourceChunk(BaseModel):
    id: str
    text: str
    metadata: dict
    rerank_score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_resources() -> rag_engine.RagResources:
    if _resources is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG index isn't loaded. Run `python ingest.py` against your "
                "source_docs/ files, then restart the API."
            ),
        )
    return _resources


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok" if _resources is not None else "index_not_loaded",
        "indexed_chunks": len(_resources.all_ids) if _resources else 0,
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    """Non-streaming: waits for the full answer, then returns it as JSON."""
    res = _require_resources()
    history = [turn.model_dump() for turn in payload.chat_history]

    chunks, gen = rag_engine.ask(res, payload.query, history)
    answer = "".join(gen)  # drain the generator into one string

    return AskResponse(
        answer=answer,
        sources=[
            SourceChunk(
                id=c["id"],
                text=c["text"],
                metadata=c["metadata"],
                rerank_score=c.get("rerank_score", 0.0),
            )
            for c in chunks
        ],
    )


@app.post("/ask/stream")
def ask_stream(payload: AskRequest):
    """
    Streaming variant: Server-Sent-Events-style plain text stream.
    Sources aren't known until retrieval finishes, so they're sent first
    as a single JSON line prefixed with `event: sources`, followed by the
    token stream prefixed with `event: token`.
    """
    res = _require_resources()
    history = [turn.model_dump() for turn in payload.chat_history]

    def event_stream():
        import json

        chunks, gen = rag_engine.ask(res, payload.query, history)
        sources_payload = [
            {
                "id": c["id"],
                "metadata": c["metadata"],
                "rerank_score": c.get("rerank_score", 0.0),
            }
            for c in chunks
        ]
        yield f"event: sources\ndata: {json.dumps(sources_payload)}\n\n"
        for token in gen:
            yield f"event: token\ndata: {json.dumps(token)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
