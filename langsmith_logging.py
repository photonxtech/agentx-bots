"""Live LangSmith logging for chat turns.

This is what makes a question asked in the frontend show up in LangSmith
immediately — one "chat_turn" run per turn, with "retrieve_documents" and
"llm" child runs nested under it (so retrieval and generation show up as
their own spans, not just buried in the parent's outputs). "retrieve_documents"
itself has two children of its own — "vector_search" (the raw pre-rerank
candidate pool) and "rerank_documents" (the final reranked + filtered set) —
so before/after reranking are each independently inspectable in the UI,
instead of only ever seeing the final hits. Every metric from api.main
(timing + RAGAS/DeepEval scores from api.service.RagService.evaluate_answer())
is attached to the parent as feedback.
It's the live-traffic counterpart to scripts/run_langsmith_eval.py, which
instead replays the curated golden-set dataset as a single batch "experiment"
— the two write to the same LangSmith project but are otherwise independent;
this module never touches a dataset.

Four calls, in order, per turn: start_chat_turn() (before retrieval) ->
log_retrieval() (right after hits come back, real search_ms/rerank_ms as the
vector_search/rerank_documents durations) -> log_generation() (right after
the answer is fully generated, real ttft_ms/generation_ms as duration) ->
end_chat_turn() (once the answer + metrics are known). They're split instead
of one shot because each child run needs to be posted as a genuine child of
the parent run — `@traceable`'s
automatic context-propagation doesn't apply here since nothing upstream runs
inside a live trace context, so the parent/child link is carried explicitly
via a `RunTree` object (returned by start_chat_turn, threaded through the
other three calls) instead. RunTree, not raw client.create_run(trace_id=...,
parent_run_id=...), is required here: the API rejects a manually-specified
trace_id without a matching `dotted_order`, and RunTree.create_child()
computes that automatically from the parent — hand-rolling it isn't worth it.
Timing is reconstructed from measured durations (end=now, start=end-duration)
rather than wrapping the live calls, since generator.answer_stream() is a
plain generator with no LangSmith context of its own — this keeps each run's
own duration accurate even though it's posted after the fact.

No-op (not an error) when LANGSMITH_TRACING/LANGSMITH_API_KEY aren't set, same
as db.py's Postgres logging. Never raises — a LangSmith outage must not break
a chat answer.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

from langsmith.run_trees import RunTree

import config

logger = logging.getLogger(__name__)

_client = None
_enabled: bool | None = None


_project_id = None


def _get_client():
    global _client, _enabled
    if _enabled is None:
        _enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(
            os.getenv("LANGSMITH_API_KEY")
        )
        if _enabled:
            from langsmith import Client

            _client = Client()
    return _client


def _get_project_id():
    """Resolves LANGCHAIN_PROJECT (a name) to the project's actual id, once,
    and caches it. create_feedback() needs session_id/project_id (a UUID) —
    it does NOT accept project_name; passing that silently no-ops via
    **kwargs and never resolves the 'feedback without session_id' deprecation
    warning, which is what was happening before this existed."""
    global _project_id
    if _project_id is not None:
        return _project_id
    client = _get_client()
    if client is None:
        return None
    try:
        project = client.read_project(
            project_name=os.getenv("LANGCHAIN_PROJECT", "pdf-copilot")
        )
        _project_id = project.id
    except Exception:
        logger.warning("LangSmith project id lookup failed", exc_info=True)
    return _project_id


def start_chat_turn(chat_id: str, question: str) -> RunTree | None:
    """Create the parent 'chat_turn' run. Returns it, or None if disabled/failed.

    Call this before retrieval so the other three calls have a run to nest
    under / finalize. The returned RunTree — not a bare run_id — is what
    lets log_retrieval()/log_generation() attach correctly-linked children.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        run = RunTree(
            name="chat_turn",
            run_type="chain",
            inputs={"chat_id": chat_id, "question": question},
            start_time=datetime.now(timezone.utc),
            client=client,
        )
        run.post()
        return run
    except Exception:
        logger.warning("LangSmith chat-turn start failed", exc_info=True)
        return None


def log_retrieval(
    run: RunTree | None,
    query: str,
    raw_contexts: list[str],
    search_ms: int | None,
    final_contexts: list[str],
    rerank_ms: int | None,
) -> None:
    """Log retrieval as a 'retrieve_documents' child run under run, with
    'vector_search' (pre-rerank candidate pool) and 'rerank_documents' (final
    reranked + filtered set) nested under that — so before/after reranking
    are each independently inspectable in the LangSmith UI.
    """
    if run is None:
        return
    try:
        search_start = datetime.now(timezone.utc) - timedelta(
            milliseconds=(search_ms or 0) + (rerank_ms or 0)
        )
        rerank_start = search_start + timedelta(milliseconds=search_ms or 0)
        rerank_end = rerank_start + timedelta(milliseconds=rerank_ms or 0)

        parent = run.create_child(
            name="retrieve_documents",
            run_type="retriever",
            inputs={"query": query},
            start_time=search_start,
            # Chunking happens once at ingestion, not per-turn, so it has no
            # model/duration of its own — but the config it ran under still
            # belongs on the retrieval span, since it's what shaped every
            # chunk vector_search/rerank_documents below are scoring.
            extra={
                "metadata": {
                    "chunking": {
                        "parent_chunk_size": config.PARENT_CHUNK_SIZE,
                        "parent_chunk_overlap": config.PARENT_CHUNK_OVERLAP,
                        "child_chunk_size": config.CHILD_CHUNK_SIZE,
                        "child_chunk_overlap": config.CHILD_CHUNK_OVERLAP,
                    }
                }
            },
        )

        vector_search = parent.create_child(
            name="vector_search",
            run_type="retriever",
            inputs={"query": query, "model": config.EMBEDDING_MODEL},
            start_time=search_start,
            extra={"metadata": {"model": config.EMBEDDING_MODEL}},
        )
        vector_search.end(outputs={"documents": raw_contexts}, end_time=rerank_start)
        vector_search.post()

        rerank_documents = parent.create_child(
            name="rerank_documents",
            run_type="chain",
            inputs={"documents": raw_contexts, "model": config.RERANK_MODEL},
            start_time=rerank_start,
            extra={"metadata": {"model": config.RERANK_MODEL}},
        )
        rerank_documents.end(outputs={"documents": final_contexts}, end_time=rerank_end)
        rerank_documents.post()

        parent.end(outputs={"documents": final_contexts}, end_time=rerank_end)
        parent.post()
    except Exception:
        logger.warning("LangSmith retrieval logging failed", exc_info=True)


def log_generation(
    run: RunTree | None,
    model: str,
    question: str,
    answer: str,
    ttft_ms: int,
    generation_ms: int,
) -> None:
    """Log the Groq call as an 'llm' child run under run.

    Without this, generation never became its own span — TTFT and generation
    time were computed in api.main but had nowhere in LangSmith to attach to.
    TTFT has no dedicated field on a non-streamed 'llm' run, so it's attached
    as metadata; the run's own start/end still spans the full generation_ms
    so the span's displayed latency is the true end-to-end duration.
    """
    if run is None:
        return
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(milliseconds=generation_ms or 0)
        child = run.create_child(
            name="llm",
            run_type="llm",
            inputs={"question": question, "model": model},
            start_time=start,
            extra={"metadata": {"ttft_ms": ttft_ms, "generation_ms": generation_ms}},
        )
        child.end(outputs={"answer": answer}, end_time=end)
        child.post()
    except Exception:
        logger.warning("LangSmith generation logging failed", exc_info=True)


def end_chat_turn(
    run: RunTree | None, answer: str, contexts: list[str], metrics: dict | None
) -> None:
    """Finalize the parent run with the answer, and attach every metric as feedback.

    Previously only the 6 RAGAS/DeepEval score keys were forwarded — timing
    metrics (ttft_ms, generation_ms, search_ms, rewrite_ms, confidence_pct)
    were computed in api.main and shown in the frontend, but silently dropped
    before ever reaching LangSmith. Now every numeric metric is sent.
    """
    if run is None:
        return
    client = _get_client()
    if client is None:
        return
    try:
        run.end(
            outputs={"answer": answer, "context": contexts, "metrics": metrics or {}},
            end_time=datetime.now(timezone.utc),
        )
        run.patch()
        for key, value in (metrics or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                client.create_feedback(
                    run_id=run.id,
                    trace_id=run.trace_id,
                    key=key,
                    score=value,
                    session_id=_get_project_id(),
                )
    except Exception:
        logger.warning("LangSmith chat-turn end failed", exc_info=True)

def attach_feedback(run_id: str | None, metrics: dict | None) -> None:
    """Attach every numeric metric as feedback to an already-finalized
    chat_turn run, given just its stored run_id — not a live RunTree.

    DeepEval scores are no longer computed automatically at generation time
    (see api.service.RagService.evaluate_message); they're computed on demand
    when the user clicks "Calculate Metrics", by which point the original
    RunTree from start_chat_turn() is long gone (a separate HTTP request, its
    own process lifetime). The run's id survives in chat_store instead, and
    that's all client.create_feedback() actually needs — feedback can be
    attached to a run at any time after it's posted, live or not.
    """
    if run_id is None:
        return
    client = _get_client()
    if client is None:
        return
    try:
        for key, value in (metrics or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                client.create_feedback(
                    run_id=run_id,
                    key=key,
                    score=value,
                    session_id=_get_project_id(),
                )
    except Exception:
        logger.warning("LangSmith feedback attach failed", exc_info=True)
