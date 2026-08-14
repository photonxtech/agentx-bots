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
    diagnostics: list[dict] | None = None,
    neighbor_expansion_ms: int | None = None,
    diversity_ms: int | None = None,
    retrieval_meta: dict | None = None,
) -> None:
    """Log retrieval as a 'retrieve_documents' child run under run, with
    'vector_search' (pre-rerank candidate pool) and 'rerank_documents' (final
    reranked + filtered set) nested under that — so before/after reranking
    are each independently inspectable in the LangSmith UI.

    `diagnostics` (see api.service.RagService._build_retrieval_diagnostics) —
    one row per final chunk with rank/chunk_id/page/section/chunk_type/
    dense/bm25/hybrid/reranker scores plus parent/child chunk identity
    (parent_id/child_index/parent_chars/child_chars/used_parent_context) — is
    attached as structured output on the rerank_documents span, so this
    per-question breakdown is visible in the LangSmith UI (Outputs tab of
    that span) without needing DEBUG-level server logs.

    `neighbor_expansion_ms`/`diversity_ms` — timed separately from
    search_ms/rerank_ms (see RagService.retrieve) so a slow vector DB, a slow
    neighbor-expansion pass, a slow cross-encoder, and a slow MMR/filter pass
    are each independently identifiable instead of bundled together.

    `retrieval_meta` — embedding_model/rerank_model/candidate_k/final_k/
    expanded_count, attached to the parent span so the models and pool sizes
    actually used for this turn are visible without cross-referencing config.
    """
    if run is None:
        return
    try:
        total_ms = (
            (search_ms or 0) + (neighbor_expansion_ms or 0)
            + (rerank_ms or 0) + (diversity_ms or 0)
        )
        search_start = datetime.now(timezone.utc) - timedelta(milliseconds=total_ms)
        expand_start = search_start + timedelta(milliseconds=search_ms or 0)
        rerank_start = expand_start + timedelta(milliseconds=neighbor_expansion_ms or 0)
        diversity_start = rerank_start + timedelta(milliseconds=rerank_ms or 0)
        rerank_end = diversity_start + timedelta(milliseconds=diversity_ms or 0)

        meta = retrieval_meta or {}
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
                    "embedding_model": meta.get("embedding_model", config.EMBEDDING_MODEL),
                    "rerank_model": meta.get("rerank_model", config.RERANK_MODEL),
                    "candidate_k": meta.get("candidate_k"),
                    "final_k": meta.get("final_k"),
                    "expanded_count": meta.get("expanded_count"),
                    "neighbor_expansion_ms": neighbor_expansion_ms,
                    "diversity_ms": diversity_ms,
                    "chunking": {
                        "parent_chunk_size": config.PARENT_CHUNK_SIZE,
                        "parent_chunk_overlap": config.PARENT_CHUNK_OVERLAP,
                        "child_chunk_size": config.CHILD_CHUNK_SIZE,
                        "child_chunk_overlap": config.CHILD_CHUNK_OVERLAP,
                    },
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
        vector_search.end(outputs={"documents": raw_contexts}, end_time=expand_start)
        vector_search.post()

        rerank_documents = parent.create_child(
            name="rerank_documents",
            run_type="chain",
            inputs={"documents": raw_contexts, "model": config.RERANK_MODEL},
            start_time=rerank_start,
            extra={"metadata": {"model": config.RERANK_MODEL}},
        )
        rerank_documents.end(
            outputs={"documents": final_contexts, "diagnostics": diagnostics or []},
            end_time=rerank_end,
        )
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
    usage: dict | None = None,
) -> None:
    """Log the Groq call as an 'llm' child run under run.

    Without this, generation never became its own span — TTFT and generation
    time were computed in api.main but had nowhere in LangSmith to attach to.
    TTFT has no dedicated field on a non-streamed 'llm' run, so it's attached
    as metadata; the run's own start/end still spans the full generation_ms
    so the span's displayed latency is the true end-to-end duration.

    `usage` — prompt_tokens/completion_tokens/total_tokens (see
    generator.answer's `usage` sink param), attached both as metadata here
    and, via LangSmith's own token-usage fields, so cost/latency dashboards
    that read `usage_metadata` pick it up natively instead of needing a
    custom metric.
    """
    if run is None:
        return
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(milliseconds=generation_ms or 0)
        usage = usage or {}
        child = run.create_child(
            name="llm",
            run_type="llm",
            inputs={"question": question, "model": model},
            start_time=start,
            extra={
                "metadata": {
                    "ttft_ms": ttft_ms,
                    "generation_ms": generation_ms,
                    "usage_metadata": {
                        "input_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                    },
                }
            },
        )
        child.end(outputs={"answer": answer}, end_time=end)
        child.post()
    except Exception:
        logger.warning("LangSmith generation logging failed", exc_info=True)


def fail_chat_turn(run: RunTree | None, error: str) -> None:
    """Close an in-progress 'chat_turn' run as FAILED instead of leaving it
    orphaned (perpetually "running") in the LangSmith UI.

    Confirmed live: retrieval raising NoDocumentError, or generation raising
    mid-stream, both re-raised straight to FastAPI's error handling in
    api.main without ever calling end_chat_turn() — the run started by
    start_chat_turn() was simply never closed. Call this from those except
    blocks, before re-raising/yielding the error to the client.
    """
    if run is None:
        return
    try:
        run.end(error=error, end_time=datetime.now(timezone.utc))
        run.patch()
    except Exception:
        logger.warning("LangSmith chat-turn failure logging failed", exc_info=True)


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
                    run_id=run.id, trace_id=run.trace_id, key=key, score=value
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
                client.create_feedback(run_id=run_id, key=key, score=value)
    except Exception:
        logger.warning("LangSmith feedback attach failed", exc_info=True)
