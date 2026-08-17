"""
RAG Evaluation Orchestrator

Flow:
  1. Build a config-driven RAG pipeline (chunk -> embed -> index -> retrieve
     -> optional rerank -> generate) and call it to get answers + contexts.
  2. Score the WHOLE BATCH with DeepEval — faithfulness, answer_relevancy,
     context_precision, context_recall. These are the library's own metric
     implementations, not hand-written judge prompts.
  3. Log the batch + aggregate scores to LangSmith -> return experiment URL.

A note on the division of labour, because it is easy to get backwards:
LangSmith does not compute metrics and never has. It is a tracing / dataset /
experiment store. DeepEval computes; LangSmith records. Both are used here, for
those two distinct jobs.
"""
import os
import json
import uuid
import logging
import re
import math
import hashlib
import asyncio
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

from backend.services.document_reader import read_document, read_documents
from backend.services.groq_keys import (
    make_client, make_async_client, key_pool_size,
    rotation_stats, reset_rotation_stats,
)
from backend.services.llm_endpoint import resolve_endpoint, auth_headers, endpoint_stats

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_MODEL     = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag-eval-ui")

INDEX_CACHE_DIR = os.getenv("RAG_INDEX_CACHE_DIR", os.path.join(os.getcwd(), ".rag_index_cache"))
os.makedirs(INDEX_CACHE_DIR, exist_ok=True)

# In-process cache: hash(config-relevant fields) -> built index bundle.
# Avoids re-chunking/re-embedding the same document for the same
# chunk_size/chunk_overlap/embedding_model combination.
_INDEX_CACHE: dict[str, dict] = {}

# Chat models actually servable by Groq. The UI offers exactly these; anything
# else (e.g. a legacy free-text "gpt-4" from an older run) falls back to
# GROQ_MODEL and the substitution is reported in the run's notes rather than
# being silently swallowed.
# Verified against Groq's live /v1/models list. The Llama family (llama-3.3-70b-
# versatile, llama-3.1-8b-instant) was removed by Groq and is NOT listed here —
# requesting either now fails outright rather than being merely deprecated.
GROQ_CHAT_MODELS = {
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "allam-2-7b",
}
# Deliberately NOT offered, though Groq serves them:
#   groq/compound, groq/compound-mini  — agentic systems with built-in web
#     search. They can answer from the internet instead of the retrieved
#     context, which silently invalidates faithfulness and context_recall.
#   openai/gpt-oss-safeguard-20b       — tuned for content moderation, not QA.
#   whisper-*, canopylabs/orpheus-*    — speech models.
#   meta-llama/llama-prompt-guard-*    — 512-token classifiers.

# The judge deliberately defaults to a different model family than the usual
# generator: letting one model grade its own output inflates faithfulness and
# relevancy. _resolve_models() below enforces generator != judge.
#
# Sourced from DEEPEVAL_MODEL rather than a separate setting, so there is one
# place that decides who judges — this guard previously keyed off a separate
# variable and would have quietly stopped working when that was removed.
JUDGE_FALLBACK = "qwen/qwen3.6-27b"

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

# The form now submits real FastEmbed model ids, so no translation is needed
# for new runs. These aliases only exist so runs saved by earlier versions
# still resolve to the model they actually used.
LEGACY_EMBEDDING_ALIASES = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
    # These never ran on OpenAI — the old map silently substituted BGE.
    "text-embedding-3-small": "BAAI/bge-small-en-v1.5",
    "text-embedding-3-large": "BAAI/bge-base-en-v1.5",
}

LEGACY_RERANKER_ALIASES = {
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "bge-reranker-large": "BAAI/bge-reranker-base",   # large is not in FastEmbed
    "cohere-rerank-v3": "BAAI/bge-reranker-base",     # closest local equivalent
}


def _supported_embedding_models() -> set[str]:
    """Model ids FastEmbed can actually serve, queried from its own registry
    rather than hard-coded, so the list cannot drift out of date."""
    try:
        from fastembed import TextEmbedding
        return {m["model"] for m in TextEmbedding.list_supported_models()}
    except Exception:
        return set()


def _supported_reranker_models() -> set[str]:
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        return {m["model"] for m in TextCrossEncoder.list_supported_models()}
    except Exception:
        return set()


def _groq_ready() -> bool:
    return key_pool_size() > 0


def _resolve_models(config, notes: list[str]) -> tuple[str, str]:
    """Pick (generator_model, judge_model), guaranteeing they differ.

    Appends a human-readable line to `notes` for every substitution made, so
    the UI can show exactly which models produced the numbers.
    """
    requested = (config.chat_model or "").strip()
    if requested in GROQ_CHAT_MODELS:
        generator = requested
    else:
        generator = GROQ_MODEL if GROQ_MODEL in GROQ_CHAT_MODELS else JUDGE_FALLBACK
        if requested:
            notes.append(
                f"Chat model '{requested}' is not available on Groq — generated with '{generator}' instead."
            )

    from backend.services.deepeval_llm import DEEPEVAL_MODEL
    judge = DEEPEVAL_MODEL
    if judge == generator:
        judge = JUDGE_FALLBACK if generator != JUDGE_FALLBACK else "openai/gpt-oss-120b"
        notes.append(
            f"Judge model matched the generator — judged with '{judge}' instead to avoid self-evaluation bias."
        )
    return generator, judge


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — RAG pipeline (chunk -> embed -> index -> retrieve -> generate)
# ═══════════════════════════════════════════════════════════════════════════════

def _read_document(document_path: str) -> str:
    """Read plain text out of pdf / docx / txt files."""
    return read_document(document_path)


def _chunk_document(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Character-based sliding window chunking, snapped to paragraph/sentence
    boundaries where possible. chunk_size/chunk_overlap come straight from
    the EvaluateRequest config."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    step = max(chunk_size - chunk_overlap, 1)
    while start < n:
        end = min(start + chunk_size, n)
        boundary = text.rfind("\n\n", start, end)
        if boundary == -1 or boundary <= start + chunk_size // 2:
            boundary = text.rfind(". ", start, end)
        if boundary == -1 or boundary <= start:
            boundary = end
        else:
            boundary += 1
        chunk = text[start:boundary].strip()
        if chunk:
            chunks.append(chunk)
        if boundary >= n:
            break
        start += step
    return chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _resolve_embedding_model(config, notes: list[str] | None = None) -> str:
    """Resolve the configured embedding model to a real FastEmbed model id."""
    raw = (config.embedding_model or "").strip()
    if raw in _supported_embedding_models():
        return raw
    alias = LEGACY_EMBEDDING_ALIASES.get(raw.lower())
    if alias:
        return alias
    if raw and notes is not None:
        notes.append(
            f"Embedding model '{raw}' is not available in FastEmbed — "
            f"embedded with '{DEFAULT_EMBEDDING_MODEL}' instead."
        )
    return DEFAULT_EMBEDDING_MODEL


def _resolve_reranker_model(raw: str) -> str | None:
    """Resolve a reranker selection to a FastEmbed cross-encoder id, or None."""
    raw = (raw or "").strip()
    if not raw:
        return None
    if raw in _supported_reranker_models():
        return raw
    return LEGACY_RERANKER_ALIASES.get(raw.lower(), DEFAULT_RERANKER_MODEL)


def _index_cache_key(document_paths: list[str], config) -> str:
    # Only fields that genuinely change the index belong here. `vectordb` and
    # `index_type` do not affect retrieval (always Chroma + its default index),
    # so including them would force pointless rebuilds.
    #
    # Every document path is part of the key, sorted so upload order does not
    # matter — adding or removing a file must rebuild, reordering must not.
    docs = "|".join(sorted(document_paths))
    raw = f"{docs}|{config.chunk_size}|{config.chunk_overlap}|{_resolve_embedding_model(config)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _build_or_get_index(document_paths: list[str], config) -> dict:
    """Build (chunks, dense collection, bm25 index) across every uploaded
    document, or reuse it from cache if already built.

    Each document is chunked separately so no chunk ever spans two files — a
    chunk straddling unrelated documents would be incoherent to retrieve and
    impossible to attribute. `sources[i]` records which file chunk `i` came from.
    """
    cache_key = _index_cache_key(document_paths, config)
    if cache_key in _INDEX_CACHE:
        return _INDEX_CACHE[cache_key]

    embedding_model_name = _resolve_embedding_model(config)
    logger.info("Building index over %d document(s) (chunk_size=%s, overlap=%s, embedding=%s)...",
                len(document_paths), config.chunk_size, config.chunk_overlap, embedding_model_name)

    docs = read_documents(document_paths)
    if not docs:
        raise RuntimeError("No readable text in the uploaded document(s).")

    chunks: list[str] = []
    sources: list[str] = []
    for filename, text in docs:
        doc_chunks = _chunk_document(text, config.chunk_size, config.chunk_overlap)
        chunks.extend(doc_chunks)
        sources.extend([filename] * len(doc_chunks))

    if not chunks:
        raise RuntimeError("Document(s) produced no chunks — check the file contents.")

    # Dense index (FastEmbed + Chroma)
    from fastembed import TextEmbedding
    import chromadb

    embedder = TextEmbedding(model_name=embedding_model_name)
    embeddings = list(embedder.embed(chunks))

    chroma_client = chromadb.PersistentClient(path=INDEX_CACHE_DIR)
    collection_name = f"idx_{cache_key}"
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass
    collection = chroma_client.create_collection(collection_name)
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=[e.tolist() for e in embeddings],
        documents=chunks,
        # Source travels as metadata, never inside the chunk text — prefixing the
        # text would put a filename into the context the judge scores.
        metadatas=[{"source": src} for src in sources],
    )

    # Sparse index (BM25) for keyword / hybrid search
    from rank_bm25 import BM25Okapi
    tokenized_chunks = [_tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    bundle = {
        "chunks": chunks,
        "sources": sources,
        "embedder": embedder,
        "collection": collection,
        "bm25": bm25,
        "embedding_model_name": embedding_model_name,
        "num_documents": len(docs),
    }
    _INDEX_CACHE[cache_key] = bundle
    return bundle


# Cross-encoders are expensive to construct, so keep one per model for the
# process lifetime instead of rebuilding it on every question.
_RERANKER_CACHE: dict[str, object] = {}


def _get_reranker(model_name: str):
    if model_name not in _RERANKER_CACHE:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _RERANKER_CACHE[model_name] = TextCrossEncoder(model_name=model_name)
    return _RERANKER_CACHE[model_name]


def _reciprocal_rank_fusion(rank_lists: list[list[int]], k: int = 60) -> list[int]:
    """Combine multiple ranked lists of chunk indices into one ranking."""
    scores: dict[int, float] = {}
    for ranked in rank_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def _retrieve(question: str, index: dict, config, notes: list[str]) -> list[str]:
    """Retrieve top_k chunks using config.search_type (keyword / semantic / hybrid)."""
    chunks = index["chunks"]
    top_k = config.top_k
    candidate_k = min(max(top_k * 3, top_k), len(chunks))

    dense_ranked, sparse_ranked = [], []

    if config.search_type in ("semantic", "hybrid"):
        query_emb = list(index["embedder"].embed([question]))[0].tolist()
        result = index["collection"].query(query_embeddings=[query_emb], n_results=candidate_k)
        dense_ranked = [int(i) for i in result["ids"][0]]

    if config.search_type in ("keyword", "hybrid"):
        tokenized_query = _tokenize(question)
        scores = index["bm25"].get_scores(tokenized_query)
        sparse_ranked = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:candidate_k]]

    if config.search_type == "semantic":
        final_ids = dense_ranked[:top_k]
    elif config.search_type == "keyword":
        final_ids = sparse_ranked[:top_k]
    else:  # hybrid — reciprocal rank fusion of both rankings
        final_ids = _reciprocal_rank_fusion([dense_ranked, sparse_ranked])[:top_k]

    retrieved = [chunks[i] for i in final_ids]

    # Optional cross-encoder reranking pass. This runs on FastEmbed's ONNX
    # rerankers, so it works out of the box — no torch, no FlagEmbedding.
    # Reranking reorders the chunks already retrieved; it can raise
    # context_precision but by construction cannot change context_recall.
    if config.reranker_model and retrieved:
        reranker_name = _resolve_reranker_model(config.reranker_model)
        try:
            reranker = _get_reranker(reranker_name)
            scores = list(reranker.rerank(question, retrieved))
            ranked = sorted(zip(scores, retrieved), key=lambda x: x[0], reverse=True)
            retrieved = [c for _, c in ranked]
        except Exception as e:
            msg = f"Reranker '{reranker_name}' failed ({e}) — results are NOT reranked."
            if msg not in notes:
                notes.append(msg)

    return retrieved


async def _generate_answer(question: str, contexts: list[str], config, model: str) -> str:
    """Generate the final answer from retrieved contexts using the resolved
    Groq chat model at the configured temperature."""
    context_text = "\n---\n".join(contexts) if contexts else "(no relevant context found)"
    prompt = f"""Answer the question using ONLY the provided context. Be concise and factual.
If the context does not contain the answer, say so explicitly.

Context:
{context_text}

Question: {question}

Answer:"""

    if not _groq_ready():
        return "[No Groq API key configured — cannot generate an answer]"

    try:
        # Rotating client: a 429 on one Groq account fails over to the next. When
        # a gateway is configured and healthy the endpoint resolves there instead,
        # adding provider failover underneath key rotation; the transport leaves
        # gateway requests untouched, so the two never fight over the auth header.
        ep = await resolve_endpoint()
        async with make_async_client(timeout=90.0) as client:
            res = await client.post(
                ep.chat_completions,
                headers=auth_headers(ep),
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": config.temperature,
                    # Explicit, not merely omitted: some gateway providers treat
                    # an absent `stream` as streaming and answer with an SSE body,
                    # which .json() below cannot parse.
                    "stream": False,
                },
            )
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error("Answer generation failed: %s", e)
        return f"[Answer generation failed: {e}]"


async def run_rag_pipeline(
    config,
    qa_pairs: list[dict],
    document_paths: list[str] | None,
    generator_model: str,
    notes: list[str],
) -> list[dict]:
    """
    Config-driven RAG pipeline. For each question:
      1. Build/reuse an index keyed by (document, chunk_size, chunk_overlap, embedding_model)
      2. Retrieve top_k chunks using search_type (keyword / semantic / hybrid),
         optionally reranked with reranker_model
      3. Generate an answer from those chunks with generator_model at config.temperature

    Returns list of dicts: {question, generated_answer, expected_answer, contexts}
    """
    paths = [p for p in (document_paths or []) if p and os.path.exists(p)]
    if not paths:
        raise RuntimeError("No document available to build the RAG index from.")

    index = await asyncio.to_thread(_build_or_get_index, paths, config)
    if index.get("num_documents", 1) > 1:
        notes.append(
            f"Retrieval ran over {index['num_documents']} documents "
            f"({len(index['chunks'])} chunks total) — chunks are never split across files."
        )

    results = []
    for qa in qa_pairs:
        contexts = await asyncio.to_thread(_retrieve, qa["question"], index, config, notes)
        answer = await _generate_answer(qa["question"], contexts, config, generator_model)
        results.append({
            "question": qa["question"],
            "generated_answer": answer,
            "expected_answer": qa["answer"],
            "contexts": contexts,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — LangSmith logging (tracing / experiment store, NOT metric computation)
# ═══════════════════════════════════════════════════════════════════════════════

def log_to_langsmith(
    session_id: str,
    run_config_id: str,
    results: list[dict],
    qa_pairs: list[dict],
    aggregate: dict,
) -> tuple[str | None, str | None]:
    """
    Upload QA pairs as a LangSmith dataset (one per session), create a run per
    sample with its per-sample metric feedback, plus one batch-summary run
    carrying the aggregate.

    Returns (project_url, summary_run_id). The summary run id is kept so human
    feedback can later be attached to the same run as the automated scores.
    """
    if not LANGCHAIN_API_KEY or LANGCHAIN_API_KEY.startswith("your_"):
        logger.info("LangSmith API key not set — skipping LangSmith logging")
        return None, None

    try:
        from langsmith import Client
        client = Client(api_key=LANGCHAIN_API_KEY)

        # ── Create/get dataset for this session ───────────────────────────────
        dataset_name = f"rag-eval-session-{str(session_id)[:8]}"
        try:
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description=f"QA pairs for RAG eval session {session_id}",
            )
            client.create_examples(
                dataset_id=dataset.id,
                examples=[
                    {
                        "inputs": {"question": qa["question"]},
                        "outputs": {"expected_answer": qa["answer"]},
                    }
                    for qa in qa_pairs
                ],
            )
            logger.info("Created LangSmith dataset: %s", dataset_name)
        except Exception:
            logger.info("LangSmith dataset already exists: %s", dataset_name)

        project_name = f"rag-eval-run-{str(run_config_id)[:8]}"
        now = datetime.now(timezone.utc)

        # Materialise the project up front so read_project() can return a URL
        # at the end even if every individual run failed to log.
        try:
            client.read_project(project_name=project_name)
        except Exception:
            try:
                client.create_project(project_name=project_name)
            except Exception:
                pass

        # ── One run per sample, with its own metric scores as feedback ────────
        for result in results:
            run_id = str(uuid.uuid4())
            try:
                client.create_run(
                    id=run_id,
                    name="rag-pipeline",
                    run_type="chain",
                    project_name=project_name,
                    inputs={"question": result["question"]},
                    outputs={
                        "answer": result.get("generated_answer", ""),
                        "contexts": result.get("contexts", []),
                        "expected_answer": result.get("expected_answer", ""),
                    },
                    start_time=now,
                    end_time=now,
                )
                for key, val in (result.get("metrics") or {}).items():
                    if isinstance(val, (int, float)):
                        # No project_id here — the API rejects it whenever a
                        # run_id is supplied ("project_id cannot be provided if
                        # run_id or trace_id is provided").
                        client.create_feedback(run_id=run_id, key=key, score=float(val))
            except Exception as e:
                logger.warning("Failed to log run to LangSmith: %s", e)

        # ── One summary run carrying the batch aggregate ──────────────────────
        summary_id: str | None = None
        try:
            summary_id = str(uuid.uuid4())
            client.create_run(
                id=summary_id,
                name="eval-batch-summary",
                run_type="chain",
                project_name=project_name,
                inputs={"num_samples": len(results)},
                outputs=aggregate,
                start_time=now,
                end_time=now,
            )
            for key, val in aggregate.items():
                if isinstance(val, (int, float)):
                    client.create_feedback(run_id=summary_id, key=f"batch_{key}", score=float(val))
        except Exception as e:
            logger.warning("Failed to log batch summary to LangSmith: %s", e)
            summary_id = None

        try:
            project = client.read_project(project_name=project_name)
            url = str(project.url) if project.url else "https://smith.langchain.com"
        except Exception:
            url = "https://smith.langchain.com"
        return url, summary_id

    except ImportError:
        logger.error("langsmith not installed — run: pip install langsmith")
        return None, None
    except Exception as e:
        logger.error("LangSmith logging failed: %s", e)
        return None, None


def backfill_summary_run(run_config) -> str | None:
    """Create the LangSmith batch-summary run for an already-evaluated config.

    Needed because feedback has to attach to a LangSmith run, and a run evaluated
    while LangSmith was unreachable (or before the key was configured) has none.
    Everything required is already in Postgres — the config, the aggregate
    metrics, and the per-case rows — so the run can be reconstructed after the
    fact rather than refusing the user's feedback.

    Returns the new run id, or None if LangSmith is unavailable.
    """
    if not LANGCHAIN_API_KEY or LANGCHAIN_API_KEY.startswith("your_"):
        return None

    try:
        from langsmith import Client
        client = Client(api_key=LANGCHAIN_API_KEY)

        project_name = f"rag-eval-run-{str(run_config.id)[:8]}"
        try:
            client.read_project(project_name=project_name)
        except Exception:
            try:
                client.create_project(project_name=project_name)
            except Exception:
                pass

        metrics = run_config.metrics or {}
        aggregate = {k: v for k, v in metrics.items() if not k.startswith("_")}
        meta = metrics.get("_meta") or {}
        results = list(run_config.results or [])

        run_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        client.create_run(
            id=run_id,
            name="eval-batch-summary",
            run_type="chain",
            project_name=project_name,
            inputs={
                "num_samples": meta.get("num_test_cases", len(results)),
                # Flag the reconstruction so a backfilled run is never mistaken
                # for one logged live during evaluation.
                "backfilled": True,
                "config": {
                    "chat_model": run_config.chat_model,
                    "embedding_model": run_config.embedding_model,
                    "chunk_size": run_config.chunk_size,
                    "chunk_overlap": run_config.chunk_overlap,
                    "search_type": run_config.search_type,
                    "top_k": run_config.top_k,
                    "temperature": run_config.temperature,
                    "reranker_model": run_config.reranker_model,
                },
            },
            outputs=aggregate or {"note": "no metrics recorded for this run"},
            start_time=now,
            end_time=now,
        )

        for key, val in aggregate.items():
            if isinstance(val, (int, float)):
                try:
                    client.create_feedback(run_id=run_id, key=f"batch_{key}", score=float(val))
                except Exception as e:
                    logger.warning("Backfill metric feedback failed for %s: %s", key, e)

        logger.info("Backfilled LangSmith summary run %s for run_config %s", run_id, run_config.id)
        return run_id

    except Exception as e:
        logger.warning("LangSmith summary backfill failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

async def call_rag_api(
    config,
    qa_pairs: list[dict],
    document_paths: list[str] | None,
    session_id: str | None = None,
    run_config_id: str | None = None,
) -> tuple[list[dict], dict, str | None]:
    """
    Full evaluation pipeline:
      1. Config-driven RAG pipeline -> answers + contexts
      2. DeepEval -> batch metrics
      3. LangSmith -> log runs + feedback, return experiment URL

    Returns: (results, batch_metrics, langsmith_url)
    """
    notes: list[str] = []
    generator_model, judge_model = _resolve_models(config, notes)

    # Count key failovers for this run only, so the notes reflect this run.
    reset_rotation_stats()

    # Stage 1 — RAG
    logger.info("Stage 1: Running RAG pipeline for %d QA pairs (model=%s)...",
                len(qa_pairs), generator_model)
    results = await run_rag_pipeline(config, qa_pairs, document_paths, generator_model, notes)

    embedding_model_name = _resolve_embedding_model(config, notes)

    # Stage 2 — score the whole batch with DeepEval
    from backend.services.deepeval_evaluator import evaluate_with_deepeval
    logger.info("Stage 2: Scoring batch with deepeval (judge=%s)...", judge_model)
    scored = await evaluate_with_deepeval(results, notes, judge_model=judge_model)

    for result, row in zip(results, scored["per_sample"]):
        result["metrics"] = row

    aggregate = scored["aggregate"]
    scored_counts = scored["scored_counts"]

    # Report partial coverage explicitly — an average over 3 of 10 samples must
    # never look like an average over 10.
    for name, count in scored_counts.items():
        if count and count < len(results):
            notes.append(
                f"{name}: only {count} of {len(results)} test cases scored "
                f"(the rest failed, usually a judge rate-limit) — the average covers those {count}."
            )

    # Surface key failover: it means an account hit its rate limit, which is
    # worth knowing even when the run completed successfully.
    rotations = rotation_stats()
    if rotations["rotations"]:
        notes.append(
            f"Groq key failover occurred {rotations['rotations']} time(s) during this run — "
            f"one account hit its rate limit and requests moved to another "
            f"(pool of {rotations['keys']} keys)."
        )

    # A run served through a gateway may have been answered by a different
    # provider than Groq, so the answers are not strictly comparable with runs
    # that were not. Record it rather than let it pass unseen.
    gateway = endpoint_stats()
    if gateway["gateway_configured"]:
        if gateway["gateway_active"]:
            notes.append(
                f"Answers were generated through the gateway at {gateway['gateway_configured']}, "
                f"which may have routed them to a provider other than Groq. "
                f"Judging was unaffected — it always runs direct against Groq."
            )
        else:
            notes.append(
                f"Gateway {gateway['gateway_configured']} was unreachable — "
                f"answers were generated directly against Groq."
            )

    # DeepEval reports the model that actually served the calls, which may be a
    # fallback if the requested judge was rate-limited.
    effective_judge = scored.get("judge_model") or judge_model

    batch_metrics = {
        **aggregate,
        "_meta": {
            "num_test_cases": len(results),
            "scored_counts": scored_counts,
            "generator_model": generator_model,
            "judge_model": effective_judge,
            "embedding_model": embedding_model_name,
            "framework": "deepeval",
            "key_pool": rotations,
            "notes": notes,
        },
    }
    if scored.get("cost_controls"):
        batch_metrics["_meta"]["cost_controls"] = scored["cost_controls"]
    if scored.get("model_fallbacks"):
        batch_metrics["_meta"]["model_fallbacks"] = scored["model_fallbacks"]
        served = ", ".join(f"{m} ({n})" for m, n in scored["model_fallbacks"].items())
        notes.append(
            f"The requested judge was rate-limited on every key; part of this scoring "
            f"was served by: {served}."
        )

    # Stage 3 — LangSmith
    logger.info("Stage 3: Logging to LangSmith...")
    langsmith_url = None
    langsmith_summary_run_id = None
    if session_id and run_config_id:
        langsmith_url, langsmith_summary_run_id = log_to_langsmith(
            session_id=str(session_id),
            run_config_id=str(run_config_id),
            results=results,
            qa_pairs=qa_pairs,
            aggregate=aggregate,
        )

    logger.info("Evaluation complete. LangSmith URL: %s", langsmith_url)
    if langsmith_summary_run_id:
        batch_metrics["_meta"]["langsmith_summary_run_id"] = langsmith_summary_run_id
    return results, batch_metrics, langsmith_url
