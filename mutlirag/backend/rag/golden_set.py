"""Golden-set lookup — matches a live question against curated Q&A pairs.

context_recall and answer_correctness (see rag.evaluation) need a ground-truth
reference answer, which a real user's question never has. This module lets
RagService.evaluate_answer() score those two metrics anyway, but ONLY when the
live question is a close semantic match to one of the hand-curated questions
in the golden set — in which case that row's ground_truth is reused. Any
other live question still falls back to the four reference-free metrics.

Source of truth: the LangSmith dataset named LANGSMITH_DATASET_NAME (default
"jsonl") when LANGSMITH_TRACING/LANGSMITH_API_KEY are configured — this is
edited directly in the LangSmith UI, no redeploy needed to change the Q&A
pairs, only a server restart to pick up edits (see caching note below). Falls
back to the local scripts/eval_dataset_osw.json file if LangSmith isn't
configured, or if fetching from it fails for any reason (network, auth, the
dataset not existing) — a LangSmith outage must never break live chat.

Fetched ONCE per process and cached in memory (like the old local-file-only
version was) — this does NOT re-check LangSmith on every question, both for
latency and to avoid hammering the API. Restart the server to pick up edits
made in the LangSmith UI.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

import config
from rag import embeddings

logger = logging.getLogger(__name__)

_questions: list[str] | None = None
_ground_truths: list[str] | None = None
_vecs: np.ndarray | None = None


def _load_from_langsmith() -> tuple[list[str], list[str]] | None:
    """Fetch golden-set Q&A pairs from LangSmith, or None if unavailable.

    Never raises — any failure (LangSmith not configured, unreachable, the
    dataset missing) just tells the caller to fall back to the local file.
    """
    if os.getenv("LANGSMITH_TRACING", "").lower() != "true" or not os.getenv("LANGSMITH_API_KEY"):
        return None
    try:
        from langsmith import Client

        dataset_name = os.getenv("LANGSMITH_DATASET_NAME", "jsonl")
        client = Client()
        if not client.has_dataset(dataset_name=dataset_name):
            return None
        dataset = client.read_dataset(dataset_name=dataset_name)
        examples = [
            ex for ex in client.list_examples(dataset_id=dataset.id) if ex.inputs.get("question")
        ]
        if not examples:
            return None
        questions = [ex.inputs["question"] for ex in examples]
        ground_truths = [ex.outputs.get("ground_truth", "") for ex in examples]
        logger.info(
            "Golden set: loaded %d example(s) from LangSmith dataset '%s'", len(questions), dataset_name
        )
        return questions, ground_truths
    except Exception:
        logger.warning(
            "Golden set: failed to load from LangSmith, falling back to local file", exc_info=True
        )
        return None


def _load() -> None:
    global _questions, _ground_truths, _vecs
    if _vecs is not None:
        return

    from_langsmith = _load_from_langsmith()
    if from_langsmith is not None:
        _questions, _ground_truths = from_langsmith
        _vecs = embeddings.embed(_questions)
        return

    if not os.path.exists(config.GOLDEN_SET_PATH):
        _questions, _ground_truths = [], []
        _vecs = np.zeros((0, embeddings.embedding_dim()), dtype="float32")
        return
    with open(config.GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)
    _questions = [r["question"] for r in rows]
    _ground_truths = [r["ground_truth"] for r in rows]
    _vecs = embeddings.embed(_questions) if _questions else np.zeros(
        (0, embeddings.embedding_dim()), dtype="float32"
    )


def lookup(question: str) -> str | None:
    """Return the ground_truth of the closest curated question, or None.

    Only returns a match when cosine similarity clears
    config.GOLDEN_SET_MATCH_THRESHOLD — a loose paraphrase of a curated
    question still counts, but an unrelated question does not.
    """
    _load()
    if not _questions or not question or not question.strip():
        return None
    q_vec = embeddings.embed([question])[0]
    sims = _vecs @ q_vec
    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= config.GOLDEN_SET_MATCH_THRESHOLD:
        return _ground_truths[best_idx]
    return None
