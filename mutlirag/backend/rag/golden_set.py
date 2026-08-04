"""Golden-set lookup — matches a live question against the curated
scripts/eval_dataset_osw.json Q&A pairs.

context_recall and answer_correctness (see rag.evaluation) need a ground-truth
reference answer, which a real user's question never has. This module lets
RagService.evaluate_answer() score those two metrics anyway, but ONLY when the
live question is a close semantic match to one of the hand-curated questions
in the golden set — in which case that row's ground_truth is reused. Any
other live question still falls back to the four reference-free metrics.
"""

from __future__ import annotations

import json
import os

import numpy as np

import config
from rag import embeddings

_questions: list[str] | None = None
_ground_truths: list[str] | None = None
_vecs: np.ndarray | None = None


def _load() -> None:
    global _questions, _ground_truths, _vecs
    if _vecs is not None:
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
