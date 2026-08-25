"""
Loads the ground_truth.csv dataset you uploaded to LangSmith Datasets &
Experiments, embeds every question once, and matches live user questions
against it by cosine similarity. Cached in memory at import time — refresh
by calling reload() if you edit the dataset in LangSmith without restarting
the server.
"""

import os
import logging
import numpy as np
from langsmith import Client

from rag_utils import _EMBEDDINGS  # reuse the already-loaded singleton

logger = logging.getLogger(__name__)

DATASET_NAME = os.getenv("GT_DATASET_NAME", "ground_truth")
SIMILARITY_THRESHOLD = float(os.getenv("GT_SIMILARITY_THRESHOLD", "0.85"))

_client = None
_questions = []       # list[str]
_ground_truths = []   # list[str], same order as _questions
_embeddings = None    # np.ndarray, shape (n, dim)


def _get_client():
    global _client
    if _client is None:
        _client = Client()
    return _client


def reload():
    """(Re)loads every example from the LangSmith dataset into memory."""
    global _questions, _ground_truths, _embeddings
    client = _get_client()

    questions, truths = [], []
    try:
        for example in client.list_examples(dataset_name=DATASET_NAME):
            q = (example.inputs or {}).get("question", "").strip()
            gt = (example.outputs or {}).get("ground_truth", "").strip()
            if q and gt:
                questions.append(q)
                truths.append(gt)
    except Exception:
        logger.warning("Failed to load ground-truth dataset '%s'", DATASET_NAME, exc_info=True)
        return

    if not questions:
        logger.warning("Ground-truth dataset '%s' loaded but had no usable rows", DATASET_NAME)
        _questions, _ground_truths, _embeddings = [], [], None
        return

    vecs = _EMBEDDINGS.embed_documents(questions)
    arr = np.array(vecs)
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)

    _questions, _ground_truths, _embeddings = questions, truths, arr
    logger.info("Loaded %d ground-truth Q&A pairs from dataset '%s'", len(questions), DATASET_NAME)


def find_ground_truth(question: str):
    """Returns (matched_question, ground_truth, similarity) for the closest
    dataset question above SIMILARITY_THRESHOLD, or None if no match / not loaded."""
    if _embeddings is None:
        return None

    q_vec = np.array(_EMBEDDINGS.embed_query(question))
    q_vec = q_vec / np.linalg.norm(q_vec)

    sims = _embeddings @ q_vec
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    if best_score < SIMILARITY_THRESHOLD:
        return None

    return _questions[best_idx], _ground_truths[best_idx], round(best_score, 3)


reload()  # populate cache at import time (server startup)
