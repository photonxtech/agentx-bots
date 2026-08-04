"""Cross-encoder reranking: a second-stage relevance re-scoring of retrieved chunks.

Hybrid search (vectorstore.search) blends two independent signals (embedding
cosine + BM25) with a fixed weight and can't reason about the query and a
chunk together. A cross-encoder scores each (query, chunk) pair jointly, which
is far more accurate at ranking but too slow to run over a whole index — so it
only re-scores the small candidate pool hybrid search already retrieved.
"""

from __future__ import annotations

import numpy as np

import config
from rag.ingestion import Document

_model = None


def get_model():
    """Load the cross-encoder once and cache it."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(config.RERANK_MODEL)
    return _model


def rerank(
    query: str, hits: list[tuple[Document, float]]
) -> list[tuple[Document, float]]:
    """Re-score and re-sort `hits` by cross-encoder relevance to `query`.

    Each hit's hybrid-search score is replaced with a 0..1 relevance
    probability (sigmoid of the cross-encoder's raw logit), so downstream
    confidence-% display keeps working the same way. List length is
    unchanged — truncation to the final top_k still happens in
    RagService._filter_hits.
    """
    if not hits:
        return hits

    model = get_model()
    pairs = [(query, d.text) for d, _ in hits]
    logits = np.asarray(model.predict(pairs), dtype="float32")
    scores = 1.0 / (1.0 + np.exp(-logits))

    order = np.argsort(-scores)
    return [(hits[i][0], float(scores[i])) for i in order]
