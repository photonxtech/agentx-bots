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
from rag.embeddings import embed
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
    query: str,
    hits: list[tuple[Document, float]],
    protect: set[int] | None = None,
) -> list[tuple[Document, float]]:
    """Re-score and re-sort `hits` by cross-encoder relevance to `query`.

    Each hit's hybrid-search score is replaced with a 0..1 relevance
    probability (sigmoid of the cross-encoder's raw logit), so downstream
    confidence-% display keeps working the same way. Hits scoring below
    config.RERANK_MIN_SCORE are dropped — without this, a fixed top_k always
    fills its quota with weak matches, diluting context_relevancy. At least
    one hit (the best-scoring) is always kept, even if it's below the floor,
    so a genuinely relevant-but-lower-scoring pool doesn't collapse to no
    context at all. Truncation to the final top_k still happens in
    RagService._filter_hits/diversity_select.

    `protect` — a set of `id(doc)` values (see RagService.retrieve's
    neighbor-expansion step) — are still scored and sorted normally, but
    exempt from the RERANK_MIN_SCORE floor: they were added for structural
    completeness (e.g. the neighboring chunk of a chunk that DID score
    well), not because they're expected to score well standing alone, so
    the floor must not be allowed to silently drop them again.
    """
    if not hits:
        return hits

    model = get_model()
    pairs = [(query, d.text) for d, _ in hits]
    logits = np.asarray(model.predict(pairs), dtype="float32")
    scores = 1.0 / (1.0 + np.exp(-logits))

    order = np.argsort(-scores)
    ranked = [(hits[i][0], float(scores[i])) for i in order]

    protect = protect or set()
    above_floor = [
        h for h in ranked if h[1] >= config.RERANK_MIN_SCORE or id(h[0]) in protect
    ]
    return above_floor if above_floor else ranked[:1]


def diversity_select(
    hits: list[tuple[Document, float]],
    k: int,
    protect: set[int] | None = None,
    lambda_mult: float = None,
) -> list[tuple[Document, float]]:
    """Pick up to `k` hits favoring high relevance + low redundancy (MMR —
    Maximal Marginal Relevance), except chunks in `protect` which are always
    kept regardless of how redundant they look.

    Reranking alone can hand back several near-duplicate restatements of the
    same idea as the top hits, crowding out genuinely different evidence
    sitting just below them. Classic greedy MMR: pick the best-scoring hit
    first, then repeatedly pick whichever remaining hit maximizes
    `lambda_mult * relevance - (1 - lambda_mult) * max_similarity_to_already_picked`.

    `protect` (neighbor-expanded structural context, see
    RagService.retrieve) is deliberately exempt: those chunks are SUPPOSED
    to be similar to their anchor (same section, adjacent chunk) — that's
    not redundancy, that's completeness, and diversity selection must never
    remove them for it. They're added first, and count against `k`; MMR
    fills whatever budget remains from the rest.
    """
    if not hits:
        return hits
    lambda_mult = config.MMR_LAMBDA if lambda_mult is None else lambda_mult
    protect = protect or set()

    protected = sorted((h for h in hits if id(h[0]) in protect), key=lambda h: -h[1])[:k]
    regular = [h for h in hits if id(h[0]) not in protect]
    budget = k - len(protected)
    selected_regular = _mmr(regular, budget, lambda_mult) if budget > 0 else []
    return protected + selected_regular


def _mmr(
    hits: list[tuple[Document, float]], k: int, lambda_mult: float
) -> list[tuple[Document, float]]:
    if not hits or k <= 0:
        return []
    if len(hits) <= k:
        return hits

    vecs = embed([d.text for d, _ in hits])
    scores = np.array([s for _, s in hits], dtype="float32")

    selected: list[int] = [int(np.argmax(scores))]
    remaining = [i for i in range(len(hits)) if i != selected[0]]

    while remaining and len(selected) < k:
        best_i, best_val = remaining[0], -np.inf
        for i in remaining:
            redundancy = max(float(vecs[i] @ vecs[j]) for j in selected)
            val = lambda_mult * scores[i] - (1.0 - lambda_mult) * redundancy
            if val > best_val:
                best_val, best_i = val, i
        selected.append(best_i)
        remaining.remove(best_i)

    return [hits[i] for i in selected]
