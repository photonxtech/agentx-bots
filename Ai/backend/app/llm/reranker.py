import math
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import settings


@lru_cache
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rerank(question: str, candidates: list[dict], top_k: int) -> list[dict]:
    # ANN/BM25 retrieval is optimized for speed over a large candidate pool, not precision.
    # A cross-encoder scores the (question, chunk) pair jointly instead of via separate
    # embeddings, which is slower but substantially more accurate — so we retrieve broadly,
    # then rerank the smaller candidate set for the final ordering.
    return rerank_best_of([question], candidates, top_k)


def rerank_best_of(queries: list[str], candidates: list[dict], top_k: int) -> list[dict]:
    # Some questions score far better against the bare question text, others against a
    # version with the site name folded in (a short/vague question like "how to contact"
    # needs that framing; a question that already names the exact thing being asked about,
    # like "who is design lead", can score dramatically WORSE with it — the extra tokens
    # dilute the cross-encoder's focus). There's no single framing that wins for every
    # question, so score each candidate against every query variant and keep its best.
    if not candidates:
        return []
    model = get_reranker()
    best_relevance = [0.0] * len(candidates)
    best_raw = [float("-inf")] * len(candidates)
    for query in queries:
        pairs = [(query, c["text"]) for c in candidates]
        raw_scores = model.predict(pairs)
        for i, raw_score in enumerate(raw_scores):
            relevance = _sigmoid(float(raw_score))
            if relevance > best_relevance[i]:
                best_relevance[i] = relevance
                best_raw[i] = float(raw_score)

    reranked = [
        {**c, "rerank_score": best_raw[i], "similarity": best_relevance[i]}
        for i, c in enumerate(candidates)
    ]
    reranked.sort(key=lambda c: c["similarity"], reverse=True)
    return reranked[:top_k]
