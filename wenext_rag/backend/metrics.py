"""Local RAGAS-style metrics — no external APIs, uses BGE embeddings from rag.py."""

from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Optional
from backend.rag import embeddings

# ---------------------------------------------------------------------------
# Thresholds (tune on eval_dataset.json)
# ---------------------------------------------------------------------------
FAITHFULNESS_THRESHOLD = 0.72
CONTEXT_RELEVANCE_THRESHOLD = 0.55
CLAIM_SUPPORT_THRESHOLD = 0.68

REFUSAL_PHRASE = "I am sorry, but I can only answer questions related to the provided documents."

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
    "because", "until", "while", "although", "though", "after", "before",
    "that", "this", "these", "those", "it", "its", "they", "them", "their",
    "what", "which", "who", "whom", "you", "your", "we", "our", "i", "me",
    "my", "he", "she", "his", "her", "about", "also", "any", "both", "each",
})


@dataclass
class RagMetrics:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    context_entity_recall: float
    answer_correctness: float

    def to_dict(self) -> dict[str, float]:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevancy": round(self.answer_relevancy, 3),
            "context_precision": round(self.context_precision, 3),
            "context_recall": round(self.context_recall, 3),
            "context_entity_recall": round(self.context_entity_recall, 3),
            "answer_correctness": round(self.answer_correctness, 3),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed(text: str) -> list[float]:
    return embeddings.embed_query(text.strip())


def _embed_batch(texts: list[str]) -> list[list[float]]:
    cleaned = [t.strip() for t in texts if t.strip()]
    if not cleaned:
        return []
    return embeddings.embed_documents(cleaned)


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\[\d+\]", "", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]


def _split_claims(text: str) -> list[str]:
    """Sentence *and* clause-level split, for reference/ground-truth text.
    Many ground truths here are a single sentence listing several facts
    ('X, Y, and Z.') - splitting only on sentence boundaries collapses that
    into one claim, so context_recall becomes a single yes/no check instead
    of measuring partial coverage. Splits on every comma (not just before
    'and'/'but') so enumerated lists yield one claim per item; the length
    filter is intentionally low (>2 chars) so short list items like 'XLS'
    or 'CSV' aren't dropped."""
    text = re.sub(r"\[\d+\]", "", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+|;\s+|,\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 2]


def _normalize_tokens(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t and t not in _STOPWORDS]


def _token_f1(pred: str, ref: str) -> float:
    pred_tokens = set(_normalize_tokens(pred))
    ref_tokens = set(_normalize_tokens(ref))
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _extract_entities(text: str) -> list[str]:
    entities: list[str] = []
    patterns = [
        r"\b[A-Z]{2,}\b",
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        r"\bTier\s+\d+\b",
        r"\b(?:XLSX|XLS|CSV|PDF|GST|WABA|API|FAQ)\b",
        r"\b\d{1,3}(?:,\d{3})*\b",
    ]
    for pattern in patterns:
        entities.extend(re.findall(pattern, text, re.IGNORECASE))

    seen: set[str] = set()
    unique: list[str] = []
    for entity in entities:
        key = entity.lower()
        if key not in seen and len(key) > 1:
            seen.add(key)
            unique.append(entity)
    return unique


def _entity_in_text(entity: str, text: str) -> bool:
    return entity.lower() in text.lower()


def _max_sim_to_contexts(text: str, context_vecs: list[list[float]], contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    text_vec = _embed(text)
    return max(_cosine(text_vec, cv) for cv in context_vecs)


def _reference_text(question: str, ground_truth: Optional[str]) -> str:
    return ground_truth if ground_truth else question


def _is_refusal(answer: str) -> bool:
    return REFUSAL_PHRASE.lower() in answer.lower()


# ---------------------------------------------------------------------------
# Metric 1 — Faithfulness
# ---------------------------------------------------------------------------
def score_faithfulness(answer: str, contexts: list[str]) -> float:
    """Fraction of answer claims supported by retrieved context."""
    if _is_refusal(answer):
        return 1.0
    if not contexts:
        return 0.0

    claims = _split_sentences(answer)
    if not claims:
        return 1.0

    context_vecs = _embed_batch(contexts)
    supported = sum(
        1 for claim in claims
        if _max_sim_to_contexts(claim, context_vecs, contexts) >= FAITHFULNESS_THRESHOLD
    )
    return supported / len(claims)


# ---------------------------------------------------------------------------
# Metric 2 — Answer Relevancy
# ---------------------------------------------------------------------------
def score_answer_relevancy(question: str, answer: str) -> float:
    """Semantic similarity between question and answer."""
    if _is_refusal(answer):
        return 0.0
    if not question.strip() or not answer.strip():
        return 0.0

    sentences = _split_sentences(answer) or [answer]
    q_vec = _embed(question)
    sims = [_cosine(q_vec, _embed(s)) for s in sentences]
    return sum(sims) / len(sims)


# ---------------------------------------------------------------------------
# Metric 3 — Context Precision
# ---------------------------------------------------------------------------
def score_context_precision(
    question: str,
    contexts: list[str],
    ground_truth: Optional[str] = None,
) -> float:
    """Average precision: relevant chunks should appear at higher ranks.

    Relevance is judged against the ground truth when one is available.
    Falling back to question-similarity as well (the old behavior) trivially
    inflates this score, since the retriever already selects chunks by
    similarity to the question - that's not independent evidence of quality.
    The question fallback is only used when there's no ground truth to
    compare against.
    """
    if not contexts:
        return 0.0

    reference_vec = _embed(ground_truth) if ground_truth else _embed(question)

    relevance_flags: list[bool] = []
    for chunk in contexts:
        chunk_vec = _embed(chunk)
        score = _cosine(chunk_vec, reference_vec)
        relevance_flags.append(score >= CONTEXT_RELEVANCE_THRESHOLD)

    total_relevant = sum(relevance_flags)
    if total_relevant == 0:
        return 0.0

    relevant_seen = 0
    precision_sum = 0.0
    for idx, is_relevant in enumerate(relevance_flags):
        if is_relevant:
            relevant_seen += 1
            precision_sum += relevant_seen / (idx + 1)

    return precision_sum / total_relevant


# ---------------------------------------------------------------------------
# Metric 4 — Context Recall
# ---------------------------------------------------------------------------
def score_context_recall(
    contexts: list[str],
    ground_truth: Optional[str] = None,
    question: str = "",
) -> float:
    """Fraction of reference claims found in retrieved contexts."""
    reference = ground_truth if ground_truth else question
    if not reference.strip() or not contexts:
        return 0.0

    claims = _split_claims(reference)
    if not claims:
        claims = [reference]

    context_vecs = _embed_batch(contexts)
    supported = sum(
        1 for claim in claims
        if _max_sim_to_contexts(claim, context_vecs, contexts) >= CLAIM_SUPPORT_THRESHOLD
    )
    return supported / len(claims)


# ---------------------------------------------------------------------------
# Metric 5 — Context Entity Recall
# ---------------------------------------------------------------------------
def score_context_entity_recall(
    contexts: list[str],
    ground_truth: Optional[str] = None,
    question: str = "",
) -> float:
    """Fraction of entities in reference found in retrieved contexts."""
    reference = ground_truth if ground_truth else question
    if not reference.strip() or not contexts:
        return 0.0

    entities = _extract_entities(reference)
    if not entities:
        return 1.0

    combined = " ".join(contexts)
    found = sum(1 for entity in entities if _entity_in_text(entity, combined))
    return found / len(entities)


# ---------------------------------------------------------------------------
# Metric 6 — Answer Correctness
# ---------------------------------------------------------------------------
def score_answer_correctness(
    answer: str,
    ground_truth: Optional[str] = None,
    contexts: Optional[list[str]] = None,
) -> float:
    """0.75 * semantic similarity + 0.25 * token F1 vs reference."""
    if _is_refusal(answer):
        return 0.0

    if ground_truth:
        reference = ground_truth
    elif contexts:
        reference = " ".join(contexts[:3])
    else:
        return 0.0

    if not answer.strip() or not reference.strip():
        return 0.0

    semantic = _cosine(_embed(answer), _embed(reference))
    f1 = _token_f1(answer, reference)
    return 0.75 * semantic + 0.25 * f1


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_all_metrics(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: Optional[str] = None,
) -> RagMetrics:
    """Compute all six RAGAS-style metrics for one Q/A pair."""
    return RagMetrics(
        faithfulness=score_faithfulness(answer, contexts),
        answer_relevancy=score_answer_relevancy(question, answer),
        context_precision=score_context_precision(question, contexts, ground_truth),
        context_recall=score_context_recall(contexts, ground_truth, question),
        context_entity_recall=score_context_entity_recall(contexts, ground_truth, question),
        answer_correctness=score_answer_correctness(answer, ground_truth, contexts),
    )