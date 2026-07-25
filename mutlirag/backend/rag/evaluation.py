"""RAGAS-style, reference-free answer evaluation.

After the pipeline produces a grounded answer, we score it on three RAGAS
metrics WITHOUT any ground-truth labels — using only the question, the answer,
and the retrieved context chunks:

  * faithfulness      — fraction of the answer's claims that the context supports
                        (catches hallucination).
  * answer_relevancy  — how well the answer addresses the question, measured by
                        generating questions from the answer and comparing them
                        (semantically) to the original question.
  * context_precision — average precision of the retrieved chunks: are the ones
                        actually relevant to the question ranked near the top?

These are hand-written (no LangChain / no `ragas` package) so the mechanics stay
visible and reuse the tools the app already has: the Groq client (a small, fast
judge model) and the local sentence-transformers embeddings.

Every metric is defensive: any Groq/JSON failure yields ``None`` for that metric
rather than raising, so evaluation can never break a chat answer.
"""

from __future__ import annotations

import json
import re

import numpy as np

import config
from rag import embeddings
from rag.generator import _client


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _judge(system: str, user: str, max_tokens: int = 500) -> str:
    """One deterministic call to the cheap judge model. Returns raw text."""
    resp = _client().chat.completions.create(
        model=config.RAGAS_EVAL_MODEL,
        temperature=0.0,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_json(text: str):
    """Best-effort JSON extraction from a model reply (handles ```json fences)."""
    if not text:
        return None
    # Strip code fences if the model wrapped its answer.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fall back to the first [...] or {...} span in the text.
    for pattern in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                continue
    return None


def _round(x: float | None) -> float | None:
    return None if x is None else round(float(x), 3)


# --------------------------------------------------------------------------- #
# 1. Faithfulness
# --------------------------------------------------------------------------- #
_CLAIMS_SYS = (
    "You break an answer into a list of standalone, atomic factual claims. "
    "Each claim must be a single, self-contained statement that can be checked "
    "as true or false. Ignore opinions, questions, filler, and citations. "
    'Return ONLY a JSON array of strings, e.g. ["Claim one.", "Claim two."]. '
    "If the answer contains no factual claims, return []."
)

_VERIFY_SYS = (
    "You are given a CONTEXT and a list of CLAIMS. For each claim, decide whether "
    "it can be directly inferred from the context. Return ONLY a JSON array of "
    "0/1 integers, one per claim in order: 1 if the context supports the claim, "
    "0 if it does not or is unclear. Return exactly as many numbers as claims."
)


def faithfulness(answer: str, context: str) -> float | None:
    """Fraction of the answer's atomic claims that are supported by the context.

    Returns 1.0 when the answer makes no factual claims (e.g. "I don't know"),
    and None if the judge/JSON step fails.
    """
    try:
        claims = _parse_json(_judge(_CLAIMS_SYS, f"Answer:\n{answer}"))
        if not isinstance(claims, list):
            return None
        claims = [str(c) for c in claims if str(c).strip()]
        if not claims:
            return 1.0  # nothing to hallucinate

        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
        verdicts = _parse_json(
            _judge(
                _VERIFY_SYS,
                f"CONTEXT:\n{context}\n\nCLAIMS:\n{numbered}",
            )
        )
        if not isinstance(verdicts, list) or not verdicts:
            return None
        supported = sum(1 for v in verdicts if int(v) == 1)
        return supported / len(claims)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 2. Answer relevancy
# --------------------------------------------------------------------------- #
_GENQ_SYS = (
    "Given an ANSWER, generate {n} diverse questions that the answer would be a "
    "direct and complete response to. Do not use outside knowledge. "
    "Return ONLY a JSON array of {n} question strings."
)


def answer_relevancy(question: str, answer: str, n: int | None = None) -> float | None:
    """How well the answer addresses the question.

    Generate `n` questions the answer would answer, embed them and the original
    question with the local model, and average the cosine similarities. Because
    embeddings are L2-normalized, a dot product IS the cosine similarity.
    """
    n = n or config.RAGAS_RELEVANCY_N
    try:
        gen = _parse_json(_judge(_GENQ_SYS.format(n=n), f"ANSWER:\n{answer}"))
        if not isinstance(gen, list):
            return None
        gen_qs = [str(q) for q in gen if str(q).strip()]
        if not gen_qs:
            return None

        vecs = embeddings.embed([question] + gen_qs)  # (1 + m, dim), normalized
        q_vec, gen_vecs = vecs[0], vecs[1:]
        sims = gen_vecs @ q_vec  # cosine similarity per generated question
        # Clamp to [0, 1]: negatives mean "unrelated", not "anti-relevant".
        return float(np.clip(sims, 0.0, 1.0).mean())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 3. Context precision
# --------------------------------------------------------------------------- #
_CTXREL_SYS = (
    "You are given a QUESTION and a numbered list of retrieved CONTEXT chunks. "
    "For each chunk, decide whether it is useful for answering the question. "
    "Return ONLY a JSON array of 0/1 integers, one per chunk in order: "
    "1 if the chunk is relevant/useful, 0 if not. Return exactly as many "
    "numbers as chunks."
)


def context_precision(question: str, contexts: list[str]) -> float | None:
    """Average precision @k of the retrieved chunks (order matters).

    Rewards ranking relevant chunks near the top:
        AP = sum_k (precision@k * relevant_k) / (total relevant)
    Returns None on judge/JSON failure; 0.0 if nothing relevant was retrieved.
    """
    if not contexts:
        return None
    try:
        numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
        verdicts = _parse_json(
            _judge(_CTXREL_SYS, f"QUESTION: {question}\n\nCONTEXT CHUNKS:\n{numbered}")
        )
        if not isinstance(verdicts, list) or not verdicts:
            return None
        rel = [1 if int(v) == 1 else 0 for v in verdicts][: len(contexts)]
        total_relevant = sum(rel)
        if total_relevant == 0:
            return 0.0

        hits = 0
        precision_sum = 0.0
        for k, r in enumerate(rel, 1):
            if r:
                hits += 1
                precision_sum += hits / k
        return precision_sum / total_relevant
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def evaluate(question: str, answer: str, contexts: list[str]) -> dict:
    """Score one answer on the three reference-free RAGAS metrics.

    `contexts` are the full text of the retrieved chunks (best relevance first).
    Returns a dict of floats in [0, 1] (rounded), with None for any metric whose
    judge call failed. Never raises.
    """
    context_blob = "\n\n".join(contexts)
    return {
        "faithfulness": _round(faithfulness(answer, context_blob)),
        "answer_relevancy": _round(answer_relevancy(question, answer)),
        "context_precision": _round(context_precision(question, contexts)),
    }
