"""RAGAS-style answer evaluation — reference-free metrics live, ground-truth
metrics offline.

Four metrics run live, after every chat answer, WITHOUT any ground-truth
labels — using only the question, the answer, and the retrieved context
chunks (see `evaluate()`):

  * faithfulness      — fraction of the answer's claims that the context supports
                        (catches hallucination).
  * answer_relevancy  — how well the answer addresses the question, measured by
                        generating questions from the answer and comparing them
                        (semantically) to the original question.
  * context_precision — average precision of the retrieved chunks: are the ones
                        actually relevant to the question ranked near the top?
  * context_relevancy — density of the retrieved context: of all the sentences
                        retrieved, what fraction are actually relevant to the
                        question (vs padding/noise)? Unlike context_precision,
                        this ignores chunk ranking and just measures signal-to-noise.

Two more metrics need a ground-truth (correct) answer, so they only make sense
for OFFLINE regression testing against a curated Q&A dataset (see
`evaluate_with_ground_truth()` and scripts/run_ragas_eval.py) — never live chat,
since a real user's question has no known-correct answer to compare against:

  * context_recall     — did retrieval pull back everything needed to produce
                        the ground-truth answer?
  * answer_correctness — does the generated answer match the ground-truth
                        answer, factually and semantically?

These are hand-written (no LangChain / no `ragas` package) so the mechanics stay
visible and reuse the tools the app already has: the Groq client (a small, fast
judge model) and the local sentence-transformers embeddings.

Every metric is defensive: any Groq/JSON failure yields ``None`` for that metric
rather than raising, so evaluation can never break a chat answer. Failures are
logged (not silent) so a metric that goes blank in the UI is diagnosable. Within
each of `evaluate()` and `evaluate_with_ground_truth()`, metrics are independent
of each other and run concurrently, so the extra latency is roughly one judge
call per stage, not one per metric.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from groq import Groq

import config
from rag import embeddings

logger = logging.getLogger(__name__)


def _client() -> Groq:
    """Groq client for judge calls, on its own API key/rate-limit bucket.

    Falls back to GROQ_API_KEY (the generation key, from rag.generator) if
    GROQ_API_KEY_RAGAS isn't set, so this works out of the box with a single
    key. Set GROQ_API_KEY_RAGAS in .env to give judge calls (faithfulness x2,
    relevancy, context precision, context relevancy — up to 5 per chat turn,
    more during offline eval) a separate rate limit from answer generation.
    """
    key = os.getenv("GROQ_API_KEY_RAGAS") or os.getenv("GROQ_API_KEY")
    if not key or key.startswith("gsk_your"):
        raise RuntimeError(
            "No Groq API key set for RAGAS evaluation. Add GROQ_API_KEY_RAGAS "
            "(or GROQ_API_KEY) to your .env file."
        )
    return Groq(api_key=key)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
# Parse Groq's "Please try again in 7m48.72s" / "3.5s" / "120ms" hint.
_RETRY_AFTER_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)\b", re.IGNORECASE)


def _retry_after_seconds(message: str) -> float | None:
    m = _RETRY_AFTER_RE.search(message or "")
    if not m:
        return None
    minutes = int(m.group(1)) if m.group(1) else 0
    value = float(m.group(2))
    seconds = value / 1000.0 if m.group(3).lower() == "ms" else value
    return minutes * 60 + seconds


def _judge_json(system: str, user: str, max_tokens: int = 500) -> dict | None:
    """One deterministic call to the cheap judge model, in JSON-object mode.

    Retries transient per-minute rate limits, honoring the API's suggested
    wait (bounded by RAGAS_MAX_RETRY_WAIT so a long/per-day quota wait gives up
    instead of stalling the whole eval run). Also retries when the judge's
    JSON reply got cut off by max_tokens before it could close (a longer
    answer/more chunks needs more room than the default budget) — each such
    retry doubles the budget, capped at RAGAS_MAX_TOKENS_CAP. Returns the
    parsed dict, or None (logged) if the call, its retries, or the JSON parse
    ultimately failed.
    """
    text = None
    current_max_tokens = max_tokens
    for attempt in range(config.RAGAS_MAX_RETRIES + 1):
        try:
            resp = _client().chat.completions.create(
                model=config.RAGAS_EVAL_MODEL,
                temperature=0.0,
                max_tokens=current_max_tokens,
                timeout=config.RAGAS_TIMEOUT_S,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            break
        except Exception as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "rate_limit" in msg.lower()
            is_token_limit = "json_validate_failed" in msg or "max completion tokens" in msg.lower()
            wait = _retry_after_seconds(msg)
            if (
                is_rate_limit
                and attempt < config.RAGAS_MAX_RETRIES
                and wait is not None
                and wait <= config.RAGAS_MAX_RETRY_WAIT
            ):
                time.sleep(wait + 0.5)
                continue
            if (
                is_token_limit
                and attempt < config.RAGAS_MAX_RETRIES
                and current_max_tokens < config.RAGAS_MAX_TOKENS_CAP
            ):
                current_max_tokens = min(current_max_tokens * 2, config.RAGAS_MAX_TOKENS_CAP)
                continue
            logger.warning("RAGAS judge call failed", exc_info=True)
            return None

    parsed = _parse_json(text)
    if not isinstance(parsed, dict):
        logger.warning("RAGAS judge returned non-JSON-object reply: %.200r", text)
        return None
    return parsed


def _parse_json(text: str):
    """Best-effort JSON extraction from a model reply (handles ```json fences).

    JSON-object mode makes this a formality in the common case, but small judge
    models occasionally still wrap the object in prose or a code fence.
    """
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def _round(x: float | None) -> float | None:
    return None if x is None else round(float(x), 3)


def _int_list(values: list) -> list[int]:
    """Coerce judge verdicts to 0/1 ints. An unparseable entry counts as 0
    (not supported) rather than aborting the whole metric."""
    out = []
    for v in values:
        try:
            out.append(1 if int(v) == 1 else 0)
        except (TypeError, ValueError):
            out.append(0)
    return out


# --------------------------------------------------------------------------- #
# 1. Faithfulness
# --------------------------------------------------------------------------- #
_CLAIMS_SYS = (
    "You break an answer into a list of standalone, atomic factual claims. "
    "Each claim must be a single, self-contained statement that can be checked "
    "as true or false. Ignore opinions, questions, filler, and citations. "
    'Respond with a JSON object of the form {"claims": ["Claim one.", "Claim two."]}. '
    'If the answer contains no factual claims, respond with {"claims": []}.'
)

_VERIFY_SYS = (
    "You are given a CONTEXT and a numbered list of CLAIMS. For each claim, decide "
    "whether it can be directly inferred from the context. Respond with a JSON "
    'object of the form {"verdicts": [1, 0, ...]}: exactly one integer per claim, '
    "in the same order — 1 if the context supports the claim, 0 if it does not "
    "or is unclear."
)


def _claim_coverage(text: str, context: str) -> float | None:
    """Fraction of `text`'s atomic claims that are supported by `context`.

    Shared by faithfulness (text=generated answer) and context_recall
    (text=ground-truth answer) — both are "break into claims, then check each
    against the context" underneath, just checking a different text.

    Returns 1.0 when `text` makes no factual claims (e.g. "I don't know"), and
    None (logged) if the judge/JSON step fails or the two judge calls disagree
    on how many claims there are.
    """
    try:
        claims_resp = _judge_json(_CLAIMS_SYS, f"Answer:\n{text}", max_tokens=1000)
        if claims_resp is None:
            return None
        claims = claims_resp.get("claims")
        if not isinstance(claims, list):
            logger.warning("RAGAS claim extraction: 'claims' missing/not a list: %r", claims_resp)
            return None
        claims = [str(c) for c in claims if str(c).strip()]
        if not claims:
            return 1.0  # nothing to hallucinate / nothing left uncovered

        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
        verify_resp = _judge_json(_VERIFY_SYS, f"CONTEXT:\n{context}\n\nCLAIMS:\n{numbered}")
        if verify_resp is None:
            return None
        verdicts = verify_resp.get("verdicts")
        if not isinstance(verdicts, list) or len(verdicts) != len(claims):
            logger.warning(
                "RAGAS claim verification: verdict count (%s) != claim count (%d) — discarding",
                len(verdicts) if isinstance(verdicts, list) else "n/a", len(claims),
            )
            return None
        supported = sum(_int_list(verdicts))
        return supported / len(claims)
    except Exception:
        logger.exception("RAGAS claim coverage: unexpected error")
        return None


def faithfulness(answer: str, context: str) -> float | None:
    """Fraction of the answer's atomic claims that are supported by the context.

    Reference-free: checks the *generated* answer against the retrieved
    context, so it can run live on every chat turn (see `evaluate()`).
    """
    return _claim_coverage(answer, context)


# --------------------------------------------------------------------------- #
# 2. Answer relevancy
# --------------------------------------------------------------------------- #
_GENQ_SYS = (
    "Given an ANSWER, generate {n} diverse questions that the answer would be a "
    "direct and complete response to. Do not use outside knowledge. "
    'Respond with a JSON object of the form {{"questions": [... {n} strings ...]}}.'
)


def answer_relevancy(question: str, answer: str, n: int | None = None) -> float | None:
    """How well the answer addresses the question.

    Generate `n` questions the answer would answer, embed them and the original
    question with the local model, and average the cosine similarities. Because
    embeddings are L2-normalized, a dot product IS the cosine similarity.
    """
    n = n or config.RAGAS_RELEVANCY_N
    try:
        resp = _judge_json(_GENQ_SYS.format(n=n), f"ANSWER:\n{answer}")
        if resp is None:
            return None
        gen = resp.get("questions")
        if not isinstance(gen, list):
            logger.warning("RAGAS answer_relevancy: 'questions' missing/not a list: %r", resp)
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
        logger.exception("RAGAS answer_relevancy: unexpected error")
        return None


# --------------------------------------------------------------------------- #
# 3. Context precision
# --------------------------------------------------------------------------- #
_CTXREL_SYS = (
    "You are given a QUESTION and a numbered list of retrieved CONTEXT chunks. "
    "For each chunk, decide whether it is useful for answering the question. "
    'Respond with a JSON object of the form {"verdicts": [1, 0, ...]}: exactly '
    "one integer per chunk, in the same order — 1 if the chunk is relevant/useful, "
    "0 if not."
)


def context_precision(question: str, contexts: list[str]) -> float | None:
    """Average precision @k of the retrieved chunks (order matters).

    Rewards ranking relevant chunks near the top:
        AP = sum_k (precision@k * relevant_k) / (total relevant)
    Returns None (logged) if the judge/JSON step fails or its verdict count
    doesn't match the number of chunks; 0.0 if nothing relevant was retrieved.
    """
    if not contexts:
        return None
    try:
        numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
        resp = _judge_json(_CTXREL_SYS, f"QUESTION: {question}\n\nCONTEXT CHUNKS:\n{numbered}")
        if resp is None:
            return None
        verdicts = resp.get("verdicts")
        if not isinstance(verdicts, list) or len(verdicts) != len(contexts):
            logger.warning(
                "RAGAS context_precision: verdict count (%s) != chunk count (%d) — discarding",
                len(verdicts) if isinstance(verdicts, list) else "n/a", len(contexts),
            )
            return None
        rel = _int_list(verdicts)
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
        logger.exception("RAGAS context_precision: unexpected error")
        return None


# --------------------------------------------------------------------------- #
# 4. Context relevancy
# --------------------------------------------------------------------------- #
_CTXREV_SYS = (
    "You are given a QUESTION and a CONTEXT passage made up of several chunks. "
    "Extract ONLY the sentences from the context that are directly relevant to "
    "answering the question, copied verbatim (do not paraphrase). "
    'Respond with a JSON object of the form {"relevant_sentences": ["sentence one.", ...]}. '
    'If no sentences are relevant, respond with {"relevant_sentences": []}.'
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _sentence_count(text: str) -> int:
    return len([s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()])


def context_relevancy(question: str, contexts: list[str]) -> float | None:
    """Fraction of the retrieved context that is actually relevant to the question.

    Unlike context_precision (which rewards ranking relevant chunks near the
    top), this measures density: of every sentence across the retrieved chunks,
    how many does the judge consider relevant vs noise/padding? The judge
    extracts relevant sentences verbatim; score = (extracted count) / (total
    sentence count). Returns None (logged) if the judge/JSON step fails.
    """
    if not contexts:
        return None
    context_blob = "\n\n".join(contexts)
    total = _sentence_count(context_blob)
    if total == 0:
        return None
    try:
        resp = _judge_json(
            _CTXREV_SYS, f"QUESTION: {question}\n\nCONTEXT:\n{context_blob}", max_tokens=800
        )
        if resp is None:
            return None
        sentences = resp.get("relevant_sentences")
        if not isinstance(sentences, list):
            logger.warning(
                "RAGAS context_relevancy: 'relevant_sentences' missing/not a list: %r", resp
            )
            return None
        relevant = len([s for s in sentences if str(s).strip()])
        return min(1.0, relevant / total)
    except Exception:
        logger.exception("RAGAS context_relevancy: unexpected error")
        return None


# --------------------------------------------------------------------------- #
# 5. Context recall (needs ground truth — offline eval only)
# --------------------------------------------------------------------------- #
def context_recall(ground_truth: str, contexts: list[str]) -> float | None:
    """Fraction of the ground-truth answer's claims that the retrieved context supports.

    Same claim-then-verify mechanics as faithfulness, but checks the *reference*
    (correct) answer against the context instead of the generated one — this
    measures whether retrieval pulled back everything needed to answer
    correctly, independent of what the generator actually said. Needs a
    ground-truth answer, so unlike the four metrics above this only makes
    sense for offline regression testing against a curated Q&A set, not live
    chat (see `evaluate_with_ground_truth()`).
    """
    if not ground_truth or not ground_truth.strip() or not contexts:
        return None
    context_blob = "\n\n".join(contexts)
    return _claim_coverage(ground_truth, context_blob)


# --------------------------------------------------------------------------- #
# 6. Answer correctness (needs ground truth — offline eval only)
# --------------------------------------------------------------------------- #
_CORRECTNESS_SYS = (
    "You compare a candidate ANSWER to a GROUND TRUTH reference answer for the "
    "same question. Judge them at the level of individual factual statements. "
    'Respond with a JSON object of the form {"tp": <int>, "fp": <int>, "fn": <int>}:\n'
    "tp = number of statements in the ANSWER that are also supported by the GROUND TRUTH.\n"
    "fp = number of statements in the ANSWER that are NOT supported by the GROUND TRUTH "
    "(wrong, invented, or contradicting it).\n"
    "fn = number of statements in the GROUND TRUTH that are missing from the ANSWER.\n"
    "If both texts are empty or contain no checkable statements, respond with "
    '{"tp": 0, "fp": 0, "fn": 0}.'
)


def answer_correctness(answer: str, ground_truth: str) -> float | None:
    """How well the answer matches a ground-truth reference answer.

    Combines a factual-overlap F1 (TP/FP/FN statement classification by the
    judge LLM) with embedding cosine similarity between the two texts, weighted
    0.75/0.25 — RAGAS's default weighting, favoring factual overlap over loose
    semantic similarity. Needs a ground-truth answer, so this is for offline
    regression testing only (see `evaluate_with_ground_truth()`), never live
    chat. Returns None (logged) on judge/JSON failure.
    """
    if not answer or not answer.strip() or not ground_truth or not ground_truth.strip():
        return None
    try:
        resp = _judge_json(_CORRECTNESS_SYS, f"GROUND TRUTH:\n{ground_truth}\n\nANSWER:\n{answer}")
        if resp is None:
            return None
        try:
            tp, fp, fn = int(resp["tp"]), int(resp["fp"]), int(resp["fn"])
        except (KeyError, TypeError, ValueError):
            logger.warning("RAGAS answer_correctness: bad tp/fp/fn in judge reply: %r", resp)
            return None
        f1 = 1.0 if tp + fp + fn == 0 else tp / (tp + 0.5 * (fp + fn))

        vecs = embeddings.embed([answer, ground_truth])
        semantic_sim = float(np.clip(vecs[0] @ vecs[1], 0.0, 1.0))

        return 0.75 * f1 + 0.25 * semantic_sim
    except Exception:
        logger.exception("RAGAS answer_correctness: unexpected error")
        return None


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def evaluate(question: str, answer: str, contexts: list[str]) -> dict:
    """Score one answer on the four reference-free RAGAS metrics.

    `contexts` are the full text of the retrieved chunks (best relevance first).
    The four metrics don't depend on each other, so they run concurrently —
    wall-clock cost is roughly one judge call, not five sequential ones.
    Returns a dict of floats in [0, 1] (rounded), with None for any metric whose
    judge call failed. Never raises.
    """
    context_blob = "\n\n".join(contexts)
    with ThreadPoolExecutor(max_workers=4) as pool:
        faith_future = pool.submit(faithfulness, answer, context_blob)
        rel_future = pool.submit(answer_relevancy, question, answer)
        prec_future = pool.submit(context_precision, question, contexts)
        ctx_rel_future = pool.submit(context_relevancy, question, contexts)
        faith = faith_future.result()
        rel = rel_future.result()
        prec = prec_future.result()
        ctx_rel = ctx_rel_future.result()
    return {
        "faithfulness": _round(faith),
        "answer_relevancy": _round(rel),
        "context_precision": _round(prec),
        "context_relevancy": _round(ctx_rel),
    }


def evaluate_with_ground_truth(
    question: str, answer: str, ground_truth: str, contexts: list[str]
) -> dict:
    """Score one answer on all six RAGAS metrics, using a curated ground truth.

    Adds context_recall and answer_correctness (both need `ground_truth`) on
    top of the four reference-free metrics from `evaluate()`. This is for
    OFFLINE regression testing against a hand-written Q&A dataset — never call
    this from the live chat path, since a real user's question has no
    ground-truth answer to compare against. See scripts/run_ragas_eval.py.
    """
    base = evaluate(question, answer, contexts)
    with ThreadPoolExecutor(max_workers=2) as pool:
        recall_future = pool.submit(context_recall, ground_truth, contexts)
        correctness_future = pool.submit(answer_correctness, answer, ground_truth)
        recall = recall_future.result()
        correctness = correctness_future.result()
    base["context_recall"] = _round(recall)
    base["answer_correctness"] = _round(correctness)
    return base
