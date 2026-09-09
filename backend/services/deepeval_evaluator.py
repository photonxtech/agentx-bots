"""
Batch metrics evaluation using DeepEval, tuned for token cost.

Scores four dimensions over the whole batch, with cost controls applied at
construction time:

  * `truths_extraction_limit` caps how many facts are pulled from the context
    before claims are checked. Uncapped, that list grows with context length
    AND is re-sent inside the verification prompt — the single most expensive
    prompt in the pass.
  * `include_reason=False` skips a whole extra call per metric per sample.
    The UI shows batch aggregates only, so those per-sample prose explanations
    were tokens spent on text nobody reads.
  * `ContextualRelevancyMetric` is available but OFF by default: it issues a
    verdict per retrieved chunk, so its cost scales with top_k while telling
    much the same story as context_precision.

Trade-off worth stating plainly: capping truths means claims beyond the limit
are not checked, so faithfulness becomes an approximation over a subset —
cheaper and less complete. That is the deal being made, deliberately.
"""
import os
import asyncio
import logging

from dotenv import load_dotenv

from backend.services.deepeval_llm import (
    build_groq_judge, groq_ready, nvidia_ready, DEEPEVAL_MODEL,
    model_fallback_stats, reset_model_fallback_stats,
)
from backend.services.groq_keys import rotation_stats, reset_rotation_stats

load_dotenv()

logger = logging.getLogger(__name__)

# Caps the facts extracted from context before claim verification. 15 is
# PhotonXRAG's tested value; raise it for completeness, lower it to save tokens.
TRUTHS_EXTRACTION_LIMIT = int(os.getenv("DEEPEVAL_TRUTHS_LIMIT", "15"))

# Per-sample prose reasons cost one extra call per metric. Off by default.
INCLUDE_REASON = os.getenv("DEEPEVAL_INCLUDE_REASON", "false").lower() == "true"

# Cost scales with top_k and largely duplicates context_precision. Opt-in.
ENABLE_CONTEXTUAL_RELEVANCY = (
    os.getenv("DEEPEVAL_ENABLE_CTX_RELEVANCY", "false").lower() == "true"
)

# Metric key -> the name exposed in the API/UI.
METRIC_NAMES = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "context_precision": "context_precision",
    "context_recall": "context_recall",
    "context_relevancy": "context_relevancy",   # DeepEval-only extra
}


def _build_metrics(judge):
    """One fresh metric instance per call — DeepEval metrics carry per-run state
    (verdicts, score, reason), so reusing an instance across samples leaks it."""
    from deepeval.metrics import (
        FaithfulnessMetric, AnswerRelevancyMetric,
        ContextualPrecisionMetric, ContextualRecallMetric,
        ContextualRelevancyMetric,
    )

    common = dict(
        model=judge,
        threshold=0.7,
        include_reason=INCLUDE_REASON,
        # Parallel evaluation is handled by ThreadPoolExecutor across cases; 
        # setting this to True would explode concurrency limits per-metric.
        async_mode=False,
    )

    metrics = {
        "faithfulness": FaithfulnessMetric(
            truths_extraction_limit=TRUTHS_EXTRACTION_LIMIT,
            # MANDATORY for a RAG evaluator, and not the default. DeepEval's
            # scorer counts any verdict that is not "no" as faithful, so a claim
            # the context merely does not mention ("idk") passes. Measured: a
            # deliberately hallucinated answer scored faithfulness 1.0 with this
            # off, where a strict scorer gave the identical sample 0.0. Turning it on
            # makes unsupported claims subtract, which is the semantics we want:
            # "supported by the context", not merely "not contradicted by it".
            penalize_ambiguous_claims=True,
            **common
        ),
        "answer_relevancy": AnswerRelevancyMetric(**common),
        "context_precision": ContextualPrecisionMetric(**common),
        "context_recall": ContextualRecallMetric(**common),
    }
    if ENABLE_CONTEXTUAL_RELEVANCY:
        metrics["context_relevancy"] = ContextualRelevancyMetric(**common)
    return metrics


def _score_one(result: dict, judge, notes: list[str]) -> dict:
    """Score a single result. A metric that fails yields None for that metric
    rather than aborting the sample or being counted as zero."""
    from deepeval.test_case import LLMTestCase

    test_case = LLMTestCase(
        input=result["question"],
        actual_output=result.get("generated_answer") or "",
        expected_output=result.get("expected_answer") or "",
        retrieval_context=result.get("contexts") or [],
    )

    scores: dict[str, float | None] = {}
    for key, metric in _build_metrics(judge).items():
        name = METRIC_NAMES[key]
        try:
            metric.measure(test_case)
            score = metric.score
            scores[name] = round(float(score), 4) if score is not None else None
        except Exception as e:
            logger.warning("DeepEval %s failed: %s", name, str(e)[:200])
            scores[name] = None
            msg = f"{name}: a judge call failed ({type(e).__name__})"
            if msg not in notes:
                notes.append(msg)
    return scores


def _run_sync(results: list[dict], notes: list[str], judge_model: str | None = None) -> dict:
    """Blocking — call via asyncio.to_thread."""
    # judge_model comes from _resolve_models, which guarantees it differs from
    # the generator. Passing it through is what keeps that guard effective.
    judge = build_groq_judge(judge_model)

    from concurrent.futures import ThreadPoolExecutor
    
    per_sample: list[dict] = []
    # 3 concurrent cases prevents instant rate limit explosions on Groq's free tier
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_score_one, result, judge, notes) for result in results]
        for future in futures:
            per_sample.append(future.result())

    active = [METRIC_NAMES[k] for k in _build_metrics(judge).keys()]

    aggregate: dict = {}
    scored_counts: dict = {}
    for name in active:
        vals = [row[name] for row in per_sample if row.get(name) is not None]
        scored_counts[name] = len(vals)
        aggregate[name] = round(sum(vals) / len(vals), 4) if vals else None

    return {"aggregate": aggregate, "per_sample": per_sample, "scored_counts": scored_counts}


async def evaluate_with_deepeval(
    results: list[dict], notes: list[str], judge_model: str | None = None,
) -> dict:
    """Score the batch with DeepEval over the whole test set."""
    if not (groq_ready() or nvidia_ready()):
        notes.append("No Groq or Nvidia API key configured — DeepEval metrics were not computed.")
        return {"aggregate": {}, "per_sample": [{} for _ in results], "scored_counts": {}}

    reset_rotation_stats()
    reset_model_fallback_stats()

    try:
        scored = await asyncio.to_thread(_run_sync, results, notes, judge_model)
    except Exception as e:
        logger.exception("DeepEval evaluation failed")
        notes.append(f"DeepEval evaluation failed: {e}")
        return {"aggregate": {}, "per_sample": [{} for _ in results], "scored_counts": {}}

    scored["cost_controls"] = {
        "truths_extraction_limit": TRUTHS_EXTRACTION_LIMIT,
        "include_reason": INCLUDE_REASON,
        "context_relevancy_enabled": ENABLE_CONTEXTUAL_RELEVANCY,
    }
    scored["key_pool"] = rotation_stats()
    scored["model_fallbacks"] = model_fallback_stats()
    scored["judge_model"] = judge_model or DEEPEVAL_MODEL
    return scored
