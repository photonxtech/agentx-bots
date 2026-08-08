"""DeepEval-based answer evaluation — reference-free metrics live, ground-truth
metrics gated on a golden-set match.

Three metrics run live, after every chat answer, WITHOUT any ground-truth
labels — using only the question, the answer, and the retrieved context
chunks (see `evaluate()`):

  * faithfulness      — fraction of the answer's claims that the context supports
                        (catches hallucination).
  * answer_relevancy  — how well the answer addresses the question.
  * context_relevancy — density of the retrieved context: of all the sentences
                        retrieved, what fraction are actually relevant to the
                        question (vs padding/noise)?

Three more metrics need a ground-truth (correct) answer, so they only run when
`golden_set.lookup()` finds a close match for the live question, or from
offline regression scripts against the curated Q&A dataset (see
`evaluate_with_ground_truth()`) — never on an arbitrary live question, since a
real user's question has no known-correct answer to compare against:

  * context_precision — are the *useful* retrieved chunks ranked near the top?
                        (DeepEval's ContextualPrecisionMetric requires
                        expected_output, so unlike RAGAS-style implementations
                        this can no longer run reference-free.)
  * context_recall     — did retrieval pull back everything needed to produce
                        the ground-truth answer?
  * answer_correctness — does the generated answer match the ground-truth
                        answer, factually and semantically? DeepEval ships no
                        built-in metric for this, so it's a GEval judge.

The judge model runs over Groq via rag.deepeval_judge.GroqJudge, since DeepEval
has no native Groq provider. Every metric is defensive: any judge/DeepEval
failure yields ``None`` for that metric rather than raising, so evaluation can
never break a chat answer. Failures are logged (not silent) so a metric that
goes blank in the UI is diagnosable. Within each of `evaluate()` and
`evaluate_with_ground_truth()`, metrics are independent of each other and run
concurrently, so the extra latency is roughly one judge call per stage, not
one per metric.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

import config
from rag.deepeval_judge import GroqJudge

logger = logging.getLogger(__name__)

_judge_instance: GroqJudge | None = None


def _judge() -> GroqJudge:
    """The Groq judge model, built once and reused across metrics/calls."""
    global _judge_instance
    if _judge_instance is None:
        _judge_instance = GroqJudge(config.DEEPEVAL_JUDGE_MODEL)
    return _judge_instance


def _metric_kwargs() -> dict:
    """Fresh kwargs for every metric instance — build new metric objects per
    call, never reuse one across turns, since DeepEval metrics keep the last
    run's score/reason as instance state."""
    return dict(
        model=_judge(),
        threshold=config.DEEPEVAL_METRIC_THRESHOLD,
        include_reason=True,
        async_mode=False,
    )


_CORRECTNESS_STEPS = [
    "Classify each claim in 'actual output' as present in 'expected output', "
    "contradicting it, or missing from it.",
    "Penalize contradictions most, then omissions.",
    "Judge meaning, not wording.",
]


def _answer_correctness_metric() -> GEval:
    """DeepEval ships no built-in answer-correctness metric, so this is a
    GEval judge instead — a weaker, judged tier than the native decompose-and-
    count metrics above, per DeepEval's own guidance."""
    return GEval(
        name="Answer Correctness",
        evaluation_steps=_CORRECTNESS_STEPS,
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=_judge(),
        threshold=config.DEEPEVAL_METRIC_THRESHOLD,
        async_mode=False,
    )


def _score(metric, test_case: LLMTestCase) -> tuple[float | None, str | None]:
    """Run one metric against `test_case`, returning (score, error_reason).

    Never raises: one metric's judge call failing (rate limit, malformed
    judge JSON, network error) must not take down the others — see the
    concurrent `pool.submit` calls in evaluate()/evaluate_with_ground_truth().
    `error_reason` is the short exception summary so a metric that goes blank
    in the UI is diagnosable there directly, not just in server logs.
    """
    try:
        metric.measure(test_case)
    except Exception as e:
        logger.warning("DeepEval metric %s failed", type(metric).__name__, exc_info=True)
        return None, f"{type(e).__name__}: {e}"
    if metric.score is None:
        return None, "judge returned no score"
    return round(float(metric.score), 3), None


def _run_metrics(metrics: dict, test_case: LLMTestCase) -> dict:
    """Run every metric concurrently; failed ones get None plus an entry in
    the returned dict's "errors" sub-dict, keyed by metric name."""
    with ThreadPoolExecutor(max_workers=max(len(metrics), 1)) as pool:
        futures = {key: pool.submit(_score, metric, test_case) for key, metric in metrics.items()}
        results = {key: future.result() for key, future in futures.items()}
    out = {key: score for key, (score, _) in results.items()}
    errors = {key: reason for key, (score, reason) in results.items() if score is None and reason}
    if errors:
        out["errors"] = errors
    return out


def evaluate(question: str, answer: str, contexts: list[str]) -> dict:
    """Score one answer on the three reference-free DeepEval metrics.

    `contexts` are the full text of the retrieved chunks (best relevance first).
    The three metrics don't depend on each other, so they run concurrently —
    wall-clock cost is roughly one judge call, not three sequential ones.
    Returns a dict of floats in [0, 1] (rounded), with None for any metric whose
    judge call failed. Never raises.
    """
    test_case = LLMTestCase(input=question, actual_output=answer, retrieval_context=contexts)
    metrics = {
        "faithfulness": FaithfulnessMetric(**_metric_kwargs()),
        "answer_relevancy": AnswerRelevancyMetric(**_metric_kwargs()),
        "context_relevancy": ContextualRelevancyMetric(**_metric_kwargs()),
    }
    return _run_metrics(metrics, test_case)


def evaluate_with_ground_truth(
    question: str, answer: str, ground_truth: str, contexts: list[str]
) -> dict:
    """Score one answer on all six metrics, using a curated ground truth.

    Adds context_precision, context_recall, and answer_correctness (all need
    `ground_truth`) on top of the three reference-free metrics from
    `evaluate()`. This is for OFFLINE regression testing / golden-set-matched
    live questions against a hand-written Q&A dataset — never call this with
    a ground truth improvised for the occasion, since a reference derived from
    the same contexts it's checked against trivially scores 1.00 forever.
    """
    base = evaluate(question, answer, contexts)
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=ground_truth,
        retrieval_context=contexts,
    )
    metrics = {
        "context_precision": ContextualPrecisionMetric(**_metric_kwargs()),
        "context_recall": ContextualRecallMetric(**_metric_kwargs()),
        "answer_correctness": _answer_correctness_metric(),
    }
    extra = _run_metrics(metrics, test_case)
    # Both evaluate() and this call can independently produce an "errors"
    # sub-dict — merge them instead of letting the second clobber the first.
    merged_errors = {**base.pop("errors", {}), **extra.pop("errors", {})}
    base.update(extra)
    if merged_errors:
        base["errors"] = merged_errors
    return base
