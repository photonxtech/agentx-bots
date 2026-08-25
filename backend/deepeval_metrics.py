"""
DeepEval-based live evaluation, using Groq as the judge model via its
OpenAI-compatible endpoint. This is the ONLY live judge pipeline — the
custom metrics.py RAGAS-style pipeline has been removed from the live
chat path.

Produces up to 6 metrics, for parity with what the old custom pipeline +
gt_metrics.py gave you: faithfulness, answer_relevancy, context_relevancy,
context_precision, context_recall, answer_correctness.

IMPORTANT — context_precision / context_recall approximation:
DeepEval's ContextualPrecisionMetric and ContextualRecallMetric both grade
retrieval quality against an `expected_output` (a reference answer). A live
user question usually doesn't have one. When no real ground-truth match is
found (see chat.py's find_ground_truth), this module substitutes the
MODEL'S OWN generated answer as a stand-in expected_output. That's not a
true reference — it can't catch cases where the model confidently answered
the wrong thing — but it still gives a real, useful signal: are the chunks
that were actually retrieved the ones that support what got said, and nothing
extraneous. When find_ground_truth DOES find a match, the real ground_truth
is used instead and the score is a genuine, accurate one.

answer_correctness is NOT approximated this way — comparing the answer to
itself would trivially score ~1.0 always and carry no signal. It stays
None unless chat.py finds a real ground-truth match and computes it
separately via gt_metrics.answer_correctness.

Each result dict also carries "context_metrics_are_proxy": True/False so
callers/UI can show an "approximate" indicator when no real ground truth
was used.
"""

import os
import time

from openai import OpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

# deepeval_metrics.py
#
# Reverted to openai/gpt-oss-120b. qwen/qwen3.6-27b was tried as a
# migration target (see git history / earlier comments), but it appears to
# be a reasoning-style model: even with response_format=json_object and a
# generous max_tokens, Groq rejected most completions with 400
# json_validate_failed and an EMPTY failed_generation — consistent with
# internal <think>-style reasoning tokens leaking into (or entirely
# consuming) the completion before any JSON is emitted, which Groq's
# strict json_object validator then rejects server-side.
#
# gpt-oss-120b's own logs (before any of these changes) show it reliably
# producing valid, parseable JSON from a PLAIN completion — no
# response_format needed. Its only real problem was ever the 8000 TPM
# ceiling, which turned out to be an ACCOUNT-WIDE limit (confirmed
# identical on both models), not something switching models could fix
# anyway. That ceiling is now handled by running metrics sequentially with
# real spacing (see STAGGER_S / evaluate_deepeval below) instead, so
# there's no remaining reason to prefer qwen here.
DEEPEVAL_MODEL = "openai/gpt-oss-120b"

# Delay between sequential metric calls (see evaluate_deepeval below).
# Confirmed the 8000 TPM ceiling is ACCOUNT/TIER-wide, not model-specific
# (it was identical whether targeting gpt-oss-120b or qwen3.6-27b), so
# model choice was never going to fix it — only spacing requests out over
# real time does. This run's logs show it working: 429s dropped sharply
# once metrics ran sequentially with this delay instead of concurrently.
# 2s isn't a guarantee against ever hitting the ceiling on a busy account,
# but it gives real separation without making a "Calculate Metrics" click
# take unreasonably long.
STAGGER_S = 2.0


class GroqDeepEvalModel(DeepEvalBaseLLM):
    """Wraps Groq's OpenAI-compatible endpoint for DeepEval's judge calls.

    IMPORTANT: GROQ_API_KEY_DEEPEVAL must point at a genuinely SEPARATE
    Groq account/organization, not just a second key on the same account —
    Groq's rate limits are scoped per-organization, not per-key.
    """

    def __init__(self, model_name=DEEPEVAL_MODEL):
        self.model_name = model_name
        api_key = (
            os.getenv("GROQ_API_KEY_DEEPEVAL")
            or os.getenv("GROQ_API_KEY_RAGAS")
            or os.getenv("GROQ_API_KEY")
        )
        if not os.getenv("GROQ_API_KEY_DEEPEVAL"):
            print(
                "[deepeval_metrics] WARNING: GROQ_API_KEY_DEEPEVAL is not set — "
                "falling back to a key shared with other Groq usage in this app. "
                "This means judge calls here share the SAME per-minute token "
                "budget as your main chat/condense LLM and/or metrics.py's "
                "evaluator, which can reintroduce the exact 429 problem this "
                "separate-org design was meant to avoid. Set GROQ_API_KEY_DEEPEVAL "
                "to a key on a genuinely separate Groq organization to fix this.",
                flush=True,
            )
        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        # CORRECTION to the comment this replaced: gpt-oss-120b does NOT
        # reliably return plain parseable JSON — it's a reasoning model, and
        # per langchain_groq's own field docs, Groq "will default to
        # enabling reasoning if left undefined". Confirmed live: faithfulness
        # and context_recall (the two metrics that issue several internal
        # judge calls per single .measure(), giving more chances to hit a
        # polluted response) were failing with deepeval's "Evaluation LLM
        # outputted an invalid JSON" — because with no reasoning_format set,
        # Groq's default wraps the response in a leading <think>...</think>
        # reasoning block before the actual JSON, and deepeval's parser
        # chokes on that preamble.
        #
        # reasoning_format="hidden" strips that block from `content` so this
        # always returns clean JSON to deepeval (the model still reasons
        # internally — this only removes it from the text we hand back,
        # it's not the same as disabling reasoning). reasoning_effort="low"
        # additionally caps how much hidden reasoning happens at all, cutting
        # the token cost of every judge call. Passed via extra_body since
        # these are Groq-specific fields the openai-python client doesn't
        # know about natively.
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            extra_body={"reasoning_format": "hidden", "reasoning_effort": "low"},
        )
        return resp.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        # No async Groq client wired up yet — reuse the sync path for now.
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


_judge_model = GroqDeepEvalModel()


def _run_one_metric(name, metric_cls, test_case):
    try:
        metric = metric_cls(threshold=0.5, model=_judge_model, include_reason=False, async_mode=False)
        metric.measure(test_case)
        return name, (round(metric.score, 3) if metric.score is not None else None)
    except Exception as e:
        print(f"[deepeval] {name} failed: {e}", flush=True)
        return name, None


def evaluate_deepeval(question, answer, contexts, ground_truth=None):
    """Score one live chat turn with DeepEval.

    ground_truth: pass the matched reference answer when chat.py's
    find_ground_truth() found one. When None, context_precision/
    context_recall fall back to using `answer` itself as a proxy
    expected_output (see module docstring) — real signal, just softer
    than a true reference would give.

    Never raises — a failed metric just comes back as None so a live
    chat answer can't break because of this.
    """
    is_proxy = ground_truth is None
    expected_output = ground_truth if ground_truth else answer

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=expected_output,
        retrieval_context=contexts,
    )

    metric_specs = [
        ("faithfulness", FaithfulnessMetric),
        ("answer_relevancy", AnswerRelevancyMetric),
        ("context_relevancy", ContextualRelevancyMetric),
        ("context_precision", ContextualPrecisionMetric),
        ("context_recall", ContextualRecallMetric),
    ]

    # Sequential, not concurrent. The 8000 TPM ceiling on this Groq
    # account is per-organization and shared across the whole minute
    # regardless of which model is targeted (confirmed: qwen/qwen3.6-27b
    # and openai/gpt-oss-120b both hit the identical "Limit 8000"). Running
    # these concurrently — even with a stagger — lands all 5 calls' token
    # cost within the same few seconds, well inside the same rolling 60s
    # window, which is what caused the 429 pile-up regardless of model.
    # Running them one at a time with a real delay between each is slower
    # per "Calculate Metrics" click, but it's the only thing that actually
    # respects a per-minute account-wide token budget instead of just
    # moving the collision around.
    results = {}
    for name, metric_cls in metric_specs:
        results[name] = _run_one_metric(name, metric_cls, test_case)[1]
        time.sleep(STAGGER_S)

    results["context_metrics_are_proxy"] = is_proxy
    return results