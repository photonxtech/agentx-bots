"""
PhotonX RAG - per-answer scoring via DeepEval
---------------------------------------------
Replaces the previous hand-rolled judge-LLM prompt. That approach asked one
Groq call to invent six numbers in a single JSON blob: the "counts" it quoted
were never verified, nothing forced the claim decomposition to actually
happen, and a model in a good mood returned 0.9s across the board. It was a
prompt pretending to be a metric.

Everything here is computed by DeepEval (https://deepeval.com), which
implements each metric as a fixed, published algorithm - statement extraction,
per-statement verdicts, then arithmetic in Python over those verdicts. The LLM
is only ever asked small, checkable classification questions; DeepEval does
the counting, not the model. That is the difference between a measurement and
an opinion.

WHAT IS MEASURED
----------------
Same six metrics the UI has always shown, now each backed by a DeepEval
implementation:

  Faithfulness          FaithfulnessMetric         truths vs. claims in the answer
  Answer Relevancy      AnswerRelevancyMetric      relevant statements / statements
  Context Precision     ContextualPrecisionMetric  rank-weighted useful contexts
  Context Recall        ContextualRecallMetric     reference sentences attributable
  Context Entity Recall GEval                      reference entities present in context
  Answer Correctness    GEval                      answer vs. reference agreement

Four are native DeepEval RAG metrics. DeepEval has no built-in entity-recall
metric, and "correctness" is by design a GEval metric there (it is the
framework's documented pattern for it), so those two are defined as GEval
metrics with explicit evaluation_steps. GEval is still a framework algorithm -
the steps are fixed, the score comes out of DeepEval's own scoring - not a
free-form "rate this 0-1" prompt.

TWO MODES, AND THE DIFFERENCE MATTERS
-------------------------------------
Four of the six metrics are defined against a REFERENCE answer, and DeepEval
enforces that: ContextualPrecisionMetric and ContextualRecallMetric will not
run without `expected_output`. Faithfulness and Answer Relevancy are
reference-free by definition and always run.

`score_answer(..., reference="...")` is BENCHMARK mode, used by evaluate.py
over eval_dataset.json, where a human-written reference exists.

`score_answer(...)` without one is LIVE mode - a real user question in the
chat, which has no reference. As before, one extra Groq call
(_synthesize_reference) drafts a reference from ONLY the question and the
retrieved contexts. It never sees the system's actual answer, so it cannot
anchor to it, and the resulting reference is then handed to DeepEval exactly
like a human-written one. If that call fails, scoring degrades to the two
reference-free metrics rather than failing outright.

WHAT THIS COSTS, AND WHY THE DEFAULTS LOOK STINGY
-------------------------------------------------
DeepEval buys its reliability by asking many small questions instead of one
big one: roughly 13 calls per scored answer, with the retrieval context
resent on most of them. That is a different cost shape from the single-call
judge this replaced, where context was sent exactly once - and the settings
below were retuned for it after a 6-context / 4000-char budget burned a
Groq free tier's entire daily token allowance in three questions.

Measured, per scored answer: ~29.5k input tokens at 6x4000, ~10k at 3x1500.
The judge also runs on its own model (see JUDGE_MODEL) so that scoring can
never exhaust the allowance the chat needs to answer at all.

Fail-safe by construction: any failure - missing key, rate limit, deepeval not
installed, one metric erroring - is contained. A metric that fails is dropped
from the result; only a total failure produces an `error` string. It never
raises into the chat flow, because a broken score must not cost the user their
answer.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# Must be set before `deepeval` is first imported. DeepEval otherwise phones
# home on import (analytics + a version check) and prints an update banner into
# the Streamlit server log on every rerun.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("ERROR_REPORTING", "NO")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "YES")

from groq import Groq

def _secret(name: str) -> str | None:
    """An API key from the environment, falling back to Streamlit secrets.
    Both are checked because evaluate.py runs outside Streamlit (env / .env)
    while the deployed app usually only has .streamlit/secrets.toml."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets[name]
    except Exception:
        return None


# ANSWERING AND SCORING MUST NOT SHARE A QUOTA
# --------------------------------------------
# rag_engine.LLM_MODEL_NAME writes the answers; this writes the scores. Keeping
# them on separate models is not a tuning preference, it is what stops scoring
# from starving the chat: DeepEval spends ~10k tokens per scored answer, so on
# a shared model it eats the whole daily allowance in a few questions and every
# later question then dies on a 429 *before it is even answered*.
#
# ONE API KEY COVERS BOTH. Groq keys are per account, not per model, so a
# GROQ_API_KEY in .streamlit/secrets.toml authenticates the answer model and
# the judge model alike. A second secret is needed only for option 2 below,
# where the judge moves to a different company.
#
# Two levels of separation, and the second is strictly better:
#
#   1. Different model, same Groq account (the default). Groq's token-per-day
#      limit is per-model, so openai/gpt-oss-20b's 200k/day pool is separate
#      from the answer model's own 200k. Nothing to sign up for. Avoid
#      llama-3.1-8b-instant despite its larger 500k/day pool - DeepEval asks
#      the judge for schema-constrained verdicts and an 8B model fails that
#      often enough to silently drop metrics.
#
#   2. Different provider entirely. Set JUDGE_PROVIDER="gemini" plus a
#      GOOGLE_API_KEY (free from aistudio.google.com) and scoring moves to
#      Gemini via DeepEval's native GeminiModel, leaving the Groq account to do
#      nothing but answer questions.
#
# JUDGE_PROVIDER is required rather than inferred from the key's presence, and
# that is a deliberate correction: an earlier version switched automatically on
# finding GOOGLE_API_KEY or GEMINI_API_KEY, which meant an unrelated Google
# credential already exported on the host - a very common thing to have - would
# silently take over scoring and fail with an error pointing nowhere near the
# cause. Provider selection is now something you state, not something the
# environment decides for you.
JUDGE_PROVIDER = (os.environ.get("JUDGE_PROVIDER") or "groq").strip().lower()

GROQ_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-oss-20b")

# gpt-oss models bill reasoning tokens like output tokens. Kept low because
# DeepEval never asks the judge an open question - it asks "is this one
# statement supported, yes or no?" a dozen times, which is classification, not
# deliberation. Sent only to models that accept the field (see _chat).
JUDGE_REASONING_EFFORT = os.environ.get("JUDGE_REASONING_EFFORT", "low")
GEMINI_JUDGE_MODEL = os.environ.get("GEMINI_JUDGE_MODEL", "gemini-2.5-flash-lite")

USE_GEMINI_JUDGE = JUDGE_PROVIDER == "gemini"

# Read only once Gemini has been explicitly asked for, so a stray Google
# credential in the environment cannot influence anything.
_GOOGLE_KEY = (
    (_secret("GOOGLE_API_KEY") or _secret("GEMINI_API_KEY"))
    if USE_GEMINI_JUDGE
    else None
)

# What the UI reports as the evaluation model. Resolved here rather than
# hardcoded so the footer names the model that actually did the judging.
JUDGE_MODEL = GEMINI_JUDGE_MODEL if USE_GEMINI_JUDGE else GROQ_JUDGE_MODEL

# The single biggest lever on cost. DeepEval resends the retrieval context on
# most of its sub-calls, so unlike the old one-shot judge - where the context
# was sent exactly once - every extra character here is paid for a dozen times
# over. Measured: 6x4000 costs ~29.5k tokens per scored answer, 3x1500 costs
# ~10k. Three reranked chunks is where the answer's claims come from anyway.
MAX_CONTEXTS = 3
MAX_CONTEXT_CHARS = 1500

# DeepEval decomposes each metric into several small LLM calls, so six metrics
# is well over a dozen Groq requests per answer. Serial by default: the binding
# free-tier constraint is tokens per MINUTE (8k on the judge model), and
# running metrics concurrently spends the same tokens in a shorter window,
# which is precisely what trips a 429. Raise it to 2-3 on a paid tier, where
# the wall-clock win is real and the TPM ceiling is not in reach.
MAX_PARALLEL_METRICS = int(os.environ.get("DEEPEVAL_MAX_WORKERS", "1"))

# Retries for a single judge call. Tuned for a per-minute token limit, which
# is what a free-tier burst actually hits: 5s, 15s, 45s, rather than the 2/4/8
# that expires long before a TPM window rolls over. A per-DAY exhaustion is
# not retried at all - see _chat.
MAX_RETRIES = 3
RETRY_BACKOFF = (5, 15, 45)

# (json key, UI label, "higher" | "lower", needs_reference)
# `needs_reference` marks the metrics DeepEval cannot compute without an
# `expected_output`. In live mode that reference is blind-synthesized (see
# _synthesize_reference); either way the metric is only scored once a real
# reference string exists, never against thin air.
METRICS: tuple[tuple[str, str, str, bool], ...] = (
    ("faithfulness", "Faithfulness", "higher", False),
    ("answer_relevancy", "Answer Relevancy", "higher", False),
    ("context_precision", "Context Precision", "higher", True),
    ("context_recall", "Context Recall", "higher", True),
    ("context_entity_recall", "Context Entity Recall", "higher", True),
    ("answer_correctness", "Answer Correctness", "higher", True),
)

LIVE_METRICS = tuple(m for m in METRICS if not m[3])


def metrics_for(has_reference: bool) -> tuple[tuple[str, str, str, bool], ...]:
    """The metric table applicable to one scoring mode. `has_reference` is
    True whenever a REFERENCE will actually be given to DeepEval - whether
    it's human-written (benchmark mode) or blind-synthesized (live mode)."""
    return METRICS if has_reference else LIVE_METRICS


@dataclass
class JudgeScores:
    """`error` is set instead of the scores when everything goes wrong; callers
    render whichever is populated and never have to handle an exception."""

    scores: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.scores)

    def as_dict(self) -> dict:
        """Flat, JSON-serialisable shape for st.session_state, so Streamlit
        replays stored scores on rerun instead of paying for them again."""
        return {"scores": self.scores, "reasons": self.reasons, "error": self.error}


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------
_client_lock = threading.Lock()
_client: Groq | None = None


def _get_client() -> Groq:
    """One Groq client for the process. Streamlit reruns the script on every
    interaction, so building this per call would churn HTTP clients - and
    DeepEval fans out across threads, which would multiply that."""
    global _client
    with _client_lock:
        if _client is None:
            api_key = _secret("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY is not configured")
            _client = Groq(api_key=api_key)
    return _client


def _is_daily_exhaustion(e: Exception) -> bool:
    """A per-day limit and a per-minute limit are both 429s, and treating them
    the same is a mistake: a TPM burst clears in under a minute and is worth
    waiting out, while a TPD exhaustion clears at midnight UTC and no amount of
    backoff will help. Retrying the latter just makes the user stare at a
    spinner for a minute before seeing the same failure."""
    return "per day" in str(e).lower() or "(tpd)" in str(e).lower()


def _chat(messages: list[dict], *, json_mode: bool, max_tokens: int) -> str:
    """One Groq completion, retried on rate limits. Judging must be
    reproducible, hence temperature=0: the same triple should not score
    differently on a rerun."""
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = _get_client().chat.completions.create(
                model=GROQ_JUDGE_MODEL,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
                **({"response_format": {"type": "json_object"}} if json_mode else {}),
                # gpt-oss only; other models reject it as an unknown field,
                # which keeps JUDGE_MODEL free to point anywhere on Groq.
                **(
                    {"reasoning_effort": JUDGE_REASONING_EFFORT}
                    if "gpt-oss" in GROQ_JUDGE_MODEL
                    else {}
                ),
            )
            return response.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 - retried below, re-raised at the end
            last = e
            if _is_daily_exhaustion(e) or attempt == MAX_RETRIES - 1:
                break
            time.sleep(RETRY_BACKOFF[attempt])
    raise last  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The DeepEval judge model
# ---------------------------------------------------------------------------
def _build_judge():
    """The model DeepEval scores with - Gemini if a Google key is configured,
    otherwise Groq.

    Gemini is used through DeepEval's own GeminiModel, which is already wired
    for structured output. Groq has no DeepEval provider, so that path is a
    DeepEvalBaseLLM over the `groq` client the app already uses.

    Defined inside a function so that `import llm_metrics` costs nothing and
    cannot fail: deepeval is imported lazily, and if it is missing the app
    still boots and reports it in the UI rather than dying at import time."""
    if USE_GEMINI_JUDGE:
        from deepeval.models import GeminiModel

        if not _GOOGLE_KEY:
            # Named explicitly rather than falling back to Groq: a silent
            # fallback would spend the answer model's quota under a config
            # that says it should not be, and the numbers in the UI would
            # credit the wrong model.
            raise RuntimeError(
                'JUDGE_PROVIDER is "gemini" but no GOOGLE_API_KEY is set. '
                "Add one (free at https://aistudio.google.com/apikey), or "
                'remove JUDGE_PROVIDER to score on Groq.'
            )
        return GeminiModel(
            model=GEMINI_JUDGE_MODEL, api_key=_GOOGLE_KEY, temperature=0
        )

    from deepeval.models import DeepEvalBaseLLM
    from pydantic import BaseModel

    class GroqJudge(DeepEvalBaseLLM):
        """DeepEval hands each metric's sub-prompt here and, for every step
        that matters, a Pydantic `schema` it needs back. Returning a validated
        instance of that schema is the whole contract - it is what lets
        DeepEval do the counting in Python instead of trusting a number the
        model wrote."""

        def __init__(self, model_name: str = JUDGE_MODEL):
            self.model_name = model_name
            super().__init__(model_name)

        def load_model(self):
            return _get_client()

        def get_model_name(self) -> str:
            return self.model_name

        def generate(self, prompt: str, schema: type[BaseModel] | None = None):
            if schema is None:
                return _chat(
                    [{"role": "user", "content": prompt}],
                    json_mode=False,
                    max_tokens=1500,
                )

            # Groq's JSON mode guarantees syntactically valid JSON but not the
            # right shape, so the shape is stated in the prompt and enforced by
            # Pydantic on the way back. One repair attempt: a near-miss is
            # cheap to fix and expensive to throw away.
            instruction = (
                f"{prompt}\n\n"
                "Respond with a single JSON object and nothing else. It must "
                "validate against this JSON schema:\n"
                f"{schema.model_json_schema()}"
            )
            messages = [{"role": "user", "content": instruction}]
            for attempt in range(2):
                raw = _chat(messages, json_mode=True, max_tokens=2000)
                try:
                    return schema.model_validate_json(raw)
                except Exception as e:  # noqa: BLE001 - retried once, then raised
                    if attempt:
                        raise
                    messages = messages + [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                f"That JSON failed schema validation: {e}. "
                                "Return the corrected JSON object only."
                            ),
                        },
                    ]

        async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
            # Metrics run with async_mode=False and are parallelised across
            # threads instead (see _measure_all), so this is only here to
            # satisfy the abstract base class.
            return self.generate(prompt, schema=schema)

    return GroqJudge()


# ---------------------------------------------------------------------------
# Metric construction
# ---------------------------------------------------------------------------
_ENTITY_RECALL_STEPS = [
    "List every specific entity in 'expected output': proper nouns, product "
    "and service names, organisation names, technologies, figures, and dates. "
    "Generic words are not entities.",
    "For each entity, check whether it appears in 'retrieval context', either "
    "verbatim or as an unambiguous variant of the same thing.",
    "The score is the number of entities found in 'retrieval context' divided "
    "by the total number of entities listed.",
    "Judge only entity presence in the retrieval context. Ignore the wording, "
    "fluency, and completeness of any answer.",
    "If 'expected output' contains no entities at all, return a score of 1.",
]

_CORRECTNESS_STEPS = [
    "Compare 'actual output' against 'expected output' and classify every "
    "factual claim as: present in both, present in the actual output but "
    "absent from or contradicting the expected output, or required by the "
    "expected output but missing from the actual output.",
    "Penalise contradictions of the expected output most heavily; penalise "
    "omissions of required information next.",
    "Wording, ordering, and level of detail may differ freely - judge meaning, "
    "not phrasing.",
    "If 'expected output' states that the documents cannot answer the "
    "question, then a correct 'actual output' is one that declines to answer; "
    "an actual output that confidently states facts anyway scores 0.",
]


def _build_metrics(judge, has_reference: bool) -> dict:
    """One fresh metric instance per key. Fresh per call, not cached: DeepEval
    metrics carry the last run's verdicts, score and reason on the instance, so
    reusing one across answers would leak state between them."""
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
        GEval,
    )

    try:  # renamed in deepeval 4.1; the old name still works but warns
        from deepeval.test_case import SingleTurnParams as Params
    except ImportError:  # pragma: no cover - older deepeval
        from deepeval.test_case import LLMTestCaseParams as Params

    # async_mode=False on every metric: DeepEval's async path drives its own
    # event loop, which fights Streamlit's script-rerun threading. Concurrency
    # comes from the thread pool in _measure_all instead.
    common = dict(model=judge, threshold=0.7, include_reason=True, async_mode=False)

    built = {
        # truths_extraction_limit caps how many facts are pulled out of the
        # contexts before claims are checked against them. Uncapped, that list
        # grows with context length and is resent on the verdict call, which is
        # the single most expensive prompt in the whole pass.
        "faithfulness": FaithfulnessMetric(truths_extraction_limit=15, **common),
        "answer_relevancy": AnswerRelevancyMetric(**common),
    }
    if has_reference:
        built["context_precision"] = ContextualPrecisionMetric(**common)
        built["context_recall"] = ContextualRecallMetric(**common)
        built["context_entity_recall"] = GEval(
            name="Context Entity Recall",
            evaluation_steps=_ENTITY_RECALL_STEPS,
            evaluation_params=[
                Params.INPUT,
                Params.EXPECTED_OUTPUT,
                Params.RETRIEVAL_CONTEXT,
            ],
            model=judge,
            threshold=0.7,
            async_mode=False,
        )
        built["answer_correctness"] = GEval(
            name="Answer Correctness",
            evaluation_steps=_CORRECTNESS_STEPS,
            evaluation_params=[
                Params.INPUT,
                Params.ACTUAL_OUTPUT,
                Params.EXPECTED_OUTPUT,
            ],
            model=judge,
            threshold=0.7,
            async_mode=False,
        )
    return built


def _measure(metric, test_case) -> None:
    """`_show_indicator` suppresses DeepEval's rich spinner, which would
    otherwise scribble ANSI escapes into the Streamlit server log once per
    metric. Both flags are private to DeepEval, so a version that drops them
    falls back to the plain call rather than breaking scoring."""
    try:
        metric.measure(test_case, _show_indicator=False, _log_metric_to_confident=False)
    except TypeError:
        metric.measure(test_case)


def _measure_all(metrics: dict, test_case) -> tuple[dict, dict, list[str]]:
    """Run every metric over the one test case and collect what succeeded.

    A metric that fails is dropped and its failure recorded, never raised: one
    rate-limited sub-call must not cost the other five metrics their scores.
    """
    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    failures: list[str] = []

    workers = max(1, min(MAX_PARALLEL_METRICS, len(metrics)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            key: pool.submit(_measure, metric, test_case)
            for key, metric in metrics.items()
        }
        for key, future in futures.items():
            try:
                future.result()
            except Exception as e:  # noqa: BLE001 - one metric down, not all six
                failures.append(f"{key}: {type(e).__name__}")
                continue
            metric = metrics[key]
            score = getattr(metric, "score", None)
            if score is None:
                failures.append(f"{key}: no score")
                continue
            # Every metric is defined on 0-1; clamp so an out-of-range value
            # can't reach the UI's progress bars.
            scores[key] = max(0.0, min(1.0, float(score)))
            reason = (getattr(metric, "reason", None) or "").strip()
            if reason:
                reasons[key] = reason

    return scores, reasons, failures


# ---------------------------------------------------------------------------
# Reference synthesis (live mode only)
# ---------------------------------------------------------------------------
_SYNTHESIS_SYSTEM_PROMPT = """\
You answer questions using ONLY the provided context. Write the single best, \
complete, accurate answer to the QUESTION using only the CONTEXTS - as if you \
were about to answer the user directly. If the CONTEXTS do not contain enough \
to answer, say so plainly instead of guessing. Plain text only, no preamble, \
no JSON, 2-5 sentences."""


def _generate_text(judge, prompt: str) -> str:
    """Plain text out of whichever judge is configured.

    DeepEval's own providers report their spend, so GeminiModel.generate
    returns a (text, cost) tuple while our GroqJudge returns a bare string.
    DeepEval unwraps that internally for metrics; anything of ours that calls
    the judge directly has to do it here."""
    result = judge.generate(prompt)
    if isinstance(result, tuple):
        result = result[0]
    return str(result or "")


def _synthesize_reference(judge, question: str, contexts: list[str]) -> str | None:
    """Drafts a reference answer from ONLY the question and contexts, in a call
    that never sees the system's actual answer. DeepEval needs an
    `expected_output` for four of the six metrics and a live chat question has
    none; this supplies one without contaminating it. The blind separation is
    the point - a reference drafted while looking at the real answer just
    measures self-agreement.

    Runs on the judge rather than a direct Groq call so that every token spent
    evaluating lands on the evaluation provider. Otherwise configuring a Gemini
    judge would still quietly bill this call to Groq, which is the exact
    cross-contamination the split exists to prevent.

    Returns None (never raises) on any failure, so a bad synthesis call costs
    the four reference-based metrics, not the whole score.
    """
    context_block = "\n\n".join(f"[Context {i}]\n{c}" for i, c in enumerate(contexts, 1))
    try:
        text = _generate_text(
            judge,
            f"{_SYNTHESIS_SYSTEM_PROMPT}\n\n"
            f"QUESTION\n{question.strip()}\n\nCONTEXTS\n{context_block}",
        )
        return text.strip() or None
    except Exception:
        return None


def _trim(contexts: list[str]) -> list[str]:
    out = []
    for c in contexts[:MAX_CONTEXTS]:
        c = (c or "").strip()
        if c:
            out.append(c[:MAX_CONTEXT_CHARS])
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def score_answer(
    question: str,
    contexts: list[str],
    answer: str,
    reference: str | None = None,
) -> JudgeScores:
    """Score one answer with DeepEval. Never raises.

    Pass `reference` (a known-correct answer, as eval_dataset.json supplies) to
    use it directly for benchmark mode. Omit it for a live chat question and a
    reference is drafted automatically via a separate, answer-blind call (see
    _synthesize_reference) so the four reference-based metrics stay available -
    if that call fails, scoring falls back to Faithfulness and Answer Relevancy
    rather than failing the whole score.

    Deliberately takes no embedder and no retrieval objects: DeepEval reads
    text, so scoring stays decoupled from the engine that produced the answer.
    """
    contexts = _trim(contexts)
    if not question.strip() or not answer.strip() or not contexts:
        # No retrieved context means the app already declined to answer -
        # there is nothing meaningful to score.
        return JudgeScores(error="nothing to score")

    # The judge is built before the reference is synthesized because synthesis
    # now runs on it (see _synthesize_reference). A missing key or a missing
    # deepeval is therefore reported here, once, instead of surfacing as a
    # mysteriously absent reference later.
    try:
        from deepeval.test_case import LLMTestCase

        judge = _build_judge()
    except Exception as e:  # noqa: BLE001 - deepeval missing, or no API key
        # Includes the message, not just the exception type: a bare
        # "ModuleNotFoundError" with no module name is impossible to diagnose
        # from the UI.
        return JudgeScores(error=f"{type(e).__name__}: {e}"[:200])

    reference = (reference or "").strip()
    if not reference:
        reference = _synthesize_reference(judge, question, contexts) or ""
    has_reference = bool(reference)

    try:
        metrics = _build_metrics(judge, has_reference)
        test_case = LLMTestCase(
            input=question.strip(),
            actual_output=answer.strip(),
            retrieval_context=contexts,
            expected_output=reference or None,
        )
    except Exception as e:  # noqa: BLE001 - a deepeval version mismatch
        # Separate from the judge-construction failure above so the UI can tell
        # "could not reach the judge" apart from "this deepeval version renamed
        # something", which are fixed in completely different places.
        return JudgeScores(error=f"metric setup failed - {type(e).__name__}: {e}"[:200])

    scores, reasons, failures = _measure_all(metrics, test_case)

    if not scores:
        detail = "; ".join(failures) or "no metric returned a score"
        return JudgeScores(error=f"DeepEval scoring failed - {detail}"[:200])

    return JudgeScores(scores=scores, reasons=reasons)
