"""
eval/eval_utils.py
Shared DeepEval setup used by both the offline golden-dataset runner
(eval/run_eval.py) and the live "Evaluate" button in the chat UI (main.py).
"""

import json
import os

# Local Ollama models are much slower than DeepEval's default cloud-judge assumptions,
# and some metrics (e.g. Faithfulness) fire 2 concurrent LLM calls that Ollama serializes
# on a single GPU. Raise the timeouts before any deepeval metric/settings object is built.
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "1200")
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "300")
# Small local models occasionally break JSON output on complex multi-verdict prompts
# (e.g. Contextual Recall/Precision, which ask for one verdict per claim). More retry
# attempts gives the flaky model more chances to produce valid JSON on a re-roll.
os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "4")

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
    GEval,
)
from deepeval.models.llms.local_model import LocalModel
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.test_case import LLMTestCaseParams
from pydantic import BaseModel
import typing

from rag_pipeline import OLLAMA_BASE_URL

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")

# Cosine similarity above this threshold means a live chat question is considered
# "the same question" as a golden one, so its ground truth can be used.
GOLDEN_MATCH_THRESHOLD = 0.82


def load_golden_dataset():
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _repair_missing_fields(data, schema_cls):
    """
    Small local judge models reliably produce syntactically valid JSON (thanks to
    response_format=json_object) but occasionally drop a required field on items
    deep in a long verdict list (e.g. Contextual Relevancy/Recall/Precision, which
    ask for one verdict per retrieved chunk/sentence) — this is a content quality
    limitation of the model, not something retries fix, since temperature=0 makes
    the judge reproduce the exact same (incomplete) output every attempt.

    This defaults a missing `verdict` field to "no", the conservative choice
    matching the metric's own semantics: an item with no clear "yes" is treated
    as not relevant/not supported, so the score is never artificially inflated.
    """
    if not isinstance(data, dict):
        return data
    repaired = dict(data)
    for field_name, field in schema_cls.model_fields.items():
        if field_name not in repaired:
            continue
        value = repaired[field_name]
        origin = typing.get_origin(field.annotation)
        if origin is not list or not isinstance(value, list):
            continue
        args = typing.get_args(field.annotation)
        if not args or not (isinstance(args[0], type) and issubclass(args[0], BaseModel)):
            continue
        item_cls = args[0]
        new_list = []
        for item in value:
            if isinstance(item, dict) and "verdict" in item_cls.model_fields and "verdict" not in item:
                item = {**item, "verdict": "no"}
            new_list.append(item)
        repaired[field_name] = new_list
    return repaired


class RepairingLocalModel(LocalModel):
    """
    LocalModel with a best-effort JSON repair step before schema validation.
    Multimodal input isn't supported here (unneeded for this project's text-only
    metrics) — this only re-implements the plain-text request/parse path.
    """

    def generate(self, prompt: str, schema=None):
        client = self.load_model(async_mode=False)
        response = client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        return self._parse(response.choices[0].message.content, schema)

    async def a_generate(self, prompt: str, schema=None):
        client = self.load_model(async_mode=True)
        response = await client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        return self._parse(response.choices[0].message.content, schema)

    def _parse(self, content: str, schema):
        if not schema:
            return content, 0.0
        json_output = _repair_missing_fields(trim_and_load_json(content), schema)
        return schema.model_validate(json_output), 0.0


# DeepEval's LocalModel accepts a `format` param but never actually sends it to the
# API call, so the judge relies purely on prompt instructions for JSON output — which
# small local models occasionally break. Forcing Ollama's real JSON-mode via
# response_format fixes this at the request level.
def build_judge_model(judge_model_name: str) -> RepairingLocalModel:
    return RepairingLocalModel(
        model=judge_model_name,
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        generation_kwargs={
            "response_format": {"type": "json_object"},
            "max_tokens": 4096,
        },
    )


def build_answer_correctness_metric(judge_model: LocalModel) -> GEval:
    """
    DeepEval has no built-in Answer Correctness metric, so this is a GEval
    metric that scores the generated answer against the golden expected_output.
    """
    return GEval(
        name="Answer Correctness",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        evaluation_steps=[
            "Compare the actual output to the expected output for the given input question.",
            "Check whether the actual output captures the key facts present in the expected output.",
            "Penalize the actual output for missing important information from the expected output.",
            "Penalize the actual output for including facts that contradict the expected output.",
            "Do not penalize differences in phrasing, wording, or extra correct detail not present in the expected output.",
        ],
        model=judge_model,
        _include_g_eval_suffix=False,
    )


def build_full_metrics(judge_model: LocalModel):
    """All 6 metrics. Requires a ground-truth expected_output + context."""
    return [
        AnswerRelevancyMetric(model=judge_model),
        FaithfulnessMetric(model=judge_model),
        ContextualPrecisionMetric(model=judge_model),
        ContextualRecallMetric(model=judge_model),
        ContextualRelevancyMetric(model=judge_model),
        build_answer_correctness_metric(judge_model),
    ]


def build_reference_free_metrics(judge_model: LocalModel):
    """The 4 metrics that don't need a ground-truth expected answer."""
    return [
        AnswerRelevancyMetric(model=judge_model),
        FaithfulnessMetric(model=judge_model),
        ContextualRelevancyMetric(model=judge_model),
        HallucinationMetric(model=judge_model),
    ]


def find_golden_match(question: str, golden_dataset: list, embed_model, threshold: float = GOLDEN_MATCH_THRESHOLD):
    """
    Returns the golden_dataset entry whose question is closest in meaning to `question`,
    if the similarity clears `threshold`. Otherwise returns None.
    Reuses the same sentence-transformers model already loaded for retrieval.
    """
    from sentence_transformers import util

    if not golden_dataset:
        return None

    golden_questions = [g["question"] for g in golden_dataset]
    live_emb = embed_model.encode([question])
    golden_embs = embed_model.encode(golden_questions)
    sims = util.cos_sim(live_emb, golden_embs)[0]

    best_idx = int(sims.argmax())
    best_score = float(sims[best_idx])
    if best_score >= threshold:
        return golden_dataset[best_idx]
    return None
