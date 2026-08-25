import csv
import json
import os
import sys
from pathlib import Path
from typing import List

# Ensure local package imports work when the script is executed directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case.llm_test_case import LLMTestCase
import importlib

from backend.metrics import evaluate_local

from deepeval.models.llms.grok_model import (
    GrokModel,
    check_if_multimodal,
    convert_to_multi_modal_array,
)
from deepeval.models.llms.openai_model import OpenAIModel

try:
    xai_chat = importlib.import_module("xai_sdk.chat")
    _orig_xai_user = xai_chat.user

    def _patched_xai_user(*args):
        fixed_args = []
        for arg in args:
            if isinstance(arg, list):
                if len(arg) == 1 and isinstance(arg[0], dict) and arg[0].get("type") == "text":
                    fixed_args.append(arg[0].get("text", ""))
                else:
                    fixed_args.append(str(arg))
            else:
                fixed_args.append(arg)
        return _orig_xai_user(*fixed_args)

    xai_chat.user = _patched_xai_user
except ImportError:
    xai_chat = None

from evaluation.deepeval_adapter import load_sessions_records, to_deepeval_cases

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEEPEVAL_PROVIDER = os.getenv("DEEPEVAL_PROVIDER", "grok").lower()
DEEPEVAL_MODEL = os.getenv("DEEPEVAL_MODEL", "grok-4.20")
DEEPEVAL_API_KEY = os.getenv("DEEPEVAL_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY_RAGAS") or os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPEVAL_FALLBACK = os.getenv("DEEPEVAL_FALLBACK", "false").lower() in ("1", "true", "yes", "on")


def use_local_fallback() -> bool:
    return DEEPEVAL_PROVIDER == "fallback" or DEEPEVAL_FALLBACK or not (DEEPEVAL_API_KEY or GROQ_API_KEY or OPENAI_API_KEY)


def build_test_cases() -> List[LLMTestCase]:
    records = load_sessions_records()
    return to_deepeval_cases(records)


def build_metrics() -> List:
    if use_local_fallback():
        print("Using local fallback evaluator instead of remote DeepEvals judge model.")
        return []

    # Instantiate a provider-specific model wrapper expected by DeepEvals
    if DEEPEVAL_PROVIDER == "grok":
        if not (DEEPEVAL_API_KEY or GROQ_API_KEY):
            raise EnvironmentError("GROQ_API_KEY_RAGAS or GROQ_API_KEY must be set for DeepEvals Groq provider")
        model = GrokModel(model=DEEPEVAL_MODEL, api_key=DEEPEVAL_API_KEY or GROQ_API_KEY)
    elif DEEPEVAL_PROVIDER == "openai":
        if not (DEEPEVAL_API_KEY or OPENAI_API_KEY):
            raise EnvironmentError("OPENAI_API_KEY or DEEPEVAL_API_KEY must be set for DeepEvals OpenAI provider")
        model = OpenAIModel(model=DEEPEVAL_MODEL, api_key=DEEPEVAL_API_KEY or OPENAI_API_KEY)
    else:
        raise ValueError(f"Unsupported DEEPEVAL_PROVIDER={DEEPEVAL_PROVIDER}. Use 'grok', 'openai', or 'fallback'.")

    return[
        metric = metric_cls(threshold=0.5, model=_judge_model, include_reason=False, async_mode=False)
        AnswerRelevancyMetric(model=model, async_mode=False, verbose_mode=False),
        ContextualPrecisionMetric(model=model, async_mode=False, verbose_mode=False),
        ContextualRecallMetric(model=model, async_mode=False, verbose_mode=False),
        ContextualRelevancyMetric(model=model, async_mode=False, verbose_mode=False),
    ]


def write_results(results, output_path: Path):
    rows = []
    for result in results.test_results:
        row = {
            "name": result.name,
            "input": result.input,
            "actual_output": result.actual_output,
            "expected_output": result.expected_output,
            "success": result.success,
            "score": result.score,
            "metric_name": result.metric_name,
            "reason": result.reason,
            "score_breakdown": json.dumps(result.score_breakdown or {}),
            **(result.metadata or {}),
        }
        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(rows[0].keys(), fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    test_cases = build_test_cases()
    if not test_cases:
        raise SystemExit("No test cases found. Run evaluation/deepeval_adapter.py first or ensure sessions.db is populated.")

    metrics = build_metrics()

    if not metrics:
        print("Running local fallback evaluation on all cases...")
        results = []
        for case in test_cases:
            question = case.input
            answer = case.actual_output
            contexts = [str(item) for item in (case.retrieval_context or [])]
            score = evaluate_local(question, answer, contexts)
            results.append({
                "name": case.name,
                "question": question,
                "answer": answer,
                "contexts": contexts,
                **score,
            })
        output_path = OUTPUT_DIR / "deepeval_local_fallback_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Local fallback results written to {output_path}")
        return

    print(f"Running DeepEvals on {len(test_cases)} cases with model {DEEPEVAL_MODEL}")
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        identifier="WeNextAI DeepEvals",
    )

    print("Evaluation complete")
    output_path = OUTPUT_DIR / "deepeval_results.csv"
    write_results(result, output_path)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
