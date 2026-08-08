"""
eval/run_eval.py
Runs the WeNext RAG pipeline against eval/golden_dataset.json and scores each
answer with DeepEval's standard RAG metric suite:

  - Answer Relevancy    -> does the answer address the question?
  - Faithfulness        -> does the answer stick to the retrieved context (no hallucination)?
  - Contextual Precision -> are the most relevant retrieved chunks ranked first?
  - Contextual Recall    -> does the retrieved context contain everything needed to answer?
  - Contextual Relevancy -> are the retrieved chunks relevant to the question at all?
  - Hallucination        -> does the answer contradict the ideal/ground-truth context?

The judge LLM is whatever DeepEval is configured to use (see `deepeval diagnose`).
This project configures it to a local Ollama model via `deepeval set-local-model`,
so no API key or internet connection is needed.

Usage:
    python eval/run_eval.py                # run all golden questions
    python eval/run_eval.py --limit 3       # smoke-test on the first 3
    python eval/run_eval.py --gen-model qwen2.5:0.5b   # generate answers with a different model
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_utils import RESULTS_PATH, build_judge_model, build_full_metrics, load_golden_dataset

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.test_case import LLMTestCase

from rag_pipeline import WeNextRAG, DEFAULT_MODEL


def run(limit: int = None, gen_model: str = DEFAULT_MODEL, judge_model: str = DEFAULT_MODEL):
    golden = load_golden_dataset()
    if limit:
        golden = golden[:limit]

    print(f"Loaded {len(golden)} golden questions.")
    print(f"Generation model: {gen_model}")
    print(f"Judge model: {judge_model}")
    print("Running the RAG pipeline for each question (sequential, this is the slow part)...\n")

    rag = WeNextRAG()
    test_cases = []
    pipeline_runs = []

    for i, item in enumerate(golden, 1):
        print(f"[{i}/{len(golden)}] {item['question']}")
        result = rag.answer(item["question"], model=gen_model)
        retrieval_context = [s["text"] for s in result["sources"]]

        test_cases.append(
            LLMTestCase(
                input=item["question"],
                actual_output=result["answer"],
                expected_output=item["expected_answer"],
                retrieval_context=retrieval_context,
                context=item["expected_context"],
            )
        )
        pipeline_runs.append({"module": item["module"], "question": item["question"], "pipeline_result": result})

    print("\nPipeline runs done. Scoring with DeepEval metrics (this calls the judge LLM many times, also slow)...\n")

    judge = build_judge_model(judge_model)
    eval_result = evaluate(
        test_cases,
        metrics=build_full_metrics(judge),
        async_config=AsyncConfig(run_async=True, max_concurrent=1),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        # A single metric occasionally failing on the local model's flaky JSON output
        # shouldn't abort the entire run — record the error and keep going.
        error_config=ErrorConfig(ignore_errors=True),
    )

    report = []
    metric_scores = {}
    metric_errors = {}
    for test_result, run_info in zip(eval_result.test_results, pipeline_runs):
        row = {
            "module": run_info["module"],
            "question": run_info["question"],
            "answer": run_info["pipeline_result"]["answer"],
            "model": run_info["pipeline_result"]["model"],
            "metrics": {},
        }
        for md in test_result.metrics_data:
            row["metrics"][md.name] = {
                "score": md.score,
                "success": md.success,
                "reason": md.reason,
                "error": md.error,
            }
            if md.error:
                metric_errors.setdefault(md.name, 0)
                metric_errors[md.name] += 1
            else:
                metric_scores.setdefault(md.name, []).append(md.score)
        report.append(row)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "generation_model": gen_model,
                "judge_model": judge_model,
                "results": report,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print(f"SUMMARY (generation model: {gen_model}, judge model: {judge_model})")
    print("=" * 70)
    all_metric_names = sorted(set(metric_scores) | set(metric_errors))
    for name in all_metric_names:
        scores = metric_scores.get(name, [])
        errors = metric_errors.get(name, 0)
        avg = sum(scores) / len(scores) if scores else float("nan")
        avg_str = f"{avg:.2f}" if scores else "n/a"
        note = f"  ({errors} errored)" if errors else ""
        print(f"  {name:<26} avg score: {avg_str:<6} (n={len(scores)}){note}")

    print(f"\nPer-question detail saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N golden questions")
    parser.add_argument("--gen-model", type=str, default=DEFAULT_MODEL, help="Ollama model used to generate answers")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_MODEL, help="Ollama model used to score the metrics")
    args = parser.parse_args()
    run(limit=args.limit, gen_model=args.gen_model, judge_model=args.judge_model)
