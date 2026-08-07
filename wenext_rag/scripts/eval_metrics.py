"""Batch evaluation of all 6 RAGAS-style metrics over eval_dataset.json."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.metrics import run_all_metrics
from backend.rag import generate_rag_response, get_relevant_context

EVAL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "eval_dataset.json")

METRIC_KEYS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "context_entity_recall",
    "answer_correctness",
]


def main() -> None:
    with open(EVAL_PATH, encoding="utf-8") as handle:
        dataset = json.load(handle)

    print(f"Evaluating {len(dataset)} samples...\n")
    aggregates: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    results = []

    for idx, item in enumerate(dataset, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        chunks, min_dist = get_relevant_context(question)
        contexts = [c["text"] for c in chunks]
        rag_result = generate_rag_response(question, [])
        answer = rag_result["reply"]

        scores = run_all_metrics(
            question=question,
            answer=answer,
            contexts=contexts if contexts else rag_result["contexts"],
            ground_truth=ground_truth,
        ).to_dict()

        for key in METRIC_KEYS:
            aggregates[key].append(scores[key])

        row = {
            "question": question,
            "min_distance": round(min_dist, 4),
            "num_contexts": len(contexts),
            "scores": scores,
        }
        results.append(row)

        print(f"[{idx}/{len(dataset)}] {question[:60]}...")
        for key in METRIC_KEYS:
            print(f"  {key}: {scores[key]:.3f}")
        print()

    print("=" * 60)
    print("AGGREGATE MEANS")
    print("=" * 60)
    for key in METRIC_KEYS:
        values = aggregates[key]
        mean = sum(values) / len(values) if values else 0.0
        print(f"  {key}: {mean:.3f}")

    out_path = os.path.join(os.path.dirname(EVAL_PATH), "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"samples": results, "means": {k: round(sum(v) / len(v), 3) for k, v in aggregates.items()}}, handle, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
