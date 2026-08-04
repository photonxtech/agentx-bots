"""Offline RAGAS regression test, run from the project root (backend/).

Loads a curated question/ground-truth dataset (default: eval_dataset_osw.json),
runs each question through the REAL retrieval + generation pipeline against
whatever is currently indexed (config.INDEX_DIR), and scores every answer on
all six RAGAS metrics via rag.evaluation.evaluate_with_ground_truth() — the
four live/reference-free ones plus context_recall and answer_correctness,
which need the ground_truth field this script provides.

Usage (run as a module from backend/, so `config`/`rag` resolve on sys.path —
running it as a plain script file only puts scripts/ on the path, not backend/):
    python -m scripts.run_ragas_eval [path/to/dataset.json]

Writes a full per-question report to scripts/ragas_eval_report.json and
prints a summary table + averages to stdout.
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

import config
from rag import evaluation, generator, reranker
from rag.vectorstore import VectorStore

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(SCRIPT_DIR, "eval_dataset_osw.json")
REPORT_PATH = os.path.join(SCRIPT_DIR, "ragas_eval_report.json")

METRIC_KEYS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_relevancy",
    "context_recall",
    "answer_correctness",
]


def run(dataset_path: str) -> None:
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} questions from {dataset_path}")

    store = VectorStore(dirpath=config.INDEX_DIR)
    print(f"Connected to vector store (backend: {store.backend}, {store.size} chunks indexed)\n")

    results = []
    for i, row in enumerate(dataset, 1):
        question = row["question"]
        ground_truth = row["ground_truth"]
        print(f"[{i}/{len(dataset)}] {question}")

        hits = store.search(question, top_k=config.TOP_K * 3)
        if config.RERANK_ENABLED:
            hits = reranker.rerank(question, hits)
        hits = hits[:config.TOP_K]
        contexts = [doc.text for doc, _ in hits if doc.text]

        try:
            answer_text = "".join(generator.answer(question, hits))
        except Exception as e:
            print(f"    generation failed: {e}")
            results.append({**row, "answer": None, "metrics": None, "error": str(e)})
            continue

        try:
            metrics = evaluation.evaluate_with_ground_truth(question, answer_text, ground_truth, contexts)
        except Exception as e:
            print(f"    evaluation failed: {e}")
            metrics = None

        print(f"    answer: {answer_text[:150].replace(chr(10), ' ')}...")
        print(f"    metrics: {metrics}\n")
        results.append({**row, "answer": answer_text, "metrics": metrics})

        time.sleep(1)  # be gentle on the judge model's rate limit

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Full report written to {REPORT_PATH}\n")

    print("=" * 70)
    print("AVERAGES")
    print("=" * 70)
    for key in METRIC_KEYS:
        scored = [r["metrics"][key] for r in results if r.get("metrics") and r["metrics"].get(key) is not None]
        if scored:
            print(f"  {key:20s} {sum(scored) / len(scored):.3f}  ({len(scored)}/{len(results)} scored)")
        else:
            print(f"  {key:20s} n/a  (0/{len(results)} scored)")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET
    run(path)
