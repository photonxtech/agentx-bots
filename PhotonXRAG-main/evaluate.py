"""
PhotonX RAG - corpus-level evaluation
-------------------------------------
Runs the real pipeline in rag_engine.py over every question in
eval_dataset.json, scores each answer with DeepEval via llm_metrics.py, and
writes aggregate numbers to eval_summary.json. app.py renders that file as the
system-level report card at the bottom of the page.

    python evaluate.py                        # full run, writes eval_summary.json
    python evaluate.py --dataset my.json      # a different question set
    python evaluate.py --limit 3              # smoke test on the first 3
    python evaluate.py --sleep 3              # pause between questions (rate limits)
    python evaluate.py --no-write             # print only, leave the summary alone

WHY THIS IS A SEPARATE SCRIPT AND NOT A BUTTON IN THE APP
---------------------------------------------------------
Each question costs a retrieval pass, an answer generation, and a full
DeepEval scoring pass - and DeepEval breaks each metric into several small
extraction/verdict calls, so that is well over a dozen Groq requests per
question. Over a full set that is minutes of wall-clock and hundreds of
requests - enough to hit a free-tier rate limit and long enough to trip a
Streamlit Cloud request timeout. So it runs here, offline, on demand; the app
only ever reads the committed result. Same reason the numbers carry a
timestamp: they describe the system as of that run, not as of page load.

Use --sleep if Groq starts returning 429s, and DEEPEVAL_MAX_WORKERS=1 to make
each question's metrics run one at a time instead of three at a time.

HOW THIS DIFFERS FROM THE PER-ANSWER SCORES
-------------------------------------------
The chat UI scores whatever a user just asked, which has no reference answer -
so one is synthesized blind from the retrieved context to satisfy the four
reference-based metrics. This scores a fixed set where every question has a
human-written reference, which buys two things the live path cannot have:
those four metrics measured against real ground truth, and comparability - the
same questions rerun after a change in chunking, reranking or prompting
produce numbers you can diff.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from llm_metrics import JUDGE_MODEL, METRICS, score_answer  # noqa: E402
from rag_engine import LLM_MODEL_NAME, ask, load_resources  # noqa: E402

ROOT = Path(__file__).parent
DEFAULT_DATASET = ROOT / "eval_dataset.json"
DEFAULT_SUMMARY = ROOT / "eval_summary.json"

# A metric mean below this is called out in the console summary. Not a pass/fail
# gate - just the line under which a number is worth looking at rather than
# skimming past.
ATTENTION_FLOOR = 0.70
# Generic mirror of ATTENTION_FLOOR for any "lower is better" metric (none
# currently defined in METRICS, but the mechanism stays generic rather than
# being tied to one specific metric's name).
LOWER_IS_BETTER_CEILING = 1.0 - ATTENTION_FLOOR


def load_dataset(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Accept both shapes: the documented {"questions": [...]} object and a bare
    # top-level list, so a hand-written file works either way.
    items = raw.get("questions", []) if isinstance(raw, dict) else raw
    if not isinstance(items, list) or not items:
        raise SystemExit(f"No questions found in {path}. Expected a non-empty list.")

    cleaned = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise SystemExit(f"Question {i} in {path} is not an object.")
        question = str(item.get("question", "")).strip()
        if not question:
            raise SystemExit(f"Question {i} in {path} has no 'question' field.")
        # `ground_truth` is what the old ragas dataset called it; accept both so
        # an older file does not silently evaluate with no reference at all.
        reference = str(item.get("reference") or item.get("ground_truth") or "").strip()
        if not reference:
            print(
                f"  ! '{question[:50]}' has no reference - it will fall back to "
                f"a synthesized one, so its four reference-based metrics are "
                f"not measured against ground truth.",
                file=sys.stderr,
            )
        cleaned.append(
            {
                "id": str(item.get("id") or f"q{i + 1}"),
                "question": question,
                "reference": reference,
                "expect_refusal": bool(item.get("expect_refusal", False)),
            }
        )
    return cleaned


def run_one(resources, item: dict) -> dict:
    """Answer one question with the real pipeline, then score it. Never raises -
    a single bad question must not abandon the rest of the run."""
    row = {
        "id": item["id"],
        "question": item["question"],
        "expect_refusal": item["expect_refusal"],
        "answer": "",
        "n_contexts": 0,
        "sources": [],
        "scores": {},
        "reasons": {},
        "error": None,
    }
    try:
        chunks, stream = ask(resources, item["question"], chat_history=[])
        row["answer"] = "".join(stream)
        row["n_contexts"] = len(chunks)
        row["sources"] = sorted(
            {
                (c["metadata"].get("h2") or c["metadata"].get("h1") or "?")
                for c in chunks
            }
        )
        if not chunks:
            # retrieve() returns at least one chunk whenever the index is
            # non-empty, so this means the index itself is empty or the query
            # matched nothing at all.
            row["error"] = "no contexts retrieved"
            return row

        judged = score_answer(
            question=item["question"],
            contexts=[c["text"] for c in chunks],
            answer=row["answer"],
            reference=item["reference"] or None,
        )
        row["scores"] = judged.scores
        row["reasons"] = judged.reasons
        row["error"] = judged.error
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:200]
    return row


def aggregate(rows: list[dict]) -> list[dict]:
    """Mean per metric over the questions that produced a score for it.

    Averaging only over questions that scored, rather than treating a missing
    score as zero, keeps one unscorable question from dragging a metric down and
    misreporting it as a regression. `n_scored` is carried alongside every mean
    so a thin average is visible rather than implied.
    """
    out = []
    for key, label, direction, needs_ref in METRICS:
        values = [r["scores"][key] for r in rows if key in r.get("scores", {})]
        out.append(
            {
                "key": key,
                "label": label,
                "direction": direction,
                "needs_reference": needs_ref,
                "score": (sum(values) / len(values)) if values else None,
                "n_scored": len(values),
                "worst": (max(values) if direction == "lower" else min(values))
                if values
                else None,
            }
        )
    return out


def build_summary(rows: list[dict], dataset_name: str) -> dict:
    """Assemble the summary object. Shared by this script's CLI and by the
    in-app "Run evaluation" button in app.py, so the report card renders from
    an identical shape no matter which one produced it."""
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "dataset": dataset_name,
        "n_questions": len(rows),
        "n_scored": sum(1 for r in rows if r["scores"]),
        "answer_model": LLM_MODEL_NAME,
        "judge_model": JUDGE_MODEL,
        "metrics": aggregate(rows),
        "questions": rows,
    }


def needs_attention(metric: dict) -> bool:
    if metric["score"] is None:
        return False
    if metric["direction"] == "lower":
        return metric["score"] > LOWER_IS_BETTER_CEILING
    return metric["score"] < ATTENTION_FLOOR


def print_report(summary: dict) -> None:
    rows = summary["questions"]
    print()
    print("=" * 72)
    print(f"  PhotonX RAG - system evaluation over {summary['n_questions']} questions")
    print("=" * 72)

    for r in rows:
        mark = "!" if (r["error"] or not r["scores"]) else " "
        print(f"\n{mark} [{r['id']}] {r['question']}")
        print(f"    contexts: {r['n_contexts']}  sources: {', '.join(r['sources']) or '-'}")
        if r["error"]:
            print(f"    ERROR: {r['error']}")
        for key, label, direction, _nr in METRICS:
            if key in r["scores"]:
                arrow = " (lower better)" if direction == "lower" else ""
                print(f"    {label:<22} {r['scores'][key]:.2f}{arrow}")

    print()
    print("-" * 72)
    print("  AGGREGATE")
    print("-" * 72)
    for m in summary["metrics"]:
        if m["score"] is None:
            print(f"  {m['label']:<24}     n/a  (0 questions scored)")
            continue
        flag = "  <-- look at this" if needs_attention(m) else ""
        arrow = "v" if m["direction"] == "lower" else "^"
        print(
            f"  {m['label']:<24} {arrow} {m['score']:.3f}"
            f"  (n={m['n_scored']}, worst {m['worst']:.2f}){flag}"
        )

    failed = [r for r in rows if r["error"]]
    if failed:
        print(f"\n  {len(failed)} question(s) did not score:")
        for r in failed:
            print(f"    - [{r['id']}] {r['error']}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the PhotonX RAG system.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds between questions")
    parser.add_argument("--no-write", action="store_true", help="print without writing --out")
    args = parser.parse_args()

    items = load_dataset(args.dataset)
    if args.limit:
        items = items[: args.limit]

    print(f"Loading RAG resources (embedder + reranker + index)...")
    try:
        resources = load_resources()
    except RuntimeError as e:
        print(f"\n{e}\nRun `python ingest.py` first.", file=sys.stderr)
        return 1

    print(f"Answering + scoring {len(items)} questions with DeepEval.")
    print(f"  answer model:     {LLM_MODEL_NAME}")
    print(f"  evaluation model: {JUDGE_MODEL}\n")

    rows = []
    for i, item in enumerate(items, 1):
        print(f"  [{i}/{len(items)}] {item['question'][:60]}...", flush=True)
        row = run_one(resources, item)
        rows.append(row)
        if row["error"]:
            print(f"        -> {row['error']}", flush=True)
        if args.sleep and i < len(items):
            time.sleep(args.sleep)

    summary = build_summary(rows, args.dataset.name)

    print_report(summary)

    if args.no_write:
        print(f"--no-write: {args.out.name} left unchanged.")
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.out.name}. The app's report card will pick it up on reload.")

    # Non-zero exit if nothing scored at all, so CI or a shell `&&` chain
    # notices a totally broken run. A partial run is still a success.
    return 0 if summary["n_scored"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
