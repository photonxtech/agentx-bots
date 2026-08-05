"""
Combines custom_ragas_results.csv (4 reference-free metrics, 109 rows) and
custom_ragas_ground_truth_results.csv (2 ground-truth metrics, 22 rows) into
one file with all 6 metrics per question — plus a summary file with the
aggregate scores you can paste straight into a report or message.

Run this in the same backend/ folder as those two CSVs.
"""

import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

REFFREE_PATH = BASE / "custom_ragas_results.csv"
GT_PATH = BASE / "custom_ragas_ground_truth_results.csv"

OUT_COMBINED = BASE / "combined_ragas_metrics.csv"
OUT_SUMMARY = BASE / "combined_ragas_summary.csv"


def main():
    if not REFFREE_PATH.exists():
        raise SystemExit(f"Missing {REFFREE_PATH}")
    if not GT_PATH.exists():
        raise SystemExit(f"Missing {GT_PATH}")

    reffree = pd.read_csv(REFFREE_PATH)
    gt = pd.read_csv(GT_PATH)

    # Left join on 'question': every one of the 109 questions is kept, and the
    # 22 that also have ground-truth scores get those extra 2 columns filled
    # in; the rest show blank/NaN for context_recall and answer_correctness.
    combined = reffree.merge(gt, on="question", how="left")

    metric_cols = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_relevancy",
        "context_recall",
        "answer_correctness",
    ]
    cols = ["question"] + [c for c in metric_cols if c in combined.columns]
    combined = combined[cols]

    combined.to_csv(OUT_COMBINED, index=False)
    print(f"Combined per-question file saved to {OUT_COMBINED} ({len(combined)} rows).")

    # Summary — aggregate mean + sample size per metric
    summary_rows = []
    for col in metric_cols:
        if col in combined.columns:
            n = combined[col].notna().sum()
            mean = combined[col].mean()
            summary_rows.append({"metric": col, "mean_score": round(mean, 3), "n": n})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    print(f"Summary saved to {OUT_SUMMARY}.\n")
    print("=== Final 6-metric summary ===")
    for row in summary_rows:
        print(f"{row['metric']}: {row['mean_score']}  (n={row['n']})")


if __name__ == "__main__":
    main()
