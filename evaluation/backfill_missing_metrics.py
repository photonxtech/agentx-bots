"""
Targeted metric backfill — scores ONLY chat_turns rows that are missing
metrics (faithfulness/answer_relevancy/context_precision/context_relevancy
IS NULL), instead of re-scoring your entire sessions.db like
eval_custom_ragas.py does.

Rows end up with NULL metrics for two reasons, both worth catching here:
  1. Live scoring failed at insert time (Groq 429/quota, malformed judge
     JSON, etc.) — see backend/routes/chat.py -> backend/metrics.py.
  2. Historical rows created before live scoring existed, or before the
     text:"" bug in the broad-summary path was fixed.

Run this any time you suspect gaps (e.g. after a batch of testing where you
saw "Faithfulness: None" in the server logs), instead of running the full
eval_custom_ragas.py batch. Much cheaper on Groq quota since it typically
touches a handful of rows, not your whole table.

Run in the SAME ragas_env venv as eval_custom_ragas.py, from the same
directory (backend/), since it imports functions directly from that file.
"""

import sqlite3

from eval_custom_ragas import (
    DB_PATH,
    evaluate_one,
    is_scoreable,
    is_broad_summary_question,
    is_fallback_answer,
)
import json


def find_incomplete_rows():
    """Rows missing at least one metric, that are actually scoreable and
    not a fallback ('I don't know') answer — same filtering logic as the
    full eval script, just scoped to NULL rows only."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, question, answer, sources FROM chat_turns "
        "WHERE faithfulness IS NULL "
        "   OR answer_relevancy IS NULL "
        "   OR context_precision IS NULL "
        "   OR context_relevancy IS NULL "
        "ORDER BY id"
    ).fetchall()
    conn.close()

    candidates = []
    skipped_unscoreable = 0
    skipped_fallback = 0

    for row_id, question, answer, sources_json in rows:
        try:
            sources = json.loads(sources_json)
        except Exception:
            continue

        if not is_scoreable(sources):
            skipped_unscoreable += 1
            continue

        if is_fallback_answer(answer):
            skipped_fallback += 1
            continue

        contexts = [s["text"] for s in sources if s.get("text", "").strip()]
        candidates.append(
            {
                "id": row_id,
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "is_broad": is_broad_summary_question(question),
            }
        )

    print(f"NULL-metric rows found : {len(rows)}")
    print(f"  -> unscoreable (skipped)     : {skipped_unscoreable}")
    print(f"  -> fallback answer (skipped) : {skipped_fallback}")
    print(f"  -> will be re-scored         : {len(candidates)}")

    return candidates


def update_row_metrics(row_id, scores):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE chat_turns SET faithfulness = ?, answer_relevancy = ?, "
        "context_precision = ?, context_relevancy = ? WHERE id = ?",
        (
            scores["faithfulness"],
            scores["answer_relevancy"],
            scores["context_precision"],
            scores["context_relevancy"],
            row_id,
        ),
    )
    conn.commit()
    conn.close()


def main():
    candidates = find_incomplete_rows()
    if not candidates:
        print("\nNothing to backfill — every scoreable row already has metrics.")
        return

    print(f"\nRe-scoring {len(candidates)} rows...\n")

    for i, c in enumerate(candidates, 1):
        scores = evaluate_one(c["question"], c["answer"], c["contexts"])
        update_row_metrics(c["id"], scores)
        tag = " [broad-summary]" if c["is_broad"] else ""
        print(
            f"[{i}/{len(candidates)}] id={c['id']}{tag} "
            f"faith={scores['faithfulness']} rel={scores['answer_relevancy']} "
            f"prec={scores['context_precision']} ctxrel={scores['context_relevancy']}"
        )

    print(f"\nDone. Updated {len(candidates)} rows in {DB_PATH}.")


if __name__ == "__main__":
    main()
