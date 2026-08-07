import json
import sqlite3
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent

DB_PATH = BASE.parent / "sessions.db"
OUT_PATH = BASE / "ground_truth_template.csv"

def is_scoreable(sources):
    if not sources:
        return False

    return any(
        s.get("text", "").strip()
        for s in sources
    )


def main():

    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute(
        """
        SELECT question,
               answer,
               sources
        FROM chat_turns
        ORDER BY id
        """
    ).fetchall()

    conn.close()

    existing = {}

    if OUT_PATH.exists():

        old = pd.read_csv(OUT_PATH).fillna("")

        for _, row in old.iterrows():

            existing[row["question"]] = row.get(
                "ground_truth",
                ""
            )

    seen = set()

    output = []

    for question, answer, sources_json in rows:

        try:
            sources = json.loads(sources_json)
        except Exception:
            continue

        if not is_scoreable(sources):
            continue

        if question in seen:
            continue

        seen.add(question)

        contexts = [
            s["text"]
            for s in sources
            if s.get("text", "").strip()
        ]

        preview = " | ".join(
            c[:150].replace("\n", " ")
            for c in contexts[:2]
        )

        output.append(
            {
                "question": question,
                "current_answer": answer,
                "context_preview": preview,
                "ground_truth": existing.get(
                    question,
                    ""
                ),
                "contexts_json": json.dumps(contexts),
            }
        )

    df = pd.DataFrame(output)

    df.insert(
        0,
        "id",
        range(1, len(df) + 1)
    )

    df.to_csv(
        OUT_PATH,
        index=False
    )

    print()
    print(f"Saved {len(df)} questions")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()