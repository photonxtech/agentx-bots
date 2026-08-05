"""
RAGAS evaluation for the WeNext AI PDF Copilot RAG pipeline.

Pulls REAL production Q&A pairs straight out of sessions.db (already has
question, answer, and the retrieved chunks that were used) and scores them
on 3 reference-free RAGAS metrics — no manually labeled ground truth needed:

  - faithfulness        : is the answer actually supported by the retrieved
                           context, or is it hallucinating?
  - answer_relevancy     : does the answer actually address the question asked?
  - context_utilization  : are the retrieved chunks relevant/useful, and are
                            the useful ones ranked near the top? (reference-free
                            stand-in for context_precision, which needs
                            ground-truth answers you don't have yet)

Run this from inside your backend/ folder (same folder as sessions.db and
rag_utils.py) using the ISOLATED venv described in the setup instructions —
do NOT run this inside your app's main venv, it will downgrade langchain-core
and break your app.
"""

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, ContextUtilization
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(__file__).parent / "sessions.db"

# Turns to exclude, because they have no real retrieved-context text and
# would produce meaningless/zero scores rather than an honest signal:
#   - whole-file "summarize/explain this pdf" answers -> sources have
#     page == "full document" and text == "" (see get_answer() in rag_utils.py)
#   - the out-of-scope fallback -> sources == [] entirely
#     ("That doesn't seem to be covered in your uploaded document(s).")
def is_scoreable(sources):
    if not sources:
        return False
    return any(s.get("text", "").strip() for s in sources)


def load_real_qa_pairs():
    if not DB_PATH.exists():
        sys.exit(
            f"Couldn't find sessions.db at {DB_PATH}. Put this script in the "
            f"SAME folder as your real sessions.db (your backend/ folder), "
            f"not a copy elsewhere."
        )

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT question, answer, sources FROM chat_turns ORDER BY id"
    ).fetchall()
    conn.close()

    records = []
    skipped = 0
    for question, answer, sources_json in rows:
        sources = json.loads(sources_json)
        if not is_scoreable(sources):
            skipped += 1
            continue
        contexts = [s["text"] for s in sources if s.get("text", "").strip()]
        records.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
        })

    print(f"Loaded {len(records)} scoreable Q&A pairs from sessions.db "
          f"(skipped {skipped} whole-file-summary / out-of-scope turns).")
    return records


def main():
    records = load_real_qa_pairs()
    if not records:
        sys.exit("No scoreable Q&A pairs found — nothing to evaluate.")

    dataset = EvaluationDataset.from_list(records)

    # Same judge model your app already uses for generation — keep this in
    # mind when comparing against peers who might've judged with GPT-4o etc.
    judge_llm = LangchainLLMWrapper(ChatGroq(model="llama-3.1-8b-instant", temperature=0))    # Same embedding model your retriever already uses
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, ContextUtilization()],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=1, max_retries=8, max_wait=90),
    )

    print("\n=== Aggregate RAGAS scores ===")
    print(result)

    df = result.to_pandas()
    df.to_csv("ragas_results.csv", index=False)
    print(f"\nPer-question breakdown saved to ragas_results.csv ({len(df)} rows).")

    print("\n=== Summary (paste this to your manager) ===")
    for col in ["faithfulness", "answer_relevancy", "context_utilization"]:
        if col in df.columns:
            print(f"{col}: {df[col].mean():.3f}  (n={df[col].notna().sum()})")


if __name__ == "__main__":
    main()
