"""LangSmith offline RAG evaluation, run from the project root (backend/).

Like scripts/run_ragas_eval.py, this runs the curated eval_dataset_osw.json
questions through the REAL retrieval + generation pipeline against whatever is
currently indexed (config.INDEX_DIR). The difference is where results go: this
script uploads the dataset to LangSmith (once) and reports scores through
LangSmith's `evaluate()`, so runs show up as experiments at smith.langchain.com
with per-example traces, instead of only a local JSON report.

Scoring is done by openevals' prebuilt, community-maintained RAG judge prompts
(faithfulness/groundedness, answer_relevancy/helpfulness, context_relevancy/
retrieval-relevance, answer_correctness), run through Groq via
create_llm_as_judge — NOT rag/evaluation.py's hand-written prompts, which the
live chat path and scripts/run_ragas_eval.py still use. The old evaluation.py-
based evaluators are kept below, commented out, in case you want to revert —
see the "OLD evaluators" section. Because of that split, a metric here is
scored differently than the same-named metric in the chat UI's live scores;
they're not directly comparable numbers.

Setup:
    1. pip install -r requirements.txt   (adds `langsmith` + `openevals`)
    2. Add to backend/.env:
           LANGSMITH_TRACING=true
           LANGSMITH_API_KEY=<your key from smith.langchain.com/settings>
           LANGSMITH_PROJECT=multirag-evaluation   (optional, defaults below)

Usage (run as a module from backend/, so `config`/`rag` resolve on sys.path):
    python -m scripts.run_langsmith_eval [path/to/dataset.json]

The dataset is uploaded to LangSmith under LANGSMITH_DATASET_NAME (default
"multirag-golden-set") the first time this runs; later runs reuse it as-is
(edit the dataset in the LangSmith UI, not by re-running this script, once it
exists — this script never mutates an existing dataset's examples).
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "multirag-evaluation")

import config
from rag import generator, reranker
# from rag import evaluation  # old approach — see commented evaluators below
from rag.vectorstore import VectorStore

from langsmith import Client
from langsmith.evaluation import evaluate

from openai import OpenAI
from openevals import prompts as openevals_prompts
from openevals.llm import create_llm_as_judge

# openevals judges via Groq's OpenAI-compatible endpoint. Only a few Groq
# models support the strict JSON-schema structured output openevals requires
# (see https://console.groq.com/docs/structured-outputs#supported-models) —
# llama-3.3-70b-versatile does NOT, so this uses one of the models that does.
OPENEVALS_MODEL = "openai/gpt-oss-120b"
_judge_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY_RAGAS") or os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(SCRIPT_DIR, "eval_dataset_osw.json")
DATASET_NAME = os.getenv("LANGSMITH_DATASET_NAME", "multirag-golden-set")

store: VectorStore | None = None  # set in main(), read by rag_target per example


def ensure_dataset(client: Client, dataset_path: str) -> str:
    """Sync eval_dataset_osw.json into LangSmith as DATASET_NAME.

    First run: creates the dataset and uploads every row. Later runs: any
    question already present (matched by exact question text) is left alone —
    edit an existing example's ground_truth in the LangSmith UI, not here —
    but new questions added to the JSON file get uploaded as new examples.

    IMPORTANT: this sync-from-local-JSON behavior only applies to the default
    dataset name ("multirag-golden-set", tied to eval_dataset_osw.json). If
    LANGSMITH_DATASET_NAME points at a DIFFERENT dataset (e.g. one you curated
    by hand in the LangSmith UI for a different indexed document), this
    function does NOT touch it — it just confirms it exists and leaves its
    examples exactly as they are. Otherwise this would "helpfully" dump
    eval_dataset_osw.json's questions (for whatever document THAT dataset was
    built from) into an unrelated dataset the moment they don't match.
    """
    if DATASET_NAME != "multirag-golden-set":
        if not client.has_dataset(dataset_name=DATASET_NAME):
            raise RuntimeError(
                f"LANGSMITH_DATASET_NAME={DATASET_NAME!r} does not exist in LangSmith — "
                "create it in the UI first (this script won't auto-populate a non-default dataset)."
            )
        return DATASET_NAME

    with open(dataset_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if not client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Curated question / ground-truth pairs for the Multi-RAG pipeline (from eval_dataset_osw.json).",
        )
        client.create_examples(
            inputs=[{"question": r["question"]} for r in rows],
            outputs=[{"ground_truth": r["ground_truth"]} for r in rows],
            dataset_id=dataset.id,
        )
        print(f"Uploaded {len(rows)} examples to LangSmith dataset '{DATASET_NAME}'")
        return DATASET_NAME

    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    existing_questions = {
        ex.inputs.get("question") for ex in client.list_examples(dataset_id=dataset.id)
    }
    new_rows = [r for r in rows if r["question"] not in existing_questions]
    if new_rows:
        client.create_examples(
            inputs=[{"question": r["question"]} for r in new_rows],
            outputs=[{"ground_truth": r["ground_truth"]} for r in new_rows],
            dataset_id=dataset.id,
        )
        print(f"Added {len(new_rows)} new example(s) to LangSmith dataset '{DATASET_NAME}'")
    else:
        print(f"No new examples — LangSmith dataset '{DATASET_NAME}' already up to date")
    return DATASET_NAME


def rag_target(inputs: dict) -> dict:
    """The pipeline under test: real retrieval + reranking + generation."""
    question = inputs["question"]
    hits = store.search(question, top_k=config.TOP_K * 3)
    if config.RERANK_ENABLED:
        hits = reranker.rerank(question, hits)
    hits = hits[: config.TOP_K]
    contexts = [doc.text for doc, _ in hits if doc.text]
    answer_text = "".join(generator.answer(question, hits))
    return {"answer": answer_text, "context": contexts}


# --------------------------------------------------------------------------- #
# OLD evaluators — wrapped rag/evaluation.py's hand-written Groq judge prompts.
# STALE since the DeepEval migration: rag/evaluation.py no longer exposes
# per-metric functions (faithfulness(), answer_relevancy(), ...), only
# evaluate()/evaluate_with_ground_truth() returning a dict. Uncommenting this
# block as-is will raise AttributeError — kept only as a historical record of
# what the hand-written prompts looked like, not a working revert path.
# --------------------------------------------------------------------------- #
# def faithfulness_evaluator(run, example):
#     answer = run.outputs.get("answer", "")
#     context_blob = "\n\n".join(run.outputs.get("context", []))
#     return {"key": "faithfulness", "score": evaluation.faithfulness(answer, context_blob)}
#
#
# def answer_relevancy_evaluator(run, example):
#     question = example.inputs.get("question", "")
#     answer = run.outputs.get("answer", "")
#     return {"key": "answer_relevancy", "score": evaluation.answer_relevancy(question, answer)}
#
#
# def context_precision_evaluator(run, example):
#     question = example.inputs.get("question", "")
#     answer = run.outputs.get("answer", "")
#     contexts = run.outputs.get("context", [])
#     return {"key": "context_precision", "score": evaluation.context_precision(question, answer, contexts)}
#
#
# def context_relevancy_evaluator(run, example):
#     question = example.inputs.get("question", "")
#     contexts = run.outputs.get("context", [])
#     return {"key": "context_relevancy", "score": evaluation.context_relevancy(question, contexts)}
#
#
# def context_recall_evaluator(run, example):
#     ground_truth = example.outputs.get("ground_truth", "")
#     contexts = run.outputs.get("context", [])
#     return {"key": "context_recall", "score": evaluation.context_recall(ground_truth, contexts)}
#
#
# def answer_correctness_evaluator(run, example):
#     ground_truth = example.outputs.get("ground_truth", "")
#     answer = run.outputs.get("answer", "")
#     return {"key": "answer_correctness", "score": evaluation.answer_correctness(answer, ground_truth)}


# --------------------------------------------------------------------------- #
# NEW evaluators — openevals' prebuilt, community-maintained RAG judge prompts
# instead of our own hand-written ones, run through Groq via create_llm_as_judge.
#
# openevals has no built-in equivalent of our context_precision (rank-aware
# average precision) or context_recall (ground-truth claim coverage), so this
# covers 4 metrics, not 6 — context_precision/context_recall are simply dropped
# here rather than faked with a mismatched prompt. Uncomment the OLD block
# above to get all 6 back.
# --------------------------------------------------------------------------- #
_groundedness_judge = create_llm_as_judge(
    prompt=openevals_prompts.RAG_GROUNDEDNESS_PROMPT,
    feedback_key="faithfulness",
    judge=_judge_client,
    model=OPENEVALS_MODEL,
    continuous=True,
)
_helpfulness_judge = create_llm_as_judge(
    prompt=openevals_prompts.RAG_HELPFULNESS_PROMPT,
    feedback_key="answer_relevancy",
    judge=_judge_client,
    model=OPENEVALS_MODEL,
    continuous=True,
)
_retrieval_relevance_judge = create_llm_as_judge(
    prompt=openevals_prompts.RAG_RETRIEVAL_RELEVANCE_PROMPT,
    feedback_key="context_relevancy",
    judge=_judge_client,
    model=OPENEVALS_MODEL,
    continuous=True,
)
_correctness_judge = create_llm_as_judge(
    prompt=openevals_prompts.CORRECTNESS_PROMPT,
    feedback_key="answer_correctness",
    judge=_judge_client,
    model=OPENEVALS_MODEL,
    continuous=True,
)


def faithfulness_evaluator_openevals(run, example):
    context_blob = "\n\n".join(run.outputs.get("context", []))
    return _groundedness_judge(outputs=run.outputs.get("answer", ""), context=context_blob)


def answer_relevancy_evaluator_openevals(run, example):
    return _helpfulness_judge(inputs=example.inputs.get("question", ""), outputs=run.outputs.get("answer", ""))


def context_relevancy_evaluator_openevals(run, example):
    context_blob = "\n\n".join(run.outputs.get("context", []))
    return _retrieval_relevance_judge(inputs=example.inputs.get("question", ""), context=context_blob)


def answer_correctness_evaluator_openevals(run, example):
    return _correctness_judge(
        inputs=example.inputs.get("question", ""),
        outputs=run.outputs.get("answer", ""),
        reference_outputs=example.outputs.get("ground_truth", ""),
    )


def main(dataset_path: str) -> None:
    global store

    client = Client()
    dataset_name = ensure_dataset(client, dataset_path)

    store = VectorStore(dirpath=config.INDEX_DIR)
    print(f"Connected to vector store (backend: {store.backend}, {store.size} chunks indexed)\n")

    evaluate(
        rag_target,
        data=dataset_name,
        evaluators=[
            faithfulness_evaluator_openevals,
            answer_relevancy_evaluator_openevals,
            context_relevancy_evaluator_openevals,
            answer_correctness_evaluator_openevals,
        ],
        experiment_prefix="multirag-rag-metrics",
    )
    print("Evaluation complete — view results at https://smith.langchain.com")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET
    main(path)
