"""
main.py
FastAPI backend for the WeNext RAG demo.

Run:
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000 in a browser.
LLM inference runs locally via Ollama (http://localhost:11434) — make sure
`ollama serve` is running and the model in rag_pipeline.py has been pulled.
"""

import json
import os
import sys
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import WeNextRAG, OLLAMA_BASE_URL, OLLAMA_HOST, DEFAULT_MODEL

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))
from eval_utils import (  # noqa: E402
    build_judge_model,
    build_full_metrics,
    build_reference_free_metrics,
    find_golden_match,
    load_golden_dataset,
)
from deepeval import evaluate as deepeval_evaluate  # noqa: E402
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

EVAL_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval", "results.json")

app = FastAPI(title="WeNext AI Features RAG Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = None  # lazy-loaded on first request so the server starts instantly


class QueryRequest(BaseModel):
    question: str
    model: Optional[str] = None


class SourceIn(BaseModel):
    title: str
    text: str


class EvaluateRequest(BaseModel):
    question: str
    answer: str
    sources: List[SourceIn] = []


@app.on_event("startup")
def check_ollama():
    try:
        httpx.get(OLLAMA_BASE_URL.replace("/v1", ""), timeout=2.0)
    except httpx.HTTPError:
        print(f"WARNING: Could not reach Ollama at {OLLAMA_BASE_URL}. Is `ollama serve` running?")


def get_rag() -> WeNextRAG:
    global rag
    if rag is None:
        rag = WeNextRAG()
    return rag


@app.post("/query")
def query(req: QueryRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        result = get_rag().answer(req.question.strip(), model=req.model or DEFAULT_MODEL)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
def evaluate_answer(req: EvaluateRequest):
    """
    Scores a single live chat Q&A with DeepEval, on demand (this is slow: several
    judge-LLM calls on a local model, roughly 1-2 minutes).

    If the question closely matches one of the 15 hand-written golden questions
    (by embedding similarity), that golden entry's ground-truth answer is used so
    all 6 metrics can run, same as the offline eval. Ground truth doesn't exist for
    arbitrary chat questions, so unmatched questions only get the 4 metrics that
    don't require a ground-truth expected answer.
    """
    if not req.question.strip() or not req.answer.strip():
        raise HTTPException(status_code=400, detail="question and answer must not be empty")

    rag = get_rag()
    retrieval_context = [s.text for s in req.sources]

    golden = load_golden_dataset()
    match = find_golden_match(req.question, golden, rag.embed_model)

    if match:
        test_case = LLMTestCase(
            input=req.question,
            actual_output=req.answer,
            expected_output=match["expected_answer"],
            retrieval_context=retrieval_context,
            context=match["expected_context"],
        )
        judge = build_judge_model(DEFAULT_MODEL)
        metrics = build_full_metrics(judge)
    else:
        # No ground truth available: use the retrieved context itself as the
        # Hallucination reference, since that metric requires a `context` field.
        test_case = LLMTestCase(
            input=req.question,
            actual_output=req.answer,
            retrieval_context=retrieval_context,
            context=retrieval_context,
        )
        judge = build_judge_model(DEFAULT_MODEL)
        metrics = build_reference_free_metrics(judge)

    try:
        eval_result = deepeval_evaluate(
            [test_case],
            metrics=metrics,
            async_config=AsyncConfig(run_async=True, max_concurrent=1),
            display_config=DisplayConfig(show_indicator=False, print_results=False),
            cache_config=CacheConfig(write_cache=False, use_cache=False),
            error_config=ErrorConfig(ignore_errors=True),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    metrics_out = {}
    for md in eval_result.test_results[0].metrics_data:
        metrics_out[md.name] = {
            "score": md.score,
            "success": md.success,
            "reason": md.reason,
            "error": md.error,
        }

    return {
        "matched_golden": {"module": match["module"], "question": match["question"]} if match else None,
        "metrics": metrics_out,
    }


@app.get("/models")
def list_models():
    try:
        resp = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return {"models": models, "default": DEFAULT_MODEL}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Ollama: {e}")


@app.get("/eval-results")
def eval_results():
    if not os.path.exists(EVAL_RESULTS_PATH):
        raise HTTPException(status_code=404, detail="No eval results found. Run eval/run_eval.py first.")
    with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/health")
def health():
    try:
        httpx.get(OLLAMA_BASE_URL.replace("/v1", ""), timeout=2.0)
        ollama_reachable = True
    except httpx.HTTPError:
        ollama_reachable = False
    return {"status": "ok", "ollama_reachable": ollama_reachable}


FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

# Serve the frontend
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
