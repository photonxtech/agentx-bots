from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel
from langsmith.run_helpers import tracing_context
from backend.deepeval_metrics import evaluate_deepeval
from backend.ground_truth_lookup import find_ground_truth
from backend.gt_metrics import answer_correctness
from langsmith_logging import start_chat_turn, log_generation, end_chat_turn, attach_feedback

from rag_utils import (
    build_hybrid_retriever,
    get_answer,
    session_exists,
    get_pdf_names,
    get_chat_history,
    append_chat_turn,
    get_langsmith_run_id,
)

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    question: str


# Metrics are NOT computed automatically on every chat turn — that was
# firing concurrent Groq judge calls on every single message, which caused
# repeated rate-limit storms. Metrics are opt-in: the frontend shows a
# "Calculate Metrics" button per response, which calls
# POST /metrics/calculate below only when the user actually wants the score
# for that specific answer.
#
# /metrics/calculate now runs ONLY DeepEval (backend.metrics' custom
# RAGAS-style pipeline has been dropped from the live path entirely — it
# was previously running CONCURRENTLY alongside DeepEval here too, which
# defeated the point of making metrics opt-in: two judge pipelines instead
# of one still doubles Groq usage every time the button is pressed, and
# that double usage is what burned through the daily token quota. DeepEval
# alone now produces up to 6 metrics — faithfulness, answer_relevancy,
# context_relevancy, context_precision, context_recall, and (when a real
# ground-truth match is found) answer_correctness. See deepeval_metrics.py
# for how context_precision/context_recall are approximated when no ground
# truth exists.
class MetricsRequest(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]] = []
    turn_id: int | None = None


@router.post("/chat")
async def chat(request: ChatRequest):
    if not session_exists(request.session_id):
        return {"error": "Invalid session ID"}

    if not get_pdf_names(request.session_id):
        return {"error": "No PDFs attached to this session yet"}

    run = start_chat_turn(chat_id=request.session_id, question=request.question)

    with tracing_context(enabled=False):
        # build_hybrid_retriever() now returns (compression_retriever,
        # ensemble_retriever) instead of just the compressed retriever —
        # the second value is the RAW pre-rerank retriever, used below only
        # for the temporary [retrieval diagnostic] print inside get_answer().
        retriever, ensemble_retriever = build_hybrid_retriever(request.session_id)

    chat_history = get_chat_history(request.session_id)

    with tracing_context(parent=run):
        answer, sources, token_usage = get_answer(
            retriever,
            request.question,
            chat_history,
            session_id=request.session_id,
            ensemble_retriever=ensemble_retriever,
        )

    print("\n===== SOURCES RETURNED FROM get_answer =====")
    print("Number of sources:", len(sources))

    for i, s in enumerate(sources):
        print("\nSOURCE", i)
        print("Keys:", s.keys())
        print("Text length:", len(s.get("text", "")))
        print("Text preview:", s.get("text", "")[:300])

    print("===========================================\n")

    # Contexts are still extracted (free — local string work, no Groq calls)
    # so the LangSmith trace below still records what was retrieved for this
    # turn, even though we no longer score it inline.
    contexts = [s["text"] for s in sources if s.get("text", "").strip()]

    # No live judge calls here anymore. Metrics stay empty until the user
    # presses "Calculate Metrics" in the UI, which hits /metrics/calculate.
    metrics: Dict[str, Any] = {}

    end_chat_turn(run, answer=answer, contexts=contexts, metrics=metrics)

    turn_id = append_chat_turn(
        request.session_id,
        request.question,
        answer,
        sources,
        metrics,
        langsmith_run_id=(str(run.id) if run is not None else None),
    )

    print("\n===== RESPONSE SENT TO FRONTEND =====")
    print({
        "answer": answer,
        "sources": sources,
        "token_usage": token_usage,
        "metrics": metrics,
        "turn_id": turn_id,
    })
    print("=====================================\n")

    return {
        "answer": answer,
        "sources": sources,
        "token_usage": token_usage,
        "metrics": metrics,
        "turn_id": turn_id,
    }


@router.post("/metrics/calculate")
async def calculate_metrics(request: MetricsRequest):
    """Computes up to 6 metrics via DeepEval ONLY, for ONE specific
    (question, answer, sources) triple, on demand. This is the
    button-triggered replacement for the metrics block that used to run
    automatically inside /chat on every turn.

    Only one judge pipeline runs here now — previously this endpoint ran
    the custom backend.metrics pipeline AND DeepEval concurrently, which
    doubled Groq token usage on every button press and was the direct
    cause of exhausting the daily TPD quota on gpt-oss-120b.
    """
    contexts = [s.get("text", "") for s in request.sources if s.get("text", "").strip()]

    if not contexts:
        return {"metrics": {}}

    # Ground truth is looked up FIRST so it can be passed into
    # evaluate_deepeval() — that's what lets context_precision/context_recall
    # use a real reference answer when one exists, instead of falling back
    # to the (weaker) proxy of the answer scoring itself. See
    # deepeval_metrics.py's module docstring for what the proxy means.
    gt_match = find_ground_truth(request.question)
    ground_truth = None
    if gt_match:
        matched_question, ground_truth, similarity = gt_match
        print(f"\n[ground truth] matched \"{matched_question}\" (sim={similarity})")
    else:
        print("\n[ground truth] no match — context_precision/recall will use the answer itself as a proxy reference")

    metrics = evaluate_deepeval(request.question, request.answer, contexts, ground_truth=ground_truth)

    # answer_correctness is never approximated — it only gets computed when
    # a real ground-truth match exists, since comparing the answer to
    # itself would trivially score ~1.0 and carry no signal.
    if ground_truth:
        metrics["answer_correctness"] = answer_correctness(request.answer, ground_truth)

    print("\n===== ON-DEMAND METRICS (deepeval-only, up to 6) =====")
    print({"metrics": metrics})
    print("========================================================\n")

    # Attach these scores as LangSmith feedback on the original chat_turn
    # trace, so they show up in the Feedback panel there instead of only
    # ever living in the local SQLite chat_turns table. Only possible when
    # this turn was created with LangSmith tracing enabled AND the frontend
    # sent back the turn_id it belongs to.
    if request.turn_id is not None:
        run_id = get_langsmith_run_id(request.turn_id)
        attach_feedback(run_id, metrics)

    return {"metrics": metrics}