"""
RAG evaluation routes — run config + full evaluation pipeline + save results.
"""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession, joinedload
from backend.database import get_db
from backend.models import Session, RunConfig, RunResult
from backend.schemas import EvaluateRequest, RunConfigOut
from backend.services.rag_evaluator import call_rag_api
from backend.services.document_reader import document_paths

router = APIRouter(prefix="/api/sessions/{session_id}", tags=["evaluate"])


@router.post("/evaluate", response_model=RunConfigOut)
async def run_evaluation(session_id: UUID, payload: EvaluateRequest, db: DBSession = Depends(get_db)):
    """
    Full pipeline:
      1. Save run config to Postgres
      2. Run the config-driven RAG pipeline → generated answers + contexts
      3. Score the whole batch with DeepEval → batch metrics
      4. LangSmith → log runs + feedback, get URL
      5. Save batch metrics + per-case rows to Postgres
      6. Return full config + results
    """
    session = db.query(Session).filter(Session.id == str(session_id)).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Build the combined test set: seed (user-uploaded) pairs first, then
    # system-generated ones.  A session is evaluable when it has either kind —
    # seed-only is fine if the user uploads pairs without ever running generation.
    seed_pairs: list[dict] = list(session.seed_qa_json or [])
    generated_pairs: list[dict] = list(session.qa_json or [])
    combined_qa: list[dict] = seed_pairs + generated_pairs

    if not combined_qa:
        raise HTTPException(
            status_code=400,
            detail="No Q&A data available yet — upload seed Q&A or generate a test set first (Step 2)"
        )

    # Ensure every pair carries a source tag so the result rows are attributable.
    for pair in seed_pairs:
        pair.setdefault("source", "seed")
    for pair in generated_pairs:
        pair.setdefault("source", "generated")

    # 1. Save run config
    run_config = RunConfig(
        session_id=session_id,
        chat_model=payload.chat_model,
        embedding_model=payload.embedding_model,
        vectordb=payload.vectordb,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
        reranker_model=payload.reranker_model,
        search_type=payload.search_type,
        index_type=payload.index_type,
        top_k=payload.top_k,
        temperature=payload.temperature,
    )
    db.add(run_config)
    db.flush()  # Get ID before calling evaluation

    # 2–4. Run full evaluation pipeline
    try:
        api_results, batch_metrics, langsmith_url = await call_rag_api(
            config=payload,
            qa_pairs=combined_qa,
            document_paths=document_paths(session),
            session_id=str(session_id),
            run_config_id=str(run_config.id),
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=502, detail=f"Evaluation pipeline failed: {str(e)}")

    # 5. Save LangSmith URL + batch-level metrics on the run config
    if langsmith_url:
        run_config.langsmith_experiment_url = langsmith_url
    run_config.metrics = batch_metrics
    # Kept so human feedback can be attached to the same LangSmith run that
    # carries the automated scores.
    run_config.langsmith_summary_run_id = (batch_metrics.get("_meta") or {}).get(
        "langsmith_summary_run_id"
    )

    # 7. Save each result row
    # api_results are returned in the same order as combined_qa, so we can
    # zip them to recover the source tag for each pair.
    for result, qa_pair in zip(api_results, combined_qa):
        # Merge the source tag into the per-case metrics so it is persisted
        # and visible in the UI without a separate DB column.
        per_case_metrics = dict(result.get("metrics") or {})
        per_case_metrics["_source"] = qa_pair.get("source", "generated")
        run_result = RunResult(
            run_config_id=run_config.id,
            question=result.get("question", ""),
            generated_answer=result.get("generated_answer", ""),
            expected_answer=result.get("expected_answer", ""),
            metrics=per_case_metrics,
        )
        db.add(run_result)


    db.commit()

    # 8. Reload with results and return
    loaded = (
        db.query(RunConfig)
        .options(joinedload(RunConfig.results), joinedload(RunConfig.feedback))
        .filter(RunConfig.id == run_config.id)
        .first()
    )
    return loaded


@router.get("/runs", response_model=list[RunConfigOut])
def get_runs(session_id: UUID, db: DBSession = Depends(get_db)):
    """Get all past run configs + their results for this session."""
    runs = (
        db.query(RunConfig)
        .options(joinedload(RunConfig.results), joinedload(RunConfig.feedback))
        .filter(RunConfig.session_id == session_id)
        .order_by(RunConfig.created_at)
        .all()
    )
    return runs
