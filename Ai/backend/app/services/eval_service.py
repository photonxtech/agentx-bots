import time
from datetime import datetime

from sqlalchemy.orm import Session

from app.database.models import EvalRun, Website
from app.llm.llm_client import generate_structured
from app.llm.prompts import CHAT_RESPONSE_SCHEMA, build_messages, build_no_context_messages
from app.llm.query_normalizer import correct_query_spelling
from app.llm.retrieval import NO_ANSWER_MESSAGE, compute_confidence, retrieve_relevant_chunks
from app.utils.logging import get_logger

logger = get_logger(__name__)


async def _generate_answer(website_id: int, website_name: str, question: str) -> tuple[str, float]:
    corrected_question = correct_query_spelling(website_id, question)
    chunks = retrieve_relevant_chunks(website_id, question, use_cache=False, website_name=website_name)
    if not chunks:
        messages = build_no_context_messages(corrected_question, website_name)
        confidence = 0.0
    else:
        messages = build_messages(corrected_question, chunks)
        confidence = compute_confidence(chunks)

    result = await generate_structured(messages, CHAT_RESPONSE_SCHEMA)
    answer = (result.get("answer") or "").strip()
    return answer, confidence


def _judge_case(question: str, answer: str, confidence: float, expected_keywords: list[str], expect_no_answer: bool) -> dict:
    is_fallback = NO_ANSWER_MESSAGE.lower() in answer.lower()

    if expect_no_answer:
        # This case is deliberately unanswerable from the site's content — the correct
        # behavior IS the fallback line, and answering anyway is a hallucination.
        passed = is_fallback
        hallucinated = not is_fallback
    elif expected_keywords:
        passed = any(kw.lower() in answer.lower() for kw in expected_keywords)
        # Only count as hallucination if it confidently answered something WRONG rather
        # than admitting it didn't know — a fallback on a genuinely answerable question is
        # a miss (recall problem), not a hallucination.
        hallucinated = (not passed) and (not is_fallback)
    else:
        passed = not is_fallback
        hallucinated = False

    return {
        "question": question,
        "answer": answer,
        "confidence": confidence,
        "passed": passed,
        "hallucinated": hallucinated,
    }


async def run_evaluation(db: Session, website_id: int, cases: list[dict]) -> EvalRun:
    website = db.query(Website).filter(Website.id == website_id).first()
    website_name = website.name if website else "this website"

    run = EvalRun(website_id=website_id, total_cases=len(cases))
    db.add(run)
    db.commit()
    db.refresh(run)

    results = []
    total_confidence = 0.0
    total_latency_ms = 0.0
    passed_count = 0
    hallucination_count = 0

    for case in cases:
        question = case["question"]
        expected_keywords = case.get("expected_keywords", [])
        expect_no_answer = case.get("expect_no_answer", False)

        started = time.perf_counter()
        try:
            answer, confidence = await _generate_answer(website_id, website_name, question)
        except Exception:
            logger.exception("Eval case failed for website %s: %r", website_id, question)
            answer, confidence = "", 0.0
        latency_ms = round((time.perf_counter() - started) * 1000, 1)

        result = _judge_case(question, answer, confidence, expected_keywords, expect_no_answer)
        result["latency_ms"] = latency_ms
        results.append(result)

        total_confidence += confidence
        total_latency_ms += latency_ms
        if result["passed"]:
            passed_count += 1
        if result["hallucinated"]:
            hallucination_count += 1

    n = len(cases) or 1
    run.passed_cases = passed_count
    run.hallucination_count = hallucination_count
    run.avg_confidence = round(total_confidence / n, 1)
    run.avg_latency_ms = round(total_latency_ms / n, 1)
    run.details = results
    run.finished_at = datetime.utcnow()
    db.commit()
    db.refresh(run)
    return run
