import csv
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from deepeval.test_case.llm_test_case import LLMTestCase, RetrievedContextData

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR.parent / "storage" / "sessions.db"
GROUND_TRUTH_CSV = BASE_DIR / "ground_truth_template.csv"


@dataclass
class QARecord:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    session_id: Optional[str] = None
    chat_turn_id: Optional[int] = None


def load_ground_truth_map() -> Dict[str, str]:
    if not GROUND_TRUTH_CSV.exists():
        return {}
    truth_map: Dict[str, str] = {}
    with open(GROUND_TRUTH_CSV, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            question = row.get("question", "")
            ground_truth = row.get("ground_truth", "")
            if question and ground_truth and ground_truth.strip():
                truth_map[question.strip()] = ground_truth.strip()
    return truth_map


def is_scoreable(sources: List[dict]) -> bool:
    if not sources:
        return False
    return any(s.get("text", "").strip() for s in sources)


def load_sessions_records() -> List[QARecord]:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Could not find sessions.db at {DB_PATH}")

    truth_map = load_ground_truth_map()
    records: List[QARecord] = []

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT rowid, session_id, question, answer, sources FROM chat_turns ORDER BY rowid"
    )
    for row in cursor.fetchall():
        rowid, session_id, question, answer, sources_json = row
        try:
            sources = json.loads(sources_json)
        except Exception:
            continue
        if not is_scoreable(sources):
            continue
        contexts = [s["text"] for s in sources if s.get("text", "").strip()]
        if not contexts:
            continue
        records.append(
            QARecord(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=truth_map.get(question.strip()),
                session_id=session_id,
                chat_turn_id=rowid,
            )
        )
    conn.close()
    return records


def to_deepeval_cases(records: List[QARecord]) -> List[LLMTestCase]:
    cases: List[LLMTestCase] = []
    for idx, record in enumerate(records, start=1):
        retrieval_context = [text for text in record.contexts]
        metadata = {
            "session_id": record.session_id,
            "chat_turn_id": record.chat_turn_id,
            "retrieved_context_sources": [f"source_{i+1}" for i, _ in enumerate(record.contexts)],
        }
        if record.ground_truth:
            metadata["ground_truth"] = record.ground_truth

        case = LLMTestCase(
            input=record.question,
            actual_output=record.answer,
            expected_output=record.ground_truth or record.answer,
            retrieval_context=retrieval_context,
            metadata=metadata,
            comments="Generated from sessions.db and evaluation/ground_truth_template.csv",
            name=f"turn_{record.chat_turn_id}",
        )
        cases.append(case)
    return cases


if __name__ == "__main__":
    records = load_sessions_records()
    print(f"Loaded {len(records)} scoreable records from {DB_PATH}")
    if records:
        print(records[0])
