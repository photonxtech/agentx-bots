import os
import re
import uuid
import json
import time
import io
import base64
import hashlib
import asyncio
import threading
import sqlite3
from typing import Any, List, Optional, Dict
from dotenv import load_dotenv

load_dotenv()

try:
    from langsmith import Client as LangSmithClient, trace as langsmith_trace
    LANGSMITH_AVAILABLE = True
except ImportError:
    LangSmithClient = None
    langsmith_trace = None
    LANGSMITH_AVAILABLE = False

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").strip().lower() in ("1", "true", "yes", "on")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "DocQnA-Chat")
_LANGSMITH_CLIENT = None

def get_langsmith_client():
    global _LANGSMITH_CLIENT
    if not LANGSMITH_TRACING or not os.getenv("LANGSMITH_API_KEY") or not LANGSMITH_AVAILABLE:
        return None
    if _LANGSMITH_CLIENT is None:
        _LANGSMITH_CLIENT = LangSmithClient()
    return _LANGSMITH_CLIENT


def save_golden_dataset_to_langsmith(session_id: str, golden_items: List[dict]) -> Optional[str]:
    """Stores the session's combined DeepEval golden dataset in one LangSmith
    dataset. Regeneration replaces the existing examples for that session,
    while preserving the same dataset ID. Each example also keeps source
    document metadata so multi-document goldens remain traceable."""
    if not LANGSMITH_AVAILABLE or not os.getenv("LANGSMITH_API_KEY"):
        print("[LangSmith] Golden dataset upload skipped: LangSmith is not configured.")
        return None

    if not golden_items:
        return None

    try:
        client = LangSmithClient()
        conn = get_db_connection()
        row = conn.execute(
            "SELECT langsmith_dataset_id FROM sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        conn.close()

        dataset_id = row["langsmith_dataset_id"] if row and row["langsmith_dataset_id"] else None

        if dataset_id:
            # Regeneration in the same session replaces the previous golden
            # examples while keeping the same LangSmith dataset for that session.
            existing_examples = list(client.list_examples(dataset_id=dataset_id))
            if existing_examples:
                client.delete_examples(example_ids=[str(example.id) for example in existing_examples])
        else:
            dataset = client.create_dataset(
                dataset_name=f"QA_Studio_Session_{session_id}",
                description=f"DeepEval golden dataset for QA Studio session {session_id}."
            )
            dataset_id = str(dataset.id)

        valid_items = [
            item for item in golden_items
            if item.get("question") and item.get("expected_output")
        ]

        if valid_items:
            inputs = [{"question": item.get("question", "")} for item in valid_items]
            outputs = [{"answer": item.get("expected_output", "")} for item in valid_items]
            metadata = [
                {
                    "source_documents": item.get("source_documents", []),
                    "source_pages": item.get("source_pages", []),
                    "multi_document": len(item.get("source_documents", [])) > 1,
                }
                for item in valid_items
            ]
            client.create_examples(
                dataset_id=dataset_id,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
            )

        conn = get_db_connection()
        conn.execute(
            "UPDATE sessions SET langsmith_dataset_id = ? WHERE session_id = ?",
            (dataset_id, session_id)
        )
        conn.commit()
        conn.close()

        print(f"[LangSmith] Golden dataset saved for session {session_id}: {dataset_id}")
        return dataset_id
    except Exception as e:
        # LangSmith must never break golden-dataset generation.
        print(f"[LangSmith] Failed to save golden dataset for session {session_id}: {e}")
        return None


import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from groq import Groq

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    GEval
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import EvolutionConfig
from deepeval.synthesizer.types import Evolution

app = FastAPI(title="QA Generator with Vision Model & Split DeepEval via Groq")

def _load_groq_api_keys() -> List[str]:
    """Collects Groq API keys from any of:
    - GROQ_API_KEYS="key1,key2,key3,key4,key5" (comma-separated, recommended
      for several free-tier accounts)
    - GROQ_API_KEY_1 .. GROQ_API_KEY_10 (one env var per key)
    - GROQ_API_KEY (single key, kept for backwards compatibility)
    Order is preserved and duplicates are dropped."""
    keys: List[str] = []
    multi = os.getenv("GROQ_API_KEYS", "")
    if multi.strip():
        keys.extend(k.strip() for k in multi.split(",") if k.strip())
    for i in range(1, 11):
        k = os.getenv(f"GROQ_API_KEY_{i}")
        if k and k.strip():
            keys.append(k.strip())
    single = os.getenv("GROQ_API_KEY")
    if single and single.strip():
        keys.append(single.strip())
    seen = set()
    unique_keys = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique_keys.append(k)
    return unique_keys


class GroqKeyPool:
    """Round-robins across several Groq API keys (e.g. from different
    free-tier accounts). Every call site in this file goes through
    generate_content_with_key_rotation() below instead of talking to a single
    Groq directly, so the instant one key hits a rate limit, the very
    next request (and the retry of the one that just failed) transparently
    uses the next key — looping back around to the first key once every key
    has been tried. A short per-key cooldown avoids immediately re-picking a
    key that *just* rate-limited, even on unrelated concurrent calls."""

    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("No Groq API keys configured.")
        self.api_keys = api_keys
        self._clients = [Groq(api_key=k) for k in api_keys]
        self._lock = threading.Lock()
        self._current = 0
        self._cooldown_until = [0.0] * len(api_keys)

    @staticmethod
    def _mask(key: str) -> str:
        return f"...{key[-4:]}" if len(key) > 4 else "****"

    def current_client(self):
        with self._lock:
            return self._clients[self._current], self._current

    def rotate(self, from_index: int, cooldown_seconds: float = 60.0):
        """Marks from_index as cooling down and switches to the next key that
        isn't currently cooling down (or just the next one in line if every
        key is cooling down — better to retry a cooling key than get stuck)."""
        with self._lock:
            self._cooldown_until[from_index] = time.time() + cooldown_seconds
            n = len(self._clients)
            chosen = (from_index + 1) % n
            for step in range(1, n + 1):
                candidate = (from_index + step) % n
                if time.time() >= self._cooldown_until[candidate]:
                    chosen = candidate
                    break
            self._current = chosen
            print(f"[GroqKeyPool] Key {from_index} ({self._mask(self.api_keys[from_index])}) "
                  f"rate-limited — switching to key {self._current} "
                  f"({self._mask(self.api_keys[self._current])}).")
            return self._clients[self._current], self._current

    def num_keys(self) -> int:
        return len(self._clients)


_groq_keys = _load_groq_api_keys()
groq_pool = GroqKeyPool(_groq_keys) if _groq_keys else None
# Kept around only for simple truthiness checks / anything expecting a client
# object directly; actual requests always go through generate_content_with_key_rotation.
groq_client = groq_pool.current_client()[0] if groq_pool else None
DB_FILE = "qa_sessions.db"

# Model Configuration
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "openai/gpt-oss-120b")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.6-27b")  # Groq multimodal model
GEVAL_JUDGE_MODEL = os.getenv("GEVAL_JUDGE_MODEL", "openai/gpt-oss-120b")
RAG_JUDGE_MODEL = os.getenv("RAG_JUDGE_MODEL", "openai/gpt-oss-120b")
SYNTHESIZER_MODEL = os.getenv("SYNTHESIZER_MODEL", "openai/gpt-oss-120b")

EVAL_CONCURRENCY_LIMIT = int(os.getenv("EVAL_CONCURRENCY", "3"))
_eval_semaphore = asyncio.Semaphore(EVAL_CONCURRENCY_LIMIT)

# Limits how many questions are processed (retrieval + RAG answer + eval)
# concurrently in a single "Run RAG Pipeline" call, independent of the
# per-metric-call semaphore above.
RAG_CONCURRENCY_LIMIT = int(os.getenv("RAG_PIPELINE_CONCURRENCY", "3"))
_rag_semaphore = asyncio.Semaphore(RAG_CONCURRENCY_LIMIT)


_RETRY_AFTER_RE = re.compile(r"retry(?:Delay|_delay)?[\"'\s:]*[\{\s]*([\d.]+)\s*(ms|s)|try again in ([\d.]+)(ms|s)", re.IGNORECASE)

# --- Proactive TPM (tokens-per-minute) throttling ---
# The semaphores above only cap how many requests are in flight at once; they
# don't cap how many tokens are used in a rolling 60s window, which is what
# Groq's free tier actually rate-limits on. Under concurrent DeepEval metric
# calls that budget is easy to blow through even with low concurrency, so we
# throttle *before* sending a request instead of only retrying after a 429.
class TokenRateLimiter:
    """Thread-safe sliding-window token-per-minute limiter. acquire() blocks
    (sleeping) until enough budget is free rather than raising, so callers
    just get slowed down instead of erroring out."""

    def __init__(self, tpm_limit: int, window_seconds: float = 60.0):
        self.tpm_limit = max(int(tpm_limit), 1)
        self.window = window_seconds
        self._lock = threading.Lock()
        self._usage: List[tuple] = []  # [(timestamp, tokens), ...]

    def _prune(self, now: float):
        cutoff = now - self.window
        while self._usage and self._usage[0][0] < cutoff:
            self._usage.pop(0)

    def acquire(self, tokens: int):
        tokens = max(int(tokens), 1)
        # If a single request exceeds the whole budget, let it through alone
        # rather than spinning forever.
        tokens = min(tokens, self.tpm_limit)
        while True:
            with self._lock:
                now = time.time()
                self._prune(now)
                used = sum(t for _, t in self._usage)
                if used + tokens <= self.tpm_limit:
                    self._usage.append((now, tokens))
                    return
                oldest_ts = self._usage[0][0] if self._usage else now
                wait_time = max(oldest_ts + self.window - now, 0.25)
            time.sleep(min(wait_time, 5.0))


def estimate_tokens(*texts: str) -> int:
    """Rough token estimate (chars/4, a standard heuristic) plus a small
    fixed overhead for message formatting and the completion itself."""
    total_chars = sum(len(t) for t in texts if t)
    return max(total_chars // 4, 1) + 400


_rate_limiters: dict = {}
_rate_limiter_registry_lock = threading.Lock()


def _tpm_env_key(model_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", model_name).upper()
    return f"TPM_{safe}"


def get_rate_limiter(model_name: str) -> "TokenRateLimiter":
    with _rate_limiter_registry_lock:
        limiter = _rate_limiters.get(model_name)
        if limiter is None:
            default_tpm = int(os.getenv("DEFAULT_TPM", "200000"))
            tpm = int(os.getenv(_tpm_env_key(model_name), str(default_tpm)))
            limiter = TokenRateLimiter(tpm)
            _rate_limiters[model_name] = limiter
        return limiter

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_EMBEDDING_MODEL_CACHE: dict = {}

# --- ChromaDB persistent vector store ---
# Vectors are persisted on disk instead of being rebuilt and kept entirely in
# Python memory for every RAG request. Set CHROMA_DB_DIR in .env to change the
# storage location.
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
CHROMA_EMBED_BATCH_SIZE = max(int(os.getenv("CHROMA_EMBED_BATCH_SIZE", "64")), 1)
_CHROMA_CLIENT = None
_CHROMA_CLIENT_LOCK = threading.Lock()
_CHROMA_COLLECTION_LOCK = threading.Lock()

# --- RAG Pipeline Config (from the "Models & Params" modal) ---
class RAGConfigRequest(BaseModel):
    chat_model: str = Field(default="openai/gpt-oss-20b")
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    chunk_size: int = Field(default=1000, ge=50, le=8000)
    chunk_overlap: int = Field(default=200, ge=0, le=4000)
    top_k: int = Field(default=4, ge=1, le=20)
    search_model: str = Field(default="similarity")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT,
            filename TEXT,
            sample_json TEXT,
            source_doc TEXT,
            generated_qa TEXT,
            deepeval_score REAL,
            deepeval_details TEXT,
            is_golden INTEGER DEFAULT 0,
            status TEXT DEFAULT 'Pending',
            test_case_count INTEGER DEFAULT 20
        )
    """)
    cursor.execute("PRAGMA table_info(sessions)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    if "test_case_count" not in existing_columns:
        cursor.execute("ALTER TABLE sessions ADD COLUMN test_case_count INTEGER DEFAULT 20")
    if "rag_config" not in existing_columns:
        cursor.execute("ALTER TABLE sessions ADD COLUMN rag_config TEXT")
    if "per_question_results" not in existing_columns:
        cursor.execute("ALTER TABLE sessions ADD COLUMN per_question_results TEXT")
    if "golden_dataset" not in existing_columns:
        cursor.execute("ALTER TABLE sessions ADD COLUMN golden_dataset TEXT")
    if "langsmith_dataset_id" not in existing_columns:
        cursor.execute("ALTER TABLE sessions ADD COLUMN langsmith_dataset_id TEXT")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            chat_model TEXT,
            embedding_model TEXT,
            chunk_size INTEGER,
            chunk_overlap INTEGER,
            top_k INTEGER,
            search_model TEXT,
            temperature REAL,
            model_config TEXT,
            retrieved_context TEXT,
            feedback TEXT,
            feedback_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)
    cursor.execute("PRAGMA table_info(chat_messages)")
    chat_columns = {row[1] for row in cursor.fetchall()}
    if "feedback_reason" not in chat_columns:
        cursor.execute("ALTER TABLE chat_messages ADD COLUMN feedback_reason TEXT")
    if "langsmith_trace_id" not in chat_columns:
        cursor.execute("ALTER TABLE chat_messages ADD COLUMN langsmith_trace_id TEXT")
    conn.commit()
    conn.close()

init_db()

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_documents_table():
    """Creates the session-document table used for multi-document sessions."""
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_documents (
            document_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            source_doc TEXT NOT NULL,
            document_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_documents_session
        ON session_documents(session_id)
    """)
    conn.commit()
    conn.close()


def _is_plans_document(filename: str) -> bool:
    """Returns True for documents whose filename contains "Plans" (case-insensitive).

    These documents are stored in the session for reference but are intentionally
    excluded from extraction, generation, evaluation, RAG, and chat processing.
    """
    return "plans" in (filename or "").lower()


def _get_processable_documents(documents: List[dict]) -> List[dict]:
    """Returns only documents that participate in processing.

    Documents whose filenames contain "Plans" are session-only documents and
    must remain visible/stored without contributing content to any pipeline.
    """
    return [doc for doc in documents if not _is_plans_document(doc.get("filename", ""))]


def _get_session_documents(session_id: str) -> List[dict]:
    """Returns all documents currently belonging to a session."""
    _ensure_documents_table()
    conn = get_db_connection()
    rows = conn.execute(
        """SELECT document_id, session_id, filename, source_doc, document_hash, created_at
           FROM session_documents WHERE session_id = ? ORDER BY created_at ASC, rowid ASC""",
        (session_id,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _ensure_legacy_document(session_id: str, row) -> List[dict]:
    """Migrates the old single-document session representation into the new
    session_documents table the first time a legacy session is accessed."""
    _ensure_documents_table()
    documents = _get_session_documents(session_id)
    if documents or not (row["source_doc"] or "").strip():
        return documents

    source_doc = row["source_doc"] or ""
    filename = row["filename"] or "document"
    document_hash = _document_hash(source_doc)
    document_id = str(uuid.uuid4())

    conn = get_db_connection()
    conn.execute(
        """INSERT INTO session_documents
           (document_id, session_id, filename, source_doc, document_hash)
           VALUES (?, ?, ?, ?, ?)""",
        (document_id, session_id, filename, source_doc, document_hash)
    )
    conn.commit()
    conn.close()
    return _get_session_documents(session_id)


def _add_uploaded_document(session_id: str, filename: str, source_doc: str) -> Optional[dict]:
    """Adds a document to the session unless the same document is already present.

    Plans documents are stored as session-only documents with no extracted content.
    All other documents retain the existing extracted-content behavior.
    """
    source_doc = (source_doc or "").strip()
    stored_only = _is_plans_document(filename)
    if not source_doc and not stored_only:
        return None

    _ensure_documents_table()
    document_hash = _document_hash(source_doc) if source_doc else _document_hash(f"__PLANS_ONLY__::{(filename or 'document').lower()}")
    conn = get_db_connection()
    existing = conn.execute(
        """SELECT document_id, session_id, filename, source_doc, document_hash, created_at
           FROM session_documents WHERE session_id = ? AND document_hash = ? LIMIT 1""",
        (session_id, document_hash)
    ).fetchone()
    if existing:
        conn.close()
        return dict(existing)

    document_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO session_documents
           (document_id, session_id, filename, source_doc, document_hash)
           VALUES (?, ?, ?, ?, ?)""",
        (document_id, session_id, filename or "document", source_doc, document_hash)
    )
    conn.commit()
    row = conn.execute(
        """SELECT document_id, session_id, filename, source_doc, document_hash, created_at
           FROM session_documents WHERE document_id = ?""",
        (document_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def _combined_document_text(documents: List[dict]) -> str:
    """Builds one clearly separated text representation for generation and RAG.
    The vector store uses this combined representation only in the RAG path;
    golden synthesis itself does not use a vector store."""
    sections = []
    for doc in documents:
        sections.append(
            f"[SOURCE DOCUMENT: {doc.get('filename', 'document')}]\n"
            f"{doc.get('source_doc', '')}".strip()
        )
    return "\n\n==================== DOCUMENT SEPARATOR ====================\n\n".join(sections).strip()


def _session_document_summary(documents: List[dict]) -> List[dict]:
    return [
        {
            "document_id": d["document_id"],
            "filename": d["filename"],
            "created_at": d.get("created_at"),
        }
        for d in documents
    ]


def _source_metadata_from_context(context) -> tuple:
    """Extracts document/page provenance from Synthesizer context strings."""
    if isinstance(context, str):
        context_parts = [context]
    elif isinstance(context, list):
        context_parts = [str(x) for x in context]
    else:
        context_parts = [str(context)] if context else []

    sources = []
    pages = []
    source_pattern = re.compile(r"\[SOURCE DOCUMENT:\s*(.*?)\]")
    page_pattern = re.compile(r"\[PAGE\s+(\d+)\]")

    for part in context_parts:
        for match in source_pattern.findall(part):
            name = match.strip()
            if name and name not in sources:
                sources.append(name)
        for match in page_pattern.findall(part):
            page = int(match)
            if page not in pages:
                pages.append(page)

    return sources, pages

_ensure_documents_table()

TEST_CASE_DEPTH_INSTRUCTIONS = {
    10: (
        "Generate exactly 10 test cases covering ONLY the primary, explicitly "
        "stated concepts in the document. One test case per major concept — "
        "do not include sub-topics, edge cases, or minor details."
    ),
    20: (
        "Generate exactly 20 test cases covering the primary concepts AND "
        "their directly related sub-topics or details explicitly mentioned "
        "in the document."
    ),
    30: (
        "Generate exactly 30 test cases with keen observation: cover primary "
        "concepts, sub-topics, edge cases, boundary/negative scenarios "
        "(e.g. what happens if a step is skipped or a value is missing), and "
        "implicit relationships or cross-references between different parts "
        "of the document. Prioritize thoroughness and nuance over repetition."
    ),
}
ALLOWED_TEST_CASE_COUNTS = (10, 20, 30)

# Mirrors TEST_CASE_DEPTH_INSTRUCTIONS but for the golden-dataset synthesis
# step: deeper depths get harder/more varied DeepEval "evolutions" applied
# to the synthetic questions, not just more of them.
SYNTHESIS_EVOLUTION_CONFIG = {
    10: EvolutionConfig(num_evolutions=1, evolutions={Evolution.REASONING: 1.0}),
    20: EvolutionConfig(num_evolutions=1, evolutions={
        Evolution.REASONING: 0.5, Evolution.MULTICONTEXT: 0.5
    }),
    30: EvolutionConfig(num_evolutions=2, evolutions={
        Evolution.REASONING: 0.25, Evolution.MULTICONTEXT: 0.25,
        Evolution.COMPARATIVE: 0.25, Evolution.HYPOTHETICAL: 0.25
    }),
}

# ─── Genesis Capital Field-Targeted QA Generation ────────────────────────────
#
# 32 testable fields derived from the Genesis Feasibility Review Template.
# The 7 pure user-input fields are excluded (Project Status, Report Created by,
# appropriate/not appropriate timeline, Draw Hold, Specify Draw Hold Items,
# Other Special Conditions, Permits Post Funding).
#
# Format rules (matching Genesis extract_fields.py conventions):
#   currency  → $#,###,###.## via _format_money()
#   date      → stored as YYYY-MM-DD (ISO), displayed as MM/DD/YYYY
#   percent   → XX.X%
#   GFA       → "41,656 SF"
#   timeline  → "X days (Y months)"
#   dropdown  → exact option string from the options list

GENESIS_FIELDS: List[dict] = [
    # ── Report Header ──────────────────────────────────────────────────────────
    {
        "field_name": "Date of Report Approved",
        "section": "Report Header",
        "type": "date",
        "source": "Auto-generated",
        "extraction_method": "deterministic",
        "expected_doc": None,
        "options": None,
        "format_hint": "ISO date YYYY-MM-DD stored internally; displayed as MM/DD/YYYY. Always equals today's date.",
        "question_template": "What should the 'Date of Report Approved' field display on a Genesis Capital feasibility review report generated today?",
    },
    {
        "field_name": "Project Address / Title",
        "section": "Report Header",
        "type": "freeform",
        "source": "Trinity Report cover page",
        "extraction_method": "RAG + LLM + Regex text scan",
        "expected_doc": "Trinity",
        "options": None,
        "format_hint": "Full street address including city, state, zip. Prepend project development name before address if present. Strip deal/loan IDs.",
        "question_template": "What is the full project address and title for this Genesis Capital feasibility review?",
    },
    {
        "field_name": "Sponsor",
        "section": "Report Header",
        "type": "freeform",
        "source": "SCA (Sponsor Construction Analysis)",
        "extraction_method": "RAG + LLM (company entity name only)",
        "expected_doc": "SCA",
        "options": None,
        "format_hint": "Full legal company/entity name including fka/dba/aka. Must NOT be an individual person's name.",
        "question_template": "Who is the sponsor (development company or legal entity) for this project according to the SCA document?",
    },
    {
        "field_name": "Borrower Entity",
        "section": "Report Header",
        "type": "freeform",
        "source": "Construction Budget / Trinity / SCA",
        "extraction_method": "RAG + LLM (header row scan)",
        "expected_doc": "Budget",
        "options": None,
        "format_hint": "Legal LLC/entity name executing the loan. No timestamps, dates, or individual names.",
        "question_template": "What is the borrower entity (legal LLC or company name) for this project?",
    },
    # ── Executive Summary ──────────────────────────────────────────────────────
    {
        "field_name": "Additional Comments",
        "section": "Executive Summary",
        "type": "narrative",
        "source": "SCA (Sponsor Construction Analysis)",
        "extraction_method": "RAG + LLM narrative synthesis (zero-hallucination)",
        "expected_doc": "SCA",
        "options": None,
        "format_hint": "2–3 paragraph narrative: sponsor track record, experience, construction approval tier, team evaluation. Plain text only, no markdown headers or bullets. Ends with: 'This report is for the [Project Type] of a [No. of Stories] [Property Type] with [No. of Units] Units totaling approximately [GFA] gross square feet.'",
        "question_template": "Based on the SCA document, what should be written in the 'Additional Comments' narrative for the executive summary of this feasibility review?",
    },
    {
        "field_name": "Third-Party Review",
        "section": "Executive Summary",
        "type": "dropdown",
        "source": "Trinity Report",
        "extraction_method": "RAG + LLM + dropdown validation",
        "expected_doc": "Trinity",
        "options": ["None", "feasibility", "budget review"],
        "format_hint": "Select 'feasibility' if plans, budget, and scope were reviewed together. Select 'budget review' if only budget was reviewed without drawings/plans.",
        "question_template": "What type of third-party review was obtained for this project — 'None', 'feasibility', or 'budget review'?",
    },
    {
        "field_name": "Third-Party Reviewer",
        "section": "Executive Summary",
        "type": "dropdown",
        "source": "Trinity Report",
        "extraction_method": "RAG + LLM + dropdown validation",
        "expected_doc": "Trinity",
        "options": ["Trinity", "Granite", "DCMI", "Northwest Monitoring"],
        "format_hint": "Must be exactly one of the 4 approved firms: Trinity, Granite, DCMI, Northwest Monitoring.",
        "question_template": "Which third-party company authored the feasibility review report for this project?",
    },
    {
        "field_name": "Third-party Review (Good/Bad)",
        "section": "Executive Summary",
        "type": "dropdown",
        "source": "Trinity Report page 1",
        "extraction_method": "RAG + LLM + dropdown validation",
        "expected_doc": "Trinity",
        "options": ["appropriate", "not appropriate"],
        "format_hint": "Select 'appropriate' if the report states cost is reasonable for the proposed scope. Select 'not appropriate' if risk or cost is flagged negatively.",
        "question_template": "Does the third-party reviewer consider the overall project cost and risk 'appropriate' or 'not appropriate' for the proposed scope?",
    },
    {
        "field_name": "Third-Party Review (Meet or Fail)",
        "section": "Executive Summary",
        "type": "dropdown",
        "source": "Trinity Report",
        "extraction_method": "RAG + LLM + Regex text scan",
        "expected_doc": "Trinity",
        "options": [
            "meets best practice and is Recommended for Approval",
            "meets best practice with Advisement or Conditions Recommended for Approval",
            "does not meet best practice and should Not be Approved",
        ],
        "format_hint": "Must be one of the 3 exact strings. Map approval language to option 1, conditional approval to option 2, rejection to option 3.",
        "question_template": "What is the third-party reviewer's formal best-practice recommendation for this project?",
    },
    {
        "field_name": "Genesis Agree (Y/N)",
        "section": "Executive Summary",
        "type": "dropdown",
        "source": "Trinity Report (derived from Meet or Fail)",
        "extraction_method": "Derived: agrees if Meet/Fail is options 1 or 2; disagree if option 3",
        "expected_doc": "Trinity",
        "options": ["agrees", "disagree"],
        "format_hint": "Defaults to 'agrees' unless the third-party review is a clear rejection (does not meet best practice). Derived automatically from Third-Party Review (Meet or Fail).",
        "question_template": "Does Genesis Capital Construction Department agree with the third-party findings for this project?",
    },
    {
        "field_name": "Project Timeline To Date",
        "section": "Executive Summary",
        "type": "freeform",
        "source": "Construction Timeline / Deal Notes",
        "extraction_method": "Scoped date regex + days/months calculation",
        "expected_doc": "Timeline",
        "options": None,
        "format_hint": "Format: 'X days (Y months)' elapsed from project start date to today. Returns 'Project has not started yet' if start date is in the future.",
        "question_template": "How much time has elapsed since this project started, expressed in days and months?",
    },
    {
        "field_name": "Remaining Timeline",
        "section": "Executive Summary",
        "type": "freeform",
        "source": "Construction Timeline / Deal Notes",
        "extraction_method": "Scoped date regex + days/months calculation",
        "expected_doc": "Timeline",
        "options": None,
        "format_hint": "Format: 'X days (Y months)' remaining until final completion/Certificate of Occupancy. Returns 'Project timeline has passed' if end date is in the past.",
        "question_template": "How much time remains until this project's final completion or Certificate of Occupancy?",
    },
    # ── Loan Summary ───────────────────────────────────────────────────────────
    {
        "field_name": "Project Type",
        "section": "Loan Summary",
        "type": "dropdown",
        "source": "Trinity Report",
        "extraction_method": "RAG + LLM + Regex text scan",
        "expected_doc": "Trinity",
        "options": [
            "Renovation",
            "Ground - Up Construction",
            "Renovation plus square footage",
            "Mid - Construction Refinance of a Renovation",
            "Mid - Construction Refinance of a Ground - Up Construction",
            "Horizontal Site Work Only",
        ],
        "format_hint": "'New construction'/'ground up'/'new build' → 'Ground - Up Construction'. 'Renovation'/'rehab'/'remodel' → 'Renovation'. 'Adding square footage' → 'Renovation plus square footage'.",
        "question_template": "What is the project type classification for this Genesis Capital feasibility review (e.g. Renovation, Ground - Up Construction)?",
    },
    {
        "field_name": "Rehab Amount",
        "section": "Loan Summary",
        "type": "freeform",
        "source": "Construction Budget (.xlsx)",
        "extraction_method": "Excel TOTAL row scan + currency formatting",
        "expected_doc": "Budget",
        "options": None,
        "format_hint": "Total construction budget. Format: $#,###,###.## (e.g. $4,250,000.00). Must be >= Construction Holdback Amount.",
        "question_template": "What is the total rehab/construction budget amount for this project?",
    },
    {
        "field_name": "Construction Holdback Amount",
        "section": "Loan Summary",
        "type": "freeform",
        "source": "Construction Budget (.xlsx)",
        "extraction_method": "Excel TOTAL/holdback row scan + currency formatting",
        "expected_doc": "Budget",
        "options": None,
        "format_hint": "Total loan holdback for construction draws. Format: $#,###,###.## Must be <= Rehab Amount and >= 1% of Rehab.",
        "question_template": "What is the construction holdback amount (loan proceeds allocated for construction draws) for this project?",
    },
    {
        "field_name": "Project Cost per Square Foot",
        "section": "Loan Summary",
        "type": "calculated",
        "source": "Calculated: Rehab Amount / Gross Buildable Square Footage (GFA)",
        "extraction_method": "Formula: Rehab Amount ÷ GFA",
        "expected_doc": None,
        "options": None,
        "format_hint": "Format: $#,###,###.## (e.g. $245.50). Recalculates live when Rehab Amount or GFA changes.",
        "question_template": "What is the project cost per square foot, calculated as Rehab Amount divided by Gross Buildable Square Footage?",
    },
    {
        "field_name": "Cost per Structure",
        "section": "Loan Summary",
        "type": "calculated",
        "source": "Calculated: Rehab Amount / No. of Structures",
        "extraction_method": "Formula: Rehab Amount ÷ No. of Structures",
        "expected_doc": None,
        "options": None,
        "format_hint": "Format: $#,###,###.## Left empty if No. of Structures is not populated.",
        "question_template": "What is the cost per structure, calculated as Rehab Amount divided by the number of structures?",
    },
    {
        "field_name": "Cost per Unit",
        "section": "Loan Summary",
        "type": "calculated",
        "source": "Calculated: Rehab Amount / No. of Units",
        "extraction_method": "Formula: Rehab Amount ÷ No. of Units",
        "expected_doc": None,
        "options": None,
        "format_hint": "Format: $#,###,###.## (e.g. $156,250.00). Recalculates live when Rehab Amount or No. of Units changes.",
        "question_template": "What is the cost per unit, calculated as Rehab Amount divided by the number of residential units?",
    },
    {
        "field_name": "Contingency Amount",
        "section": "Loan Summary",
        "type": "freeform",
        "source": "Construction Budget / Trinity Report",
        "extraction_method": "Budget summary row scan → line items sum → Trinity text scan → LLM extraction",
        "expected_doc": "Budget",
        "options": None,
        "format_hint": "Total contingency reserve in dollars. Format: $#,###,###.## Sourced from 'Total Contingency Included | $X | Y%' summary row in budget spreadsheet.",
        "question_template": "What is the total contingency amount allocated in the construction budget for this project?",
    },
    {
        "field_name": "Contingency (%)",
        "section": "Loan Summary",
        "type": "calculated",
        "source": "Calculated: Contingency Amount / (Construction Holdback - Contingency Amount)",
        "extraction_method": "Stated % from budget row (confidence 0.95) or formula fallback",
        "expected_doc": None,
        "options": None,
        "format_hint": "Format: XX.X% (e.g. 7.5%). Formula: (Contingency Amount / (Construction Holdback - Contingency Amount)) × 100.",
        "question_template": "What is the contingency percentage relative to the net construction holdback for this project?",
    },
    {
        "field_name": "Project Complete Percentage",
        "section": "Loan Summary",
        "type": "calculated",
        "source": "Calculated: (today - start_date) / (end_date - start_date)",
        "extraction_method": "Formula: elapsed timeline / total timeline × 100",
        "expected_doc": None,
        "options": None,
        "format_hint": "Format: XX.X% (e.g. 42.5%). Returns 0.0% if project not started, 100.0% if past end date.",
        "question_template": "What percentage of the construction timeline has been completed as of today, based on start and end dates?",
    },
    {
        "field_name": "Budget Review",
        "section": "Loan Summary",
        "type": "dropdown",
        "source": "Trinity Report page 1",
        "extraction_method": "RAG + LLM + dropdown validation",
        "expected_doc": "Trinity",
        "options": [
            "The budget presented accurately reflects the scope of the project and the cost allocations have been determined to meet the minimum threshold to complete",
            'The budget is considered to be a higher than typical "cost per square foot", but it is acceptable',
            "The budget presented accurately reflects the scope of the project and the cost allocations have been determined to meet the minimum threshold to complete. However, certain line items require further review",
        ],
        "format_hint": "Must be one of the 3 exact standardized reviewer statements about budget adequacy.",
        "question_template": "What is the budget review determination for this project — does it accurately reflect the scope and meet the minimum threshold to complete?",
    },
    {
        "field_name": "Additional Budget Comments",
        "section": "Loan Summary",
        "type": "narrative",
        "source": "Trinity Report",
        "extraction_method": "RAG + LLM (max 2 lines, specific dollar amounts, plain text)",
        "expected_doc": "Trinity",
        "options": None,
        "format_hint": "1–2 sentence narrative: overall budget adequacy, $/SF market comparison, flagged line items. Plain text only.",
        "question_template": "What additional budget comments does the third-party report provide about the construction budget adequacy and cost per square foot?",
    },
    {
        "field_name": "Plan Status",
        "section": "Loan Summary",
        "type": "dropdown",
        "source": "Trinity Report / Plans",
        "extraction_method": "RAG + LLM + Regex text scan",
        "expected_doc": "Trinity",
        "options": [
            "Pre - Submittal",
            "Submittal/Plan Check (PC)",
            "RTI",
            "City approved",
            "Not Required",
        ],
        "format_hint": "Explicit approval language/stamps → 'City approved'. Submittal dates → 'Submittal/Plan Check (PC)'. RTI stamp → 'RTI'.",
        "question_template": "What is the architectural and engineering plan review/approval status with the municipal building department for this project?",
    },
    {
        "field_name": "Plan Review Status",
        "section": "Loan Summary",
        "type": "dropdown",
        "source": "Trinity Report / Plans",
        "extraction_method": "Key drawing sets check (Civil, Structural, Architectural) + Regex scan",
        "expected_doc": "Trinity",
        "options": [
            "The plans received are sufficient to support the project scope as needed",
            "Supplemental plan documentation is needed",
            "N/A",
        ],
        "format_hint": "Civil + Structural + Architectural all provided without defects → option 1. Key sets missing or major deficiencies → option 2. No plans provided → 'N/A'.",
        "question_template": "Are the drawing sets provided sufficient to support the construction scope, or is supplemental plan documentation needed?",
    },
    {
        "field_name": "Permit Status",
        "section": "Loan Summary",
        "type": "user_input",
        "source": "Trinity Report / Deal Notes",
        "extraction_method": "LLM + anti-hallucination text scan gate",
        "expected_doc": "Trinity",
        "options": [
            "Building permits have been issued prior to funding this loan. The Construction Department has received all necessary building permits",
            'The borrower has applied for building permits and they are currently "RTI" Ready-To-Issue.',
            "The borrower has been issued partial permits on this project. Permits are expected to be issued.",
            "The borrower has not yet obtained permits for this loan",
            "There will not be permits issued/required on this loan",
        ],
        "format_hint": "Must match one of the 5 exact permit status strings. LLM is prevented from claiming permits issued unless Trinity text explicitly confirms it.",
        "question_template": "What is the current building permit status for this project at the time of loan review?",
    },
    # ── Finished Product Details ───────────────────────────────────────────────
    {
        "field_name": "Property Type",
        "section": "Finished Product Details",
        "type": "dropdown",
        "source": "Trinity Report page 1 para 5-6",
        "extraction_method": "RAG + LLM + Trinity text scan",
        "expected_doc": "Trinity",
        "options": [
            "Single - Family Home",
            "Single - Family Home plus ADU",
            "Multifamily Building",
            "Multi Unit",
            "Multi Unit (duplexes) buildings",
            "Planned Urban Development (PUD)",
            "Horizontal Land Improvements",
            "Subdivision",
        ],
        "format_hint": "townhome/condo/apartment/hotel → 'Multifamily Building'. duplex → 'Multi Unit (duplexes) buildings'. single family → 'Single - Family Home'.",
        "question_template": "What is the property type classification of the completed project (e.g. Multifamily Building, Single - Family Home)?",
    },
    {
        "field_name": "Region",
        "section": "Finished Product Details",
        "type": "freeform",
        "source": "Trinity Report cover page",
        "extraction_method": "RAG + LLM + Regex text scan",
        "expected_doc": "Trinity",
        "options": None,
        "format_hint": "City and state where property is located (e.g. 'Richmond Heights, OH').",
        "question_template": "What is the geographic region (city and state) where this project is located?",
    },
    {
        "field_name": "No. of Units",
        "section": "Finished Product Details",
        "type": "dropdown",
        "source": "Trinity Report",
        "extraction_method": "Dedicated full-text LLM + Budget Cost/Unit header scan + Regex",
        "expected_doc": "Trinity",
        "options": ["SFR", "SFR + ADU", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                    "21", "22", "23", "24", "25"],
        "format_hint": "Integer count of residential dwelling units. Custom values above 25 accepted as strings.",
        "question_template": "What is the total number of residential dwelling units in this project?",
    },
    {
        "field_name": "No. of Stories",
        "section": "Finished Product Details",
        "type": "dropdown",
        "source": "Trinity Report",
        "extraction_method": "Dedicated full-text LLM + Regex scan",
        "expected_doc": "Trinity",
        "options": [
            "One - story", "two - story", "three - story", "four - story",
            "five - story", "six - story", "seven - story", "eight - story",
            "nine - story", "ten - story",
            "11 - story", "12 - story", "13 - story", "14 - story", "15 - story",
        ],
        "format_hint": "Above-grade floors only. Excludes basements, parking garages, underground levels.",
        "question_template": "How many above-grade stories does the primary building have in this project?",
    },
    {
        "field_name": "No. of Structures",
        "section": "Finished Product Details",
        "type": "freeform",
        "source": "Trinity Report",
        "extraction_method": "Budget/Trinity Cost/Structure header scan + Full-text LLM + Regex + BTR logic",
        "expected_doc": "Trinity",
        "options": None,
        "format_hint": "Integer count of physical buildings including amenity structures (Clubhouse, Grill House, Pool House). Excludes detached garages, carports, leasing offices.",
        "question_template": "What is the total number of physical building structures in this project, including amenity buildings?",
    },
    {
        "field_name": "Gross Buildable Square Footage (GFA)",
        "section": "Finished Product Details",
        "type": "freeform",
        "source": "Trinity Report / Plans / Budget",
        "extraction_method": "RAG + LLM + Regex scan + SF format",
        "expected_doc": "Trinity",
        "options": None,
        "format_hint": "Total cumulative gross building floor area across all structures. Format: '41,656 SF' (with SF suffix, comma-formatted).",
        "question_template": "What is the total gross buildable square footage (GFA) across all structures in this project?",
    },
]

# Fields excluded from GENESIS_FIELDS (pure user input — no document answer exists):
# 1.5  Project Status          (analyst decision dropdown)
# 1.6  Report Created by       (analyst name dropdown)
# 2.9  appropriate/not appropriate timeline  (analyst judgment)
# 2.10 Draw Hold               (analyst decision)
# 2.11 Specify Draw Hold Items (analyst checklist)
# 2.12 Other Special Conditions (analyst freeform)
# 3.15 Permits (Post Funding)  (analyst deadline dropdown)

_GENESIS_FIELD_NAMES = {f["field_name"] for f in GENESIS_FIELDS}


async def generate_genesis_field_goldens(
    documents: List[dict],
    sample_json: str,
    field_count: int = 32,
) -> tuple[List[dict], dict]:
    """Generate golden QA pairs for a random sample of Genesis fields.

    field_count controls how many of the 32 testable fields to include:
    - The date field (Date of Report Approved) is always included.
    - The remaining slots are filled by random sampling from the other fields.
    - field_count=32 means all fields (no sampling).

    Uses parallel LLM calls grouped by source document.
    """
    import random
    from datetime import date as _date

    # ── Sample fields ─────────────────────────────────────────────────────────
    date_fields = [f for f in GENESIS_FIELDS if f["type"] == "date"]
    other_fields = [f for f in GENESIS_FIELDS if f["type"] != "date"]

    if field_count >= len(GENESIS_FIELDS):
        # All fields — no sampling
        active_fields = GENESIS_FIELDS
    else:
        # Always include the date field; randomly sample the rest
        remaining_slots = max(field_count - len(date_fields), 0)
        sampled = random.sample(other_fields, min(remaining_slots, len(other_fields)))
        # Keep original GENESIS_FIELDS order for consistent display
        sampled_names = {f["field_name"] for f in sampled} | {f["field_name"] for f in date_fields}
        active_fields = [f for f in GENESIS_FIELDS if f["field_name"] in sampled_names]

    print(f"[Genesis QA] Sampling {len(active_fields)}/{len(GENESIS_FIELDS)} fields")

    doc_text = _combined_document_text(documents)
    today_display = _date.today().strftime("%m/%d/%Y")
    source_doc_names = [d["filename"] for d in documents]

    import re as _re

    def _extract_text_only(text: str) -> str:
        """Extract only the --- EXTRACTED TEXT --- sections, discarding all
        visual descriptions and think blocks regardless of nesting."""
        sections = re.findall(
            r'--- EXTRACTED TEXT ---\s*(.*?)(?=--- EXTRACTED TEXT ---|--- VISUAL & DIAGRAM DESCRIPTION ---|$)',
            text, flags=_re.DOTALL
        )
        if not sections:
            # Fallback: strip think blocks the old way
            text = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL)
            text = _re.sub(r'<think>.*$', '', text, flags=_re.DOTALL)
            return _re.sub(r'\n{3,}', '\n\n', text).strip()
        combined = '\n\n'.join(s.strip() for s in sections if s.strip())
        return _re.sub(r'\n{3,}', '\n\n', combined).strip()

    # Use _extract_text_only instead of strip_think for all documents
    _strip_think_blocks = _extract_text_only

    # Build clean per-document text, skipping binary files
    doc_text_by_label: dict = {}  # label -> full clean text
    for doc in documents:
        fname = doc.get("filename", "").lower()
        raw = doc.get("source_doc", "") or ""
        if not raw.strip() or raw.strip().startswith("PK") or "\x00" in raw[:100]:
            continue
        clean = _strip_think_blocks(raw)
        if not clean.strip():
            continue
        if any(k in fname for k in ["feasibility", "trinity", "feasibility report"]):
            doc_text_by_label["trinity"] = doc_text_by_label.get("trinity", "") + "\n\n" + clean
        if any(k in fname for k in ["sca", "sponsor", "borrower construction due diligence", "track record"]):
            doc_text_by_label["sca"] = doc_text_by_label.get("sca", "") + "\n\n" + clean
        if any(k in fname for k in ["timeline", "schedule", "gantt"]):
            doc_text_by_label["timeline"] = doc_text_by_label.get("timeline", "") + "\n\n" + clean
        if any(k in fname for k in ["deal notes", "deal_notes", "loan notes"]):
            doc_text_by_label["deal_notes"] = doc_text_by_label.get("deal_notes", "") + "\n\n" + clean

    # Everything except binary budget xlsx
    combined_clean = "\n\n".join(v.strip() for v in doc_text_by_label.values() if v.strip())
    if not combined_clean:
        combined_clean = _strip_think_blocks(doc_text)

    print(f"[Genesis QA] Clean text ready: {len(combined_clean):,} chars | docs: {list(doc_text_by_label.keys())}")

    # ── Map each field to its source document text ───────────────────────────
    label_map = {
        "Trinity":    "trinity",
        "Budget":     "trinity",   # budget table is inside the Feasibility Report
        "SCA":        "sca",
        "Timeline":   "timeline",
        "Deal Notes": "deal_notes",
    }

    def _context_for_field(field: dict, max_chars: int = 8000) -> str:
        label = label_map.get(field.get("expected_doc", ""), None)
        source = (doc_text_by_label.get(label, "") if label else combined_clean).strip()
        if not source:
            source = combined_clean
        return source[:max_chars]

    system_prompt = (
        "You are a precise data extraction assistant for Genesis Capital feasibility reviews.\n"
        "You will receive a batch of fields to extract from the provided document context.\n"
        "Documents may include: Trinity Reports (feasibility reviews with budget tables), "
        "SCA (Sponsor Construction Analysis), Construction Timelines, and Deal Notes.\n\n"
        "STRICT RULES:\n"
        "1. Extract ONLY from the provided context. Never fabricate data.\n"
        "2. If a field is not found in the context, use exactly: NOT_FOUND\n"
        "3. Currency format: $#,###,###.## (e.g. $4,250,000.00)\n"
        "4. Date format: MM/DD/YYYY (e.g. 08/27/2026)\n"
        "5. Percentage format: XX.X% (e.g. 7.5%)\n"
        "6. GFA format: number with SF suffix (e.g. 41,656 SF)\n"
        "7. Timeline format: X days (Y months)\n"
        "8. For dropdown fields, return ONLY one of the exact option strings listed.\n"
        "9. For calculated fields, find the input values in the context, compute, and show "
        "   the formula + result (e.g. '$9,649,588.08 / 20,868 SF = $462.41').\n"
        "10. Return a JSON object: each key = exact field_name, each value = extracted string."
    )

    # ── Group fields by their source document label ──────────────────────────
    groups: dict = {}
    formula_fields = []
    for f in active_fields:
        if f["type"] == "date":
            continue
        if f["type"] == "calculated":
            formula_fields.append(f)
            continue
        lbl = label_map.get(f.get("expected_doc", ""), "combined")
        groups.setdefault(lbl, []).append(f)

    # For each group, build (context_slice, fields) pairs.
    # If the doc is longer than MAX_CHARS, split into 3 equal parts and send
    # ALL fields against each part — best-answer-wins after.
    # All parts run in PARALLEL so total time = 1 LLM call latency, not N×latency.
    MAX_CHARS = 8000
    MAX_FIELDS = 10
    batches: List[tuple] = []  # (context_text, fields_list)

    for lbl, fields in groups.items():
        src = (doc_text_by_label.get(lbl, combined_clean) if lbl != "combined" else combined_clean).strip() or combined_clean
        if len(src) <= MAX_CHARS:
            for i in range(0, len(fields), MAX_FIELDS):
                batches.append((src, fields[i:i + MAX_FIELDS]))
        else:
            # Split into 3 equal non-overlapping parts
            part_size = len(src) // 3
            parts = [src[:part_size], src[part_size:2*part_size], src[2*part_size:]]
            for part in parts:
                for i in range(0, len(fields), MAX_FIELDS):
                    batches.append((part, fields[i:i + MAX_FIELDS]))

    print(f"[Genesis QA] {len(batches)} parallel LLM calls + {len(formula_fields)} Python formula fields")

    async def _run_batch(batch_idx: int, ctx: str, batch: List[dict]) -> dict:
        field_descriptors = []
        for f in batch:
            desc = (f"FIELD: {f['field_name']}\n"
                    f"  TYPE: {f['type']}\n"
                    f"  SOURCE DOC: {f.get('expected_doc', 'any')}\n"
                    f"  FORMAT: {f['format_hint']}")
            if f.get("options"):
                desc += "\n  ALLOWED OPTIONS (pick exactly one): " + " | ".join(f["options"])
            field_descriptors.append(desc)
        fields_block = "\n\n".join(field_descriptors)
        user_prompt = (
            f"DOCUMENT CONTEXT:\n{ctx}\n\n"
            f"FIELDS TO EXTRACT:\n\n{fields_block}\n\n"
            "Return a JSON object: key = exact field_name, value = extracted answer string.\n"
            "Unknown fields: value = \"NOT_FOUND\"."
        )
        for attempt in range(3):
            try:
                limiter = get_rate_limiter(GENERATION_MODEL)
                limiter.acquire(estimate_tokens(system_prompt, user_prompt))
                response = await asyncio.to_thread(
                    generate_content_with_key_rotation,
                    GENERATION_MODEL,
                    user_prompt,
                    {"temperature": 0.0, "system_instruction": system_prompt,
                     "response_format": {"type": "json_object"}},
                )
                result = json.loads(response.choices[0].message.content.strip())
                found = sum(1 for v in result.values() if v != "NOT_FOUND")
                print(f"[Genesis QA] Batch {batch_idx+1} done: {found}/{len(batch)} found")
                return result
            except Exception as e:
                if _is_rate_limit_error(e) and attempt < 2:
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                print(f"[Genesis QA] Batch {batch_idx+1} failed: {e}")
                return {f["field_name"]: "NOT_FOUND" for f in batch}
        return {f["field_name"]: "NOT_FOUND" for f in batch}

    # Run all batches in parallel
    tasks = [_run_batch(i, ctx, batch) for i, (ctx, batch) in enumerate(batches)]
    results = await asyncio.gather(*tasks)

    # Best-answer-wins merge
    extracted_values: dict = {}
    for batch_result in results:
        for k, v in batch_result.items():
            if v and v != "NOT_FOUND":
                extracted_values[k] = v
            elif k not in extracted_values:
                extracted_values[k] = v

    # ── Compute formula fields in Python using extracted base values ─────────
    def _parse_money(s: str) -> Optional[float]:
        if not s or s == "NOT_FOUND":
            return None
        try:
            return float(_re.sub(r"[^\d.]", "", s))
        except (ValueError, TypeError):
            return None

    def _parse_sf(s: str) -> Optional[float]:
        if not s or s == "NOT_FOUND":
            return None
        try:
            return float(_re.sub(r"[^\d.]", "", s.replace(",", "")))
        except (ValueError, TypeError):
            return None

    def _parse_int(s: str) -> Optional[float]:
        if not s or s == "NOT_FOUND":
            return None
        try:
            return float(_re.sub(r"[^\d.]", "", s))
        except (ValueError, TypeError):
            return None

    def _fmt_money(v: float) -> str:
        return f"${v:,.2f}"

    def _fmt_pct(v: float) -> str:
        return f"{v:.1f}%"

    from datetime import date as _date2

    rehab     = _parse_money(extracted_values.get("Rehab Amount", "NOT_FOUND"))
    holdback  = _parse_money(extracted_values.get("Construction Holdback Amount", "NOT_FOUND"))
    gfa       = _parse_sf(extracted_values.get("Gross Buildable Square Footage (GFA)", "NOT_FOUND"))
    units     = _parse_int(extracted_values.get("No. of Units", "NOT_FOUND"))
    structures= _parse_int(extracted_values.get("No. of Structures", "NOT_FOUND"))
    contingency = _parse_money(extracted_values.get("Contingency Amount", "NOT_FOUND"))

    # Project Complete % from timeline
    start_str = extracted_values.get("Project Timeline To Date", "NOT_FOUND")
    end_str   = extracted_values.get("Remaining Timeline", "NOT_FOUND")
    today_d   = _date2.today()

    for f in formula_fields:
        fname = f["field_name"]
        result = "NOT_FOUND"

        if fname == "Project Cost per Square Foot":
            if rehab and gfa:
                val = rehab / gfa
                result = f"{_fmt_money(rehab)} / {gfa:,.0f} SF = {_fmt_money(val)}"
            elif rehab and not gfa:
                result = "NOT_FOUND (GFA not extracted)"

        elif fname == "Cost per Structure":
            if rehab and structures:
                val = rehab / structures
                result = f"{_fmt_money(rehab)} / {structures:.0f} structures = {_fmt_money(val)}"

        elif fname == "Cost per Unit":
            if rehab and units:
                val = rehab / units
                result = f"{_fmt_money(rehab)} / {units:.0f} units = {_fmt_money(val)}"

        elif fname == "Contingency (%)":
            if contingency and holdback and holdback > contingency:
                pct = contingency / (holdback - contingency) * 100
                result = f"{_fmt_money(contingency)} / ({_fmt_money(holdback)} - {_fmt_money(contingency)}) = {_fmt_pct(pct)}"
            elif contingency and rehab and rehab > contingency:
                # Fallback: use rehab as denominator if holdback unknown
                pct = contingency / rehab * 100
                result = f"{_fmt_money(contingency)} / {_fmt_money(rehab)} = {_fmt_pct(pct)} (holdback not found, used rehab)"

        elif fname == "Project Complete Percentage":
            # Extract days-elapsed from Project Timeline To Date field
            # Expected format: "392 days (13.1 months)"
            days_match = _re.search(r"(\d+)\s*days", start_str or "")
            rem_match  = _re.search(r"(\d+)\s*days", end_str or "")
            if days_match and rem_match:
                elapsed = int(days_match.group(1))
                remaining = int(rem_match.group(1))
                total = elapsed + remaining
                if total > 0:
                    pct = elapsed / total * 100
                    result = f"{elapsed} / ({elapsed} + {remaining}) = {_fmt_pct(pct)}"
            else:
                result = "NOT_FOUND (timeline dates not extracted)"

        extracted_values[fname] = result

    # ── Assemble golden_items in original active_fields order ──────────────
    golden_items: List[dict] = []
    for field in active_fields:
        field_name = field["field_name"]
        if field["type"] == "date":
            value = today_display
            method = "auto-generated (date.today())"
        else:
            value = extracted_values.get(field_name, "NOT_FOUND")
            method = field.get("extraction_method", "")

        golden_items.append({
            "question": field["question_template"],
            "expected_output": value,
            "field_name": field_name,
            "section": field["section"],
            "type": field["type"],
            "source": field.get("source", ""),
            "source_documents": source_doc_names,
            "extraction_method": method,
        })

    # ── Build generated_json (respects sample_json schema if provided) ───────
    slim_items = [
        {
            "field_name": g["field_name"],
            "section": g["section"],
            "type": g["type"],
            "question": g["question"],
            "expected_output": g["expected_output"],
            "source": g["source"],
            "extraction_method": g["extraction_method"],
        }
        for g in golden_items
    ]

    if sample_json and sample_json.strip() not in ("{}", ""):
        format_system = (
            "You are a QA schema formatter. Reformat each Genesis field result into the target JSON schema. "
            "Do not invent facts, drop items, or add extra items. Preserve field names, questions, and expected outputs exactly."
        )
        format_prompt = (
            f"GENESIS FIELD RESULTS:\n{json.dumps(slim_items, indent=2)}\n\n"
            f"TARGET JSON SCHEMA:\n{sample_json}\n\n"
            "Return a JSON object with a 'test_cases' array — one entry per field result."
        )
        try:
            limiter = get_rate_limiter(GENERATION_MODEL)
            limiter.acquire(estimate_tokens(format_prompt))
            fmt_response = generate_content_with_key_rotation(
                model=GENERATION_MODEL,
                contents=format_prompt,
                config={
                    "temperature": 0.1,
                    "system_instruction": format_system,
                    "response_format": {"type": "json_object"},
                },
            )
            generated_json = json.loads(fmt_response.choices[0].message.content)
        except Exception as e:
            print(f"[Genesis QA] Schema formatting failed, using default structure: {e}")
            generated_json = {"test_cases": slim_items}
    else:
        generated_json = {"test_cases": slim_items}

    return golden_items, generated_json


def _is_rate_limit_error(e: Exception) -> bool:
    text = str(e).lower()
    return "429" in text or "resource_exhausted" in text or "rate limit" in text or "quota" in text


def _build_groq_messages(contents, config=None):
    system_instruction = None
    temperature = None
    response_format = None
    if config:
        system_instruction = config.get("system_instruction")
        temperature = config.get("temperature")
        response_format = config.get("response_format")
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    if isinstance(contents, list):
        parts = []
        for item in contents:
            if isinstance(item, str):
                parts.append({"type": "text", "text": item})
            elif isinstance(item, bytes):
                encoded = base64.b64encode(item).decode("utf-8")
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                })
            else:
                parts.append({"type": "text", "text": str(item)})
        messages.append({"role": "user", "content": parts})
    else:
        messages.append({"role": "user", "content": str(contents)})
    return messages

def generate_content_with_key_rotation(model: str, contents, config=None, cooldown_seconds: float = 60.0):
    """The single choke point every Groq call in this file goes through.
    Tries the pool's currently active key; if that key comes back rate
    limited, immediately rotates to the next key and retries the SAME
    request — looping through every configured key (multiple free-tier
    accounts) before giving up. Non-rate-limit errors are raised straight
    away so callers' own retry/backoff logic (transient 5xx, etc.) still
    applies on top of this."""
    if not groq_pool:
        raise HTTPException(status_code=500, detail="No GROQ_API_KEY(s) configured.")

    n = groq_pool.num_keys()
    last_err = None
    # Try every key at most once per call; if even a full loop through all
    # keys is rate limited, let the error bubble up to the caller's own
    # retry/backoff loop rather than spinning here.
    for _ in range(n):
        client, idx = groq_pool.current_client()
        try:
            request_kwargs = dict(config or {})
            request_kwargs.pop("system_instruction", None)
            request_kwargs.pop("response_mime_type", None)
            return client.chat.completions.create(
                model=model,
                messages=_build_groq_messages(contents, config),
                **request_kwargs
            )
        except Exception as e:
            last_err = e
            if _is_rate_limit_error(e):
                groq_pool.rotate(idx, cooldown_seconds=cooldown_seconds)
                continue
            raise
    raise last_err


# --- Custom Groq Evaluator ---
class GroqEvaluatorLLM(DeepEvalBaseLLM):
    def __init__(self, model_name="openai/gpt-oss-120b"):
        self.model_name = model_name

    def load_model(self):
        return groq_pool

    def generate(self, prompt: str) -> str:
        # DeepEval's metric templates put the "respond ONLY with this JSON
        # schema" instructions at the END of the prompt (after the
        # context/question). Blindly slicing prompt[:4000] chopped that tail
        # off, so the judge model never saw the JSON instructions and replied
        # in plain text -> "Evaluation LLM outputted an invalid JSON".
        # Fix: only truncate if we actually need to, and when we do, cut
        # from the MIDDLE (keep the head with task setup and the tail with
        # the output-format spec, which DeepEval always needs intact).
        # Keep the actual request comfortably below Groq's 8,000 TPM limit.
        # DeepEval adds its own instructions/schema around the supplied prompt,
        # so the source prompt must be kept well below the provider limit.
        max_chars = 6000
        if len(prompt) > max_chars:
            head_len = int(max_chars * 0.6)
            tail_len = max_chars - head_len
            truncated_prompt = (
                prompt[:head_len]
                + "\n...[truncated]...\n"
                + prompt[-tail_len:]
            )
        else:
            truncated_prompt = prompt
        max_retries = 5
        base_delay = 3.0
        limiter = get_rate_limiter(self.model_name)

        # Groq (like OpenAI) requires the literal word "json" to appear
        # somewhere in the prompt when response_format=json_object is set,
        # or the request is rejected outright. DeepEval's metric templates
        # always include JSON output instructions, so this holds in
        # practice — but fall back to an unconstrained call rather than
        # hard-failing if some template variant ever doesn't satisfy it.
        force_json = "json" in truncated_prompt.lower()

        for attempt in range(max_retries):
            try:
                limiter.acquire(estimate_tokens(truncated_prompt))
                request_config = {"temperature": 0.0}
                if force_json:
                    # DeepEval's metric templates only ASK for JSON in the
                    # prompt text; without Groq's actual JSON mode the judge
                    # model can drift into prose or markdown-fenced JSON,
                    # which DeepEval's json.loads then rejects with
                    # "Evaluation LLM outputted an invalid JSON." Forcing
                    # json_object here (same as the QA-formatting calls)
                    # makes Groq guarantee syntactically valid JSON.
                    request_config["response_format"] = {"type": "json_object"}
                response = generate_content_with_key_rotation(
                    model=self.model_name,
                    contents=truncated_prompt,
                    config=request_config,
                )
                content = (response.choices[0].message.content or "").strip()
                # Extra safety net: even in JSON mode some models still wrap
                # the object in a ```json ... ``` fence. Strip it so DeepEval's
                # own json.loads doesn't choke on the fence markers.
                fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", content, re.DOTALL | re.IGNORECASE)
                if fence_match:
                    content = fence_match.group(1).strip()
                return content
            except Exception as e:
                if _is_rate_limit_error(e) and attempt < max_retries - 1:
                    # Every key in the pool was already tried and rate limited
                    # inside generate_content_with_key_rotation — at this
                    # point we back off for real before looping the pool again.
                    sleep_time = self._resolve_retry_delay(str(e), attempt, base_delay)
                    print(f"[DeepEval] All keys rate limited on {self.model_name}. Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    continue
                if force_json and "json" in str(e).lower() and attempt < max_retries - 1:
                    # response_format=json_object was rejected by the
                    # provider for this particular prompt — retry once
                    # without it rather than failing the whole evaluation.
                    print(f"[DeepEval] json_object response_format rejected, retrying without it: {e}")
                    force_json = False
                    continue
                raise e

    @staticmethod
    def _resolve_retry_delay(error_text: str, attempt: int, base_delay: float) -> float:
        match = _RETRY_AFTER_RE.search(error_text)
        if match:
            groups = match.groups()
            value, unit = (groups[0], groups[1]) if groups[0] is not None else (groups[2], groups[3])
            suggested = float(value) / 1000.0 if unit.lower() == "ms" else float(value)
            return suggested + 0.5
        return base_delay * (attempt + 1)

    async def a_generate(self, prompt: str) -> str:
        async with _eval_semaphore:
            return await asyncio.to_thread(self.generate, prompt)

    def get_model_name(self) -> str:
        return self.model_name

# --- Vision Analysis Helper ---
def analyze_page_image_with_vision(image_bytes: bytes) -> str:
    """Passes page rendering image to Groq's multimodal model to describe visual content."""
    if not groq_pool:
        return ""

    try:
        response = generate_content_with_key_rotation(
            model=VISION_MODEL,
            contents=[
                (
                    "Examine this document page image. Describe all visual components, "
                    "including diagrams, flowcharts, UI mockups, infographics, visual tables, "
                    "and embedded image annotations in explicit detail."
                ),
                image_bytes,
            ],
            config={"temperature": 0.1},
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[Vision Analysis Failed]: {e}")
        return ""

def extract_document_text(file_bytes: bytes, filename: str) -> str:
    text_content = ""
    try:
        if filename.lower().endswith(".pdf"):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for i, page in enumerate(doc, start=1):
                # 1. Standard text extraction
                extracted_text = page.get_text().strip()
                
                # 2. Render page to image for Vision model
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                
                visual_description = analyze_page_image_with_vision(img_bytes)
                
                text_content += f"[PAGE {i}]\n"
                if extracted_text:
                    text_content += f"--- EXTRACTED TEXT ---\n{extracted_text}\n"
                if visual_description:
                    text_content += f"--- VISUAL & DIAGRAM DESCRIPTION ---\n{visual_description}\n"
                text_content += "\n"
        else:
            text_content = file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Text extraction error: {e}")
    return text_content.strip()

def chunk_document(text: str, chunk_size: int = 400, overlap: int = 40, max_chunks: int = 3) -> List[str]:
    text = text.strip()
    if not text:
        return ["No context provided."]

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text) and len(chunks) < max_chunks:
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks if chunks else [text[:chunk_size]]

# --- RAG Pipeline: field extraction (mirrors the frontend's flexible
# question/answer key matching in index.html, so any custom sample_json
# schema the user typed still resolves to a question/ground-truth pair) ---
QUESTION_FIELD_KEYS = ["question", "ques", "query", "input"]
ANSWER_FIELD_KEYS = ["answer", "ground_truth", "groundtruth", "expected", "expected_output", "response"]

def find_field(obj: dict, candidates: List[str]):
    if not isinstance(obj, dict):
        return None
    lower_map = {k.lower(): k for k in obj.keys()}
    for candidate in candidates:
        if candidate in lower_map:
            return obj[lower_map[candidate]]
    return None

def extract_items_array(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
    return None

def extract_questions_for_rag(generated_json) -> List[dict]:
    items = extract_items_array(generated_json)
    if not items:
        return []
    questions = []
    for idx, item in enumerate(items):
        q = find_field(item, QUESTION_FIELD_KEYS)
        a = find_field(item, ANSWER_FIELD_KEYS)
        if q:
            questions.append({
                "index": idx,
                "question": str(q),
                "expected_output": str(a) if a not in (None, "") else None
            })
    return questions

# --- RAG Pipeline: chunking, embedding, retrieval ---
def chunk_document_full(text: str, chunk_size: int = 1000, overlap: int = 200, max_chunks: Optional[int] = None) -> List[str]:
    """Chunks the full document. max_chunks is optional: the RAG pipeline
    passes None so large documents are not artificially capped at 300 chunks;
    the golden-data synthesis path can still pass an explicit cap."""
    text = text.strip()
    if not text:
        return []
    chunk_size = max(chunk_size, 50)
    overlap = min(max(overlap, 0), chunk_size - 1)
    step = max(chunk_size - overlap, 1)

    chunks = []
    start = 0
    while start < len(text) and (max_chunks is None or len(chunks) < max_chunks):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def select_contexts_for_synthesis(chunk_texts: List[str], count: int) -> List[List[str]]:
    """Evenly samples `count` chunks across the full document and wraps each
    as its own single-chunk context group, so each golden the synthesizer
    produces is grounded in one contiguous, spread-out piece of the doc
    rather than clustering near the start. If the doc has fewer chunks than
    `count`, every chunk is used once (synthesize_golden_dataset tops the
    count off via max_goldens_per_context instead)."""
    if not chunk_texts:
        return []
    n = len(chunk_texts)
    if n >= count:
        indices = sorted({int(i * n / count) for i in range(count)})
        if len(indices) < count:
            remaining = [i for i in range(n) if i not in indices]
            indices = sorted(indices + remaining[: count - len(indices)])
        return [[chunk_texts[i]] for i in indices]
    return [[c] for c in chunk_texts]


async def synthesize_golden_dataset(documents: List[dict], count: int) -> List[dict]:
    """Generates one combined golden QA dataset from all documents in a
    session. Each synthesis context is explicitly tagged with its source
    document, and some contexts can contain chunks from multiple documents so
    the resulting golden set can test cross-document questions as well."""
    if not documents:
        return []

    # Build source-tagged chunks independently so provenance is never lost.
    document_chunks = []
    for doc in documents:
        filename = doc.get("filename", "document")
        chunks = chunk_document_full(
            doc.get("source_doc", ""),
            chunk_size=800,
            overlap=100,
            max_chunks=max(count * 3, 60),
        )
        tagged = [f"[SOURCE DOCUMENT: {filename}]\n{chunk}" for chunk in chunks if chunk.strip()]
        if tagged:
            document_chunks.append((filename, tagged))

    if not document_chunks:
        return []

    # Distribute contexts across all uploaded documents. When multiple
    # documents exist, periodically provide a two-document context so the
    # synthesizer can create multi-document questions where appropriate.
    contexts: List[List[str]] = []
    pointers = [0] * len(document_chunks)
    while len(contexts) < count:
        made_progress = False
        for i, (_, chunks) in enumerate(document_chunks):
            if len(contexts) >= count:
                break
            if pointers[i] < len(chunks):
                primary = chunks[pointers[i]]
                pointers[i] += 1
                if len(document_chunks) > 1 and len(contexts) % 3 == 2:
                    j = (i + 1) % len(document_chunks)
                    if pointers[j] < len(document_chunks[j][1]):
                        contexts.append([primary, document_chunks[j][1][pointers[j]]])
                        pointers[j] += 1
                    else:
                        contexts.append([primary])
                else:
                    contexts.append([primary])
                made_progress = True
        if not made_progress:
            break

    if not contexts:
        return []

    evolution_config = SYNTHESIS_EVOLUTION_CONFIG.get(count, SYNTHESIS_EVOLUTION_CONFIG[20])
    synth_llm = GroqEvaluatorLLM(model_name=SYNTHESIZER_MODEL)
    synthesizer = Synthesizer(model=synth_llm, async_mode=True, evolution_config=evolution_config)

    # Groq has a strict per-request token limit. Sending every synthesis
    # context in one call can exceed that limit for larger documents. Process
    # the already-constructed contexts in small batches and combine the
    # resulting goldens. The context construction above is intentionally
    # unchanged so multi-document contexts are preserved.
    # One context per synthesizer request keeps the provider request small.
    SYNTHESIS_BATCH_SIZE = 1
    goldens = []
    try:
        for batch_start in range(0, len(contexts), SYNTHESIS_BATCH_SIZE):
            batch_contexts = contexts[batch_start:batch_start + SYNTHESIS_BATCH_SIZE]
            batch_goldens = await synthesizer.a_generate_goldens_from_contexts(
                contexts=batch_contexts,
                include_expected_output=True,
                max_goldens_per_context=1,
            )
            if batch_goldens:
                goldens.extend(batch_goldens)
            if len(goldens) >= count:
                break
    except Exception as e:
        print(f"[Synthesizer] Golden dataset generation failed, falling back to direct generation: {e}")
        return []

    golden_items = []
    for g in goldens:
        if not g or not g.input:
            continue
        context = g.context or []
        source_documents, source_pages = _source_metadata_from_context(context)
        golden_items.append({
            "question": g.input,
            "expected_output": g.expected_output,
            "context": context,
            "source_documents": source_documents,
            "source_pages": source_pages,
            "multi_document": len(source_documents) > 1,
        })

    return golden_items[:count]


def get_embedding_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="sentence-transformers is not installed. Run: pip install sentence-transformers"
        )
    model_name = (model_name or "").strip() or DEFAULT_EMBEDDING_MODEL
    if model_name not in _EMBEDDING_MODEL_CACHE:
        try:
            _EMBEDDING_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load embedding model '{model_name}': {e}"
            )
    return _EMBEDDING_MODEL_CACHE[model_name]

def mmr_select(chunk_embeddings: np.ndarray, sims_to_query: np.ndarray, top_k: int, lambda_param: float = 0.5) -> List[int]:
    """Maximal Marginal Relevance re-ranking over the small candidate set
    returned by Chroma. Only candidate vectors are held in memory."""
    if len(sims_to_query) == 0:
        return []
    top_k = max(1, min(top_k, len(sims_to_query)))
    selected: List[int] = []
    candidates = list(range(len(sims_to_query)))
    while len(selected) < top_k and candidates:
        if not selected:
            best = max(candidates, key=lambda i: float(sims_to_query[i]))
        else:
            def mmr_score(i):
                relevance = float(sims_to_query[i])
                diversity = max(float(chunk_embeddings[i] @ chunk_embeddings[j]) for j in selected)
                return lambda_param * relevance - (1 - lambda_param) * diversity
            best = max(candidates, key=mmr_score)
        selected.append(best)
        candidates.remove(best)
    return selected


def _get_chroma_client():
    """Creates one process-wide persistent Chroma client."""
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        with _CHROMA_CLIENT_LOCK:
            if _CHROMA_CLIENT is None:
                try:
                    import chromadb
                except ImportError:
                    raise HTTPException(
                        status_code=500,
                        detail="chromadb is not installed. Run: pip install chromadb"
                    )
                os.makedirs(CHROMA_DB_DIR, exist_ok=True)
                _CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return _CHROMA_CLIENT


def _chroma_collection_name(session_id: str) -> str:
    """Chroma collection names must satisfy Chroma's naming constraints."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", session_id).strip("-_")
    name = f"rag-{safe or 'session'}"
    return name[:63]


def _document_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _iter_document_chunks(text: str, chunk_size: int, overlap: int):
    """Yield chunks one at a time so embedding/indexing does not require a
    second full in-memory array of all chunks."""
    text = text.strip()
    if not text:
        return
    chunk_size = max(chunk_size, 50)
    overlap = min(max(overlap, 0), chunk_size - 1)
    step = max(chunk_size - overlap, 1)
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            yield chunk
        start += step


def _collection_matches_config(collection, document_hash: str, embedding_model: str, chunk_size: int, chunk_overlap: int) -> bool:
    metadata = collection.metadata or {}
    return (
        metadata.get("document_hash") == document_hash
        and metadata.get("embedding_model") == embedding_model
        and int(metadata.get("chunk_size", -1)) == int(chunk_size)
        and int(metadata.get("chunk_overlap", -1)) == int(chunk_overlap)
        and collection.count() > 0
    )


def _build_or_get_chroma_collection(
    session_id: str,
    doc_text: str,
    filename: str,
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
):
    """Return a persistent Chroma collection containing the current document
    indexed with the current embedding/chunk configuration. If the document
    or embedding/chunk configuration changed, the session collection is rebuilt."""
    client = _get_chroma_client()
    collection_name = _chroma_collection_name(session_id)
    doc_hash = _document_hash(doc_text)

    with _CHROMA_COLLECTION_LOCK:
        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            collection = None

        if collection is not None and _collection_matches_config(
            collection, doc_hash, embedding_model_name, chunk_size, chunk_overlap
        ):
            return collection

        if collection is not None:
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass

        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "session_id": session_id,
                "document_hash": doc_hash,
                "embedding_model": embedding_model_name,
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
                "filename": filename or "",
            },
        )

        embed_model = get_embedding_model(embedding_model_name)
        batch_texts: List[str] = []
        batch_embeddings = []
        batch_metadatas = []
        batch_ids = []
        chunk_index = 0

        def flush_batch():
            if not batch_texts:
                return
            embeddings = np.asarray(batch_embeddings).tolist()
            collection.add(
                ids=list(batch_ids),
                documents=list(batch_texts),
                embeddings=embeddings,
                metadatas=list(batch_metadatas),
            )
            batch_texts.clear()
            batch_embeddings.clear()
            batch_metadatas.clear()
            batch_ids.clear()

        for chunk in _iter_document_chunks(doc_text, chunk_size, chunk_overlap):
            batch_texts.append(chunk)
            batch_metadatas.append({
                "session_id": session_id,
                "document_hash": doc_hash,
                "filename": filename or "",
                "chunk_id": chunk_index,
            })
            batch_ids.append(f"{session_id}-{doc_hash[:12]}-{chunk_index}")
            chunk_index += 1

            if len(batch_texts) >= CHROMA_EMBED_BATCH_SIZE:
                encoded = embed_model.encode(batch_texts, normalize_embeddings=True)
                batch_embeddings.extend(np.asarray(encoded).tolist())
                flush_batch()

        if batch_texts:
            encoded = embed_model.encode(batch_texts, normalize_embeddings=True)
            batch_embeddings.extend(np.asarray(encoded).tolist())
            flush_batch()

        if collection.count() == 0:
            raise HTTPException(status_code=400, detail="Document produced no chunks for ChromaDB.")

        return collection


def retrieve_from_chroma(
    query: str,
    collection,
    embed_model,
    top_k: int,
    search_mode: str,
) -> List[str]:
    """Retrieve context directly from Chroma. Only the query embedding and the
    small candidate set are held in Python memory; the full document vectors
    remain in the persistent vector store."""
    top_k = max(1, int(top_k))
    query_vec = np.asarray(
        embed_model.encode([query], normalize_embeddings=True)[0],
        dtype=np.float32,
    )

    use_mmr = bool(search_mode and "mmr" in search_mode.lower())
    candidate_k = min(max(top_k * 4, top_k), 50) if use_mmr else top_k

    result = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=candidate_k,
        include=["documents", "metadatas", "embeddings", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    if not documents:
        return []

    if not use_mmr:
        return [str(doc) for doc in documents[:top_k]]

    raw_embeddings = (result.get("embeddings") or [[]])[0]
    if not raw_embeddings:
        return [str(doc) for doc in documents[:top_k]]

    candidate_embeddings = np.asarray(raw_embeddings, dtype=np.float32)
    similarities = candidate_embeddings @ query_vec
    selected_idx = mmr_select(candidate_embeddings, similarities, top_k)
    return [str(documents[i]) for i in selected_idx]


async def generate_rag_answer(question: str, context_chunks: List[str], chat_model: str, temperature: float, improvement_feedback: Optional[dict] = None) -> str:
    """Calls the configured chat model with ONLY the retrieved chunks as
    context — this is the actual RAG answer that gets evaluated, distinct
    from any ground_truth/expected_output the QA generator produced."""
    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant context was retrieved."
    system_instruction = (
        "You are a RAG assistant. Answer the user's question using ONLY the "
        "provided context. If the context does not contain the answer, say "
        "you don't know — do not use outside knowledge."
    )
    improvement_note = ""
    if improvement_feedback:
        previous_answer = str(improvement_feedback.get("answer") or "").strip()
        previous_reason = str(improvement_feedback.get("feedback_reason") or "").strip()
        improvement_note = (
            "\n\nPREVIOUS NEGATIVE FEEDBACK:\n"
            "A previous answer to this same question was rated negatively. "
            "Improve the new answer instead of blindly repeating the previous response.\n"
            f"Previous answer: {previous_answer}\n"
            f"Reason for negative feedback: {previous_reason or 'No reason was provided.'}\n"
            "Use the retrieved document context to correct or improve the answer. "
            "Do not mention this feedback to the user."
        )
    user_content = f"CONTEXT:\n{context_text}\n\nQUESTION:\n{question}\n\nAnswer concisely using only the context above.{improvement_note}"

    max_retries = 4
    base_delay = 2.0
    limiter = get_rate_limiter(chat_model)
    for attempt in range(max_retries):
        try:
            await asyncio.to_thread(limiter.acquire, estimate_tokens(context_text, question))
            response = await asyncio.to_thread(
                generate_content_with_key_rotation,
                model=chat_model,
                contents=user_content,
                config={
                    "temperature": temperature,
                    "system_instruction": system_instruction,
                },
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < max_retries - 1:
                # All keys in the pool were already tried inside
                # generate_content_with_key_rotation — back off before the
                # next full loop through the pool.
                sleep_time = GroqEvaluatorLLM._resolve_retry_delay(str(e), attempt, base_delay)
                print(f"[RAG Pipeline] All keys rate limited on {chat_model}. Retrying in {sleep_time:.1f}s...")
                await asyncio.sleep(sleep_time)
                continue
            raise

async def evaluate_single_question(question: str, rag_answer: str, context_chunks: List[str],
                                    expected_output: Optional[str], eval_llm: "GroqEvaluatorLLM") -> dict:
    """Runs DeepEval metrics for ONE question/answer pair produced by the RAG
    pipeline. ContextualPrecision/Recall are skipped when the question has no
    ground_truth/expected_output in the generated dataset, since those two
    metrics require an expected_output to compare against."""
    test_case = LLMTestCase(
        input=question,
        actual_output=rag_answer,
        retrieval_context=context_chunks if context_chunks else ["No context retrieved."],
        expected_output=expected_output
    )

    metrics = {
        "faithfulness": FaithfulnessMetric(threshold=0.7, model=eval_llm),
        "answer_relevancy": AnswerRelevancyMetric(threshold=0.7, model=eval_llm),
        "contextual_relevancy": ContextualRelevancyMetric(threshold=0.7, model=eval_llm),
    }
    if expected_output:
        metrics["contextual_precision"] = ContextualPrecisionMetric(threshold=0.7, model=eval_llm)
        metrics["contextual_recall"] = ContextualRecallMetric(threshold=0.7, model=eval_llm)

    await asyncio.gather(*(m.a_measure(test_case) for m in metrics.values()))

    metrics_out = {}
    scores = []
    all_passed = True
    for name, m in metrics.items():
        score = round(float(m.score), 2) if m.score is not None else 0.0
        passed = bool(m.is_successful())
        metrics_out[name] = {"score": score, "passed": passed, "reason": m.reason}
        scores.append(score)
        all_passed = all_passed and passed

    overall = round(sum(scores) / len(scores), 2) if scores else 0.0

    return {
        "question": question,
        "expected_output": expected_output,
        "rag_answer": rag_answer,
        "retrieved_context": context_chunks,
        "metrics": metrics_out,
        "overall_score": overall,
        "passed": all_passed
    }

def build_reference_answer(eval_llm: "GroqEvaluatorLLM", context_chunks: List[str]) -> Optional[str]:
    try:
        context_text = "\n---\n".join(context_chunks)[:4000]
        prompt = (
            "Extract 3 to 5 key factual statements directly present in the text below. "
            "Return them as a plain numbered list, one statement per line, with no extra commentary.\n\n"
            f"TEXT:\n{context_text}"
        )
        result = eval_llm.generate(prompt)
        return result.strip() if result else None
    except Exception as e:
        print(f"[DeepEval] Reference answer generation failed: {e}")
        return None

async def _run_genesis_geval(golden_items: List[dict], doc_text: str) -> dict:
    """Runs a GEval pass over genesis field extractions.

    For each extracted field, checks:
    1. The answer is not NOT_FOUND (coverage score)
    2. The answer matches the expected format for its field type
    3. Dropdown answers are one of the valid options

    Returns a metrics dict compatible with the existing renderEvaluation() UI.
    """
    total = len(golden_items)
    if total == 0:
        return {"overall_score": 0.0, "skipped": False, "genesis_coverage": 0.0}

    found = sum(1 for g in golden_items if g.get("expected_output") not in ("NOT_FOUND", None, ""))
    coverage = round(found / total, 2)

    # Format validation per field type
    format_ok = 0
    dropdown_ok = 0
    dropdown_total = 0
    for g in golden_items:
        val = g.get("expected_output", "")
        ftype = g.get("type", "")
        options = next((f.get("options") for f in GENESIS_FIELDS if f["field_name"] == g.get("field_name")), None)

        if val in ("NOT_FOUND", None, ""):
            continue

        # Format checks
        if ftype == "date":
            format_ok += 1 if re.match(r"\d{2}/\d{2}/\d{4}", val) else 0
        elif ftype == "calculated":
            format_ok += 1 if any(c in val for c in ["$", "%", "SF", "days"]) else 0
        elif ftype == "freeform":
            format_ok += 1 if len(val.strip()) > 2 else 0
        elif ftype in ("dropdown", "user_input"):
            format_ok += 1
        elif ftype == "narrative":
            format_ok += 1 if len(val.strip()) > 20 else 0
        else:
            format_ok += 1

        # Dropdown validation
        if options:
            dropdown_total += 1
            if any(val.lower() == o.lower() or val.lower() in o.lower() for o in options):
                dropdown_ok += 1

    format_score = round(format_ok / found, 2) if found > 0 else 0.0
    dropdown_score = round(dropdown_ok / dropdown_total, 2) if dropdown_total > 0 else 1.0
    overall = round((coverage + format_score + dropdown_score) / 3, 2)

    not_found_fields = [g["field_name"] for g in golden_items if g.get("expected_output") in ("NOT_FOUND", None, "")]

    return {
        "overall_score": overall,
        "skipped": False,
        "passed": overall >= 0.7,
        "genesis_coverage": coverage,
        "format_score": format_score,
        "dropdown_accuracy": dropdown_score,
        "fields_found": found,
        "fields_total": total,
        "not_found_fields": not_found_fields,
        "reason": (
            f"Coverage: {found}/{total} fields extracted ({coverage*100:.0f}%) | "
            f"Format valid: {format_ok}/{found} | "
            f"Dropdown accuracy: {dropdown_ok}/{dropdown_total} | "
            f"NOT_FOUND: {', '.join(not_found_fields) if not_found_fields else 'none'}"
        ),
    }


async def run_deepeval_evaluation(doc_text: str, generated_json: dict, sample_json: str = "") -> dict:
    """Aggregate, single-score DeepEval pass over the whole generated dataset,
    run immediately after generation (mirrors the original app's flow). This
    is intentionally NOT a per-question breakdown — that only happens later,
    once the user submits a model config and the RAG pipeline runs
    (see run_rag_pipeline / evaluate_single_question below)."""
    if not groq_pool:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing for evaluation.")

    try:
        rag_eval_llm = GroqEvaluatorLLM(model_name=RAG_JUDGE_MODEL)
        geval_eval_llm = GroqEvaluatorLLM(model_name=GEVAL_JUDGE_MODEL)

        context_chunks = chunk_document(doc_text) if doc_text else ["No context provided."]

        input_prompt = (
            "Generate structured QA test cases extracted strictly from the provided "
            "source document, adhering to the target JSON schema."
        )
        actual_output_str = json.dumps(generated_json)

        used_independent_reference = False
        expected_output = sample_json.strip() if sample_json and sample_json.strip() else None
        if not expected_output:
            expected_output = await asyncio.to_thread(build_reference_answer, rag_eval_llm, context_chunks)
            used_independent_reference = True
        if not expected_output:
            expected_output = actual_output_str
            used_independent_reference = False

        test_case = LLMTestCase(
            input=input_prompt,
            actual_output=actual_output_str,
            retrieval_context=context_chunks,
            expected_output=expected_output
        )

        faithfulness = FaithfulnessMetric(threshold=0.7, model=rag_eval_llm)
        answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=rag_eval_llm)
        contextual_relevancy = ContextualRelevancyMetric(threshold=0.7, model=rag_eval_llm)
        contextual_precision = ContextualPrecisionMetric(threshold=0.7, model=rag_eval_llm)
        contextual_recall = ContextualRecallMetric(threshold=0.7, model=rag_eval_llm)
        g_eval = GEval(
            name="Schema Structure & Quality",
            criteria=(
                "Step 1: Check that actual_output is valid JSON matching target schema fields. "
                "Step 2: Check that no extra/undocumented fields are present. "
                "Step 3: Check ground_truth values against retrieval_context facts. "
                "Step 4: Penalize score proportionally to unsupported ground_truth values."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
            model=geval_eval_llm,
            strict_mode=False
        )

        await asyncio.gather(
            faithfulness.a_measure(test_case),
            answer_relevancy.a_measure(test_case),
            contextual_relevancy.a_measure(test_case),
            contextual_precision.a_measure(test_case),
            contextual_recall.a_measure(test_case),
            g_eval.a_measure(test_case),
        )

        faith_score = round(float(faithfulness.score), 2)
        ans_rel_score = round(float(answer_relevancy.score), 2)
        ctx_rel_score = round(float(contextual_relevancy.score), 2)
        ctx_prec_score = round(float(contextual_precision.score), 2)
        ctx_rec_score = round(float(contextual_recall.score), 2)
        geval_score = round(float(g_eval.score), 2)

        scores = [faith_score, ans_rel_score, ctx_rel_score, ctx_prec_score, ctx_rec_score, geval_score]
        overall = round(sum(scores) / len(scores), 2)

        all_passed = (
            faithfulness.is_successful() and
            answer_relevancy.is_successful() and
            contextual_relevancy.is_successful() and
            contextual_precision.is_successful() and
            contextual_recall.is_successful() and
            g_eval.is_successful()
        )

        return {
            "overall_score": overall,
            "faithfulness": faith_score,
            "answer_relevancy": ans_rel_score,
            "contextual_relevancy": ctx_rel_score,
            "contextual_precision": ctx_prec_score,
            "contextual_recall": ctx_rec_score,
            "g_eval": geval_score,
            "passed": all_passed,
            "used_independent_reference": used_independent_reference,
            "retrieval_chunk_count": len(context_chunks),
            "reason": (
                f"Faithfulness: {faithfulness.reason} | "
                f"Answer Rel: {answer_relevancy.reason} | "
                f"Context Rel: {contextual_relevancy.reason} | "
                f"Context Prec: {contextual_precision.reason} | "
                f"Context Rec: {contextual_recall.reason} | "
                f"G-Eval: {g_eval.reason}"
            )
        }
    except Exception as e:
        print(f"DeepEval Execution Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"DeepEval evaluation failed via Groq: {str(e)}"
        )

# --- FastAPI Endpoints ---

@app.get("/api/sessions")
async def list_sessions():
    conn = get_db_connection()
    rows = conn.execute("SELECT session_id, title, status, is_golden FROM sessions ORDER BY session_id DESC").fetchall()
    conn.close()
    return [{"id": r["session_id"], "title": r["title"], "status": r["status"], "is_golden": bool(r["is_golden"])} for r in rows]

@app.post("/api/sessions")
async def create_session():
    new_id = str(uuid.uuid4())
    conn = get_db_connection()
    count = conn.execute('SELECT COUNT(*) FROM sessions').fetchone()[0] + 1
    title = f"Session {count}"
    conn.execute(
        "INSERT INTO sessions (session_id, title, sample_json, status, is_golden) VALUES (?, ?, ?, ?, ?)",
        (new_id, title, "", "Pending", 0)
    )
    conn.commit()
    conn.close()
    return {"id": new_id, "title": title}

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    documents = _ensure_legacy_document(session_id, row)
    processable_documents = _get_processable_documents(documents)
    combined_doc_text = _combined_document_text(processable_documents)

    return {
        "session_id": row["session_id"],
        "title": row["title"],
        "filename": row["filename"],
        "sample_json": row["sample_json"],
        "has_document": bool(combined_doc_text),
        "documents": _session_document_summary(documents),
        "status": row["status"],
        "is_golden": bool(row["is_golden"]),
        "generated_data": json.loads(row["generated_qa"]) if row["generated_qa"] else None,
        "deepeval_score": row["deepeval_score"],
        "deepeval_details": json.loads(row["deepeval_details"]) if row["deepeval_details"] else None,
        "test_case_count": row["test_case_count"] if row["test_case_count"] else 20,
        "rag_config": json.loads(row["rag_config"]) if row["rag_config"] else None,
        "chat_enabled": bool(row["rag_config"] and combined_doc_text),
        "per_question_results": json.loads(row["per_question_results"]) if row["per_question_results"] else None,
        "golden_dataset": json.loads(row["golden_dataset"]) if row["golden_dataset"] else None
    }


@app.delete("/api/sessions/{session_id}/documents/{document_id}")
async def delete_session_document(session_id: str, document_id: str):
    """Removes one document from a session. The session's QA/golden dataset is
    intentionally not regenerated here; the user clicks Generate to rebuild
    the current combined dataset from the remaining documents."""
    _ensure_documents_table()
    conn = get_db_connection()
    row = conn.execute(
        "SELECT document_id FROM session_documents WHERE session_id = ? AND document_id = ?",
        (session_id, document_id)
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found in this session.")

    conn.execute(
        "DELETE FROM session_documents WHERE session_id = ? AND document_id = ?",
        (session_id, document_id)
    )
    conn.commit()
    conn.close()

    documents = _get_session_documents(session_id)
    processable_documents = _get_processable_documents(documents)
    combined_doc_text = _combined_document_text(processable_documents)
    filenames = ", ".join(d["filename"] for d in documents)

    conn = get_db_connection()
    conn.execute(
        "UPDATE sessions SET filename = ?, source_doc = ? WHERE session_id = ?",
        (filenames, combined_doc_text, session_id)
    )
    conn.commit()
    conn.close()

    return {
        "message": "Document removed.",
        "documents": _session_document_summary(documents),
        "has_document": bool(combined_doc_text),
    }


@app.post("/api/sessions/{session_id}/generate")
async def generate_qa_testcases(
    session_id: str,
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
    sample_json: str = Form(...),
    test_case_count: int = Form(20),
    genesis_mode: bool = Form(False),
    genesis_field_count: int = Form(32),  # 10, 20, or 32 (all)
):
    if not genesis_mode and test_case_count not in ALLOWED_TEST_CASE_COUNTS:
        raise HTTPException(
            status_code=400,
            detail=f"test_case_count must be one of {ALLOWED_TEST_CASE_COUNTS}"
        )

    conn = get_db_connection()
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
    conn.close()

    # Existing sessions created before multi-document support are migrated into
    # the new document table automatically.
    documents = _ensure_legacy_document(session_id, row)

    upload_list: List[UploadFile] = []
    if files:
        upload_list.extend([f for f in files if f and f.filename])
    if file and file.filename:
        upload_list.append(file)

    if upload_list:
        for upload in upload_list:
            if _is_plans_document(upload.filename):
                # Store Plans files in the session, but do not extract or process them.
                _add_uploaded_document(session_id, upload.filename, "")
                continue

            content = await upload.read()
            extracted = extract_document_text(content, upload.filename)
            if not extracted.strip():
                continue
            _add_uploaded_document(session_id, upload.filename, extracted)
        documents = _get_session_documents(session_id)

    processable_documents = _get_processable_documents(documents)
    if not processable_documents:
        raise HTTPException(status_code=400, detail="Upload at least one processable document. Files with 'Plans' in the filename are stored only and are not used for generation.")

    if not groq_pool:
        raise HTTPException(status_code=500, detail="No GROQ_API_KEY(s) configured.")

    # One combined representation is used by the generation/evaluation path.
    # The golden synthesizer itself works directly from source-tagged document
    # contexts and does not use the vector store.
    doc_text = _combined_document_text(processable_documents)
    filename = ", ".join(d["filename"] for d in processable_documents)
    truncated_context = doc_text[:8000] if doc_text else "No document context uploaded."

    # ── Step 1: Generate goldens & QA test cases ──────────────────────────────
    if genesis_mode:
        # Genesis field-targeted mode: one golden per testable field (32 total).
        # Skips the generic DeepEval Synthesizer and directly queries the LLM
        # for each of the 32 extractable/calculable Genesis fields.
        n = min(max(genesis_field_count, 1), len(GENESIS_FIELDS))
        print(f"[Generate] Genesis mode: generating {n} field-targeted goldens (of {len(GENESIS_FIELDS)} total)...")
        golden_items, generated_json = await generate_genesis_field_goldens(
            processable_documents, sample_json, field_count=n
        )
        langsmith_dataset_id = save_golden_dataset_to_langsmith(session_id, golden_items)
    else:
        # Standard mode: synthesize ONE combined golden QA dataset from all documents.
        golden_items = await synthesize_golden_dataset(processable_documents, test_case_count)

        # Store this session's combined golden dataset in its one LangSmith dataset.
        # Regeneration replaces the previous examples for this same session.
        langsmith_dataset_id = save_golden_dataset_to_langsmith(session_id, golden_items)

        has_page_markers = "[PAGE " in truncated_context
        page_instruction = (
            "The document context contains [SOURCE DOCUMENT: filename] markers and [PAGE n] markers. "
            "For every test case, preserve the source document identity. If the target schema supports it, "
            "add a field named \"page_no\" set to the integer page number the underlying fact was drawn from. "
            "If a question requires more than one document, use the relevant source documents."
            if has_page_markers else
            "The document context contains [SOURCE DOCUMENT: filename] markers. Preserve the source document identity. "
            "If the target schema supports source_document/source_documents, populate it from the relevant document(s). "
            "Omit page_no when the source has no page markers."
        )

        if golden_items:
            system_prompt = (
                "You are an expert QA Automation Engineer. You are given multiple source documents AND a "
                "synthetically generated golden QA dataset derived from those documents. Reformat EACH golden "
                "item, in the same order, into the target JSON schema — do not invent new facts, do not drop items, "
                "do not merge items, do not add extra items. Preserve the original question intent and answer content. "
                "When a golden item is based on multiple documents, preserve that cross-document intent. "
                "Use the source document markers and page markers to fill schema fields when available."
            )
            max_retries = 4
            base_delay = 2.0
            QA_FORMAT_BATCH_SIZE = 2
            generated_items = []

            for batch_start in range(0, len(golden_items), QA_FORMAT_BATCH_SIZE):
                batch_items = golden_items[batch_start:batch_start + QA_FORMAT_BATCH_SIZE]

                # Build this batch's document context from ONLY the source
                # contexts attached to the golden items in the batch, instead of
                # sending the entire (truncated) document on every batch. Each
                # golden already carries the exact source-tagged chunk(s) it was
                # synthesized from, so this is sufficient grounding and is far
                # smaller than the 8,000-char whole-document context, which is
                # what was pushing batches over the Groq per-request token limit.
                batch_context_chunks: List[str] = []
                seen_chunks = set()
                for item in batch_items:
                    for chunk in (item.get("context") or []):
                        if chunk and chunk not in seen_chunks:
                            seen_chunks.add(chunk)
                            batch_context_chunks.append(chunk)
                batch_context = (
                    "\n\n".join(batch_context_chunks)
                    if batch_context_chunks
                    else "No document context uploaded."
                )
                # Defensive cap: chunk_document_full(chunk_size=800) keeps
                # individual chunks small, but guard against any unexpectedly
                # large context (e.g. a chunk carrying a long vision-model
                # description) still blowing the batch past the TPM limit.
                BATCH_CONTEXT_CHAR_CAP = 4000
                if len(batch_context) > BATCH_CONTEXT_CHAR_CAP:
                    batch_context = batch_context[:BATCH_CONTEXT_CHAR_CAP] + "\n...[truncated]..."

                # The raw "context" chunks are already included above in
                # DOCUMENT CONTEXT, and source_documents/source_pages were
                # already extracted from that same context at golden-synthesis
                # time (_source_metadata_from_context). Re-embedding the full
                # "context" list inside every golden item here duplicated that
                # same text a second time in the same request and was the
                # remaining cause of oversized batches. Send only the fields the
                # formatter actually needs to reformat each item.
                slim_batch_items = [
                    {
                        "question": item.get("question"),
                        "expected_output": item.get("expected_output"),
                        "source_documents": item.get("source_documents"),
                        "source_pages": item.get("source_pages"),
                        "multi_document": item.get("multi_document"),
                    }
                    for item in batch_items
                ]
                golden_dataset_json = json.dumps(slim_batch_items, indent=2)

                user_prompt = f"""DOCUMENT CONTEXT:
{batch_context}

GOLDEN QA DATASET (produce exactly one output item per golden item below, in this same order):
{golden_dataset_json}

TARGET SAMPLE JSON SCHEMA:
{sample_json if sample_json else "{}"}

SOURCE/PAGE INSTRUCTION:
{page_instruction}
"""
                batch_generated_json = None
                for attempt in range(max_retries):
                    try:
                        response = generate_content_with_key_rotation(
                            model=GENERATION_MODEL,
                            contents=user_prompt,
                            config={
                                "temperature": 0.2,
                                "system_instruction": system_prompt,
                                "response_format": {"type": "json_object"},
                            },
                        )
                        batch_generated_json = json.loads(response.choices[0].message.content)
                        break
                    except Exception as e:
                        if _is_rate_limit_error(e) and attempt < max_retries - 1:
                            sleep_time = GroqEvaluatorLLM._resolve_retry_delay(str(e), attempt, base_delay)
                            print(f"[Generate] All keys rate limited on {GENERATION_MODEL}. Retrying in {sleep_time:.1f}s...")
                            time.sleep(sleep_time)
                            continue
                        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

                if isinstance(batch_generated_json, list):
                    generated_items.extend(batch_generated_json)
                elif isinstance(batch_generated_json, dict):
                    for key in ("test_cases", "items", "data"):
                        if isinstance(batch_generated_json.get(key), list):
                            generated_items.extend(batch_generated_json[key])
                            break
                    else:
                        generated_items.append(batch_generated_json)

            generated_json = {"test_cases": generated_items[:len(golden_items)]}

        else:
            depth_instruction = TEST_CASE_DEPTH_INSTRUCTIONS[test_case_count]
            system_prompt = (
                "You are an expert QA Automation Engineer. Generate QA test cases extracted strictly from the "
                "provided source documents, including their text and visual/diagram descriptions, adhering to the "
                "target JSON schema. Do not use outside knowledge."
            )
            user_prompt = f"""DOCUMENT CONTEXT:
{truncated_context}

TARGET SAMPLE JSON SCHEMA:
{sample_json if sample_json else "{}"}

GENERATION DEPTH INSTRUCTION:
{depth_instruction}

SOURCE/PAGE INSTRUCTION:
{page_instruction}
"""
            max_retries = 4
            base_delay = 2.0
            for attempt in range(max_retries):
                try:
                    response = generate_content_with_key_rotation(
                        model=GENERATION_MODEL,
                        contents=user_prompt,
                        config={
                            "temperature": 0.2,
                            "system_instruction": system_prompt,
                            "response_format": {"type": "json_object"},
                        },
                    )
                    generated_json = json.loads(response.choices[0].message.content)
                    break
                except Exception as e:
                    if _is_rate_limit_error(e) and attempt < max_retries - 1:
                        sleep_time = GroqEvaluatorLLM._resolve_retry_delay(str(e), attempt, base_delay)
                        print(f"[Generate] All keys rate limited on {GENERATION_MODEL}. Retrying in {sleep_time:.1f}s...")
                        time.sleep(sleep_time)
                        continue
                    raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    await asyncio.sleep(1)

    # Step 3: aggregate DeepEval pass over the whole generated dataset.
    # Skip in genesis mode — the aggregate eval is meaningless for 32 individual
    # field extractions and the large payload causes JSON parse failures on Groq.
    # Per-field evaluation happens later via the RAG pipeline run.
    if genesis_mode:
        # For genesis mode, run a lightweight GEval pass that checks whether
        # each extracted field value is a valid, grounded answer.
        try:
            eval_results = await _run_genesis_geval(golden_items, doc_text)
        except Exception as e:
            print(f"[Generate] Genesis GEval error (non-fatal): {e}")
            eval_results = {
                "overall_score": None,
                "skipped": True,
                "reason": f"Genesis evaluation error: {str(e)}",
            }
    else:
        try:
            eval_results = await run_deepeval_evaluation(doc_text, generated_json, sample_json)
        except Exception as e:
            print(f"[Generate] DeepEval evaluation error (non-fatal): {e}")
            eval_results = {
                "overall_score": None,
                "skipped": True,
                "reason": f"Evaluation error: {str(e)}",
            }

    conn = get_db_connection()
    conn.execute(
        """UPDATE sessions
           SET filename = ?, sample_json = ?, source_doc = ?, generated_qa = ?,
               golden_dataset = ?, deepeval_score = ?, deepeval_details = ?, is_golden = 0, status = 'Pending',
               test_case_count = ?
           WHERE session_id = ?""",
        (filename, sample_json, doc_text, json.dumps(generated_json),
         json.dumps(golden_items), eval_results.get("overall_score"), json.dumps(eval_results),
         test_case_count, session_id)
    )
    conn.commit()
    conn.close()

    actual_count = None
    if isinstance(generated_json, list):
        actual_count = len(generated_json)
    elif isinstance(generated_json, dict):
        for v in generated_json.values():
            if isinstance(v, list):
                actual_count = len(v)
                break

    return {
        "generated_qa": generated_json,
        "golden_dataset": golden_items,
        "langsmith_dataset_id": langsmith_dataset_id,
        "deepeval": eval_results,
        "status": "Pending",
        "is_golden": False,
        "filename": filename,
        "documents": _session_document_summary(documents),
        "requested_count": len(golden_items) if genesis_mode else test_case_count,
        "actual_count": actual_count,
        "genesis_mode": genesis_mode,
    }

def _normalize_question_for_match(value: Any) -> str:
    """Normalizes a question for safe exact/near-exact golden lookup."""
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\s\?\.!]+$", "", text)
    return text

def _extract_qa_items(value: Any) -> List[dict]:
    """Finds QA objects anywhere in a generated JSON structure."""
    found = []
    if isinstance(value, list):
        for item in value:
            found.extend(_extract_qa_items(item))
    elif isinstance(value, dict):
        question = value.get("question")
        answer = value.get("expected_output", value.get("answer"))
        if question is not None and answer is not None:
            found.append({"question": str(question), "expected_output": str(answer)})
        else:
            for child in value.values():
                found.extend(_extract_qa_items(child))
    return found

def _append_manual_goldens(session_id: str, old_generated: Any, new_generated: Any, golden_dataset: Any) -> List[dict]:
    """Adds newly added/edited generated QA items as NEW goldens. Existing
    goldens are intentionally never modified or removed."""
    old_items = _extract_qa_items(old_generated)
    new_items = _extract_qa_items(new_generated)
    if not isinstance(golden_dataset, list):
        golden_dataset = []

    # Compare by position where possible. Any changed existing QA becomes a
    # new golden; extra items are also new goldens.
    additions = []
    for idx, item in enumerate(new_items):
        is_new_or_edited = idx >= len(old_items) or (
            _normalize_question_for_match(old_items[idx].get("question")) != _normalize_question_for_match(item.get("question"))
            or str(old_items[idx].get("expected_output", "")).strip() != str(item.get("expected_output", "")).strip()
        )
        if is_new_or_edited:
            additions.append({
                "question": item["question"],
                "expected_output": item["expected_output"],
                "context": [],
                "source_documents": [],
                "source_pages": [],
                "multi_document": False,
                "manual_from_qa_edit": True
            })

    if additions:
        golden_dataset.extend(additions)
    return golden_dataset

@app.put("/api/sessions/{session_id}/qa")
async def update_generated_qa(session_id: str, payload: dict):
    """Persists generated-QA edits. New/edited QA becomes a NEW golden
    testcase; previous goldens are preserved."""
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")

    raw = payload.get("generated_qa")
    if raw is None:
        conn.close()
        raise HTTPException(status_code=400, detail="Missing 'generated_qa' in request body.")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as e:
        conn.close()
        raise HTTPException(status_code=400, detail=f"generated_qa must be valid JSON: {e}")

    try:
        old_generated = json.loads(row["generated_qa"]) if row["generated_qa"] else []
        golden_dataset = json.loads(row["golden_dataset"]) if row["golden_dataset"] else []
        updated_goldens = _append_manual_goldens(session_id, old_generated, parsed, golden_dataset)
        conn.execute(
            "UPDATE sessions SET generated_qa = ?, golden_dataset = ?, status = 'Edited' WHERE session_id = ?",
            (json.dumps(parsed), json.dumps(updated_goldens), session_id)
        )
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise
    conn.close()

    # Keep the same per-session LangSmith golden dataset synchronized.
    langsmith_dataset_id = save_golden_dataset_to_langsmith(session_id, updated_goldens)
    return {"message": "Edits saved", "status": "Edited", "golden_dataset": updated_goldens, "langsmith_dataset_id": langsmith_dataset_id}

@app.post("/api/sessions/{session_id}/re-evaluate")
async def re_evaluate_generated_qa(session_id: str):
    """Re-runs the aggregate DeepEval evaluation on the session's (saved)
    generated QA dataset — useful right after editing it in the UI, without
    re-generating from scratch. Same aggregate scoring used right after
    generation; the per-question breakdown still only comes from /run-rag."""
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")

    if not row["generated_qa"]:
        conn.close()
        raise HTTPException(status_code=400, detail="No generated QA dataset on this session.")

    generated_json = json.loads(row["generated_qa"])
    documents = _ensure_legacy_document(session_id, row)
    processable_documents = _get_processable_documents(documents)
    doc_text = _combined_document_text(processable_documents)
    sample_json = row["sample_json"] or ""

    eval_results = await run_deepeval_evaluation(doc_text, generated_json, sample_json)

    conn.execute(
        "UPDATE sessions SET deepeval_score = ?, deepeval_details = ? WHERE session_id = ?",
        (eval_results["overall_score"], json.dumps(eval_results), session_id)
    )
    conn.commit()
    conn.close()
    return {"deepeval": eval_results}

@app.post("/api/sessions/{session_id}/save-rag-config")
async def save_rag_config(session_id: str, config: RAGConfigRequest):
    """Saves the RAG configuration for this session without running the pipeline.
    Saving the configuration is enough to enable the document chat UI."""
    if not groq_pool:
        raise HTTPException(status_code=500, detail="No GROQ_API_KEY(s) configured.")

    conn = get_db_connection()
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
    documents = _ensure_legacy_document(session_id, row)
    processable_documents = _get_processable_documents(documents)
    if not processable_documents:
        conn.close()
        raise HTTPException(status_code=400, detail="Upload at least one processable document before saving the RAG configuration. Files with 'Plans' in the filename are stored only and are not used for RAG.")

    config_dict = config.dict()
    conn.execute(
        "UPDATE sessions SET rag_config = ? WHERE session_id = ?",
        (json.dumps(config_dict), session_id)
    )
    conn.commit()
    conn.close()
    return {"rag_config": config_dict, "chat_enabled": True, "message": "Configuration saved."}

@app.post("/api/sessions/{session_id}/run-rag")
async def run_rag_pipeline(session_id: str, config: Optional[RAGConfigRequest] = None):
    """Runs an actual RAG pipeline (chunk -> embed -> retrieve -> generate
    answer) over the session's source document, once per generated question,
    using the config from the 'Models & Params' modal, then evaluates each
    question individually with DeepEval and returns/stores a per-question
    results table. The ground truth used for evaluation (expected_output)
    comes from the session's synthesized golden dataset — matched to each
    generated question by its original index — not from whatever the
    generated QA item itself happens to contain, since the golden dataset is
    the actual source of truth. Falls back to the generated item's own
    answer field only if no matching golden is available."""
    if not groq_pool:
        raise HTTPException(status_code=500, detail="No GROQ_API_KEY(s) configured.")

    conn = get_db_connection()
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")

    documents = _ensure_legacy_document(session_id, row)
    processable_documents = _get_processable_documents(documents)
    doc_text = _combined_document_text(processable_documents)
    generated_qa_raw = row["generated_qa"]
    golden_items = json.loads(row["golden_dataset"]) if row["golden_dataset"] else []

    # Run always uses the last saved configuration when no body is supplied.
    if config is None:
        if not row["rag_config"]:
            conn.close()
            raise HTTPException(status_code=400, detail="Save the RAG configuration before running the pipeline.")
        try:
            config = RAGConfigRequest(**json.loads(row["rag_config"]))
        except Exception as e:
            conn.close()
            raise HTTPException(status_code=400, detail=f"Saved RAG configuration is invalid: {e}")

    if not doc_text.strip():
        conn.close()
        raise HTTPException(status_code=400, detail="No source document on this session. Upload a document and generate test cases first.")
    if not generated_qa_raw:
        conn.close()
        raise HTTPException(status_code=400, detail="No generated QA test cases on this session. Generate test cases before running the RAG pipeline.")

    generated_json = json.loads(generated_qa_raw)
    questions = extract_questions_for_rag(generated_json)
    if not questions:
        conn.close()
        raise HTTPException(status_code=400, detail="Could not find any question fields in the generated QA dataset.")

    for q_item in questions:
        idx = q_item.get("index")
        if idx is not None and idx < len(golden_items):
            golden_expected = golden_items[idx].get("expected_output")
            if golden_expected:
                q_item["expected_output"] = golden_expected

    try:
        collection = await asyncio.to_thread(
            _build_or_get_chroma_collection,
            session_id,
            doc_text,
            ", ".join(d["filename"] for d in documents),
            config.embedding_model,
            config.chunk_size,
            config.chunk_overlap,
        )
    except HTTPException:
        conn.close()
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"ChromaDB indexing failed: {str(e)}")

    embed_model = get_embedding_model(config.embedding_model)
    eval_llm = GroqEvaluatorLLM(model_name=RAG_JUDGE_MODEL)
    chat_model = config.chat_model.strip() if config.chat_model and config.chat_model.strip() else GEVAL_JUDGE_MODEL

    async def process_one(q_item: dict) -> dict:
        async with _rag_semaphore:
            retrieved = await asyncio.to_thread(
                retrieve_from_chroma, q_item["question"], collection,
                embed_model, config.top_k, config.search_model
            )
            rag_answer = await generate_rag_answer(q_item["question"], retrieved, chat_model, config.temperature)
            return await evaluate_single_question(
                q_item["question"], rag_answer, retrieved, q_item["expected_output"], eval_llm
            )

    try:
        per_question_results = await asyncio.gather(*(process_one(q) for q in questions))
    except Exception as e:
        conn.close()
        print(f"RAG Pipeline Execution Error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG pipeline evaluation failed: {str(e)}")

    per_question_results = list(per_question_results)
    overall_scores = [r["overall_score"] for r in per_question_results]
    pipeline_overall = round(sum(overall_scores) / len(overall_scores), 2) if overall_scores else 0.0
    pipeline_passed = all(r["passed"] for r in per_question_results) if per_question_results else False

    config_dict = config.dict()

    conn.execute(
        "UPDATE sessions SET rag_config = ?, per_question_results = ? WHERE session_id = ?",
        (json.dumps(config_dict), json.dumps(per_question_results), session_id)
    )
    conn.commit()
    conn.close()

    return {
        "rag_config": config_dict,
        "per_question_results": per_question_results,
        "pipeline_overall_score": pipeline_overall,
        "pipeline_passed": pipeline_passed,
        "chunk_count": collection.count(),
        "question_count": len(per_question_results)
    }

@app.post("/api/sessions/{session_id}/chat")
async def chat_with_document(session_id: str, payload: dict):
    """Answers a document question using the session's saved RAG configuration.
    Each chat turn is stored in SQLite and, when LangSmith is configured, traced
    as a root RAG Chat trace with nested retrieval and Groq-generation spans."""
    if not groq_pool:
        raise HTTPException(status_code=500, detail="No GROQ_API_KEY(s) configured.")

    question = str(payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    conn = get_db_connection()
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")

    documents = _ensure_legacy_document(session_id, row)
    processable_documents = _get_processable_documents(documents)
    doc_text = _combined_document_text(processable_documents)
    if not doc_text.strip():
        conn.close()
        raise HTTPException(status_code=400, detail="No source documents on this session.")
    if not row["rag_config"]:
        conn.close()
        raise HTTPException(status_code=400, detail="Save the RAG configuration before using chat.")

    try:
        config = RAGConfigRequest(**json.loads(row["rag_config"]))
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=400, detail=f"Saved RAG configuration is invalid: {e}")

    previous_negative = conn.execute(
        """SELECT answer, feedback_reason FROM chat_messages
           WHERE session_id = ? AND LOWER(TRIM(question)) = LOWER(TRIM(?)) AND feedback = 'no'
           ORDER BY id DESC LIMIT 1""",
        (session_id, question)
    ).fetchone()
    improvement_feedback = dict(previous_negative) if previous_negative else None

    config_dict = config.dict()
    ls_client = get_langsmith_client()
    tracing_enabled = ls_client is not None and langsmith_trace is not None
    trace_metadata = {
        "session_id": session_id,
        "chat_model": config.chat_model,
        "embedding_model": config.embedding_model,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k": config.top_k,
        "search_model": config.search_model,
        "temperature": config.temperature,
        "document_filenames": [d["filename"] for d in processable_documents],
        "feedback_improvement_used": bool(improvement_feedback),
    }

    if tracing_enabled:
        trace_ctx = langsmith_trace(
            name="RAG Chat",
            run_type="chain",
            inputs={"question": question},
            metadata=trace_metadata,
            tags=["rag", "chat", "groq"],
            project_name=LANGSMITH_PROJECT,
            client=ls_client,
        )
    else:
        from contextlib import nullcontext
        trace_ctx = nullcontext(None)

    try:
        with trace_ctx as root_run:
            collection = await asyncio.to_thread(
                _build_or_get_chroma_collection,
                session_id,
                doc_text,
                ", ".join(d["filename"] for d in documents),
                config.embedding_model,
                config.chunk_size,
                config.chunk_overlap,
            )
            embed_model = get_embedding_model(config.embedding_model)

            if tracing_enabled:
                with langsmith_trace(
                    name="Retrieve Context",
                    run_type="retriever",
                    inputs={
                        "question": question,
                        "top_k": config.top_k,
                        "search_model": config.search_model,
                        "vector_store": "ChromaDB",
                        "collection": _chroma_collection_name(session_id),
                        "chunk_count": collection.count(),
                    },
                    metadata={
                        "session_id": session_id,
                        "embedding_model": config.embedding_model,
                        "chunk_size": config.chunk_size,
                        "chunk_overlap": config.chunk_overlap,
                        "vector_store": "ChromaDB",
                    },
                    tags=["rag", "retrieval", "chroma"],
                    project_name=LANGSMITH_PROJECT,
                    client=ls_client,
                ) as retrieval_run:
                    retrieved = await asyncio.to_thread(
                        retrieve_from_chroma, question, collection,
                        embed_model, config.top_k, config.search_model
                    )
                    if retrieval_run is not None:
                        retrieval_run.outputs = {
                            "retrieved_context": retrieved,
                            "retrieved_chunk_count": len(retrieved),
                            "vector_store": "ChromaDB",
                        }
            else:
                retrieved = await asyncio.to_thread(
                    retrieve_from_chroma, question, collection,
                    embed_model, config.top_k, config.search_model
                )

            generation_inputs = {
                "question": question,
                "context": retrieved,
                "chat_model": config.chat_model,
                "temperature": config.temperature,
                "improvement_feedback_used": bool(improvement_feedback),
            }
            if tracing_enabled:
                with langsmith_trace(
                    name="Groq Generation",
                    run_type="llm",
                    inputs=generation_inputs,
                    metadata={
                        "session_id": session_id,
                        "chat_model": config.chat_model,
                        "temperature": config.temperature,
                    },
                    tags=["llm", "groq", "rag"],
                    project_name=LANGSMITH_PROJECT,
                    client=ls_client,
                ) as generation_run:
                    answer = await generate_rag_answer(
                        question, retrieved, config.chat_model, config.temperature, improvement_feedback
                    )
                    if generation_run is not None:
                        generation_run.outputs = {"answer": answer}
            else:
                answer = await generate_rag_answer(
                    question, retrieved, config.chat_model, config.temperature, improvement_feedback
                )
            trace_id = str(root_run.id) if root_run is not None else None
            if root_run is not None:
                root_run.outputs = {
                    "question": question,
                    "answer": answer,
                    "retrieved_chunk_count": len(retrieved),
                    "chat_message_pending": True,
                }

            cursor = conn.execute(
                """INSERT INTO chat_messages
                   (session_id, question, answer, chat_model, embedding_model, chunk_size,
                    chunk_overlap, top_k, search_model, temperature, model_config,
                    retrieved_context, langsmith_trace_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id, question, answer, config.chat_model, config.embedding_model,
                    config.chunk_size, config.chunk_overlap, config.top_k, config.search_model,
                    config.temperature, json.dumps(config_dict), json.dumps(retrieved), trace_id
                )
            )
            message_id = cursor.lastrowid
            conn.commit()
            conn.close()

            if root_run is not None:
                root_run.outputs["message_id"] = message_id

            if root_run is not None:
                pass

            return {
                "message_id": message_id,
                "question": question,
                "answer": answer,
                "feedback": None,
                "langsmith_trace_id": trace_id,
            }
    except HTTPException:
        conn.close()
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")

@app.post("/api/chat/{message_id}/metrics")
async def calculate_chat_metrics(message_id: int):
    """Evaluates one stored chatbot answer. If the question exactly matches
    a golden question (ignoring case/whitespace/final punctuation), its golden
    expected_output is used as ground truth, enabling Context Precision/Recall.
    Otherwise only the three ground-truth-free metrics are calculated."""
    conn = get_db_connection()
    row = conn.execute(
        "SELECT id, session_id, question, answer, retrieved_context FROM chat_messages WHERE id = ?",
        (message_id,)
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Chat message not found")

    session = conn.execute(
        "SELECT golden_dataset FROM sessions WHERE session_id = ?",
        (row["session_id"],)
    ).fetchone()
    conn.close()

    golden_match = None
    if session and session["golden_dataset"]:
        try:
            goldens = json.loads(session["golden_dataset"])
        except Exception:
            goldens = []
        target = _normalize_question_for_match(row["question"])
        for golden in goldens if isinstance(goldens, list) else []:
            if _normalize_question_for_match(golden.get("question")) == target:
                golden_match = golden
                break

    try:
        retrieved = json.loads(row["retrieved_context"] or "[]")
    except Exception:
        retrieved = []
    if not isinstance(retrieved, list):
        retrieved = [str(retrieved)]

    eval_llm = GroqEvaluatorLLM(model_name=RAG_JUDGE_MODEL)
    result = await evaluate_single_question(
        row["question"], row["answer"], retrieved,
        (golden_match or {}).get("expected_output") if golden_match else None,
        eval_llm
    )
    result["golden_match"] = bool(golden_match)
    return result

@app.get("/api/sessions/{session_id}/chat-history")
async def get_chat_history(session_id: str):
    conn = get_db_connection()
    row = conn.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
    rows = conn.execute(
        """SELECT id, question, answer, feedback, feedback_reason, created_at, langsmith_trace_id
           FROM chat_messages WHERE session_id = ? ORDER BY id ASC""",
        (session_id,)
    ).fetchall()
    conn.close()
    return {"messages": [dict(r) for r in rows]}


@app.post("/api/chat/{message_id}/feedback")
async def save_chat_feedback(message_id: int, payload: dict):
    feedback = str(payload.get("feedback") or "").strip().lower()
    reason = str(payload.get("reason") or "").strip()[:500] or None
    if feedback not in ("yes", "no"):
        raise HTTPException(status_code=400, detail="Feedback must be 'yes' or 'no'.")
    if feedback == "yes":
        reason = None

    conn = get_db_connection()
    row = conn.execute(
        "SELECT id, langsmith_trace_id, question, answer, session_id FROM chat_messages WHERE id = ?",
        (message_id,)
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Chat message not found")

    conn.execute(
        "UPDATE chat_messages SET feedback = ?, feedback_reason = ? WHERE id = ?",
        (feedback, reason if feedback == "no" else None, message_id)
    )
    conn.commit()
    conn.close()

    # Keep user feedback in LangSmith attached to the exact RAG Chat trace.
    # Passing trace_id enables background/batched feedback ingestion.
    ls_client = get_langsmith_client()
    if ls_client is not None and row["langsmith_trace_id"]:
        try:
            ls_client.create_feedback(
                trace_id=str(row["langsmith_trace_id"]),
                key="user_feedback",
                score=1 if feedback == "yes" else 0,
                value=feedback,
                comment=reason if feedback == "no" else None,
                extra={
                    "message_id": message_id,
                    "session_id": row["session_id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "feedback_reason": reason,
                },
            )
        except Exception as e:
            # LangSmith must never break the application's own feedback flow.
            print(f"[LangSmith] Failed to record chat feedback: {e}")

    return {
        "message_id": message_id,
        "feedback": feedback,
        "feedback_reason": reason if feedback == "no" else None,
        "langsmith_recorded": bool(ls_client is not None and row["langsmith_trace_id"]),
    }

@app.post("/api/sessions/{session_id}/approve")
async def approve_golden_dataset(session_id: str):
    conn = get_db_connection()
    conn.execute("UPDATE sessions SET is_golden = 1, status = 'Approved' WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
    return {"message": "Saved to Golden Dataset", "status": "Approved", "is_golden": True}

@app.get("/api/sessions/{session_id}/golden-dataset/export")
async def export_golden_dataset(session_id: str):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT session_id, filename, golden_dataset FROM sessions WHERE session_id = ?",
        (session_id,)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    golden_dataset = json.loads(row["golden_dataset"]) if row["golden_dataset"] else []
    safe_session_id = re.sub(r"[^A-Za-z0-9_-]+", "_", session_id)
    filename = f"golden_dataset_{safe_session_id}.json"

    return Response(
        content=json.dumps(golden_dataset, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
app.mount("/", StaticFiles(directory="static", html=True), name="static")