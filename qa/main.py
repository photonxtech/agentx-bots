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
from typing import Optional, List
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

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "qwen/qwen3-32b")
VISION_MODEL = os.getenv("VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")  # Groq multimodal model
GEVAL_JUDGE_MODEL = os.getenv("GEVAL_JUDGE_MODEL", "openai/gpt-oss-120b")
RAG_JUDGE_MODEL = os.getenv("RAG_JUDGE_MODEL", "openai/gpt-oss-120b")
SYNTHESIZER_MODEL = os.getenv("SYNTHESIZER_MODEL", "qwen/qwen3-32b")

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
    chat_model: str = Field(default="llama-3.3-70b-versatile")
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
    def __init__(self, model_name="llama-3.3-70b-versatile"):
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
        max_chars = 12000
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

        for attempt in range(max_retries):
            try:
                limiter.acquire(estimate_tokens(truncated_prompt))
                response = generate_content_with_key_rotation(
                    model=self.model_name,
                    contents=truncated_prompt,
                    config={"temperature": 0.0},
                )
                return (response.choices[0].message.content or "")
            except Exception as e:
                if _is_rate_limit_error(e) and attempt < max_retries - 1:
                    # Every key in the pool was already tried and rate limited
                    # inside generate_content_with_key_rotation — at this
                    # point we back off for real before looping the pool again.
                    sleep_time = self._resolve_retry_delay(str(e), attempt, base_delay)
                    print(f"[DeepEval] All keys rate limited on {self.model_name}. Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
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


async def synthesize_golden_dataset(doc_text: str, count: int) -> List[dict]:
    """Uses DeepEval's Synthesizer to generate a golden QA dataset grounded
    directly in the source document. This becomes the ground truth for both
    the generation prompt (each golden is reformatted 1:1 into the target
    schema) and the per-item evaluation afterwards — replacing the old
    approach of using sample_json (or a single auto-summarized reference)
    as the expected_output for evaluation."""
    if not doc_text or not doc_text.strip():
        return []

    chunk_texts = chunk_document_full(doc_text, chunk_size=800, overlap=100, max_chunks=max(count * 3, 60))
    if not chunk_texts:
        return []

    contexts = select_contexts_for_synthesis(chunk_texts, count)
    if not contexts:
        return []

    max_goldens_per_context = 1 if len(chunk_texts) >= count else max(1, -(-count // len(contexts)))
    evolution_config = SYNTHESIS_EVOLUTION_CONFIG.get(count, SYNTHESIS_EVOLUTION_CONFIG[20])

    synth_llm = GroqEvaluatorLLM(model_name=SYNTHESIZER_MODEL)
    synthesizer = Synthesizer(model=synth_llm, async_mode=True, evolution_config=evolution_config)

    try:
        goldens = await synthesizer.a_generate_goldens_from_contexts(
            contexts=contexts,
            include_expected_output=True,
            max_goldens_per_context=max_goldens_per_context,
        )
    except Exception as e:
        print(f"[Synthesizer] Golden dataset generation failed, falling back to direct generation: {e}")
        return []

    golden_items = [
        {
            "question": g.input,
            "expected_output": g.expected_output,
            "context": g.context or [],
        }
        for g in goldens if g and g.input
    ]
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

    return {
        "session_id": row["session_id"],
        "title": row["title"],
        "filename": row["filename"],
        "sample_json": row["sample_json"],
        "has_document": bool(row["source_doc"]),
        "status": row["status"],
        "is_golden": bool(row["is_golden"]),
        "generated_data": json.loads(row["generated_qa"]) if row["generated_qa"] else None,
        "deepeval_score": row["deepeval_score"],
        "deepeval_details": json.loads(row["deepeval_details"]) if row["deepeval_details"] else None,
        "test_case_count": row["test_case_count"] if row["test_case_count"] else 20,
        "rag_config": json.loads(row["rag_config"]) if row["rag_config"] else None,
        "chat_enabled": bool(row["rag_config"] and row["source_doc"]),
        "per_question_results": json.loads(row["per_question_results"]) if row["per_question_results"] else None,
        "golden_dataset": json.loads(row["golden_dataset"]) if row["golden_dataset"] else None
    }

@app.post("/api/sessions/{session_id}/generate")
async def generate_qa_testcases(
    session_id: str,
    file: Optional[UploadFile] = File(None),
    sample_json: str = Form(...),
    test_case_count: int = Form(20)
):
    if test_case_count not in ALLOWED_TEST_CASE_COUNTS:
        raise HTTPException(
            status_code=400,
            detail=f"test_case_count must be one of {ALLOWED_TEST_CASE_COUNTS}"
        )

    conn = get_db_connection()
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")

    filename = row["filename"]
    doc_text = row["source_doc"] or ""

    if file and file.filename:
        filename = file.filename
        content = await file.read()
        doc_text = extract_document_text(content, filename)

    if not groq_pool:
        conn.close()
        raise HTTPException(status_code=500, detail="No GROQ_API_KEY(s) configured.")

    truncated_context = doc_text[:8000] if doc_text else "No document context uploaded."

    # Step 1: synthesize a golden QA dataset from the actual document via
    # DeepEval's Synthesizer. This is the ground truth from here on —
    # sample_json is used ONLY for output schema/shape, never as expected_output.
    golden_items = await synthesize_golden_dataset(doc_text, test_case_count)

    has_page_markers = "[PAGE " in truncated_context
    page_instruction = (
        "The document context contains [PAGE n] markers showing where each page starts. "
        "For every test case, add a field named \"page_no\" set to the integer page number "
        "the underlying fact was drawn from — cross-reference each item's question/answer "
        "against the document context below to find the right page."
        if has_page_markers else
        "The document has no page markers (non-PDF source). Omit any page_no field, "
        "or set it to null if your schema requires the key to be present."
    )

    if golden_items:
        # Step 2: give the LLM the golden dataset + the document, and have it
        # reformat each golden 1:1 (same order, same count) into the target
        # schema — this keeps generation aligned with the goldens so the
        # per-item eval below can match by index with no fuzzy matching.
        golden_dataset_json = json.dumps(golden_items, indent=2)
        system_prompt = (
            "You are an expert QA Automation Engineer. You are given a source document AND a "
            "synthetically generated golden QA dataset already derived from that same document. "
            "Reformat EACH golden item, in the same order, into the target JSON schema — do not "
            "invent new facts, do not drop items, do not merge items, do not add extra items. "
            "Preserve the original question intent and answer content; only adapt structure and "
            "field names to match the schema, and fill in any schema fields (like page_no) using "
            "the source document."
        )
        user_prompt = f"""DOCUMENT CONTEXT:
{truncated_context}

GOLDEN QA DATASET (produce exactly one output item per golden item below, in this same order):
{golden_dataset_json}

TARGET SAMPLE JSON SCHEMA:
{sample_json if sample_json else "{}"}

PAGE NUMBER INSTRUCTION:
{page_instruction}
"""
    else:
        # Fallback: golden synthesis produced nothing (e.g. very short/empty
        # doc, or the synthesizer call failed) — generate directly from the
        # document like before, and skip per-item eval since there's no
        # golden ground truth to evaluate against.
        depth_instruction = TEST_CASE_DEPTH_INSTRUCTIONS[test_case_count]
        system_prompt = (
            "You are an expert QA Automation Engineer. Generate QA test cases "
            "extracted strictly from the provided text and visual diagram descriptions in the source document, "
            "adhering to the target JSON schema."
        )
        user_prompt = f"""DOCUMENT CONTEXT:
{truncated_context}

TARGET SAMPLE JSON SCHEMA:
{sample_json if sample_json else "{}"}

GENERATION DEPTH INSTRUCTION:
{depth_instruction}

PAGE NUMBER INSTRUCTION:
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
            conn.close()
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    await asyncio.sleep(1)

    # Step 3: aggregate DeepEval pass over the whole generated dataset. This
    # is a single overall score shown right after generation — the detailed
    # per-question breakdown is deliberately NOT shown here; it only appears
    # later, once the user submits a model config and the RAG pipeline runs
    # (see /run-rag below).
    eval_results = await run_deepeval_evaluation(doc_text, generated_json, sample_json)

    conn.execute(
        """UPDATE sessions 
           SET filename = ?, sample_json = ?, source_doc = ?, generated_qa = ?, 
               golden_dataset = ?, deepeval_score = ?, deepeval_details = ?, is_golden = 0, status = 'Pending',
               test_case_count = ?
           WHERE session_id = ?""",
        (filename, sample_json, doc_text, json.dumps(generated_json),
         json.dumps(golden_items), eval_results["overall_score"], json.dumps(eval_results),
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
        "deepeval": eval_results,
        "status": "Pending",
        "is_golden": False,
        "filename": filename,
        "requested_count": test_case_count,
        "actual_count": actual_count
    }

@app.put("/api/sessions/{session_id}/qa")
async def update_generated_qa(session_id: str, payload: dict):
    """Persists user edits made to the generated QA dataset in the UI. Marks
    the session 'Edited' so it's visually distinct from a freshly generated,
    unreviewed dataset; the stored golden_dataset is left untouched so the
    edited QA can be re-evaluated against it via /re-evaluate."""
    conn = get_db_connection()
    row = conn.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")

    raw = payload.get("generated_qa")
    if raw is None:
        conn.close()
        raise HTTPException(status_code=400, detail="Missing 'generated_qa' in request body.")

    # Accept either an already-parsed JSON value or a JSON string.
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as e:
        conn.close()
        raise HTTPException(status_code=400, detail=f"generated_qa must be valid JSON: {e}")

    conn.execute(
        "UPDATE sessions SET generated_qa = ?, status = 'Edited' WHERE session_id = ?",
        (json.dumps(parsed), session_id)
    )
    conn.commit()
    conn.close()
    return {"message": "Edits saved", "status": "Edited"}

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
    doc_text = row["source_doc"] or ""
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
    row = conn.execute("SELECT source_doc FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
    if not (row["source_doc"] or "").strip():
        conn.close()
        raise HTTPException(status_code=400, detail="Upload a document before saving the RAG configuration.")

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

    doc_text = row["source_doc"] or ""
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
            row["filename"] or "",
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

    doc_text = row["source_doc"] or ""
    if not doc_text.strip():
        conn.close()
        raise HTTPException(status_code=400, detail="No source document on this session.")
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
        "document_filename": row["filename"] or "",
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
                row["filename"] or "",
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

@app.get("/api/golden-dataset/export")
async def export_golden_dataset():
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM sessions WHERE is_golden = 1").fetchall()
    conn.close()

    return [{
        "session_id": r["session_id"],
        "filename": r["filename"],
        "qa_data": json.loads(r["generated_qa"]) if r["generated_qa"] else {},
        "deepeval_score": r["deepeval_score"],
        "status": r["status"]
    } for r in rows]

app.mount("/", StaticFiles(directory="static", html=True), name="static")