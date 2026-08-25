import json
import pickle  # converts python objects into 0/1s to store
import threading
import uuid  # unique session ids
import shutil  # recursively deletes session folders
import os
import logging
import re  # detecting broad "summarize/explain this document/photo" style questions
import sqlite3  # session persistence — replaces the old sessions.json flat file
from flashrank import Ranker
from contextlib import contextmanager
from pathlib import Path  # sets path
from datetime import datetime  # timestamps creation
from collections import defaultdict, Counter
from langchain_community.document_loaders import PyPDFLoader  # load PDF -> Document objects
from langchain_text_splitters import RecursiveCharacterTextSplitter  # split pages into chunks
from langchain_huggingface import HuggingFaceEmbeddings  # embedding model (text -> vectors)
from langchain_community.vectorstores import Chroma  # vector DB for semantic search
from langchain_community.retrievers import BM25Retriever  # keyword/lexical retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever  # combine BM25+vector, then rerank
from langchain_community.document_compressors import FlashrankRerank  # cross-encoder reranker
from langchain_groq import ChatGroq  # Groq-hosted LLM client
from groq import APIStatusError  # to catch 413 Payload Too Large specifically
from langchain_core.prompts import ChatPromptTemplate  # prompt templating
from langchain_core.output_parsers import StrOutputParser  # extract plain text from LLM response
from langchain_core.documents import Document  # generic doc object (used to wrap image content)
from langchain_core.messages import HumanMessage  # multimodal (text+image) message for the vision LLM
from dotenv import load_dotenv
from langsmith import traceable

import base64
from PIL import Image
import easyocr  # replaces pytesseract — noticeably better on noisy/rotated real-world photos

load_dotenv()

# --- Paths: everything scoped PER SESSION (per chat, which may now hold multiple PDFs) ---
BASE_DIR = Path(__file__).parent
STORAGE_DIR = BASE_DIR / "storage"

DATA_DIR = STORAGE_DIR / "data"
CHROMA_ROOT = STORAGE_DIR / "chroma_db"
CHUNKS_ROOT = STORAGE_DIR / "chunks"

DB_PATH = STORAGE_DIR / "sessions.db"
LEGACY_SESSIONS_FILE = STORAGE_DIR / "sessions.json"
for d in (DATA_DIR, CHROMA_ROOT, CHUNKS_ROOT):
    d.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# --- Cached Chroma clients, one per session_id ---
# Constructing Chroma(persist_directory=...) opens a fresh SQLite connection
# to the same chroma.sqlite3 file. Doing this on every chat turn (previously
# happening inside build_hybrid_retriever) plus on every upload merge (in
# process_pdf/process_image) meant multiple concurrent connections could hit
# the same file at once, which surfaced as intermittent
# "attempt to write a readonly database" errors when setting hnsw:search_ef.
# Caching one Chroma client per session and reusing it everywhere avoids the
# repeated opens. invalidate_vectordb_cache() must be called anywhere the
# underlying persist_dir is deleted/rebuilt from scratch, so the next
# get_vectordb() call constructs a fresh client instead of reusing a stale one.
_vectordb_cache: dict[str, Chroma] = {}
_vectordb_lock = threading.Lock()


def get_vectordb(session_id: str) -> Chroma:
    with _vectordb_lock:
        if session_id not in _vectordb_cache:
            persist_dir = CHROMA_ROOT / session_id
            _vectordb_cache[session_id] = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=_EMBEDDINGS,
            )
        return _vectordb_cache[session_id]


def invalidate_vectordb_cache(session_id: str):
    with _vectordb_lock:
        _vectordb_cache.pop(session_id, None)


# Vision-capable Groq model used to describe photos and transcribe any text in them.
# Read from an env var (with a hardcoded default as fallback) so a model
# deprecation can be fixed with a config change + restart instead of a code
# deploy. GROQ_VISION_MODEL_FALLBACK is tried automatically if the primary
# model call fails — see _describe_image() below. Groq's vision lineup changes
# fairly often; check https://console.groq.com/docs/vision for current ids.
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
VISION_MODEL_FALLBACK = os.getenv("GROQ_VISION_MODEL_FALLBACK", "meta-llama/llama-4-scout-17b-16e-instruct")

# EasyOCR reader — loaded once at import time (loading it per-call would reload
# the model weights on every single image, which is slow and pointless).
_OCR_READER = easyocr.Reader(["en"], gpu=False)

# --- Shared model singletons ---
# These used to be re-instantiated INSIDE build_hybrid_retriever() and
# process_pdf()/process_image()/remove_file_from_session() — meaning the
# embedding model and reranker were reloaded from scratch on every single
# chat message and every single upload. Traced via LangSmith, that reload
# was costing ~5.8s of a ~7.5s chat turn (build_hybrid_retriever span).
# Loading them once here, at import time, means every call just reuses the
# already-loaded model in memory instead of reloading weights each time.
_EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
_RERANKER = Ranker(
    model_name="ms-marco-MultiBERT-L-12",
    cache_dir=str(BASE_DIR / "cache" / "flashrank_cache")
)

# Same singleton fix applied to the LLM clients — these used to be built
# fresh inside get_answer() on every single chat message. Not as costly as
# the embedding/reranker reload (no local model weights involved, just an
# HTTP client), but it's needless churn on every request and keeps the
# pattern consistent.
# rag_utils.py
#
# reasoning_effort="low" on both clients below: openai/gpt-oss-120b is a
# reasoning model, and langchain_groq's own field docstring says Groq
# "will default to enabling reasoning if left undefined" — i.e. every call
# was already generating hidden <think>...</think> reasoning tokens before
# ever producing the visible output, on top of whatever temperature/prompt
# was set. Measured directly: even the OLD one-line "return only [1, 3]"
# attribution prompt burned ~450 hidden completion tokens and 6+ seconds
# before emitting a 4-character answer. That hidden cost is what actually
# blew up turn latency after the prompt-grounding fixes below — the richer
# prompts gave the model more to silently deliberate over. Since the
# grounding fixes now force the model to show its work as EXPLICIT, VISIBLE
# output instead (attribute_sources requires a written justification line
# per source; PROMPT_TEMPLATE spells out exactly what not to invent), heavy
# hidden reasoning on top of that is redundant, not additive — capping it to
# "low" removes the redundant cost without undoing those fixes. Not
# disabled entirely: some reasoning still helps the model actually apply the
# grounding rules rather than pattern-matching past them.
_CONDENSE_LLM = ChatGroq(model="openai/gpt-oss-120b", temperature=0, reasoning_effort="low")          # was: llama-3.3-70b-versatile
# Was 0.8. Confirmed via a real case (a "how do I run this meeting" question
# against the Red/Yellow/Green document) that 0.8 produces fluent but
# UNGROUNDED additions on top of what the Context actually supports — e.g.
# inventing specific manager dialogue lines and role-delegation advice that
# appear nowhere in the retrieved chunks, even though PROMPT_TEMPLATE already
# says to answer "using only the Context above". A RAG answer's job is
# faithfulness to retrieved content, not creative variation, so this is
# dropped to 0.2 (still non-zero — some natural phrasing/summarizing
# latitude is fine, near-zero would read as stilted) rather than matching
# _CONDENSE_LLM's temperature=0 (condensation is a narrower rewrite task,
# not open-ended answer composition).
_ANSWER_LLM = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, reasoning_effort="low")          # was: llama-3.3-70b-versatile

IMAGE_DESCRIBE_PROMPT = (
    "Describe this image in detail: objects, people, setting, layout, colors, mood. "
    "Then, on a new line, transcribe any text visible in the image verbatim, word for "
    "word. If there is no text visible, write 'No text visible.'"
)

# Answer prompt — includes conversation history for tone/continuity
PROMPT_TEMPLATE = """You are a helpful assistant answering questions about the uploaded document(s).

The conversation history is ONLY for understanding what the current question means — resolving
pronouns like "it"/"that", or understanding a follow-up like "what about the other one?". It is NOT
a source of facts. Do not carry over names, claims, or details from a previous answer into this one
unless the CURRENT Context below also supports them. A previous answer may have been wrong — treat
each new answer as sourced only from the Context section, never from what was said earlier in the
conversation.

Use ONLY the Context below to answer. If the answer isn't in the Context, say you don't know — even
if something related was discussed earlier in the conversation.

Do not invent specifics that aren't in the Context — no example dialogue, dates, names, statistics,
or advice you reasoned out yourself, even if it sounds like a natural extension of the topic. If the
question asks about something the Context touches on generally but doesn't give a specific answer
for (e.g. it asks about a scenario or a person's role that the Context never addresses), say what the
Context DOES cover and note that it doesn't address that specific part — don't fill the gap with a
plausible-sounding invention. Where the Context itself supplies example phrases, wording, or
statements for the situation being asked about, prefer quoting or closely paraphrasing those over
composing your own new example.

Format your answer using markdown where it improves clarity: use **bold** for key
terms or important values, and bullet points or numbered lists when listing multiple
items or steps. Don't over-format simple one-line answers — plain sentences are fine
when a list or bold text wouldn't add clarity.

Conversation history (for understanding the question only — not a source of facts):
{history}

Context:
{context}

Question: {question}

Answer clearly and concisely, using only the Context above:"""

# Dedicated prompt for direct "what's on page N" lookups (see
# extract_page_lookup / get_chunks_by_page in get_answer). PROMPT_TEMPLATE
# above was being reused here at first, but its "if the answer isn't in the
# Context, say you don't know" instruction was written for questions where
# the Context is a set of chunks that MIGHT contain the answer somewhere.
# For a page lookup, the Context IS the page — there's nothing to search
# for, it just needs to be described. Reusing the strict-grounding prompt
# caused the model to refuse ("the excerpt doesn't specify what's on page
# 98") even when the excerpt WAS page 98, because it was looking for an
# explicit sentence answering the question rather than treating the whole
# Context as the answer.
PAGE_LOOKUP_PROMPT_TEMPLATE = """You are a helpful assistant. The Context below is the exact, complete
content of the specific page the person asked about — it is not a set of search results, it IS the page.

Describe what's on this page: summarize its topic and, if it contains a list, exercise, table, or
distinct sections, mention the specific items/headings so the person knows what's there. Do not say
the page doesn't contain an answer to their question — describing the page's content IS the answer.

Format your answer using markdown where it improves clarity: use **bold** for headings/key terms and
bullet points or numbered lists for multi-item content. Keep it concise — a short paragraph or a
handful of bullets is usually enough, not a full transcription.

Context (the page's content):
{context}

Question: {question}

Answer by describing what's on this page:"""

# Condensation prompt — rewrites a follow-up question into a standalone one BEFORE retrieval.
# This is the piece that was missing: without it, retrieval only ever sees the raw follow-up
# text ("what about pricing?") with no idea what "that" refers to.
#
# MERGED FROM THE STRICTER VARIANT: this version adds explicit anti-overreach
# guardrails that the simpler "just rewrite it if it depends on history" prompt
# didn't have. Without these, the condenser was prone to importing assumptions,
# framework names, or terminology from the PREVIOUS answer into a follow-up
# question that never actually referenced them — turning a genuinely new
# question into a mangled continuation of the old topic, which then sent
# retrieval off after the wrong chunks entirely.
CONDENSE_PROMPT_TEMPLATE = """Given the conversation history and a follow-up question, rewrite the
follow-up question as a standalone question ONLY if it depends on the conversation history to make
sense — for example, if it contains a pronoun ("it", "that", "they", "this") or an incomplete
reference ("what about pricing?", "the other one?") that only resolves using the previous turn.

If the follow-up question already names its own specific subject and can be understood on its own —
even if that subject sounds similar to something discussed earlier — return it UNCHANGED. Do NOT
import assumptions, framework names, or specific terminology from the previous answer into a
question that doesn't explicitly reference it. A new question introducing its own noun/subject is a
NEW question, not a continuation, even if a keyword overlaps with the prior topic.

When a rewrite IS needed, make the SMALLEST possible edit — substitute only the specific word(s) the
pronoun/reference stands for. Do NOT add descriptive framing, rephrase the question's structure, or
expand it into a longer or more formal version. The retrieval system searches for the user's exact
wording, so preserving their original phrase (e.g. "red person", not "personality type of someone
referred to as a 'red person'") matters more than making the question read more smoothly. When in
doubt, prefer returning the question closer to its original wording over a more "complete-sounding"
rewrite.

Example of what NOT to do: if the previous turn discussed strategies that mentioned "Carol Dweck"
in passing, and the follow-up asks "who is the author of this book" — do NOT rewrite it to mention
Carol Dweck. "This book" clearly refers to the uploaded document itself, not to any person named in
the previous answer. The correct rewrite is "who is the author of this book" — UNCHANGED. A person's
name appearing in the previous answer does not make it fair game to pull into an unrelated new
question, even if that name is the most recent salient detail in the conversation.

A second example of the SAME mistake, with numbers instead of names: if the previous turn was about
"12 Techniques to Increase Your Positivity and Outlook", and the follow-up asks "7 strategies align
with 4D-i" — do NOT rewrite it to "which seven of the 12 Techniques align with 4D-i". The follow-up
names its OWN subject ("7 strategies", "4D-i") that has nothing to do with "12 Techniques" or
"Positivity and Outlook" — those words never appear in the follow-up. The correct rewrite is "7
strategies align with 4D-i" — UNCHANGED. A shared word like a number ("7") or a repeated topic word
("strategies") is not evidence the two questions are related; only an actual pronoun or incomplete
reference justifies pulling in anything from the previous turn.

Do NOT change an explicit reference to a specific file type (e.g. "this photo", "this pdf",
"this image", "this document") — if the user names a type, keep that exact wording in your rewrite.
Output ONLY the rewritten question — no preamble, no quotes.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""

# Attribution prompt — asks the answer-model itself which of the retrieved
# sources it actually drew from, instead of guessing via word-overlap
# heuristics. Word overlap can't tell "genuinely used" apart from "shares
# real vocabulary because it's a related topic" — a topically-adjacent page
# can score HIGHER than the true source (e.g. a document's own separate
# "21 Strategies" list scoring 0.393 against a "7 strategies" question,
# while the real source sat at 0.964), and a genuinely-used page that gets
# paraphrased rather than quoted can score LOWER than any fixed threshold
# (e.g. a Fixed/Growth mindset table the model clearly drew from, dropped
# entirely once the threshold was raised to kill the first false positive).
# No single overlap threshold separates those two populations because their
# score distributions overlap by construction — every retune fixes one case
# and breaks another. Asking the model directly is exact, not approximate.
SOURCE_ATTRIBUTION_PROMPT = """You are given an ANSWER and a numbered list of SOURCE EXCERPTS that were
available as context when the answer was written.

Return ONLY a JSON array of the source numbers whose content was ACTUALLY USED to write the answer — i.e.
a specific claim, fact, list item, or detail in the answer genuinely came from that source. Do not include
a source just because it covers a similar or related topic; only include it if you can point to something
in the answer that source specifically supports.

This document reuses the same handful of terms (e.g. "red", "yellow", "green", "white", "decision-making",
"critical thinking", "understanding") across dozens of otherwise-unrelated pages — a chapter on writing
drills, a section on company history, and the actual how-to-communicate pages can all use this vocabulary
in passing. Shared terminology is NOT evidence a source was used. Before including a source number, check
that the excerpt contains the SAME specific content the answer states — the same named do/don't item, the
same example phrase or quote, the same list, the same named tactic — not just an excerpt that happens to
mention the same color word or dimension name in a different context (e.g. a page briefly summarizing what
"red decision-making" produces, in a paragraph about something else entirely, does not support an answer's
specific claims about how to communicate with a Red-style person unless it's actually the source of those
claims).

If no source was actually used (e.g. the answer says it doesn't know), return an empty array: []

Work through the sources ONE AT A TIME before deciding — a snap judgment is exactly what lets shared
vocabulary slip a wrong source past you on a document like this. For each numbered source, write one
line: either "[n] USED — <the specific claim/quote in the answer this source actually supports>" or
"[n] not used — <why: no matching specific content, or only shared terminology>". Only after going
through every source, output the final line in EXACTLY this format (nothing after it):
FINAL: [n, n, ...]

ANSWER:
{answer}

SOURCE EXCERPTS:
{numbered_sources}
"""

# Questions like "explain this pdf", "summarize this document", or "what's in
# this photo" don't closely match any ONE chunk by similarity search, so normal
# retrieval grabs a couple of semi-random chunks (often from the WRONG file
# entirely, since a big PDF has far more chunks than a single photo and can
# dominate ranking) and the LLM either hallucinates or says it doesn't have
# enough context. These patterns detect that style of broad, whole-file
# question so get_answer() can bypass retrieval and feed the LLM the entire
# file(s) instead of a few retrieved chunks.
#
# Every noun below uses "s?" instead of a bare word so PLURALS match too —
# \b needs a non-word character right after it, and "s" is a word character,
# so "document\b" alone silently misses "documents". This was a real bug:
# "explain the documents" (plural) fell through to normal retrieval while
# "explain the document" (singular) correctly used the whole-file path.

logger = logging.getLogger(__name__) 

_FILE_NOUN = r"(pdf|document|doc|file|photo|image|picture|pic)s?"

BROAD_SUMMARY_PATTERNS = [
    r"\bsummar(y|ize|ise|isation|ization)\b",
    rf"\bexplain\b.{{0,20}}\b{_FILE_NOUN}\b",  # "explain this pdf", "explain the documents", "explain all these 3 files"
    r"\bexplain (this|these|them|it|that)\b",   # "explain them", "explain these" — refers back to what's attached
    r"\b(give|provide) (me )?(an )?overview\b",
    r"\ball (of )?the details\b",
    rf"\bwhat('?s| is| does) (this|the) {_FILE_NOUN} (about|consist of|have|contain|show)\b",
    r"\btl;?dr\b",
    rf"\bwhat('?s| is|s)( there)? in (this|the) {_FILE_NOUN}\b",
    rf"\bwalk me through (this|the) {_FILE_NOUN}\b",
    # Table-of-contents questions are just as bad a fit for similarity
    # retrieval as "summarize this" — a TOC is dozens of near-identical
    # short lines scattered across several pages, with no single chunk
    # containing "the whole TOC". Whether normal retrieval happens to land
    # on the right fragment chunks is a coin flip depending on exact
    # phrasing, which is why "table of contents" worked on one rephrasing
    # and failed on another. Routing this into the whole-document path
    # (build_summary_context) instead feeds the model the actual front
    # matter text directly — the TOC lives near the start of the document,
    # comfortably inside MAX_SUMMARY_CONTEXT_CHARS, so this is reliable.
    r"\btable of contents\b",
    r"\btoc\b",
    r"\bcontents (page|list|section)\b",
]

# Detects an explicit request to treat multiple attached files SEPARATELY,
# rather than blending them into one combined answer — e.g. "explain all
# these 3 files give them separately". Covers the common "seperately" typo too.
SEPARATE_FILES_PATTERNS = [
    r"\bsepe?rately\b",
    r"\beach (file|document|pdf|photo|image)\b",
    r"\bindividually\b",
    r"\bone by one\b",
]

# Total character budget across ALL matching files when doing a whole-file
# answer. Split evenly per file, and each file's text is truncated to its
# share if it runs over. This is a simple, cheap safeguard against blowing
# past the model's context window on very large PDFs — not a token-perfect
# budget, just a conservative cap.
# Was 40000: that's ~10k+ tokens for a single request, which on its own blows
# past the account's 8000 TPM limit on openai/gpt-oss-120b — before anything
# else that minute even gets sent. 18000 chars is comfortably under budget,
# and FALLBACK_SUMMARY_CONTEXT_CHARS below is the second-chance size if a
# request still comes back 413 (e.g. a per-file split with few files, so
# each file's slice is still large).
MAX_SUMMARY_CONTEXT_CHARS = 18000
FALLBACK_SUMMARY_CONTEXT_CHARS = 9000

# Minimum FlashrankRerank relevance score for a retrieved chunk to be considered
# usable context. Below this, we treat the question as unrelated to the
# uploaded document(s) and skip the LLM call entirely — saves tokens on
# clearly out-of-scope questions (e.g. "who is Virat Kohli" against a FAQ PDF).
RELEVANCE_THRESHOLD = 0.2


def is_broad_summary_question(question):
    q = question.lower()
    return any(re.search(pattern, q) for pattern in BROAD_SUMMARY_PATTERNS)


def wants_separate_answers(question):
    q = question.lower()
    return any(re.search(pattern, q) for pattern in SEPARATE_FILES_PATTERNS)


def classify_summary_scope(question):
    """
    Decides which ATTACHED FILES a whole-file question should pull context
    from, based on which file type the person actually named. This is what
    stops "what's in this photo" from being answered using the PDF's content
    (or vice versa) when both are attached to the same chat.
    """
    q = question.lower()
    mentions_image = bool(re.search(r"\b(photo|image|picture|pic)s?\b", q))
    mentions_pdf = bool(re.search(r"\b(pdf|document|doc|file)s?\b", q))
    if mentions_image and not mentions_pdf:
        return "image"
    if mentions_pdf and not mentions_image:
        return "pdf"
    return "all"


# =================== SESSION PERSISTENCE (SQLite) ===================
#
# Replaces the old sessions.json flat file. That approach read the ENTIRE
# file, modified it in Python, and wrote the ENTIRE file back on every
# single action — two concurrent requests could interleave and one would
# silently clobber the other's changes. Every function below instead runs
# ONE atomic SQL statement per operation, and WAL mode lets reads and
# writes happen concurrently without locking each other out. This is what
# actually removes the race condition, not just switching file formats.

def _get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def _connect():
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_files (
                session_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                content_hash TEXT,
                PRIMARY KEY (session_id, filename),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        # Migration: your DB may already exist from before content_hash was
        # added — ALTER TABLE has no "IF NOT EXISTS" for columns in SQLite,
        # so check first and only add it if it's missing.
        existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(session_files)").fetchall()]
        if "content_hash" not in existing_cols:
            conn.execute("ALTER TABLE session_files ADD COLUMN content_hash TEXT")

        existing_turn_cols = [row[1] for row in conn.execute("PRAGMA table_info(chat_turns)").fetchall()]
        for col in [
            "faithfulness", "answer_relevancy", "context_precision", "context_relevancy",
            "context_recall", "answer_correctness",
        ]:
            if col not in existing_turn_cols:
                conn.execute(f"ALTER TABLE chat_turns ADD COLUMN {col} REAL")
        if "langsmith_run_id" not in existing_turn_cols:
            conn.execute("ALTER TABLE chat_turns ADD COLUMN langsmith_run_id TEXT")


def _migrate_legacy_json_once():
    """One-time import of any existing sessions.json into sessions.db, so
    upgrading doesn't lose sessions people already created. Safe to run on
    every startup — it only inserts a session if that session_id isn't
    already in the database (INSERT OR IGNORE), so it's a no-op after the
    first successful run."""
    if not LEGACY_SESSIONS_FILE.exists():
        return
    try:
        with open(LEGACY_SESSIONS_FILE, "r") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    with _connect() as conn:
        for session_id, data in raw.items():
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, created_at) VALUES (?, ?)",
                (session_id, data.get("created_at", datetime.now().isoformat())),
            )
            for filename in data.get("pdf_names", []):
                conn.execute(
                    "INSERT OR IGNORE INTO session_files (session_id, filename) VALUES (?, ?)",
                    (session_id, filename),
                )
            existing_turns = conn.execute(
                "SELECT COUNT(*) FROM chat_turns WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
            if existing_turns == 0:
                for turn in data.get("chat_history", []):
                    conn.execute(
                        "INSERT INTO chat_turns (session_id, question, answer, sources) VALUES (?, ?, ?, ?)",
                        (session_id, turn["question"], turn["answer"], json.dumps(turn.get("sources", []))),
                    )


_init_db()
_migrate_legacy_json_once()


def create_session():
    session_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, created_at) VALUES (?, ?)",
            (session_id, datetime.now().isoformat()),
        )
    return session_id


def session_exists(session_id):
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    return row is not None


def get_pdf_names(session_id):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT filename FROM session_files WHERE session_id = ? ORDER BY rowid",
            (session_id,),
        ).fetchall()
    return [r[0] for r in rows]


def add_pdf_to_session(session_id, filename, content_hash=None):
    """Registers one more filename against a session (no duplicates).
    content_hash (sha256 of the file's bytes) is optional so existing calls
    that don't pass one still work — it's what lets duplicate CONTENT be
    detected even when the filename differs (e.g. "report.pdf" vs a
    re-downloaded "report-2.pdf" that's byte-for-byte identical)."""
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO session_files (session_id, filename, content_hash) VALUES (?, ?, ?)",
            (session_id, filename, content_hash),
        )


def get_content_hashes(session_id):
    """All content hashes already attached to this session, for
    duplicate-CONTENT detection regardless of filename."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT content_hash FROM session_files WHERE session_id = ? AND content_hash IS NOT NULL",
            (session_id,),
        ).fetchall()
    return {r[0] for r in rows}


def append_chat_turn(session_id, question, answer, sources, metrics=None, langsmith_run_id=None):
    """Saves a chat turn. Returns the new row's id (sqlite3's lastrowid) so
    a caller that saves the turn BEFORE metrics are known (see
    update_chat_turn_metrics below) can patch them in later without a
    separate lookup query.

    langsmith_run_id (the RunTree.id from start_chat_turn) is stored so
    /metrics/calculate can look it up later and attach the on-demand
    DeepEval scores back to the correct LangSmith trace as feedback."""
    metrics = metrics or {}
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO chat_turns (session_id, question, answer, sources, "
            "faithfulness, answer_relevancy, context_precision, context_relevancy, "
            "context_recall, answer_correctness, langsmith_run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id, question, answer, json.dumps(sources),
                metrics.get("faithfulness"), metrics.get("answer_relevancy"),
                metrics.get("context_precision"), metrics.get("context_relevancy"),
                metrics.get("context_recall"), metrics.get("answer_correctness"),
                langsmith_run_id,
            ),
        )
        return cursor.lastrowid


def update_chat_turn_metrics(turn_id, metrics):
    """Patches all 6 score columns onto an already-saved chat turn.

    Exists so the chat endpoint can save+return the answer immediately
    (append_chat_turn, with metrics still unknown) and fill in the scores
    afterward once background evaluation finishes — instead of making the
    person wait on judge-model calls before they see their answer.
    """
    metrics = metrics or {}
    with _connect() as conn:
        conn.execute(
            "UPDATE chat_turns SET faithfulness = ?, answer_relevancy = ?, "
            "context_precision = ?, context_relevancy = ?, "
            "context_recall = ?, answer_correctness = ? WHERE id = ?",
            (
                metrics.get("faithfulness"), metrics.get("answer_relevancy"),
                metrics.get("context_precision"), metrics.get("context_relevancy"),
                metrics.get("context_recall"), metrics.get("answer_correctness"),
                turn_id,
            ),
        )

def get_langsmith_run_id(turn_id):
    """Looks up the stored LangSmith run_id for a chat turn, so
    /metrics/calculate can attach on-demand DeepEval scores back to the
    correct trace as feedback. Returns None if not found or never stored
    (e.g. LangSmith tracing was disabled when this turn was created)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT langsmith_run_id FROM chat_turns WHERE id = ?", (turn_id,)
        ).fetchone()
        return row[0] if row else None


def get_chat_history(session_id):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT question, answer, sources, faithfulness, answer_relevancy, "
            "context_precision, context_relevancy, context_recall, answer_correctness "
            "FROM chat_turns WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [
        {
            "question": q,
            "answer": a,
            "sources": json.loads(s),
            "metrics": {
                "faithfulness": faith,
                "answer_relevancy": rel,
                "context_precision": prec,
                "context_relevancy": ctxrel,
                "context_recall": recall,
                "answer_correctness": correctness,
            },
        }
        for q, a, s, faith, rel, prec, ctxrel, recall, correctness in rows
    ]


def list_sessions_summary():
    """Lightweight listing for the sidebar — pdf_names + created_at only,
    no chat history loaded for every session (avoids the old JSON approach's
    'load literally everything just to render a sidebar row' cost)."""
    with _connect() as conn:
        rows = conn.execute("SELECT session_id, created_at FROM sessions").fetchall()

    result = [
        {"session_id": sid, "pdf_names": get_pdf_names(sid), "created_at": created_at}
        for sid, created_at in rows
    ]
    result.sort(key=lambda x: x["created_at"], reverse=True)
    return result


def get_session_detail(session_id):
    """Full session detail, including chat_history. Returns None if the
    session doesn't exist."""
    if not session_exists(session_id):
        return None
    with _connect() as conn:
        created_at = conn.execute(
            "SELECT created_at FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
    return {
        "session_id": session_id,
        "pdf_names": get_pdf_names(session_id),
        "created_at": created_at,
        "chat_history": get_chat_history(session_id),
    }


def delete_session(session_id):
    with _connect() as conn:
        conn.execute("DELETE FROM chat_turns WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM session_files WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    chroma_dir = CHROMA_ROOT / session_id
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)

    chunks_path = CHUNKS_ROOT / f"{session_id}.pkl"
    if chunks_path.exists():
        chunks_path.unlink()

    page_lookup_path = _page_lookup_path(session_id)
    if page_lookup_path.exists():
        page_lookup_path.unlink()

    data_dir = DATA_DIR / session_id
    if data_dir.exists():
        shutil.rmtree(data_dir, ignore_errors=True)


def remove_file_from_session(session_id, filename):
    """
    Removes ONE attached file from a session without touching the rest:
    - drops it from session_files
    - strips its chunks out of this session's chunk pickle
    - rebuilds the Chroma store from what's left (Chroma has no clean
      "delete by source filename" without tracking per-chunk ids, so a full
      rebuild from the remaining chunks is the simplest correct approach —
      cheap here since embeddings are a fast local model, not an API call)
    - deletes the saved file itself from disk
    """
    with _connect() as conn:
        conn.execute(
            "DELETE FROM session_files WHERE session_id = ? AND filename = ?",
            (session_id, filename),
        )

    # --- Strip this file's chunks out of the pickle ---
    chunks_path = CHUNKS_ROOT / f"{session_id}.pkl"
    remaining_chunks = []
    if chunks_path.exists():
        with open(chunks_path, "rb") as f:
            existing_chunks = pickle.load(f)
        remaining_chunks = [
            c for c in existing_chunks if c.metadata.get("source") != filename
        ]
        with open(chunks_path, "wb") as f:
            pickle.dump(remaining_chunks, f)

    # --- Rebuild this session's Chroma store from the remaining chunks ---
    persist_dir = CHROMA_ROOT / session_id
    if persist_dir.exists():
        shutil.rmtree(persist_dir, ignore_errors=True)
    invalidate_vectordb_cache(session_id)
    if remaining_chunks:
        Chroma.from_documents(
            documents=remaining_chunks,
            embedding=_EMBEDDINGS,
            persist_directory=str(persist_dir)
        )
        invalidate_vectordb_cache(session_id)

    # --- Strip this file's entries out of the page lookup ---
    _remove_source_from_page_lookup(session_id, filename)

    # --- Delete the saved file itself ---
    file_path = DATA_DIR / session_id / filename
    if file_path.exists():
        file_path.unlink()


# =================== PDF PROCESSING / RETRIEVAL (per session_id) ===================

def _strip_repeated_boilerplate_lines(documents):
    """
    Detects and removes running headers/footers (e.g. a copyright line or
    book title printed on nearly every page) BEFORE chunking.

    Left in, a line like "© Author Name, Publisher" repeated on most pages
    pollutes retrieval two ways: BM25 loses the ability to treat that term
    as discriminating (it's in almost every chunk, so its IDF collapses),
    and vector search skews toward the boilerplate's semantic neighborhood
    instead of the actual page content, since the boilerplate dominates by
    sheer repetition. This showed up concretely as "who wrote this book"
    failing to retrieve the actual authorship sentence, because the
    repeated "© Bob Wiele, OneSmartWorld®" footer diluted "Bob Wiele" as a
    useful retrieval signal.

    This is frequency-based, not a hardcoded string — it works on any PDF,
    not just this one. A line is treated as boilerplate only if it appears
    on a large majority of pages, since a section heading that legitimately
    repeats across one chapter (but not the whole document) shouldn't be
    stripped.
    """
    if len(documents) < 5:
        # Too few pages for frequency-based detection to be meaningful —
        # skip rather than risk stripping something legitimate.
        return documents

    def _normalize(line):
        # Strip standalone digit runs (page numbers) so a footer like
        # "50 © Bob Wiele, OneSmartWorld®" and "93 © Bob Wiele, OneSmartWorld®"
        # are recognized as the SAME repeated line. Without this, a footer
        # containing a running page number never matches itself across
        # pages and frequency-based detection finds nothing — confirmed on
        # a real 516-page PDF where the footer appeared on 407 pages once
        # digits were normalized, but 0 times on exact match.
        line = re.sub(r"\b\d+\b", "", line)
        return re.sub(r"\s+", " ", line).strip()

    total_pages = len(documents)
    norm_counts = Counter()
    # Track one representative original line per normalized pattern, so we
    # can log something readable rather than the digit-stripped version.
    norm_examples = {}

    for doc in documents:
        # Count each normalized line at most once per page, so a line that
        # legitimately repeats several times WITHIN one page's body isn't
        # over-counted.
        page_lines = {
            line.strip() for line in doc.page_content.split("\n") if line.strip()
        }
        seen_norm_this_page = set()
        for line in page_lines:
            norm = _normalize(line)
            if not norm or norm in seen_norm_this_page:
                continue
            seen_norm_this_page.add(norm)
            norm_counts[norm] += 1
            norm_examples.setdefault(norm, line)

    BOILERPLATE_MIN_PAGE_FRACTION = 0.5   # appears on 50%+ of pages
    BOILERPLATE_MAX_LINE_LENGTH = 120     # short lines only — real headers/footers, not body text

    boilerplate_patterns = {
        norm for norm, count in norm_counts.items()
        if count >= max(3, int(total_pages * BOILERPLATE_MIN_PAGE_FRACTION))
        and len(norm) <= BOILERPLATE_MAX_LINE_LENGTH
    }

    if not boilerplate_patterns:
        return documents

    print(
        f"[boilerplate strip] removing {len(boilerplate_patterns)} repeated "
        f"header/footer pattern(s) found on {int(BOILERPLATE_MIN_PAGE_FRACTION * 100)}%+ of pages: "
        f"{[norm_examples[p] for p in boilerplate_patterns]}"
    )

    for doc in documents:
        kept_lines = [
            line for line in doc.page_content.split("\n")
            if _normalize(line.strip()) not in boilerplate_patterns
        ]
        doc.page_content = "\n".join(kept_lines)

    # SECOND PASS — substring cleanup for footer variants the exact-line
    # check above misses. Two-column PDF layouts (or footers glued directly
    # to page numbers) sometimes produce a slightly different exact string
    # on every page — e.g. the footer doubled onto one physical line
    # ("28 ©2003 Bob Wiele, OneSmartWorld ©2003 Bob Wiele, OneSmartWorld 29")
    # — so it never repeats identically often enough to clear the frequency
    # threshold above, even though it's boilerplate on nearly every page.
    # Once we've confirmed at least one real boilerplate pattern via the
    # strict pass, build a permissive regex from its core WORDS only
    # (ignoring digits, ©/®, and whitespace between them) and strip every
    # occurrence of that phrase anywhere in the text — including multiple
    # copies glued onto the same line — not just isolated full-line matches.
    for pattern in boilerplate_patterns:
        core_words = re.findall(r"[A-Za-z]{3,}", pattern)
        if len(core_words) < 2:
            continue  # too short/generic a pattern to safely match as a substring
        fuzzy = r"[^A-Za-z]{0,15}".join(re.escape(w) for w in core_words)
        compiled = re.compile(fuzzy, re.IGNORECASE)
        for doc in documents:
            doc.page_content = compiled.sub(" ", doc.page_content)

    return documents


# =================== PAGE-LEVEL LOOKUP (for expanding a matched chunk ===
# =================== back out to its full source page)             ===
#
# Root cause of the "3 key guidelines" bug: chunk_size=450 means an answer
# spanning ~2400 characters on one PDF page gets split across 4-5 separate
# chunks. The retriever only surfaces the top-ranked chunk(s) — often just
# the one containing guideline #1 — so guidelines #2/#3 never make it into
# context even though they're a few sentences later on the SAME page.
#
# Rather than re-architecting chunking/retrieval (which would risk
# reintroducing the "chunk mixes multiple unrelated FAQ topics" problem
# chunk_size=450 was specifically tuned to fix), this keeps 450/80 chunking
# for MATCHING (small chunks embed/match more precisely) but adds a
# {(source, page): full_page_text} lookup so that once a chunk is matched,
# get_answer() can swap it out for the FULL TEXT of the page it came from
# before building context. Small chunks for retrieval precision, full pages
# for what the LLM actually reads — same idea as LangChain's
# ParentDocumentRetriever, implemented against this project's existing
# pickle-based storage instead of introducing a new retriever class.

def _page_lookup_path(session_id):
    return CHUNKS_ROOT / f"{session_id}_pages.pkl"


def _load_page_lookup(session_id):
    path = _page_lookup_path(session_id)
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_page_lookup_entries(session_id, entries):
    """entries: {(source, page): full_page_text} to merge into this
    session's page-lookup pickle. Keyed by the same (source, page) metadata
    already used elsewhere (e.g. the de-dupe step in get_answer), so lookups
    line up with retrieved chunks with no extra bookkeeping."""
    page_lookup = _load_page_lookup(session_id)
    page_lookup.update(entries)
    with open(_page_lookup_path(session_id), "wb") as f:
        pickle.dump(page_lookup, f)


def _remove_source_from_page_lookup(session_id, filename):
    page_lookup = _load_page_lookup(session_id)
    if not page_lookup:
        return
    remaining = {k: v for k, v in page_lookup.items() if k[0] != filename}
    with open(_page_lookup_path(session_id), "wb") as f:
        pickle.dump(remaining, f)


# Hard cap on how much of one page's full text can replace a matched chunk.
# Kept at 3000 (not lowered) — a flat smaller cap looked appealing after
# seeing context_relevancy=0.097, but page 48 (the "3 Key Guidelines" page)
# is 2374 chars end-to-end; a 1500-char cap would truncate it mid-Guideline
# #3, reintroducing the exact bug this feature exists to fix. The real fix
# for context_relevancy is WHERE the window is taken from (see
# _expand_docs_to_full_pages below), not how big it is.
MAX_EXPANDED_PAGE_CHARS = 3000


# Only the top-ranked doc(s) get expanded to a full page/window. Retrieved
# docs 2-5 are frequently off-topic pages that just happen to score within
# ~0.0002 of the top score (this document's rerank scores cluster very
# tightly — see the [relevance filter] print in get_answer). Expanding ALL
# 5 to full pages was the actual cause of context_relevancy dropping to
# 0.097 in testing: 4 genuinely unrelated pages (a 40-line FAQ index, a
# Do's/Don'ts worksheet, etc.) each went from a small ~400-char chunk to a
# ~1700-2700 char full page, diluting the sentence-relevance fraction even
# though the TOP doc's expansion is what actually fixed the answer.
# Expanding only the top-ranked doc(s) keeps the fix where it matters
# (the doc the LLM is most likely to actually use) without also
# amplifying noise from docs that were only borderline-relevant to begin
# with.
EXPAND_TOP_N_DOCS = 2

# ===== CROSS-PAGE LIST CONTINUATION =====
# Fixes a real gap surfaced by the "12 Techniques" case: page 368 lists
# items #1-5 under a heading that states "12 Techniques...", but items
# #6-12 physically continue onto pages 369-370. _expand_docs_to_full_pages
# above only ever expands WITHIN a single matched page, so a numbered list
# that spans a page boundary gets silently truncated — the LLM then
# (correctly, but unhelpfully) says "the rest isn't in my context."
#
# _LIST_HEADING_COUNT_RE looks for a stated total near the top of a doc's
# content (e.g. "12 Techniques", "21 Strategies", "7 Mindsets"). Bounded to
# 3-50 to avoid false-triggering on ordinary sentences with a small number
# in them (e.g. "4 essential dimensions"). The (?<!\.) negative lookbehind
# rejects a digit that's part of a decimal section number like "3.7" (a
# real false positive found in testing: "3.7 Smart How To's: 12
# Techniques..." matched "7" — the section number, not the actual "12" a
# few words later — because a plain \b(\d{1,3})\b still matches the digit
# right after a period).
_LIST_HEADING_COUNT_RE = re.compile(
    r'(?<!\.)\b(\d{1,3})\s+(?:key\s+)?[A-Za-z][a-zA-Z]*(?:\s+[A-Za-z][a-zA-Z]*){0,3}\b'
)
# _LIST_ITEM_NUMBER_RE finds actual numbered list items in the document's
# own vocabulary style (this book uses "TECHNIQUE #N: ALL-CAPS TITLE",
# "GUIDELINE #N: ALL-CAPS TITLE", etc.). Requires a colon followed by a
# run of UPPERCASE title text, since that's the real content-heading
# format this document uses — this is what distinguishes an actual
# numbered technique/guideline from a false positive like a
# table-of-contents entry ("Lesson 4: How to Increase Your Sense of
# Control", mixed-case) that happens to contain one of these keywords.
# "LESSON" was dropped from the keyword list entirely: every real "Lesson
# N" occurrence found in testing was a TOC/section heading, never an
# actual numbered list item worth continuing across pages.
_LIST_ITEM_NUMBER_RE = re.compile(
    r'\b(?:TECHNIQUE|STRATEGY|GUIDELINE|STEP|RULE|TIP|PRINCIPLE)\s*#?\s*(\d{1,3})\s*:\s*[A-Z][A-Z0-9 ,\'"/&-]{4,79}(?=\n|$)',
)

CONTINUATION_MAX_EXTRA_CHARS = 2500  # total extra budget across all continuation pages
CONTINUATION_MAX_EXTRA_PAGES = 3     # hard cap on how many following pages to pull in


def _append_next_page_continuation(docs, session_id):
    """
    For the SINGLE top-ranked doc only: if its content looks like a
    numbered list with a stated total (e.g. a heading says "12
    Techniques...") but the highest item number actually visible falls
    short of that total, pulls in subsequent pages via the page-lookup
    dict and appends their text until either every implied item is
    covered or a hard budget/page-count cap is hit.

    Only the top doc is extended — same reasoning as EXPAND_TOP_N_DOCS
    above: extending every retrieved doc risks pulling in unrelated
    continuation pages for docs that were only borderline-relevant to
    begin with. Purely additive and capped — if no list/continuation
    pattern is detected, docs are returned completely unchanged.
    """
    if not docs or not session_id:
        return docs

    top = docs[0]
    text = top.page_content

    heading_match = _LIST_HEADING_COUNT_RE.search(text[:200])
    if not heading_match:
        return docs
    expected_count = int(heading_match.group(1))
    if expected_count < 3 or expected_count > 50:
        return docs

    seen_numbers = {int(n) for n in _LIST_ITEM_NUMBER_RE.findall(text)}
    if not seen_numbers or max(seen_numbers) >= expected_count:
        return docs  # nothing recognizable to check, or already complete

    page_lookup = _load_page_lookup(session_id)
    if not page_lookup:
        return docs

    source = top.metadata.get("source")
    current_page = top.metadata.get("page")
    if not isinstance(current_page, int):
        return docs

    extra_text_parts = []
    extra_chars_used = 0
    next_page = current_page + 1
    pages_pulled = 0

    while (
        max(seen_numbers) < expected_count
        and pages_pulled < CONTINUATION_MAX_EXTRA_PAGES
        and extra_chars_used < CONTINUATION_MAX_EXTRA_CHARS
    ):
        next_text = page_lookup.get((source, next_page))
        if not next_text:
            break  # ran off the end of the document, or a gap in the lookup

        budget_left = CONTINUATION_MAX_EXTRA_CHARS - extra_chars_used
        chunk_to_add = next_text[:budget_left]
        extra_text_parts.append(chunk_to_add)
        extra_chars_used += len(chunk_to_add)

        seen_numbers |= {int(n) for n in _LIST_ITEM_NUMBER_RE.findall(next_text)}
        pages_pulled += 1
        next_page += 1

    if extra_text_parts:
        print(f"[continuation] page {current_page} listed items up to #{max(seen_numbers)} "
              f"of {expected_count} expected — pulled {pages_pulled} more page(s) "
              f"({extra_chars_used} chars)", flush=True)
        combined_text = text + "\n\n[continued on next page(s)]\n\n" + "\n\n".join(extra_text_parts)
        docs = [Document(page_content=combined_text, metadata=top.metadata)] + docs[1:]

    return docs
# ===== END CROSS-PAGE LIST CONTINUATION =====


def _expand_docs_to_full_pages(docs, session_id):
    """Replaces the TOP-RANKED doc(s)' page_content with a WINDOW of the full
    page they came from, centered on where that chunk actually sits in the
    page — not just the first MAX_EXPANDED_PAGE_CHARS from the top. Docs
    beyond EXPAND_TOP_N_DOCS are left as their original small chunk (see
    EXPAND_TOP_N_DOCS comment for why — expanding every retrieved doc
    amplifies noise from borderline/off-topic matches).

    Why centered, not from-the-top: a page like the FAQ index (page 506/507)
    is a long list of unrelated Q&A lines. If the matched chunk happens to
    be line 20 of 40, taking the page from the top just grabs 19 unrelated
    lines before ever reaching the relevant one. Centering the window on the
    match's own position keeps the neighborhood around what was ACTUALLY
    matched, while still dropping distant unrelated content on long pages.

    Falls back to the original chunk text if no page-lookup entry exists
    (e.g. sessions created before this feature, or a lookup miss) — this
    must never make a doc's context WORSE, only potentially fuller.
    """
    if not session_id:
        return docs
    page_lookup = _load_page_lookup(session_id)
    if not page_lookup:
        return docs

    expanded = []
    for i, d in enumerate(docs):
        if i >= EXPAND_TOP_N_DOCS:
            expanded.append(d)
            continue

        key = (d.metadata.get("source"), d.metadata.get("page"))
        full_text = page_lookup.get(key)
        if not full_text:
            expanded.append(d)
            continue

        if len(full_text) <= MAX_EXPANDED_PAGE_CHARS:
            windowed = full_text
        else:
            # Locate roughly where this chunk sits in the full page. The
            # chunk's own text (post-splitting) should appear verbatim in
            # the page's raw text; if find() fails for any reason (e.g. the
            # splitter's whitespace normalization drifted from the raw
            # page), fall back to centering on the start of the page rather
            # than erroring.
            match_pos = full_text.find(d.page_content[:80])
            if match_pos == -1:
                match_pos = 0

            half = MAX_EXPANDED_PAGE_CHARS // 2
            start = max(0, match_pos - half)
            end = start + MAX_EXPANDED_PAGE_CHARS
            if end > len(full_text):
                end = len(full_text)
                start = max(0, end - MAX_EXPANDED_PAGE_CHARS)
            windowed = full_text[start:end]

        expanded.append(Document(page_content=windowed, metadata=d.metadata))
    return expanded


@traceable(name="process_pdf")
def process_pdf(pdf_path, session_id):
    """
    Loads ONE pdf, splits it, and INCREMENTALLY adds it to this session's
    existing chunk list + vector DB (rather than rebuilding from scratch).
    This means uploading a 2nd/3rd PDF only embeds the new pages, not
    everything again. Each chunk keeps 'source' metadata (the PDF filename)
    from PyPDFLoader, which is what lets citations say "Source 2 — invoice.pdf, page 4"
    once multiple PDFs are in the same session.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    documents = _strip_repeated_boilerplate_lines(documents)

    # Tag each chunk with a clean filename (PyPDFLoader's default 'source' is
    # the full temp file path, which is useless to show a user). Moved up
    # here (was previously set right before chunk-tagging below) so it's
    # available for the page-lookup save below too.
    filename = os.path.basename(pdf_path)

    # Save FULL per-page text before it gets split into small chunks — see
    # the "PAGE-LEVEL LOOKUP" block above for why. Keyed by (filename, page)
    # to match the metadata already on every chunk from this same page.
    _save_page_lookup_entries(
        session_id,
        {(filename, doc.metadata.get("page")): doc.page_content for doc in documents},
    )

    print("\n========== PDF DEBUG ==========")
    print("PDF:", pdf_path)
    print("Pages:", len(documents)) 
    # Shrunk from 800/200 to 450/80, and added "? " / bullet markers as
    # separators. At 800 chars, this document's dense FAQ/bullet-list pages
    # (many short lines, one topic per line/question) were getting fused
    # into a single chunk spanning several unrelated topics — e.g. a chunk
    # about "the red zone" that also contained five unrelated FAQ questions
    # about billing/results. Sentence-level context-relevancy metrics score
    # each chunk by what fraction of its sentences are actually relevant to
    # the question, so a chunk padded with unrelated FAQ lines tanks the
    # score even when it does contain the right answer. Splitting closer to
    # natural bullet/question boundaries keeps each chunk on one topic.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=80,
        separators=["\n\n", "\n", "? ", "• ", ". ", " ", ""]
    )
    new_chunks = splitter.split_documents(documents)
    print("Chunks created:", len(new_chunks))

    print("\n========== SEARCHING FOR BUSINESS VERIFICATION ==========")

    keywords = [
        "gst registration",
        "certificate of incorporation",
        "business license",
        "trade license",
        "utility bill",
        "bank statement",
        "business verification"
    ]

    found = False

    for i, chunk in enumerate(new_chunks):
        text = chunk.page_content.lower()

        if any(k in text for k in keywords):
            found = True
            print(f"\nFOUND IN CHUNK {i+1}")
            print("Page:", chunk.metadata.get("page"))
            print(chunk.page_content)

    if not found:
        print("❌ BUSINESS VERIFICATION NOT FOUND IN ANY CHUNK")

    print("====================================")

    # filename was already set above (before the page-lookup save)
    for c in new_chunks:
        c.metadata["source"] = filename

    # --- Merge into this session's chunk pickle (append, don't overwrite) ---
    chunks_path = CHUNKS_ROOT / f"{session_id}.pkl"
    if chunks_path.exists():
        with open(chunks_path, "rb") as f:
            existing_chunks = pickle.load(f)
    else:
        existing_chunks = []
    all_chunks = existing_chunks + new_chunks
    with open(chunks_path, "wb") as f:
        pickle.dump(all_chunks, f)

    # --- Merge into this session's Chroma vector store (add, don't overwrite) ---
    persist_dir = CHROMA_ROOT / session_id
    if persist_dir.exists() and any(persist_dir.iterdir()):
        vectordb = get_vectordb(session_id)
        vectordb.add_documents(new_chunks)
    else:
        Chroma.from_documents(
            documents=new_chunks,
            embedding=_EMBEDDINGS,
            persist_directory=str(persist_dir)
        )
        invalidate_vectordb_cache(session_id)

    return len(documents), len(new_chunks)


def is_image_file(filename):
    """Used by the upload routes to decide process_pdf vs process_image."""
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


def _encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _ocr_text(image_path):
    """Best-effort local text extraction via EasyOCR. Returns '' if nothing found
    or if OCR fails for any reason — OCR failing should never block the image
    from being indexed via the vision description. EasyOCR handles rotated/
    noisy real-world photos noticeably better than Tesseract did."""
    try:
        results = _OCR_READER.readtext(str(image_path), detail=0)
        return "\n".join(results).strip()
    except Exception:
        return ""


def _describe_image(image_path):
    """Asks a Groq vision model to describe the photo and transcribe any visible
    text. Tries VISION_MODEL first; if that call fails (e.g. the model was
    deprecated/renamed upstream), automatically retries once with
    VISION_MODEL_FALLBACK instead of failing the whole upload. If both fail,
    returns a clear placeholder so the image still gets indexed (just without
    a visual description) rather than blocking the upload entirely."""
    b64 = _encode_image_base64(image_path)
    ext = Path(image_path).suffix.lstrip(".").lower()
    mime = "jpeg" if ext == "jpg" else ext

    message = HumanMessage(content=[
        {"type": "text", "text": IMAGE_DESCRIBE_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
    ])

    for model_name in (VISION_MODEL, VISION_MODEL_FALLBACK):
        try:
            llm = ChatGroq(model=model_name, temperature=0)
            response = llm.invoke([message])
            return response.content.strip()
        except Exception as e:
            print(f"[vision] model '{model_name}' failed: {e}")
            continue

    return "Image description unavailable (vision model call failed). Any OCR text found is still included below."


@traceable(name="process_image")
def process_image(image_path, session_id):
    """
    Mirrors process_pdf(): builds ONE pseudo-document out of a photo (vision-model
    description + OCR'd text, when present), chunks it, and INCREMENTALLY merges it
    into this session's existing chunk list + vector DB. This is what makes photos
    show up in the same hybrid retriever as any PDFs already in the chat, so a
    question can pull context from a PDF and a photo in the same answer.
    """
    filename = os.path.basename(image_path)

    ocr_text = _ocr_text(image_path)
    description = _describe_image(image_path)

    combined = f"[Image: {filename}]\n\nVisual description:\n{description}"
    if ocr_text:
        combined += f"\n\nText found in image (OCR):\n{ocr_text}"

    # page="image" (rather than a number) is what render_sources() in app.py /
    # the frontend uses to know not to do the "+1" page-number conversion.
    doc = Document(page_content=combined, metadata={"source": filename, "page": "image"})

    # Same page-lookup save as process_pdf, so a retrieved image chunk can
    # also be expanded back to the full description+OCR text if it was
    # split into multiple chunks.
    _save_page_lookup_entries(session_id, {(filename, "image"): combined})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    new_chunks = splitter.split_documents([doc])
    for c in new_chunks:
        c.metadata["source"] = filename
        c.metadata["page"] = "image"

    # --- Merge into this session's chunk pickle (append, don't overwrite) ---
    chunks_path = CHUNKS_ROOT / f"{session_id}.pkl"
    if chunks_path.exists():
        with open(chunks_path, "rb") as f:
            existing_chunks = pickle.load(f)
    else:
        existing_chunks = []
    all_chunks = existing_chunks + new_chunks
    with open(chunks_path, "wb") as f:
        pickle.dump(all_chunks, f)

    # --- Merge into this session's Chroma vector store (add, don't overwrite) ---
    persist_dir = CHROMA_ROOT / session_id
    if persist_dir.exists() and any(persist_dir.iterdir()):
        vectordb = get_vectordb(session_id)
        vectordb.add_documents(new_chunks)
    else:
        Chroma.from_documents(
            documents=new_chunks,
            embedding=_EMBEDDINGS,
            persist_directory=str(persist_dir)
        )
        invalidate_vectordb_cache(session_id)

    # Returned as (n_pages, n_chunks) to match process_pdf's signature — an image
    # counts as "1 page" for the summary message shown after upload.
    return 1, len(new_chunks)


@traceable(name="build_hybrid_retriever")
def build_hybrid_retriever(session_id):
    chunks_path = CHUNKS_ROOT / f"{session_id}.pkl"
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    # Raised from 15 -> 30. Confirmed via a direct chunk-pickle inspection
    # (bypassing the retriever entirely) that a near-perfect keyword+semantic
    # match chunk for a real query ("12 Techniques to Increase Your
    # Positivity and Outlook", chunk #1209, page 368) exists cleanly in the
    # index but STILL never appeared in the top-5 retrieved set across 3
    # query phrasings. That rules out an indexing bug — this document's
    # 1980 chunks are dense enough, with this much repeated/similar
    # structural language ("Technique", "outlook", "positivity" appearing
    # across dozens of pages), that k=15 was too narrow a candidate pool to
    # reliably guarantee the correct chunk survives into reranking.
    bm25_retriever.k = 30

    vectordb = get_vectordb(session_id)

    # Widen HNSW's search exploration so repeated identical queries return
    # consistent results — this document has many near-tied chunks (the word
    # "red" appears throughout), and Chroma's default search depth isn't
    # thorough enough to reliably break those ties the same way every time.
    try:
        vectordb._collection.modify(metadata={"hnsw:search_ef": 200})
    except Exception as e:
        print(f"[build_hybrid_retriever] could not set hnsw:search_ef: {e}")

    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 30})  # was 15, see bm25_retriever.k comment above

    # Rebalanced from [0.45, 0.55] -> [0.3, 0.7]. The 0.45/0.55 split was
    # tuned for a case where BM25's exact keyword matching needed to win
    # out over loose semantic similarity (see the original comment below).
    # But a generic query like "types of spirals" exposed the opposite
    # failure: this document repeats the word "types" constantly ("types
    # of thinking", "types of feedback", "types of intelligence", "types
    # of work"...), so BM25 rewards every one of those chunks on pure
    # keyword overlap while the actual "positive spiral / negative spiral"
    # content — which may never contain the literal word "types" — gets no
    # BM25 credit at all and has to survive purely on vector similarity.
    # Shifting weight toward the vector retriever gives semantic matches
    # (query words paraphrasing the source, not literally repeating it) a
    # better chance of surviving into the candidate pool.
    #
    # Original reasoning, still true for the reverse case (kept for
    # context): this document reuses similar vocabulary ("tough
    # questions", "get to the point", "shift perspective") across several
    # unrelated frameworks (Red/Yellow/Green, Judger/Learner, mind traps,
    # generic writing skills), where a small embedding model can lean on
    # loose semantic similarity and pull in topically-wrong chunks that
    # just sound similar. There's a genuine tension between these two
    # failure modes on a document this dense and repetitive — 0.3/0.7 is a
    # rebalance, not a guaranteed fix for every case; keep an eye on both
    # directions as you test more queries.
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.3, 0.7]
    )

    # Raised from top_n=6 -> 10 alongside the k=15->30 bump above. With a
    # bigger raw candidate pool (30+30 instead of 15+15), top_n needs more
    # room too, or the wider pool doesn't actually help — the reranker
    # would just be choosing its top 6 from a bigger haystack without any
    # more of the haystack surviving into what get_answer() actually sees.
    compressor = FlashrankRerank(client=_RERANKER, top_n=10)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    # Returns the raw ensemble retriever alongside the compressed one.
    # DIAGNOSTIC ADDITION: this is what lets get_answer() print the
    # pre-rerank candidate pool (see the "[retrieval diagnostic]" block
    # below), so a missing-chunk failure (like the "7 strategies" /
    # Perkins page-332 case) can be told apart from a
    # retrieved-but-filtered-out failure. build_hybrid_retriever() callers
    # that only expect one return value need a one-line update — see
    # get_answer() below for the unpacking pattern.
    return compression_retriever, ensemble_retriever


def get_full_documents_by_source(session_id):
    """Reconstructs each attached file's FULL text (all its chunks stitched back
    together, in the order they were indexed) grouped by filename. Used for
    whole-document questions where retrieval-by-similarity isn't the right tool."""
    chunks_path = CHUNKS_ROOT / f"{session_id}.pkl"
    if not chunks_path.exists():
        return {}
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    docs_by_source = {}
    for c in chunks:
        source = c.metadata.get("source", "unknown")
        docs_by_source.setdefault(source, []).append(c.page_content)

    return {source: "\n\n".join(parts) for source, parts in docs_by_source.items()}


def build_summary_context(session_id, scope="all"):
    """
    Builds one big context block covering every ATTACHED FILE THAT MATCHES
    SCOPE (rather than a handful of retrieved chunks), with a per-file
    character budget so a single huge PDF can't crowd out the others or blow
    the model's context.

    scope="image" -> only photos/images attached to this session
    scope="pdf"   -> only non-image files (PDFs) attached to this session
    scope="all"   -> every attached file (used when the question doesn't
                     clearly name one file type, e.g. "summarize everything")
    """
    docs_by_source = get_full_documents_by_source(session_id)
    if not docs_by_source:
        return "", []

    if scope == "image":
        filtered = {s: t for s, t in docs_by_source.items() if is_image_file(s)}
    elif scope == "pdf":
        filtered = {s: t for s, t in docs_by_source.items() if not is_image_file(s)}
    else:
        filtered = docs_by_source

    # Fallback: if the question named a type that isn't actually attached
    # (e.g. asked about "the pdf" but only a photo is in this chat), don't
    # return an empty context — fall back to everything so the LLM can at
    # least say what's really there instead of getting nothing to work with.
    if not filtered:
        filtered = docs_by_source

    per_doc_budget = max(MAX_SUMMARY_CONTEXT_CHARS // len(filtered), 2000)

    parts = []
    for source, text in filtered.items():
        if not text.strip():
            logger.warning("Empty full-document text for source=%s session=%s", source, session_id)
        truncated = text[:per_doc_budget]
        note = "\n...[truncated — document continues beyond this excerpt]" if len(text) > per_doc_budget else ""
        parts.append(f"=== {source} ===\n{truncated}{note}")

    return "\n\n".join(parts), list(filtered.keys())


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_history(chat_history, max_turns=3):
    if not chat_history:
        return "No previous conversation."
    recent = chat_history[-max_turns:]
    lines = []
    for turn in recent:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")

    
    return "\n".join(lines)


@traceable(name="condense_question")
def condense_question(question, chat_history, llm):
    """
    Rewrites a follow-up question into a standalone one using recent history,
    so RETRIEVAL (not just the final answer) has the context it needs.
    Skipped entirely on turn 1, since there's no history to fold in.
    Returns (standalone_question, token_usage_dict).
    """
    if not chat_history:
        return question, {}

    history_text = format_history(chat_history)
    prompt = ChatPromptTemplate.from_template(CONDENSE_PROMPT_TEMPLATE)
    parser = StrOutputParser()

    filled_prompt = prompt.invoke({"history": history_text, "question": question})
    response = llm.invoke(filled_prompt)
    standalone = parser.invoke(response).strip()
    token_usage = response.response_metadata.get("token_usage", {})

    # Guard against the LLM returning an empty string / going off the rails
    if not standalone:
        return question, token_usage
    return standalone, token_usage


@traceable(name="attribute_sources")
def attribute_sources(answer, docs, llm):
    """
    Asks the LLM which of the retrieved docs it actually drew from to write
    `answer`, and filters `docs` down to just those. Replaces word-overlap-
    fraction-based citation filtering.

    Why: word overlap can't tell "genuinely used" apart from "shares real
    vocabulary because it's a related topic". Two concrete failures found
    while tuning an overlap threshold on this document:
      - A topically-adjacent page (the document's own separate "21
        Strategies" list) scored HIGHER (0.393) against a "7 strategies"
        question than a threshold tuned to kill earlier false positives,
        because it's a genuinely similar but wrong list.
      - A genuinely-used page (a Fixed/Growth mindset table the model
        clearly paraphrased into its answer) got dropped once the
        threshold was raised to fix the case above, because paraphrased
        content doesn't share enough literal words with the source.
    No single threshold separates those two populations — their score
    distributions overlap by construction. Asking the model directly which
    sources it used is exact instead of approximate, at the cost of one
    extra cheap LLM call per turn.

    Always keeps at least one source as a safety net (never shows an empty
    citation list for an answer that isn't "I don't know"), and falls back
    to keeping everything unfiltered if the attribution call itself fails
    or returns something unparseable — a broken filter should never be
    able to silently erase all citations.
    """
    if not docs:
        return docs

    def _numbered_entry(i, d):
        # rescued_from_ensemble means this doc was force-included on raw
        # keyword/vector overlap because FlashrankRerank dropped it (see the
        # "ENSEMBLE TOP-PICK SAFETY NET" in get_answer) — it never earned a
        # real relevance score, so flag it here to counter the attributor's
        # tendency to trust it just because it's in the list.
        rescued_tag = (
            ", ⚠ included via raw keyword/vector overlap, not reranked as relevant"
            if d.metadata.get("rescued_from_ensemble") else ""
        )
        page = d.metadata.get("page_label") or d.metadata.get("page")
        return (
            f"[{i+1}] (source: {d.metadata.get('source')}, page: {page}{rescued_tag})\n"
            f"{d.page_content[:1000]}"
        )

    numbered = "\n\n".join(_numbered_entry(i, d) for i, d in enumerate(docs))

    prompt = ChatPromptTemplate.from_template(SOURCE_ATTRIBUTION_PROMPT)
    parser = StrOutputParser()
    filled = prompt.invoke({"answer": answer, "numbered_sources": numbered})

    try:
        response = llm.invoke(filled)
        raw = parser.invoke(response).strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        # The prompt now asks for a per-source reasoning line before the
        # verdict (see SOURCE_ATTRIBUTION_PROMPT) — the actual JSON array is
        # only the tail of the response, after a "FINAL:" marker. Pull just
        # that part; if the marker is missing for any reason, fall back to
        # scanning the whole response for a JSON array (covers a model that
        # ignores the marker instruction but still emits the array somewhere).
        if "FINAL:" in raw:
            raw = raw.rsplit("FINAL:", 1)[1].strip()
        array_match = re.search(r"\[[^\[\]]*\]", raw)
        if not array_match:
            raise ValueError(f"no JSON array found in attribution response: {raw!r}")
        used_indices = {
            i for i in json.loads(array_match.group(0))
            if isinstance(i, int) and 1 <= i <= len(docs)
        }
    except Exception as e:
        # Fail safe: never let a broken attribution call silently nuke the
        # citation list. Keep everything, same as if no filtering happened.
        print(f"[source attribution] parse failed, keeping all {len(docs)} docs: {e}")
        return docs

    if not used_indices:
        # Either the model genuinely used nothing (answer was "I don't
        # know") or attribution under-reported — either way, showing zero
        # sources for a real answer is worse than showing an unfiltered
        # list, so fall back rather than returning [].
        return docs

    filtered = [d for i, d in enumerate(docs) if (i + 1) in used_indices]
    print(f"[source attribution] kept {len(filtered)}/{len(docs)} docs "
          f"(LLM reported: {sorted(used_indices)})", flush=True)
    return filtered


def get_chunks_by_page(session_id, page_label):
    """Returns every chunk whose metadata page/page_label matches page_label
    (a string, e.g. "247"), across all files attached to this session.

    Used for direct "what's on page N?" lookups, which vector similarity
    search is fundamentally the wrong tool for — there's no reason the
    EMBEDDING of the words "what's on page 247" would score highly against
    a chunk just because that chunk happens to BE page 247. This bypasses
    the vector store entirely and matches on metadata instead.
    """
    chunks_path = CHUNKS_ROOT / f"{session_id}.pkl"
    if not chunks_path.exists():
        return []
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    page_label = str(page_label).strip()
    matches = []
    for c in chunks:
        meta_page = c.metadata.get("page_label")
        meta_page_idx = c.metadata.get("page")
        # page_label is usually the human-facing number printed on the
        # page (what a person means by "page 247"); page is PyPDF's
        # 0-indexed internal page count, which is often off-by-one or more
        # from page_label depending on front matter — checked as a
        # fallback in case a PDF has no page_label set at all.
        if str(meta_page) == page_label or str(meta_page_idx) == page_label:
            matches.append(c)
    return matches


_PAGE_LOOKUP_RE = re.compile(r"\bpage\s*(?:no\.?|number|#)?\s*(\d+)\b", re.IGNORECASE)


def extract_page_lookup(question):
    """Returns the page number as a string if this question is a direct
    "what's on page N" / "show me page N" style lookup, else None.

    Deliberately narrow (just looks for "page <number>") rather than trying
    to distinguish intent with an LLM call — a false positive here just
    means we fetch a page's real content instead of running normal
    retrieval, which is a safe failure mode, not a wrong-answer one.
    """
    match = _PAGE_LOOKUP_RE.search(question)
    return match.group(1) if match else None


def _invoke_with_413_fallback(llm, prompt_template, context, question, history, filename_note=""):
    """Runs prompt -> llm.invoke, and if Groq rejects it with 413 Payload Too
    Large (context still too big for the 8000 TPM budget even after the
    MAX_SUMMARY_CONTEXT_CHARS cut), retries ONCE with a much smaller,
    hard-truncated context instead of letting the 413 bubble up into a 500
    ("Error getting response." in the UI).

    This is a safety net, not the primary fix — the primary fix is the
    MAX_SUMMARY_CONTEXT_CHARS reduction above, which should mean this retry
    path rarely triggers.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    parser = StrOutputParser()

    filled_prompt = prompt.invoke({
        "context": context,
        "question": question,
        "history": history,
    })

    try:
        response = llm.invoke(filled_prompt)
    except APIStatusError as e:
        if e.status_code == 413:
            print(f"[413 fallback] context still too large ({len(context)} chars){filename_note} "
                  f"— retrying with a {FALLBACK_SUMMARY_CONTEXT_CHARS}-char context", flush=True)
            shrunk_context = context[:FALLBACK_SUMMARY_CONTEXT_CHARS] + "\n...[truncated further due to size]"
            filled_prompt = prompt.invoke({
                "context": shrunk_context,
                "question": question,
                "history": history,
            })
            response = llm.invoke(filled_prompt)
        else:
            raise

    answer = parser.invoke(response)
    usage = response.response_metadata.get("token_usage", {})
    return answer, usage


def get_answer(retriever, question, chat_history, session_id=None, ensemble_retriever=None):
    """
    Runs condensation -> retrieval -> LLM call manually (not LCEL), so we can
    inject history into condensation and still read response_metadata for tokens.

    EXCEPTION 1: for direct "what's on page N?" style questions, semantic
    retrieval is the wrong tool — there's no similarity reason the embedding
    of the QUESTION would match the embedding of that PAGE'S CONTENT. This
    branch bypasses the vector store and looks the page up by metadata
    directly. Requires session_id; if not passed, falls through to normal
    retrieval (which will likely fail to find the right page, same as
    before this fix).

    EXCEPTION 2: for broad "summarize/explain this document/photo" (and now
    "table of contents") style questions, retrieval-by-similarity doesn't
    work well (nothing matches a vague query closely, and a big PDF's
    chunks can drown out a single photo's chunk in ranking). Instead we
    skip straight to feeding the LLM the ENTIRE content of whichever
    attached file type was actually named (photo vs PDF vs everything).
    This requires session_id — if it's not passed, this branch is simply
    skipped and the normal retrieval path runs.

    ensemble_retriever (optional): the RAW pre-rerank retriever, passed in
    by the caller alongside the compressed `retriever`. If given, this
    function prints the raw candidate pool for the current query BEFORE
    reranking/compression, so a "correct chunk never retrieved" failure
    (missing from this list) can be told apart from a "correct chunk
    retrieved but filtered/outranked" failure (present here, but absent
    from `docs` after compression). This is a diagnostic aid only — it
    doesn't change retrieval behavior. Safe to leave None; the print is
    simply skipped if not provided.
    """
    # Deterministic — only for rewriting follow-ups into standalone questions
    # (and now also for source attribution, which needs the same
    # deterministic, no-creative-liberty behavior).
    condense_llm = _CONDENSE_LLM

    # Creative — used for the actual answer shown to the user
    answer_llm = _ANSWER_LLM

    print("Question:", question)
    print("Broad summary?", is_broad_summary_question(question))

    if session_id:
        page_lookup = extract_page_lookup(question)
        if page_lookup:
            print(f"Direct page lookup detected: page {page_lookup}")
            page_chunks = get_chunks_by_page(session_id, page_lookup)

            if not page_chunks:
                # Being honest here matters more than forcing an answer —
                # a page number outside the document's range, or a PDF
                # whose PyPDF page_label extraction didn't line up with
                # what's printed on the page, should say so rather than
                # silently falling back to unrelated chunks (which is
                # exactly the failure mode this branch exists to avoid).
                return (
                    f"I couldn't find page {page_lookup} in the attached document(s). "
                    "It may be outside the document's page range, or this PDF's page "
                    "numbering doesn't match what's printed on the page.",
                    [],
                    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                )

            page_text = "\n\n".join(c.page_content for c in page_chunks)
            prompt = ChatPromptTemplate.from_template(PAGE_LOOKUP_PROMPT_TEMPLATE)
            parser = StrOutputParser()
            filled_prompt = prompt.invoke({
                "context": page_text,
                "question": question,
            })
            response = answer_llm.invoke(filled_prompt)
            answer = parser.invoke(response)
            usage = response.response_metadata.get("token_usage", {})
            token_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            sources = [
                {"source": c.metadata.get("source", "unknown"), "page": page_lookup, "text": c.page_content}
                for c in page_chunks
            ]
            return answer, sources, token_usage

    if session_id and is_broad_summary_question(question):        
        scope = classify_summary_scope(question)
        separate = wants_separate_answers(question)
        docs_by_source = get_full_documents_by_source(session_id)

        if separate and len(docs_by_source) > 1:
            # One LLM call PER FILE instead of one call over a blended
            # context — this is what actually answers "explain these 3
            # files, give them separately" correctly, instead of merging
            # everything into a single confused answer.
            if scope == "image":
                targets = {s: t for s, t in docs_by_source.items() if is_image_file(s)}
            elif scope == "pdf":
                targets = {s: t for s, t in docs_by_source.items() if not is_image_file(s)}
            else:
                targets = docs_by_source
            if not targets:
                targets = docs_by_source

            history_text = format_history(chat_history)

            answer_parts = []
            total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            sources = []

            for filename, text in targets.items():
                per_file_budget = max(MAX_SUMMARY_CONTEXT_CHARS // len(targets), 2000)
                truncated = text[:per_file_budget]
                note = "\n...[truncated]" if len(text) > per_file_budget else ""

                file_answer, usage = _invoke_with_413_fallback(
                    answer_llm,
                    PROMPT_TEMPLATE,
                    context=f"=== {filename} ===\n{truncated}{note}",
                    question=f"Explain this specific file: {filename}",
                    history=history_text,
                    filename_note=f" (file: {filename})",
                )
                for k in total_tokens:
                    total_tokens[k] += usage.get(k, 0)

                answer_parts.append(f"### {filename}\n{file_answer}")
                sources.append({"source": filename, "page": "full document", "text": truncated})

            return "\n\n".join(answer_parts), sources, total_tokens

        context, source_names = build_summary_context(session_id, scope=scope)
        history_text = format_history(chat_history)

        answer, answer_tokens = _invoke_with_413_fallback(
            answer_llm,
            PROMPT_TEMPLATE,
            context=context,
            question=question,
            history=history_text,
        )

        token_usage = {
            "prompt_tokens": answer_tokens.get("prompt_tokens", 0),
            "completion_tokens": answer_tokens.get("completion_tokens", 0),
            "total_tokens": answer_tokens.get("total_tokens", 0),
        }

        sources = []

        for name in source_names:
            text = docs_by_source.get(name, "")

            # Keep context reasonably sized
            text = text[:10000]

            sources.append({
                "source": name,
                "page": "full document",
                "text": text
            })

        return answer, sources, token_usage

    # 1. Condense (multi-turn aware retrieval)
    standalone_question, condense_tokens = condense_question(question, chat_history, condense_llm)

    # ===== RETRIEVAL DIAGNOSTIC (temporary) =====
    # Prints every chunk in the raw candidate pool BEFORE FlashrankRerank
    # narrows it down to top_n=6. Lets us tell apart two very different
    # failure modes when an expected chunk is missing from the final
    # answer's sources:
    #   1. RECALL failure — the chunk never shows up here at all. Fix:
    #      raise bm25_retriever.k / vector search_kwargs k (both 30 now).
    #   2. RANKING/FLOOR failure — the chunk IS here, but doesn't survive
    #      compression (FlashrankRerank top_n=10) or the relevance-floor
    #      filter further down. Fix: raise compressor top_n, widen
    #      ENSEMBLE_TOP_N_TRUST, and/or loosen RELEVANCE_SCORE_DROP /
    #      RELEVANCE_SCORE_FLOOR.
    # Only runs if the caller passed the raw ensemble_retriever alongside
    # the compressed one — see build_hybrid_retriever()'s new return value.
    # Safe to delete this whole block once the retrieval bug is resolved.
    if ensemble_retriever is not None:
        raw_candidates = ensemble_retriever.invoke(standalone_question)
        print(f"\n[retrieval diagnostic] {len(raw_candidates)} pre-rerank candidates "
              f"for standalone question: {standalone_question!r}")
        for d in raw_candidates:
            print(f"  page={d.metadata.get('page')} page_label={d.metadata.get('page_label')} "
                  f"preview={d.page_content[:80]!r}")
        print("[retrieval diagnostic] end\n")
    # ===== END RETRIEVAL DIAGNOSTIC =====

    # 2. Retrieve using the standalone question, not the raw follow-up
    docs = retriever.invoke(standalone_question)

    # ===== POST-RERANK DIAGNOSTIC (temporary) =====
    # Prints EVERY doc FlashrankRerank actually kept (up to top_n=10),
    # before de-dup/relevance-floor/top-5 slicing touch it. Answers one
    # specific question: did the correct chunk (e.g. page 368 for the "12
    # Techniques" query) survive reranking at all, just outside the final
    # top-5 slice — or did FlashrankRerank itself drop it before even that?
    # These are two different bugs with two different fixes (raise the
    # final slice size vs. fix/replace the reranker), so this print exists
    # to tell them apart with certainty instead of guessing. Safe to delete
    # once the reranking questions are fully resolved.
    print(f"\n[post-rerank diagnostic] {len(docs)} docs survived FlashrankRerank (top_n=10):")
    for i, d in enumerate(docs):
        print(f"  rank={i+1} page={d.metadata.get('page')} "
              f"score={d.metadata.get('relevance_score')} "
              f"preview={d.page_content[:60]!r}")
    print("[post-rerank diagnostic] end\n")
    # ===== END POST-RERANK DIAGNOSTIC =====

    if docs:
        print("First doc metadata keys:", docs[0].metadata.keys())
        print("First doc relevance_score:", docs[0].metadata.get("relevance_score"))

    # ===== ENSEMBLE TOP-PICK SAFETY NET =====
    # Confirmed via direct testing (the "12 Techniques" query): FlashrankRerank
    # (ms-marco-MultiBERT-L-12) can completely drop the #1-ranked raw ensemble
    # candidate from its own top-10 output on this document — not just rank it
    # low, but exclude it entirely — even when BM25 AND vector search both
    # independently agreed it was the single best match out of 52 candidates.
    # This document's dense, repetitive vocabulary (the same "outlook",
    # "positivity", "technique" language reused across dozens of pages) seems
    # to be genuinely outside what this reranker model can discriminate
    # reliably; see the [relevance filter] print elsewhere in this function
    # for the same symptom (scores clustering within ~0.0002-0.0004 of each
    # other regardless of true relevance).
    #
    # Rather than trusting the reranker as the sole authority, this treats
    # ENSEMBLE_TOP_N_TRUST top raw candidates as "innocent until proven
    # irrelevant": if the raw ensemble's own top pick(s) aren't already
    # present in the reranked `docs`, they get prepended back in with a
    # synthetic high relevance_score so they survive the relevance-floor
    # filter and the final top-5 slice below. This is a targeted patch for
    # a demonstrated reranker blind spot, not a general "ignore the
    # reranker" change — FlashrankRerank's own ordering for everything else
    # is left untouched.
    #
    # Raised from 2 -> 5: the "types of spirals" case showed even the raw
    # ensemble's own top-2 picks can be wrong on a document this repetitive
    # (BM25 rewarding the generic word "types" over the actual topic word),
    # so trusting more of the ensemble's own ranking gives borderline-but-
    # correct candidates a wider net to be rescued through.
    ENSEMBLE_TOP_N_TRUST = 5
    if ensemble_retriever is not None:
        # Reuse the same raw_candidates computed above in the retrieval
        # diagnostic block when available, to avoid a second identical
        # ensemble_retriever.invoke() call.
        if "raw_candidates" not in locals():
            raw_candidates = ensemble_retriever.invoke(standalone_question)

        already_present = {(d.metadata.get("source"), d.metadata.get("page")) for d in docs}
        rescued = []
        for d in raw_candidates[:ENSEMBLE_TOP_N_TRUST]:
            key = (d.metadata.get("source"), d.metadata.get("page"))
            if key not in already_present:
                # Give it a relevance_score at/above the current top score so
                # it isn't immediately re-filtered out by the relevance floor
                # below — this chunk is being trusted on the ENSEMBLE's own
                # top ranking, not on a fabricated rerank score.
                d.metadata["relevance_score"] = 1.0
                d.metadata["rescued_from_ensemble"] = True  # for debugging/log clarity only
                rescued.append(d)
                already_present.add(key)

        if rescued:
            print(f"[ensemble safety net] rescuing {len(rescued)} top ensemble candidate(s) "
                  f"FlashrankRerank dropped entirely: "
                  f"{[(d.metadata.get('page')) for d in rescued]}")
            docs = rescued + docs
    else:
        rescued = []
    # ===== END ENSEMBLE TOP-PICK SAFETY NET =====

    # De-dupe by (source, page) BEFORE slicing to top 3 — the retriever can
    # return the same page as two separate chunks (BM25 + vector both
    # surfacing it, or two overlapping chunks from the same page), which
    # wastes a context slot on redundant content instead of a new page.
    seen_pages = set()
    deduped_docs = []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key not in seen_pages:
            seen_pages.add(key)
            deduped_docs.append(d)
    docs = deduped_docs

    # Relevance floor — drop chunks whose rerank score trails too far behind
    # the top-ranked chunk. NOTE: this document's FlashrankRerank scores
    # cluster very tightly near 1.0 (e.g. 0.9996 down to 0.9986) even when a
    # chunk is genuinely off-topic, so RELEVANCE_SCORE_DROP starts small and
    # is meant to be tuned from the printed spread below — watch a few real
    # questions, see where clearly-irrelevant chunks actually sit relative to
    # the top score, and raise/lower this constant accordingly.
    if docs:
        scores = [d.metadata.get("relevance_score") for d in docs if d.metadata.get("relevance_score") is not None]
        if scores:
            top_score = max(scores)
            print(f"[relevance filter] top={top_score:.4f} spread={top_score - min(scores):.4f} "
                  f"scores={[round(float(s), 4) for s in scores]}", flush=True)
            # RELATIVE drop-off: mostly a no-op on this document — FlashrankRerank
            # scores here cluster within ~0.0002-0.0015 of each other even between
            # genuinely on-topic and off-topic chunks (shared vocabulary across
            # Red/Yellow/Green, Judger/Learner, etc.), so "how far below the top
            # score" barely discriminates anything. Kept as a light backstop.
            RELEVANCE_SCORE_DROP = 0.01
            # ABSOLUTE floor: the real filter for this document. Chunks that
            # truly don't answer the question still tend to land noticeably
            # below chunks that do, in absolute terms, even when the relative
            # spread is tiny — this catches what the relative filter above
            # can't. Tune against real off-topic queries the same way the
            # relative drop was tuned, using the printed [relevance filter] line.
            RELEVANCE_SCORE_FLOOR = 0.985
            floor_filtered = [
                d for d in docs
                if d.metadata.get("relevance_score") is None
                or (
                    d.metadata.get("relevance_score") >= top_score - RELEVANCE_SCORE_DROP
                    and d.metadata.get("relevance_score") >= RELEVANCE_SCORE_FLOOR
                )
            ]
            # Only apply the floor if something survives it. This threshold
            # was tuned against ONE document's score distribution (where
            # irrelevant chunks still cluster at 0.999+); a rephrased query
            # or a short/sparse passage can legitimately score below it
            # everywhere, and wiping ALL candidates to zero — forcing an
            # "I don't know" from an empty context — is a worse failure
            # than occasionally keeping one extra borderline chunk.
            if floor_filtered:
                docs = floor_filtered
            else:
                print(f"[relevance filter] floor ({RELEVANCE_SCORE_FLOOR}) would have removed "
                      f"all {len(docs)} docs — keeping unfiltered set instead", flush=True)

    # Keep only the top 5 retrieved chunks (raised from 3 — this document's
    # relevance_scores frequently land within ~0.0002 of each other across
    # many candidates, since it's mostly repeated, structurally-similar
    # trait-list content for Red/Yellow/Green. With scores that close, the
    # single correct chunk sometimes lands at rank 4-5 rather than 1-3, and
    # a top-3 cutoff was silently discarding it before the LLM ever saw it.
    #
    # Widened by len(rescued) when the ensemble safety net above prepended
    # anything: confirmed via testing that a flat docs[:5] silently evicted
    # a genuinely good, independently-reranked candidate (page 331, rank 5
    # in FlashrankRerank's own top-10) purely because rescuing 1 extra doc
    # shifted the window by one. Rescuing a dropped top-pick should never
    # come at the cost of a candidate the reranker itself ranked well.
    docs = docs[:5 + len(rescued)]

    print("\n" + "="*80)
    print("QUESTION:", question)
    print("STANDALONE QUESTION:", standalone_question)
    print("DOCUMENTS RETRIEVED:", len(docs))

    for i, doc in enumerate(docs):
        print(f"\n----- DOCUMENT {i+1} -----")
        print("Source:", doc.metadata.get("source"))
        print("Page:", doc.metadata.get("page"))
        print("Metadata:", doc.metadata)
        print("Content:")
        print(doc.page_content[:500])
        print("="*80 + "\n")

    if not docs:
        return (
            "That doesn't seem to be covered in your uploaded document(s).",
            [],
            {
                "prompt_tokens": condense_tokens.get("prompt_tokens", 0),
                "completion_tokens": condense_tokens.get("completion_tokens", 0),
                "total_tokens": condense_tokens.get("total_tokens", 0),
            }
        )

    # Expand each matched ~450-char chunk back out to the FULL TEXT of the
    # page it came from (see "PAGE-LEVEL LOOKUP" near process_pdf). This is
    # what fixes answers that span multiple chunks of one page — e.g. the
    # "3 key guidelines" case, where only the chunk containing guideline #1
    # was being retrieved. Docs were already de-duped by (source, page)
    # above, so expanding here can't create duplicate page text.
    docs = _expand_docs_to_full_pages(docs, session_id)
    docs = _append_next_page_continuation(docs, session_id)

    context = format_docs(docs)
    history_text = format_history(chat_history)

    # 3. Answer using the ORIGINAL question (so the response still reads naturally
    #    as a reply to what the user actually typed) but context from the condensed retrieval
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    parser = StrOutputParser()
    filled_prompt = prompt.invoke({
        "context": context,
        "question": question,
        "history": history_text
    })
    response = answer_llm.invoke(filled_prompt)
    answer = parser.invoke(response)
    answer_tokens = response.response_metadata.get("token_usage", {})

    # Combine token usage across both LLM calls so the UI shows the true cost of the turn
    token_usage = {
        "prompt_tokens": condense_tokens.get("prompt_tokens", 0) + answer_tokens.get("prompt_tokens", 0),
        "completion_tokens": condense_tokens.get("completion_tokens", 0) + answer_tokens.get("completion_tokens", 0),
        "total_tokens": condense_tokens.get("total_tokens", 0) + answer_tokens.get("total_tokens", 0),
    }

    # 4. Attribute — ask the model which of the retrieved docs it actually
    #    used, and filter the citation list down to those. Replaces
    #    word-overlap-based filtering; see attribute_sources()'s docstring
    #    for why overlap fraction can't reliably separate "genuinely used"
    #    from "shares vocabulary because it's a related topic".
    docs = attribute_sources(answer, docs, condense_llm)

    # De-dupe by (source, page): the retriever can return multiple chunks
    # from the same page/image (e.g. a photo split into several chunks, or
    # BM25 + vector both surfacing the same PDF page), which showed up as
    # repeated identical entries in the Sources list. Keep the first chunk
    # seen per (source, page) — that's enough to show where the answer came from.
    # Keep each retrieved chunk separately for RAG evaluation.
    # Context precision needs individual chunks, not merged documents.

    sources = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        # Prefer page_label (the number actually printed on the page) over
        # the raw PyPDF 'page' field (0-indexed loader count, which drifts
        # from what's printed once there's front matter/blank pages).
        # Falls back to 'page' for docs that never had a page_label set,
        # e.g. images (metadata page="image").
        page = doc.metadata.get("page_label") or doc.metadata.get("page")

        sources.append({
            "source": source,
            "page": page,
            "text": doc.page_content
        })

    print("\n===== INDIVIDUAL SOURCES =====")
    for i, s in enumerate(sources):
        print(
            f"SOURCE {i}: {s['source']} -> Page {s['page']}"
        )

    return answer, sources, token_usage