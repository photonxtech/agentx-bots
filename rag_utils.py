import json
import pickle  # converts python objects into 0/1s to store
import uuid  # unique session ids
import shutil  # recursively deletes session folders
import os
import re  # detecting broad "summarize/explain this document/photo" style questions
import sqlite3  # session persistence — replaces the old sessions.json flat file
from contextlib import contextmanager
from pathlib import Path  # sets path
from datetime import datetime  # timestamps creation

from langchain_community.document_loaders import PyPDFLoader  # load PDF -> Document objects
from langchain_text_splitters import RecursiveCharacterTextSplitter  # split pages into chunks
from langchain_huggingface import HuggingFaceEmbeddings  # embedding model (text -> vectors)
from langchain_community.vectorstores import Chroma  # vector DB for semantic search
from langchain_community.retrievers import BM25Retriever  # keyword/lexical retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever  # combine BM25+vector, then rerank
from langchain_community.document_compressors import FlashrankRerank  # cross-encoder reranker
from langchain_groq import ChatGroq  # Groq-hosted LLM client
from langchain_core.prompts import ChatPromptTemplate  # prompt templating
from langchain_core.output_parsers import StrOutputParser  # extract plain text from LLM response
from langchain_core.documents import Document  # generic doc object (used to wrap image content)
from langchain_core.messages import HumanMessage  # multimodal (text+image) message for the vision LLM
from dotenv import load_dotenv

import base64
from PIL import Image
import easyocr  # replaces pytesseract — noticeably better on noisy/rotated real-world photos

load_dotenv()

# --- Paths: everything scoped PER SESSION (per chat, which may now hold multiple PDFs) ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_ROOT = BASE_DIR / "chroma_db"
CHUNKS_ROOT = BASE_DIR / "chunks"
DB_PATH = BASE_DIR / "sessions.db"
LEGACY_SESSIONS_FILE = BASE_DIR / "sessions.json"  # only read once, for one-time migration

for d in (DATA_DIR, CHROMA_ROOT, CHUNKS_ROOT):
    d.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# Vision-capable Groq model used to describe photos and transcribe any text in them.
# Read from an env var (with a hardcoded default as fallback) so a model
# deprecation can be fixed with a config change + restart instead of a code
# deploy. GROQ_VISION_MODEL_FALLBACK is tried automatically if the primary
# model call fails — see _describe_image() below. Groq's vision lineup changes
# fairly often; check https://console.groq.com/docs/vision for current ids.
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
VISION_MODEL_FALLBACK = os.getenv("GROQ_VISION_MODEL_FALLBACK", "llama-3.2-11b-vision-preview")

# EasyOCR reader — loaded once at import time (loading it per-call would reload
# the model weights on every single image, which is slow and pointless).
_OCR_READER = easyocr.Reader(["en"], gpu=False)

IMAGE_DESCRIBE_PROMPT = (
    "Describe this image in detail: objects, people, setting, layout, colors, mood. "
    "Then, on a new line, transcribe any text visible in the image verbatim, word for "
    "word. If there is no text visible, write 'No text visible.'"
)

# Answer prompt — includes conversation history for tone/continuity
PROMPT_TEMPLATE = """You are a helpful assistant answering questions about the uploaded document(s).
Use ONLY the context below and the recent conversation history to answer.
If the answer isn't in the context, say you don't know.

Format your answer using markdown where it improves clarity: use **bold** for key
terms or important values, and bullet points or numbered lists when listing multiple
items or steps. Don't over-format simple one-line answers — plain sentences are fine
when a list or bold text wouldn't add clarity.

Conversation history:
{history}

Context:
{context}

Question: {question}

Answer clearly and concisely:"""

# Condensation prompt — rewrites a follow-up question into a standalone one BEFORE retrieval.
# This is the piece that was missing: without it, retrieval only ever sees the raw follow-up
# text ("what about pricing?") with no idea what "that" refers to.
CONDENSE_PROMPT_TEMPLATE = """Given the conversation history and a follow-up question, rewrite the
follow-up question as a standalone question that includes any context it implicitly depends on
(pronouns, prior topic, etc). If the follow-up question is already standalone, return it unchanged.
Do NOT change an explicit reference to a specific file type (e.g. "this photo", "this pdf",
"this image", "this document") — if the user names a type, keep that exact wording in your rewrite.
Output ONLY the rewritten question — no preamble, no quotes.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""

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
MAX_SUMMARY_CONTEXT_CHARS = 40000

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


def append_chat_turn(session_id, question, answer, sources):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chat_turns (session_id, question, answer, sources) VALUES (?, ?, ?, ?)",
            (session_id, question, answer, json.dumps(sources)),
        )


def get_chat_history(session_id):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT question, answer, sources FROM chat_turns WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [
        {"question": q, "answer": a, "sources": json.loads(s)}
        for q, a, s in rows
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
    if remaining_chunks:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Chroma.from_documents(
            documents=remaining_chunks,
            embedding=embeddings,
            persist_directory=str(persist_dir)
        )

    # --- Delete the saved file itself ---
    file_path = DATA_DIR / session_id / filename
    if file_path.exists():
        file_path.unlink()


# =================== PDF PROCESSING / RETRIEVAL (per session_id) ===================

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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    new_chunks = splitter.split_documents(documents)

    # Tag each chunk with a clean filename (PyPDFLoader's default 'source' is
    # the full temp file path, which is useless to show a user)
    filename = os.path.basename(pdf_path)
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
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_dir = CHROMA_ROOT / session_id
    if persist_dir.exists() and any(persist_dir.iterdir()):
        vectordb = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
        vectordb.add_documents(new_chunks)
    else:
        Chroma.from_documents(
            documents=new_chunks,
            embedding=embeddings,
            persist_directory=str(persist_dir)
        )

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
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_dir = CHROMA_ROOT / session_id
    if persist_dir.exists() and any(persist_dir.iterdir()):
        vectordb = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
        vectordb.add_documents(new_chunks)
    else:
        Chroma.from_documents(
            documents=new_chunks,
            embedding=embeddings,
            persist_directory=str(persist_dir)
        )

    # Returned as (n_pages, n_chunks) to match process_pdf's signature — an image
    # counts as "1 page" for the summary message shown after upload.
    return 1, len(new_chunks)


def build_hybrid_retriever(session_id):
    """Loads this session's FULL chunk set (across all its PDFs) + vector DB, builds hybrid retriever."""
    chunks_path = CHUNKS_ROOT / f"{session_id}.pkl"
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_dir = CHROMA_ROOT / session_id
    vectordb = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 10})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )

    compressor = FlashrankRerank(top_n=4)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )


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


def get_answer(retriever, question, chat_history, session_id=None):
    """
    Runs condensation -> retrieval -> LLM call manually (not LCEL), so we can
    inject history into condensation and still read response_metadata for tokens.

    EXCEPTION: for broad "summarize/explain this document/photo" style
    questions, retrieval-by-similarity doesn't work well (nothing matches a
    vague query closely, and a big PDF's chunks can drown out a single
    photo's chunk in ranking). Instead we skip straight to feeding the LLM
    the ENTIRE content of whichever attached file type was actually named
    (photo vs PDF vs everything). This requires session_id — if it's not
    passed, this branch is simply skipped and the normal retrieval path runs.
    """
    # temperature=0 for both calls: deterministic question rewriting and
    # deterministic, factual answers
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

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
            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            parser = StrOutputParser()

            answer_parts = []
            total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            sources = []

            for filename, text in targets.items():
                per_file_budget = max(MAX_SUMMARY_CONTEXT_CHARS // len(targets), 2000)
                truncated = text[:per_file_budget]
                note = "\n...[truncated]" if len(text) > per_file_budget else ""
                filled_prompt = prompt.invoke({
                    "context": f"=== {filename} ===\n{truncated}{note}",
                    "question": f"Explain this specific file: {filename}",
                    "history": history_text
                })
                response = llm.invoke(filled_prompt)
                file_answer = parser.invoke(response)
                usage = response.response_metadata.get("token_usage", {})
                for k in total_tokens:
                    total_tokens[k] += usage.get(k, 0)

                answer_parts.append(f"### {filename}\n{file_answer}")
                sources.append({"source": filename, "page": "full document", "text": ""})

            return "\n\n".join(answer_parts), sources, total_tokens

        context, source_names = build_summary_context(session_id, scope=scope)
        history_text = format_history(chat_history)

        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        parser = StrOutputParser()
        filled_prompt = prompt.invoke({
            "context": context,
            "question": question,
            "history": history_text
        })
        response = llm.invoke(filled_prompt)
        answer = parser.invoke(response)
        answer_tokens = response.response_metadata.get("token_usage", {})

        token_usage = {
            "prompt_tokens": answer_tokens.get("prompt_tokens", 0),
            "completion_tokens": answer_tokens.get("completion_tokens", 0),
            "total_tokens": answer_tokens.get("total_tokens", 0),
        }

        sources = [
            {"source": name, "page": "full document", "text": ""}
            for name in source_names
        ]

        return answer, sources, token_usage

    # 1. Condense (multi-turn aware retrieval)
    standalone_question, condense_tokens = condense_question(question, chat_history, llm)

    # 2. Retrieve using the standalone question, not the raw follow-up
    docs = retriever.invoke(standalone_question)

    relevant_docs = [
        doc for doc in docs
        if doc.metadata.get("relevance_score", 0) >= RELEVANCE_THRESHOLD
    ]

    if not relevant_docs:
        return (
            "That doesn't seem to be covered in your uploaded document(s).",
            [],
            {
                "prompt_tokens": condense_tokens.get("prompt_tokens", 0),
                "completion_tokens": condense_tokens.get("completion_tokens", 0),
                "total_tokens": condense_tokens.get("total_tokens", 0),
            }
        )

    docs = relevant_docs
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
    response = llm.invoke(filled_prompt)
    answer = parser.invoke(response)
    answer_tokens = response.response_metadata.get("token_usage", {})

    # Combine token usage across both LLM calls so the UI shows the true cost of the turn
    token_usage = {
        "prompt_tokens": condense_tokens.get("prompt_tokens", 0) + answer_tokens.get("prompt_tokens", 0),
        "completion_tokens": condense_tokens.get("completion_tokens", 0) + answer_tokens.get("completion_tokens", 0),
        "total_tokens": condense_tokens.get("total_tokens", 0) + answer_tokens.get("total_tokens", 0),
    }

    # De-dupe by (source, page): the retriever can return multiple chunks
    # from the same page/image (e.g. a photo split into several chunks, or
    # BM25 + vector both surfacing the same PDF page), which showed up as
    # repeated identical entries in the Sources list. Keep the first chunk
    # seen per (source, page) — that's enough to show where the answer came from.
    seen = set()
    sources = []
    for doc in docs:
        page = doc.metadata.get("page", "unknown")
        source_file = doc.metadata.get("source", "unknown")
        key = (source_file, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "source": source_file,
            "page": page,
            "text": doc.page_content
        })

    return answer, sources, token_usage