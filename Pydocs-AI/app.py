import os
import re
import gc
import json
import uuid
import hashlib
import shutil
import time
from pathlib import Path
from datetime import datetime

# On Windows without Developer Mode, huggingface_hub can't create symlinks in
# its cache and prints noisy warnings / retries. This disables the warning;
# the copy-based cache still works fine for loading on subsequent runs.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import streamlit as st
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

load_dotenv()

# ==========================================
# Streamlit Page Config
# ==========================================
st.set_page_config(page_title="PyDocs AI", page_icon="🐍", layout="wide")

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:0.6rem;">
        <span style="font-size:2.1rem;">🐍</span>
        <div>
            <h1 style="margin-bottom:0;">PyDocs AI</h1>
            <p style="color:#888; margin-top:0;">
                Your persistent, always-on Python documentation assistant — indexed once, queried forever.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ==========================================
# Constants
# ==========================================
CORPUS_DIR = "python-docs"                       # drop your .txt docs here
PERSIST_DIR = "faiss_store_pydocs"               # persistent vector DB, built once
INDEX_MANIFEST_PATH = os.path.join(PERSIST_DIR, "manifest.json")
FAISS_INDEX_NAME = "index"                        # -> index.faiss + index.pkl in PERSIST_DIR

CHAT_DIR = "chat_sessions"                        # persistent chat history, one file per chat

# The corpus is Sphinx-generated plain-text Python docs: every section has a
# title line followed by an underline of repeated punctuation (***, ===, ---, ...),
# and code examples are indented ">>>" blocks. Chunking is structure-aware so
# sections and code blocks stay intact instead of being cut at arbitrary
# character boundaries, and each chunk carries its heading breadcrumb as context.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250

# Embedding model & runtime: BAAI/bge-base-en-v1.5 scores well on retrieval
# benchmarks (MTEB), but running the raw PyTorch/sentence-transformers model
# on CPU for ~18K chunks is what was actually causing the "forever" indexing
# time (a 109M-param transformer forward pass, 18K times, no GPU).
#
# Fix: use fastembed instead of sentence-transformers as the runtime.
# fastembed runs the SAME bge model family through ONNX Runtime with INT8
# quantized weights - same embedding quality/space, much faster CPU inference
# (typically 3-5x) because ONNX+quantization skips a lot of the overhead
# PyTorch carries for single-model, CPU-only inference.
#
# Two-tier strategy while you're iterating (a normal RAG dev pattern, not a
# hack): use the SMALL model while you're still tweaking chunking or prompts
# and re-indexing often, then switch to the BASE model for the one final
# "production quality" build once you're happy with the pipeline.
#   - "BAAI/bge-small-en-v1.5"  -> 384-dim, ~33M params, fast iteration
#   - "BAAI/bge-base-en-v1.5"   -> 768-dim, ~109M params, better recall, slower
# Both are natively supported by fastembed. Changing this string alone
# triggers an automatic rebuild (see compute_corpus_hash below), because the
# two models produce embeddings in different, incompatible vector spaces.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE is an ASYMMETRIC retrieval model: the QUERY side is meant to carry a
# short instruction prefix while the passages are embedded as-is. fastembed
# does NOT add this prefix automatically (its query_embed() just calls the
# same embed() as documents - verified against the installed package), so we
# add it by hand to every search query. Passages are embedded without it.
# For bge-*-v1.5 this mainly helps short-query -> long-passage retrieval,
# which is exactly this app's shape (a typed question against doc sections).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Batch size for embedding requests during index building. Larger batches
# mean fewer Python-level calls (faster) but more RAM used per batch.
# Lowered from 256 -> 64 after hitting an ONNX Runtime "bad allocation"
# (genuine out-of-memory inside the C++ inference layer) on an 8GB machine.
# Smaller batches mean each individual ONNX Runtime call has to allocate
# less at once, which matters when free memory is already tight.
EMBED_BATCH_SIZE = 64

# FAISS is an in-memory index that is snapshotted to disk with save_local().
# Snapshotting rewrites the whole growing index file, so we don't do it after
# every 64-chunk batch - we do it at file boundaries once at least this many
# chunks have accumulated since the last snapshot. That bounds crash-loss to
# roughly this many chunks of work while keeping snapshots infrequent.
SAVE_EVERY_CHUNKS = 2000

# Retrieval tuning: the corpus has a lot of near-duplicate content across
# Python versions (whatsnew/2.x, 3.x, changelog...), so plain top-k similarity
# tends to return five near-identical chunks.
#
# The pipeline is two-stage:
#   1. Bi-encoder + MMR (Maximal Marginal Relevance) pulls a DIVERSE candidate
#      pool - fast, recall-oriented, de-duplicates near-identical chunks.
#   2. A cross-encoder reranker re-scores that pool against the query with a
#      full query+passage attention pass - slow but far more PRECISE than the
#      bi-encoder - and we keep only the top RETRIEVAL_K.
# Diversity first (MMR), then precision (reranker): the reranker never sees a
# pool that's already collapsed into duplicates.
RETRIEVAL_K = 6            # final chunks handed to the LLM
RERANK_POOL_K = 20         # diverse candidates MMR feeds into the reranker
RETRIEVAL_FETCH_K = 40     # raw neighbours MMR selects the pool from
RETRIEVAL_LAMBDA = 0.5     # MMR relevance/diversity trade-off (1.0 = pure relevance)

# Cross-encoder reranker, also run on CPU through fastembed's ONNX runtime so
# it adds no new heavyweight dependency (no torch). bge-reranker-base pairs
# naturally with the bge embedding family.
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# ==========================================
# Session State Initialization
# ==========================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "corpus_stats" not in st.session_state:
    st.session_state.corpus_stats = None  # {"files": n, "chunks": n, "hash": str}

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# Cached Resources
# ==========================================
@st.cache_resource
def get_embeddings_model():
    # threads=None (default) lets ONNX Runtime auto-tune per-session threading;
    # os.cpu_count() pins it to use every core available on this machine,
    # which matters a lot when there's no GPU to fall back on.
    return FastEmbedEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        max_length=512,
        batch_size=EMBED_BATCH_SIZE,
        threads=os.cpu_count(),
        cache_dir="fastembed_cache",  # model weights downloaded once, reused
    )

@st.cache_resource
def get_reranker():
    # fastembed's cross-encoder runs on ONNX/CPU - same runtime story as the
    # embedder, no torch pulled in. Weights are downloaded once and cached.
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(
        model_name=RERANKER_MODEL_NAME,
        cache_dir="fastembed_cache",
    )

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

# ==========================================
# FAISS open helper
# ==========================================
def _open_faiss(persist_dir, embeddings):
    """
    Open a persisted FAISS store, ALWAYS re-passing normalize_L2=True.

    This is not optional: FAISS.save_local() pickles only the docstore and the
    id map - it does NOT persist normalize_L2 or distance_strategy (verified in
    the installed langchain_community). load_local() reconstructs the store
    with those flags at their defaults unless we pass them back in. Since the
    index was BUILT with normalize_L2=True (L2-normalized vectors + an L2 index
    == cosine-similarity ranking, which is what BGE expects), every load - for
    querying AND for resuming a build - must re-pass it, or newly added vectors
    would be un-normalized and silently rank against the normalized ones.

    allow_dangerous_deserialization=True is required because index.pkl is a
    pickle; that's safe here because WE wrote it, on this machine.
    """
    return FAISS.load_local(
        persist_dir,
        embeddings,
        index_name=FAISS_INDEX_NAME,
        allow_dangerous_deserialization=True,
        normalize_L2=True,
    )

# ==========================================
# Helper Functions — Corpus Indexing
# ==========================================
def collect_txt_files(folder):
    """Recursively collect every .txt file under folder."""
    return sorted(Path(folder).rglob("*.txt"))

def compute_corpus_hash(text_files):
    """
    Fingerprint the corpus (paths + sizes) AND the embedding model name, so
    a model swap invalidates the old index and triggers an automatic
    rebuild instead of silently mixing embedding spaces.

    Deliberately does NOT include file modification times. mtimes get
    touched by things that don't change content at all - OneDrive/cloud
    sync re-writing files during a sync pass, antivirus scans, or simply
    re-extracting the same zip archive - and on a synced or scanned folder
    this was silently invalidating the index and forcing a full ~18K-chunk
    rebuild on every single run, even though nothing had actually changed.
    Path + size together are enough to catch the changes that matter here
    (added, removed, or resized doc files) without that false-positive risk.
    """
    h = hashlib.md5()
    h.update(EMBEDDING_MODEL_NAME.encode("utf-8"))
    for f in text_files:
        stat = f.stat()
        h.update(str(f).encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
    return h.hexdigest()

def load_manifest():
    if os.path.exists(INDEX_MANIFEST_PATH):
        with open(INDEX_MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_manifest(manifest):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(INDEX_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

# --- Structure-aware section splitting -----------------------------------

_UNDERLINE_RE = re.compile(r"^([*=\-~^\"'.+#_:])\1{3,}\s*$")

def _split_into_sections(text):
    """
    Split a Sphinx plain-text doc into (breadcrumb, content) sections using
    its title-line + underline convention. The underline character used for
    each heading level is discovered per-file (Sphinx assigns them in order
    of first appearance: level 1 first, then level 2, etc.), and a breadcrumb
    stack tracks nesting so every chunk knows exactly where it lives, e.g.
    "3. An Informal Introduction to Python > 3.1 Using Python as a Calculator".
    """
    lines = text.split("\n")
    level_for_char = {}          # underline char -> heading level (1 = top)
    breadcrumb_stack = []        # list of (level, title)
    sections = []                # list of (breadcrumb_str, [content_lines])
    current_lines = []

    def flush():
        if current_lines and "".join(current_lines).strip():
            breadcrumb = " > ".join(t for _, t in breadcrumb_stack)
            sections.append((breadcrumb, current_lines[:]))
        current_lines.clear()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < n else ""
        m = _UNDERLINE_RE.match(nxt.strip()) if nxt.strip() else None
        is_heading = (
            m is not None
            and line.strip() != ""
            and len(nxt.strip()) >= len(line.strip()) * 0.6  # underline roughly spans title
        )
        if is_heading:
            char = m.group(1)
            if char not in level_for_char:
                level_for_char[char] = len(level_for_char) + 1
            level = level_for_char[char]

            flush()
            while breadcrumb_stack and breadcrumb_stack[-1][0] >= level:
                breadcrumb_stack.pop()
            breadcrumb_stack.append((level, line.strip()))

            i += 2  # skip title + underline
            continue

        current_lines.append(line)
        i += 1

    flush()

    if not sections:
        sections = [("", lines)]

    return sections

# Safety cap for a single Sphinx doc page. Real Python-docs pages are well
# under 1MB; anything past this is almost certainly a stray non-doc file
# (a build artifact, a concatenated dump, something mislabeled .txt) rather
# than content worth indexing - and reading + decoding it can spike memory
# enough to crash the whole run on machines with limited RAM.
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB

def _iter_chunks(text_files, skipped_files, start_index=0):
    """
    Generator version of the chunker: yields (Document, file_index) one at a
    time instead of building a 17,915-item list up front.

    Why this matters on a memory-constrained machine: the old version read
    every file and appended every chunk into one big Python list before
    embedding started, so peak memory was O(whole corpus). This version
    only ever holds the current file's text plus whatever's in the current
    embedding batch (see build_index) - peak memory is O(batch_size), not
    O(corpus_size). Python's refcounting GC frees each file's raw_text as
    soon as we move to the next file, since nothing else references it.

    file_index (1-based, over ALL text_files including skipped ones so it
    stays stable across runs) lets the caller drive a progress bar off
    "files seen so far / total files" and lets a resumed build skip straight
    past files that were already embedded in a previous run.

    start_index lets a resumed build skip straight past files that were
    already embedded and persisted in a previous (crashed) run - those
    files are never even opened again, so resuming is fast and doesn't
    redo completed work.

    skipped_files is a list the caller owns; oversized/unreadable files are
    appended to it as (path, reason) tuples as they're encountered.
    """
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n\n", "\n\n", "\n", " ", ""],
    )

    for file_index, file in enumerate(text_files, start=1):
        if file_index <= start_index:
            continue  # already embedded in a previous (crashed) run - don't reopen it

        size = file.stat().st_size
        if size > MAX_FILE_SIZE_BYTES:
            skipped_files.append((str(file), f"{size / 1e6:.1f}MB, over the {MAX_FILE_SIZE_BYTES/1e6:.0f}MB cap"))
            continue

        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
        except (MemoryError, OSError) as e:
            skipped_files.append((str(file), f"read failed: {e}"))
            continue

        for breadcrumb, content_lines in _split_into_sections(raw_text):
            section_text = "\n".join(content_lines).strip("\n")
            if not section_text.strip():
                continue

            header = f"Section: {breadcrumb}\n\n" if breadcrumb else ""

            if len(section_text) <= CHUNK_SIZE:
                pieces = [section_text]
            else:
                pieces = sub_splitter.split_text(section_text)

            for piece in pieces:
                if not piece.strip():
                    continue
                yield Document(
                    page_content=header + piece,
                    metadata={
                        "source": str(file),
                        "section": breadcrumb or Path(file).stem,
                    },
                ), file_index

        # raw_text and content_lines drop out of scope here - nothing keeps
        # them alive once the next loop iteration starts.



def _checkpoint_path(persist_dir):
    return os.path.join(persist_dir, "build_checkpoint.json")


def _load_checkpoint(persist_dir, expected_hash):
    """
    Returns (files_completed, chunks_so_far, skipped_files) from a
    checkpoint left by a previous, interrupted build - but only if that
    checkpoint's corpus/model hash matches the one we're building now.
    A hash mismatch means the docs or embedding model changed since the
    crash, so resuming would silently mix incompatible data - treated the
    same as "no checkpoint" (0, 0, []), which triggers a clean full rebuild.
    """
    path = _checkpoint_path(persist_dir)
    if not os.path.exists(path):
        return 0, 0, []
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return 0, 0, []
    if data.get("hash") != expected_hash:
        return 0, 0, []
    return data.get("files_completed", 0), data.get("chunks_so_far", 0), data.get("skipped_files", [])


def _save_checkpoint(persist_dir, corpus_hash, files_completed, chunks_so_far, skipped_files):
    # Write-to-temp-then-rename: os.replace is atomic on both POSIX and
    # Windows, so a crash mid-write can never leave a half-written,
    # unreadable checkpoint file behind - worst case we lose only the
    # progress since the last successful snapshot, not the checkpoint itself.
    path = _checkpoint_path(persist_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(
            {
                "hash": corpus_hash,
                "files_completed": files_completed,
                "chunks_so_far": chunks_so_far,
                "skipped_files": skipped_files,
            },
            f,
        )
    os.replace(tmp_path, path)


def build_index(text_files, embeddings, persist_dir=PERSIST_DIR, progress_cb=None, corpus_hash=None):
    """One-time ingestion: read every .txt file, chunk it, embed it, persist it.

    Storage is FAISS, snapshotted to `persist_dir` with save_local() (writing
    index.faiss + index.pkl). FAISS was chosen over Chroma for this workload
    specifically: the app builds once and is then read-only, and FAISS is a
    couple of flat files with no SQLite process holding OS file locks - which
    is what made the old Chroma path so painful on Windows (PermissionError
    [WinError 32] on the post-build directory move, which aborted before the
    manifest was written and forced a full rebuild on every run).

    Resumable: the build snapshots + checkpoints at FILE boundaries once
    SAVE_EVERY_CHUNKS chunks have accumulated. Snapshotting only at a file
    boundary guarantees the on-disk index always contains WHOLE files, so a
    resume never re-adds a half-finished file's chunks (which FAISS, having no
    upsert/dedup, would otherwise duplicate). If the process crashes and this
    is called again with the same persist_dir and corpus_hash, it loads the
    last snapshot and continues from the next unprocessed file.

    Chunks are consumed from `_iter_chunks` one at a time and embedded in
    batches (rather than one FAISS.from_documents(all_chunks) call) so peak
    memory stays O(batch_size) instead of O(corpus_size), and so the UI can
    report progress after each batch instead of hanging on one opaque call.

    `progress_cb(fraction, message)` is called after each batch if provided.
    """
    os.makedirs(persist_dir, exist_ok=True)
    resume_files, resume_chunks, resume_skipped = _load_checkpoint(persist_dir, corpus_hash)

    total_files = len(text_files)

    # Try to reopen the last snapshot to resume. If it's missing or corrupt
    # (e.g. a crash mid-save_local left inconsistent files), fall back to a
    # clean rebuild rather than crashing or resuming onto a broken index.
    vectorstore = None
    index_file = os.path.join(persist_dir, f"{FAISS_INDEX_NAME}.faiss")
    if resume_files and os.path.exists(index_file):
        try:
            vectorstore = _open_faiss(persist_dir, embeddings)
            print(
                f"[build_index] Resuming from snapshot: {resume_files}/{total_files} "
                f"files done, {resume_chunks} chunks embedded."
            )
        except Exception as e:  # corrupt/incompatible snapshot -> start over
            print(f"[build_index] Snapshot unreadable ({e}); rebuilding from scratch.")
            vectorstore = None
            resume_files, resume_chunks, resume_skipped = 0, 0, []

    skipped_files = list(resume_skipped)
    n_chunks = resume_chunks
    batch = []
    chunks_since_save = 0
    start = time.time()

    def _add_batch():
        """Embed + add the current batch to the in-memory index (no snapshot)."""
        nonlocal vectorstore, batch
        if not batch:
            return
        if vectorstore is None:
            # normalize_L2=True -> normalized vectors + L2 index == cosine
            # ranking, which is what BGE embeddings are trained for.
            vectorstore = FAISS.from_documents(batch, embeddings, normalize_L2=True)
        else:
            vectorstore.add_documents(batch)
        batch = []

    def _snapshot(files_completed):
        """Persist the in-memory index + checkpoint. Safe only at a file boundary."""
        if vectorstore is not None:
            vectorstore.save_local(persist_dir, index_name=FAISS_INDEX_NAME)
        _save_checkpoint(persist_dir, corpus_hash, files_completed, n_chunks, skipped_files)

    prev_file_index = resume_files  # highest file index fully queued so far
    for doc, file_index in _iter_chunks(text_files, skipped_files, start_index=resume_files):
        # A change in file_index means the previous file is fully chunked and
        # (once we flush the batch) fully in the index - a safe snapshot point.
        if file_index != prev_file_index and prev_file_index != resume_files:
            _add_batch()  # ensure ALL of prev_file_index is in the index
            if chunks_since_save >= SAVE_EVERY_CHUNKS:
                _snapshot(prev_file_index)
                chunks_since_save = 0
                gc.collect()
        prev_file_index = file_index

        batch.append(doc)
        n_chunks += 1
        chunks_since_save += 1

        if len(batch) >= EMBED_BATCH_SIZE:
            _add_batch()  # memory flush only - may be mid-file, so NO snapshot here
            gc.collect()
            if progress_cb:
                elapsed = time.time() - start
                files_done_this_run = file_index - resume_files
                rate = files_done_this_run / elapsed if elapsed > 0 else 0
                remaining = (total_files - file_index) / rate if rate > 0 else 0
                progress_cb(
                    file_index / total_files,
                    f"Processed {file_index:,}/{total_files:,} files, "
                    f"{n_chunks:,} chunks embedded so far "
                    f"(~{remaining:.0f}s remaining)",
                )

    _add_batch()             # flush the final partial batch
    _snapshot(total_files)   # final snapshot covers everything
    gc.collect()

    if progress_cb:
        progress_cb(1.0, f"Done: {n_chunks:,} chunks embedded.")

    n_files = total_files - len(skipped_files)
    return vectorstore, n_files, n_chunks, skipped_files


def _remove_persist_dir(path, retries=6, delay=1.0):
    """Delete the persist directory, retrying briefly on transient file locks."""
    if not os.path.exists(path):
        return

    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except (PermissionError, OSError):
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def load_or_build_vectorstore(force_rebuild=False, progress_cb=None):
    """
    Loads the persistent Python-docs index from disk if it already matches
    the current corpus (and embedding model). Otherwise (first run, docs
    changed, model changed, or forced), it re-ingests everything once and
    persists it for all future sessions.
    """
    embeddings = get_embeddings_model()

    text_files = collect_txt_files(CORPUS_DIR)
    if not text_files:
        return None, {"files": 0, "chunks": 0, "hash": None}, "no_files", []

    current_hash = compute_corpus_hash(text_files)
    manifest = load_manifest()

    index_exists = os.path.exists(PERSIST_DIR) and manifest is not None
    hash_matches = index_exists and manifest.get("hash") == current_hash

    if index_exists and hash_matches and not force_rebuild:
        try:
            vectorstore = _open_faiss(PERSIST_DIR, embeddings)
            stats = {
                "files": manifest["files"],
                "chunks": manifest["chunks"],
                "hash": current_hash,
            }
            return vectorstore, stats, "loaded", []
        except Exception as e:
            # A valid manifest but an unreadable index (corrupt/partial). Wipe
            # and rebuild rather than getting stuck reporting a phantom index.
            print(f"[load] existing index failed to open ({e}); rebuilding.")
            _remove_persist_dir(PERSIST_DIR)

    # Need to (re)build. We build DIRECTLY into PERSIST_DIR (no temp dir + move
    # step - that move on top of Chroma's open file handles was the old source
    # of the "DB error at the end"). FAISS's build is resumable via a snapshot
    # + checkpoint left in PERSIST_DIR (see build_index).
    #
    # force_rebuild=True (the "Re-index" button) means an explicit clean
    # rebuild, so wipe any existing store + checkpoint first.
    if force_rebuild:
        _remove_persist_dir(PERSIST_DIR)

    vectorstore, n_files, n_chunks, skipped_files = build_index(
        text_files, embeddings, persist_dir=PERSIST_DIR, progress_cb=progress_cb, corpus_hash=current_hash
    )

    # Build finished - drop the checkpoint (it only exists to resume an
    # interrupted build; a completed store doesn't need it).
    checkpoint = _checkpoint_path(PERSIST_DIR)
    if os.path.exists(checkpoint):
        os.remove(checkpoint)

    # Persist the manifest LAST, once the store is complete on disk. Its
    # presence is what tells the next run "a valid index matching this
    # corpus/model already exists - just load it, don't rebuild".
    manifest = {"files": n_files, "chunks": n_chunks, "hash": current_hash}
    save_manifest(manifest)

    return vectorstore, manifest, "built", skipped_files


def force_load_existing_index():
    """
    Open whatever's currently sitting in PERSIST_DIR as-is, with no
    corpus-hash check at all. This exists as a fast escape hatch: if the
    hash check is ever producing false-positive "needs rebuild" results
    and you already know a complete, valid index is sitting on disk, this
    loads it directly instead of waiting through a rebuild.

    Returns (None, None) if PERSIST_DIR doesn't look like a real FAISS store
    (no index.faiss present) - there's nothing safe to load.
    """
    index_file = os.path.join(PERSIST_DIR, f"{FAISS_INDEX_NAME}.faiss")
    if not os.path.exists(index_file):
        return None, None

    embeddings = get_embeddings_model()
    vectorstore = _open_faiss(PERSIST_DIR, embeddings)
    manifest = load_manifest()
    stats = manifest if manifest else {"files": "?", "chunks": "?", "hash": None}
    return vectorstore, stats

# ==========================================
# Retrieval — history-aware rewrite + MMR pool + cross-encoder rerank
# ==========================================
def rewrite_query_with_history(llm, history_msgs, question):
    """
    Turn a follow-up like "what about its keyword arguments?" into a
    standalone query ("what are the keyword arguments of str.split()?") using
    the recent conversation, so retrieval - which only ever sees this one
    string - isn't handed a context-free fragment. With no prior history the
    question is returned unchanged. Failures fall back to the raw question;
    a bad rewrite must never block answering.
    """
    if not history_msgs:
        return question

    convo = ""
    for m in history_msgs[-6:]:
        role = "User" if m["role"] == "user" else "Assistant"
        convo += f"{role}: {m['content']}\n"

    prompt = (
        "Given the conversation below and a follow-up question, rewrite the "
        "follow-up as a fully standalone question that can be understood "
        "without the conversation. If it is already standalone, return it "
        "unchanged. Output ONLY the rewritten question, nothing else.\n\n"
        f"Conversation:\n{convo}\n"
        f"Follow-up question: {question}\n"
        "Standalone question:"
    )
    try:
        rewritten = llm.invoke(prompt).content.strip()
        return rewritten or question
    except Exception:
        return question


def retrieve(vectorstore, reranker, retrieval_query):
    """
    Two-stage retrieval:
      1. MMR pulls a diverse candidate pool (BGE query instruction prepended
         for the bi-encoder search only).
      2. The cross-encoder reranks that pool against the RAW query and we keep
         the top RETRIEVAL_K.
    Returns a list of (Document, rerank_score) already sorted best-first.
    """
    embed_query = BGE_QUERY_INSTRUCTION + retrieval_query
    pool = vectorstore.max_marginal_relevance_search(
        embed_query,
        k=RERANK_POOL_K,
        fetch_k=RETRIEVAL_FETCH_K,
        lambda_mult=RETRIEVAL_LAMBDA,
    )
    if not pool:
        return []

    # Cross-encoder wants natural text pairs - no BGE instruction prefix here.
    scores = list(reranker.rerank(retrieval_query, [d.page_content for d in pool]))
    ranked = sorted(zip(pool, scores), key=lambda ds: ds[1], reverse=True)
    return ranked[:RETRIEVAL_K]

# ==========================================
# Helper Functions — Persistent Chat History
# ==========================================
def _session_path(session_id):
    return os.path.join(CHAT_DIR, f"{session_id}.json")

def save_session(session_id, messages):
    """Persist the current chat to disk so it survives app restarts."""
    if not messages:
        return
    os.makedirs(CHAT_DIR, exist_ok=True)
    existing = load_session_raw(session_id) or {}
    title = existing.get("title") or messages[0]["content"][:60]
    data = {
        "title": title,
        "created": existing.get("created", datetime.now().isoformat()),
        "updated": datetime.now().isoformat(),
        "messages": messages,
    }
    with open(_session_path(session_id), "w", encoding="utf-8") as f:
        json.dump(data, f)

def load_session_raw(session_id):
    path = _session_path(session_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_sessions():
    """All saved chats, most recently updated first."""
    if not os.path.exists(CHAT_DIR):
        return []
    sessions = []
    for fname in os.listdir(CHAT_DIR):
        if not fname.endswith(".json"):
            continue
        session_id = fname[:-5]
        data = load_session_raw(session_id)
        if data:
            sessions.append((session_id, data))
    sessions.sort(key=lambda x: x[1].get("updated", ""), reverse=True)
    return sessions

def delete_session(session_id):
    path = _session_path(session_id)
    if os.path.exists(path):
        os.remove(path)

def switch_to_session(session_id, messages=None):
    st.session_state.session_id = session_id
    st.session_state.messages = messages if messages is not None else []

def start_new_chat():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []

def clear_chat():
    st.session_state.messages = []
    save_session(st.session_state.session_id, [])

# ==========================================
# Sidebar
# ==========================================
with st.sidebar:
    st.header("📚 Knowledge Base")
    st.caption(f"Source folder: `{CORPUS_DIR}/`")

    if st.session_state.vectorstore is None:
        # First run (or corpus/model changed) needs a real embedding pass.
        # A progress bar + live ETA replaces the old plain spinner, so it's
        # obvious the app is working through chunks rather than stuck.
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def _update_progress(fraction, message):
            progress_bar.progress(fraction)
            status_text.caption(message)

        # Warm the embedding model FIRST, with its own explicit status. On the
        # very first run this triggers a one-time ~130 MB model download, which
        # happens before any chunk can be embedded - so without this message the
        # progress bar would just sit at 0% with no caption (looking frozen)
        # while the model loads. Attributing that wait to a clear message makes
        # "downloading model" distinguishable from "stuck".
        status_text.caption("Loading embedding model (first run downloads ~130 MB, please wait)…")
        try:
            get_embeddings_model().embed_query("warmup")
        except Exception as e:
            print(f"[warmup] embedding model load failed: {e}")
        status_text.caption("Indexing documents…")

        vectorstore, stats, status, skipped_files = load_or_build_vectorstore(
            progress_cb=_update_progress
        )
        st.session_state.vectorstore = vectorstore
        st.session_state.corpus_stats = stats
        progress_bar.empty()
        status_text.empty()

        if skipped_files:
            with st.expander(f"⚠️ Skipped {len(skipped_files)} file(s) during indexing"):
                for path, reason in skipped_files:
                    st.caption(f"`{path}` — {reason}")

        if status == "no_files":
            st.warning(
                f"No `.txt` files found in `{CORPUS_DIR}/`. "
                "Add your Python documentation there and refresh."
            )
        elif status == "built":
            st.success(
                f"Indexed {stats['files']} files → {stats['chunks']} chunks. "
                "This only happens once."
            )
        else:
            st.info(
                f"Loaded existing index instantly: {stats['files']} files, "
                f"{stats['chunks']} chunks."
            )
    else:
        stats = st.session_state.corpus_stats
        if stats and stats["files"]:
            st.success(f"Ready — {stats['files']} files / {stats['chunks']} chunks indexed.")
        else:
            st.warning(f"No documents indexed yet. Add `.txt` files to `{CORPUS_DIR}/`.")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Reset Chat"):
            clear_chat()
            st.success("Chat history cleared.")

    with col2:
        if st.button("🔁 Re-index"):
            reindex_bar = st.progress(0.0)
            reindex_status = st.empty()

            def _update_reindex_progress(fraction, message):
                reindex_bar.progress(fraction)
                reindex_status.caption(message)

            vectorstore, stats, status, skipped_files = load_or_build_vectorstore(
                force_rebuild=True, progress_cb=_update_reindex_progress
            )
            st.session_state.vectorstore = vectorstore
            st.session_state.corpus_stats = stats
            reindex_bar.empty()
            reindex_status.empty()
            if skipped_files:
                with st.expander(f"⚠️ Skipped {len(skipped_files)} file(s) during indexing"):
                    for path, reason in skipped_files:
                        st.caption(f"`{path}` — {reason}")
            st.success(f"Re-indexed: {stats['files']} files, {stats['chunks']} chunks.")

    if st.button("⚡ Load existing index (skip check)", use_container_width=True):
        # Bypasses compute_corpus_hash entirely - use this when you already
        # know faiss_store_pydocs/ holds a complete, valid index and just
        # want it loaded immediately, no verification.
        vectorstore, stats = force_load_existing_index()
        if vectorstore is None:
            st.error(
                f"No `{FAISS_INDEX_NAME}.faiss` found in `{PERSIST_DIR}/` - there's "
                "nothing to load. Use Re-index to build one."
            )
        else:
            st.session_state.vectorstore = vectorstore
            st.session_state.corpus_stats = stats
            st.success(f"Loaded directly: {stats['files']} files, {stats['chunks']} chunks (unverified).")

    st.divider()

    # --- Persistent, multi-session chat history ---------------------------
    st.header("💬 Chats")
    if st.button("➕ New Chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    saved_sessions = list_sessions()
    if not saved_sessions:
        st.caption("Your conversations will show up here once you start chatting.")
    else:
        for session_id, data in saved_sessions:
            is_active = session_id == st.session_state.session_id
            title = data.get("title", "New chat") or "New chat"
            label = f"{'🟢 ' if is_active else ''}{title}"
            row_a, row_b = st.columns([5, 1])
            with row_a:
                if st.button(label, key=f"open_{session_id}", use_container_width=True):
                    switch_to_session(session_id, data.get("messages", []))
                    st.rerun()
            with row_b:
                if st.button("🗑️", key=f"del_{session_id}"):
                    delete_session(session_id)
                    if is_active:
                        start_new_chat()
                    st.rerun()

    st.divider()
    st.caption(
        "Drop any `.txt` Python documentation files (nested folders are fine) "
        f"into `{CORPUS_DIR}/`, then hit **Re-index**. After that, this Q&A "
        "runs forever against the persisted vector store — no re-uploading, "
        "no re-processing. Chats are saved to disk automatically."
    )

# ==========================================
# Display Chat History
# ==========================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# Main Chat Input
# ==========================================
if user_query := st.chat_input("Ask anything about Python..."):

    if st.session_state.vectorstore is None:
        st.error(f"No index available. Add `.txt` files to `{CORPUS_DIR}/` and re-index.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_query)

    # History for the query-rewrite step is everything BEFORE this new turn.
    history_msgs = list(st.session_state.messages)

    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    llm = get_llm()

    # ==========================================
    # Retrieval
    # ==========================================
    with st.chat_message("assistant"):
        with st.spinner("Consulting the docs..."):
            # 1. Make the question standalone using recent history.
            retrieval_query = rewrite_query_with_history(llm, history_msgs, user_query)

            # 2. Diverse MMR pool -> cross-encoder rerank -> top-k.
            reranker = get_reranker()
            ranked = retrieve(st.session_state.vectorstore, reranker, retrieval_query)

            context_text = ""
            citations_data = []
            for i, (doc, score) in enumerate(ranked, 1):
                source_name = doc.metadata.get("source", "Unknown")
                section_name = doc.metadata.get("section", "")

                context_text += f"""
[Chunk {i}]
Source: {source_name}
Section: {section_name}
Content:
{doc.page_content}
"""
                citations_data.append({
                    "chunk": i,
                    "source": source_name,
                    "section": section_name,
                    "score": round(float(score), 4),
                    "content": doc.page_content,
                })

            # ==========================================
            # Chat History Formatting (for the answer prompt)
            # ==========================================
            history_text = ""
            for m in st.session_state.messages[-6:]:
                role = "User" if m["role"] == "user" else "Assistant"
                history_text += f"{role}: {m['content']}\n"

            # ==========================================
            # Prompt
            # ==========================================
            final_prompt = f"""
You are a precise, expert Python documentation assistant.

Your job is to answer ONLY from the retrieved context below, which comes
from the official Python documentation.

Rules:
1. Use only the retrieved context.
2. Prefer the chunk that most directly answers the question.
3. If multiple chunks are relevant, combine them carefully.
4. Do not guess or add outside information.
5. If the answer is not clearly available in the context, say:
   "I don't know based on the indexed documentation."
6. Keep the answer concise but complete, and use code blocks for code.
7. When useful, mention which source file the answer came from.

Recent Chat History:
{history_text}

Retrieved Context:
{context_text}

Question:
{user_query}

Answer:
"""

            response = llm.invoke(final_prompt)
            answer = response.content

        st.markdown(answer)

        with st.expander("🔍 View Sources & Rerank Scores"):
            if retrieval_query.strip() != user_query.strip():
                st.caption(f"🔎 Searched for: _{retrieval_query}_")
            if not citations_data:
                st.caption("No matching chunks were retrieved.")
            for cite in citations_data:
                st.markdown(
                    f"**Chunk {cite['chunk']}** | "
                    f"**Source:** `{cite['source']}` | "
                    f"**Section:** {cite['section']} | "
                    f"**Rerank score:** `{cite['score']}`"
                )
                st.caption(cite["content"][:400] + "...")
                st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # Persist this chat to disk after every turn.
    save_session(st.session_state.session_id, st.session_state.messages)
