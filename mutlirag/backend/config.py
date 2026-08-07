"""Central configuration for the Multi-RAG app.

Everything tweakable lives here so you can tune the pipeline without hunting
through the code.
"""

import os

# Project root = the parent of backend/ (this file lives at backend/config.py).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where generated data (vector index, chat history, extracted images) is stored.
# Defaults to the project root. On a host with an ephemeral filesystem (e.g.
# Render), set MULTIRAG_DATA_DIR to a mounted persistent disk so data survives
# restarts and redeploys.
DATA_DIR = os.getenv("MULTIRAG_DATA_DIR", BASE_DIR)

# --- Embedding model (runs locally via sentence-transformers) ---
# BAAI/bge-small-en-v1.5: 384-dim, ~130MB, fast, high accuracy on dense retrieval (MTEB benchmark leader).
# Swap for "BAAI/bge-base-en-v1.5" (768-dim, ~440MB) for maximum semantic quality.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# --- Chunking ---
# Characters per chunk and overlap between consecutive chunks.
# Overlap keeps sentences that straddle a boundary retrievable from both sides.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# --- Retrieval ---
# How many chunks to pull back and stuff into the prompt as context.
TOP_K = 10  # base chunks per query; auto-scales up with more indexed documents
# Hybrid search: final score = HYBRID_ALPHA * semantic (embedding cosine)
#              + (1 - HYBRID_ALPHA) * keyword (BM25, normalized).
# Embeddings catch meaning ("who earns most"); BM25 catches exact tokens
# ("ZEBRA-42", invoice numbers). 0.65 favors meaning but keeps exact matches.
HYBRID_ALPHA = 0.65

# --- Reranking (second-stage, cross-encoder) ---
# Hybrid search above blends two INDEPENDENT signals (embedding cosine + BM25)
# with a fixed weight — it can't reason about the query and a chunk together.
# A cross-encoder scores each (query, chunk) pair jointly, which is far more
# accurate at ranking but too slow to run over the whole index, so it only
# re-scores the small candidate pool hybrid search already retrieved.
RERANK_ENABLED = True
# cross-encoder/ms-marco-MiniLM-L-6-v2: 22M params, fast enough for per-request
# reranking of ~30 chunks. Swap for "BAAI/bge-reranker-base" (278M params, pairs
# naturally with the bge embedding model above) for higher quality at more latency.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Vector DB backend ---
# "weaviate" -> self-hosted Weaviate (see docker-compose.yml); "chroma" -> the
# original embedded ChromaDB. If Weaviate is selected but unreachable at
# startup, the app automatically falls back to Chroma so nothing breaks.
VECTOR_BACKEND = "weaviate"

# --- Weaviate connection (only used when VECTOR_BACKEND == "weaviate") ---
# Matches the ports published in docker-compose.yml. We supply our own vectors
# (local sentence-transformers), so no Weaviate vectorizer module is needed.
WEAVIATE_HOST = "localhost"
WEAVIATE_HTTP_PORT = 8085     # host port mapped to the container's 8080
WEAVIATE_GRPC_PORT = 50051
WEAVIATE_COLLECTION = "MultiRagChunk"   # class names must start uppercase

# --- Persistence ---
# The index (chunks + vectors) is saved here and reloaded on app start, so
# documents survive restarts. Delete the folder (or use the app's Reset
# button) to start fresh.
#   * Weaviate: data lives in the Docker volume (weaviate_data), not here.
#   * Chroma:   data lives under INDEX_DIR/chroma.
INDEX_DIR = os.path.join(DATA_DIR, "index_store")
# Per-file ingestion cache, keyed by content hash: re-uploading the same file
# skips OCR + vision entirely.
CACHE_DIR = os.path.join(INDEX_DIR, "cache")

# --- Chat memory ---
# Follow-up questions are rewritten into standalone search queries using the
# recent chat history, with a small fast model.
REWRITE_MODEL = "llama-3.1-8b-instant"
REWRITE_HISTORY_TURNS = 6   # how many recent messages to give the rewriter

# --- Groq generation defaults ---
# Curated fallback list, used only if the live /models call fails.
# The app fetches the real, current list from Groq at runtime.
FALLBACK_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
]
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.2   # low => grounded, factual answers for RAG
MAX_TOKENS = 1024

# --- Vision (image understanding via Groq) ---
# Every ingested image (standalone, embedded in PDF/DOCX, or a rendered
# scanned page) is also described by a multimodal model so charts, photos,
# tables-as-images, and layouts become searchable — not just their OCR text.
# Set VISION_ENABLED = False to fall back to OCR-only (faster, no API calls).
VISION_ENABLED = True
# Groq's vision model. The old llama-4 scout/maverick models were decommissioned;
# qwen3.6-27b is the current multimodal option. It's a reasoning model, so vision.py
# passes reasoning_effort="none" and strips any <think> block from the output.
VISION_MODEL = "qwen/qwen3.6-27b"
VISION_MAX_TOKENS = 500
VISION_MAX_DIM = 768   # px; images are downscaled to this before upload. Lower =
# fewer tokens per image (cost scales ~quadratically), so more images fit under
# the per-day/per-minute limits. 768 is plenty for photos; OCR is unaffected
# (it reads from a separate 2x render, not this). Raise toward 1280 if fine
# detail in dense diagrams matters.
# Rate-limit handling. The vision model has both per-minute (TPM) and per-day
# (TPD) token limits. A multi-image file fires vision calls in a burst that can
# trip the per-minute limit, so we throttle calls and retry transient 429s.
VISION_MIN_INTERVAL = 2.0    # min seconds between vision calls (avoids TPM bursts)
VISION_MAX_RETRIES = 4       # retries on a transient (per-minute) rate limit
VISION_MAX_RETRY_WAIT = 25.0  # s; if the API says wait longer (per-day limit), give up instead

# --- RAGAS-style answer evaluation (reference-free, computed live per answer) ---
# After each grounded answer we score it on four RAGAS metrics — WITHOUT any
# ground-truth labels — using a small, fast judge model:
#   * faithfulness      — is every claim in the answer grounded in the context?
#   * answer_relevancy  — does the answer actually address the question?
#   * context_precision — are the *useful* retrieved chunks ranked near the top?
#   * context_relevancy — of all retrieved text, what fraction is actually relevant
#                        (signal-to-noise), regardless of chunk ranking?
# Each metric is an extra Groq call (faithfulness is two), so this adds latency.
# Set RAGAS_ENABLED = False to turn evaluation off entirely (no extra API calls).
RAGAS_ENABLED = True
# llama-3.1-8b-instant: separate TPD bucket from DEFAULT_MODEL (llama-3.3-70b-versatile),
# so judge calls no longer compete with generation for the same daily token quota.
# It used to be unreliable at producing valid JSON under load (frequent
# json_validate_failed / truncated output on faithfulness and context_precision,
# which send the largest prompts), but _judge_verdicts()'s retry + escalation to
# RAGAS_STRICT_FALLBACK_MODEL, plus the token-budget-doubling retry in
# _judge_json(), now absorb that instability.
RAGAS_EVAL_MODEL = "llama-3.1-8b-instant"
RAGAS_RELEVANCY_N = 3   # how many questions to generate from the answer for relevancy
RAGAS_TIMEOUT_S = 15.0  # per-call timeout on the judge model; a hang must not stall a request
# Rate-limit handling: judge calls run several-at-once (see evaluate()) and each
# carries the full retrieved context, so a burst can trip the judge model's TPM
# limit. Retry transient 429s using the API's own suggested wait; give up if it
# asks for longer than this (a real per-day quota exhaustion, not a blip).
RAGAS_MAX_RETRIES = 4
RAGAS_MAX_RETRY_WAIT = 65.0  # s; covers a full per-minute (TPM) reset window
# Ceiling for the token-budget-doubling retry (see _judge_json): a long answer
# can need more than the default max_tokens to finish its JSON reply without
# getting cut off mid-object.
RAGAS_MAX_TOKENS_CAP = 2000
# Verdict-list metrics (faithfulness's claim verification, context_precision)
# ask the judge for exactly one 0/1 verdict per claim/chunk; occasionally it
# miscounts (e.g. 20 verdicts for 19 claims), which _judge_verdicts() retries
# once, then escalates to this model as a last resort — it's forced to comply
# via Groq's strict JSON-schema structured output (minItems == maxItems ==
# the required count), which RAGAS_EVAL_MODEL does NOT support (see
# scripts/run_langsmith_eval.py's OPENEVALS_MODEL note on the same models).
RAGAS_STRICT_FALLBACK_MODEL = "openai/gpt-oss-120b"

# --- Golden-set lookup (live chat) ---
# context_recall/answer_correctness need a ground-truth answer, which a real
# user's question never has. If a live question closely matches one of the
# curated questions in GOLDEN_SET_PATH (cosine similarity via the same local
# embedding model), RagService.evaluate_answer() reuses that row's
# ground_truth to score those two metrics too. Anything below the threshold
# still gets only the four reference-free metrics.
GOLDEN_SET_PATH = os.path.join(BASE_DIR, "backend", "scripts", "eval_dataset_osw.json")
GOLDEN_SET_MATCH_THRESHOLD = 0.92

# --- Postgres (Q&A + RAGAS metrics logging) ---
# Every chat turn's question/answer/sources + the 6 RAGAS-style metrics
# (faithfulness, answer_relevancy, context_precision, context_relevancy,
# context_recall, answer_correctness) are logged here — see db.py. This runs
# ALONGSIDE the existing JSON chat_store.py, not instead of it: chat_store.py
# stays the source of truth for chat/message state; Postgres is purely an
# additive log for history/analytics. If DATABASE_URL is unset or Postgres is
# unreachable, logging is skipped (with a warning) — it never blocks a chat answer.
DATABASE_URL = os.getenv("DATABASE_URL")

# --- OCR ---
OCR_LANGUAGES = ["en"]   # add e.g. "fr", "de" — see EasyOCR supported languages

# --- PDF handling ---
# Pages with fewer than this many extracted characters are treated as scanned
# and OCR'd from a rendered page image.
PDF_OCR_MIN_CHARS = 20
# Embedded images pulled out of PDFs are saved here (for future vision-model
# use). Ignore tiny images below this pixel area — usually logos/icons/noise.
EXTRACTED_IMAGES_DIR = os.path.join(DATA_DIR, "extracted_images")
MIN_EMBEDDED_IMAGE_AREA = 100 * 100
