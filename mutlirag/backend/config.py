"""Central configuration for the Multi-RAG app.

Everything tweakable lives here so you can tune the pipeline without hunting
through the code.
"""

import os

# Opt out of DeepEval's anonymous usage telemetry — set before anything else
# imports deepeval, since chat content flows through its metrics (rag/evaluation.py)
# and this app has no business phoning that home. config.py is imported first by
# every module that needs it, so this always wins the race against deepeval's own
# import-time telemetry setup.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

# Force offline mode for HuggingFace's libraries — rag/embeddings.py and
# rag/reranker.py's models (EMBEDDING_MODEL/RERANK_MODEL below) are static and
# already downloaded/cached locally; without this, sentence-transformers still
# does an online HEAD check against huggingface.co on every single load (e.g.
# for an optional adapter_config.json) even when the model is fully cached —
# so one transient DNS/network hiccup during startup ([Errno 11001]
# getaddrinfo failed, seen live) took down the ENTIRE app, since RagService()
# loads synchronously in FastAPI's lifespan startup. These models never need
# a live network check to run correctly, so skip it entirely.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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

# --- Chunking: parent-child ---
# Two tiers instead of one. CHILD chunks are small and are what actually gets
# embedded/BM25-indexed/reranked — retrieval needs a tight, focused match
# between query and text (smaller chunks raise relevant-sentence density,
# which is what DeepEval's context_relevancy metric rewards). Each child is
# tagged with the full text of its containing PARENT chunk; generation and
# DeepEval evaluation use the parent text instead of the child snippet, so a
# precise-but-narrow retrieval match doesn't starve the LLM of surrounding
# context. Overlap keeps sentences that straddle a boundary retrievable from
# both sides. Re-ingest existing documents for a change here to take effect —
# already-indexed chunks predate this and have no parent_text, so they fall
# back to using their own (single-tier) text as their own parent.
PARENT_CHUNK_SIZE = 1600
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 220
CHILD_CHUNK_OVERLAP = 40

# --- Semantic boundary chunking (parent tier only) ---
# A large section with no sub-heading (e.g. a long headingless .txt, or a PDF
# page whose font never jumps) currently splits purely on the PARENT_CHUNK_SIZE
# character budget, which can blend two unrelated ideas into one parent chunk
# just because they happened to fit. When enabled, rag.chunking embeds each
# sentence in that section and cuts a new chunk wherever the topic actually
# shifts — an unusually large similarity drop relative to the REST OF THAT
# SAME SECTION's own sentence-to-sentence drops (see
# SEMANTIC_BREAKPOINT_PERCENTILE) — in addition to, never instead of, the
# PARENT_CHUNK_SIZE hard cap. CHILD chunking is untouched: children are
# already small/precise retrieval units, not meant to represent a whole topic,
# so the extra embedding calls wouldn't pay for themselves there.
# Set False to fall back to the original pure character-budget splitting (same
# output as before this feature existed) for direct comparison. Re-ingest
# existing documents for a change here to take effect — it only affects how
# NEW ingestion splits text, not chunks already in the index.
SEMANTIC_CHUNKING_ENABLED = True
# A FIXED cosine threshold doesn't generalize — what counts as "a big jump"
# differs by document and by embedding model. Instead, for each section,
# every consecutive-sentence distance is compared only to that section's OWN
# distribution: a jump at or above this percentile of the section's own
# jumps is treated as a real topic boundary. 95 = only the most pronounced
# ~5% of jumps in a section trigger an early cut.
SEMANTIC_BREAKPOINT_PERCENTILE = 95
# A semantic cut is never taken before the current chunk-so-far reaches this
# many characters, so a handful of short, genuinely unrelated sentences don't
# fragment into one-sentence chunks — an early narrow miss on a text's own
# noisy percentile.
SEMANTIC_MIN_CHUNK_CHARS = 300
# Below this many sentence units, a percentile over the drops isn't a
# meaningful signal (too few samples) — falls back to plain splitting. Above
# it, embedding every sentence individually stops paying for itself against a
# document with literally no heading structure at all (e.g. one huge
# headingless .txt) — also falls back, so one pathological document can't
# make ingestion of a single page/section disproportionately slow.
SEMANTIC_CHUNKING_MIN_UNITS = 4
SEMANTIC_CHUNKING_MAX_UNITS = 500

# --- Retrieval ---
# Table-of-contents / section-index pages (see ingestion._looks_like_toc) are
# tagged is_index=True at ingestion but still indexed, not dropped, so the tag
# is inspectable/reversible if the heuristic ever misfires. This is what
# actually keeps them out of results: excluded at search time, in the same
# boolean mask vectorstore.py already uses for chat_id isolation. They're pure
# noise for retrieval — dense with the same keywords/phrases as real content
# elsewhere in the document, so they can out-rank real content on both BM25
# and even the cross-encoder reranker — but never worth answering from.
#
# This only governs NORMAL_QUERY (see rag.query_intent). An explicit TOC
# request ("give me the table of contents") or page request ("what's on page
# 247") bypasses this exclusion entirely and retrieves those pages directly
# from metadata — see RagService._retrieve_toc / _retrieve_pages and
# vectorstore.get_toc_chunks / get_by_pages. `is_index` is named for the
# heuristic that sets it (an "index"/TOC page), not a general "exclude this"
# flag — vectorstore._is_toc() also accepts a future `is_toc` key so a rename
# would be additive, never a breaking change to already-indexed data.
EXCLUDE_INDEX_PAGES = True

# How many chunks to pull back and stuff into the prompt as context.
# Was 10 — dropped to 5 because a wide net with no score cutoff let weak
# chunks ride along just to fill the quota, diluting context_relevancy and
# answer_relevancy. Pairs with RERANK_MIN_SCORE below.
TOP_K = 5  # base chunks per query; auto-scales up with more indexed documents
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
# BAAI/bge-reranker-base: 278M params, pairs naturally with the bge embedding
# model above. Higher quality than the previous cross-encoder/ms-marco-MiniLM-L-6-v2
# (22M params) at more latency — worth it since TOP_K dropped to 5, so there's
# less to score.
RERANK_MODEL = "BAAI/bge-reranker-base"
# Hits scoring below this (post-sigmoid, 0..1) are dropped instead of always
# filling TOP_K's quota with weak matches. At least one hit is always kept
# (the best available) so a genuinely relevant-but-lower-scoring pool doesn't
# collapse to an "I don't know" answer.
RERANK_MIN_SCORE = 0.15

# --- Retrieval: candidate pool vs. final context (kept distinct on purpose) ---
# Before this, RagService.retrieve() used ONE number (effective_top_k, scaled
# from TOP_K) for BOTH the search candidate pool (x3) AND the final chunk
# count — i.e. a ~15-chunk candidate pool feeding a reranker that then got
# truncated straight to 5. Confirmed live (see the "three key guidelines"
# investigation) that this is too narrow: a chunk that's genuinely necessary
# evidence can rank far outside even a generously widened candidate pool on
# its own (its own wording alone doesn't score well — see
# NEIGHBOR_EXPANSION_WINDOW below for how that specific case is actually
# fixed) — but plenty of ordinary cases DO just need a wider net for the
# reranker to have real signal to work with instead of settling for
# whatever the top ~15 happened to be.
# RETRIEVAL_CANDIDATE_K: size of the pool handed to the reranker.
RETRIEVAL_CANDIDATE_K = 40
# FINAL_CONTEXT_K: size of what actually reaches the LLM, after reranking +
# neighbor expansion + diversity selection. Kept small and clean on purpose —
# widening the candidate pool is not license to also widen what the LLM sees.
FINAL_CONTEXT_K = 10

# --- Neighbor/parent expansion (structural completeness) ---
# A chunk can be essential evidence for a multi-part question ("what are the
# three guidelines") yet score far outside ANY reasonable candidate pool on
# its own — e.g. it starts mid-sentence, so it shares few keywords/little
# standalone meaning with the query even though the FULL answer needs it.
# Confirmed live: a genuine "guideline #3" chunk ranked ~140th out of ~500
# candidates for its own query, entirely below RETRIEVAL_CANDIDATE_K=40 —
# widening the pool alone cannot fix this class of miss.
# Fix: for each candidate that DID make the pool, also pull the immediately
# adjacent chunk(s) from the SAME page (same document/chat, same parent-chunk
# reading order — see vectorstore.get_neighbors) and let the reranker score
# them normally, but exempt them from RERANK_MIN_SCORE / diversity trimming
# (see reranker.rerank's `protect` param and reranker.diversity_select) — they
# were added for structural completeness, not because they scored well on
# their own, so the normal score-based filters must not remove them.
# 0 disables expansion entirely (exact pre-existing behavior).
NEIGHBOR_EXPANSION_WINDOW = 1  # chunks before/after, within the same page
# Only probe neighbors for the top-scoring candidates, not the whole pool —
# a low-ranked candidate's neighbors are rarely worth the extra lookups, and
# this bounds the cost added to every query.
NEIGHBOR_EXPANSION_MAX_ANCHORS = 8
# Hard cap on TOTAL chunks added via expansion across ALL anchors, not just
# per-anchor. Without this, several different anchors each contributing 1-2
# neighbors can add up to more "protected" (always-kept) chunks than
# FINAL_CONTEXT_K has room for — since protected chunks are reserved space
# in diversity_select BEFORE regular candidates, an unbounded total would
# let expansion crowd out the genuinely best-scoring regular candidates
# instead of just supplementing them. Confirmed capable of happening even
# with well-scattered candidate pages (each candidate's positional
# neighbor in reading order isn't necessarily numerically nearby).
NEIGHBOR_EXPANSION_MAX_ADDED = 4

# --- Diversity-aware final selection (MMR) ---
# After reranking, several of the top-scoring chunks can be near-duplicates
# of each other (same idea restated) while genuinely different evidence sits
# just below them — MMR trades a little top-1 relevance for coverage.
# 1.0 = pure relevance (no diversity effect); lower values favor covering
# more distinct ideas within FINAL_CONTEXT_K. Never applied to neighbor-
# expanded chunks (see above) — those are protected, not competing for a
# diversity slot, since being similar to their anchor is expected and fine.
MMR_LAMBDA = 0.7

# --- Vector DB backend ---
# "weaviate" -> self-hosted Weaviate (see docker-compose.yml); "chroma" -> the
# original embedded ChromaDB. If Weaviate is selected but unreachable at
# startup, the app automatically falls back to Chroma so nothing breaks.
VECTOR_BACKEND = "chroma"

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

# --- Query understanding / rewrite validation ---
# Confirmed live: a fully standalone question ("What is the difference between
# analytical thinking and critical thinking?") got rewritten by the model into
# an unrelated EARLIER topic from the chat history — a prompt instruction
# ("NO TOPIC POISONING") alone wasn't enough to prevent it. Instead of trying
# to enumerate every way a question can be "already standalone" (a keyword/
# regex list never fully covers real phrasing), the classification of
# standalone vs. contextual is itself an LLM judgment (see
# generator._query_understanding), and its output gets an independent,
# non-LLM safety net before being trusted: embedding similarity between the
# original question and the rewritten query. A genuine follow-up rewrite
# stays close in meaning to the original ("and when is it due?" ~ "when is
# the electricity bill due" — shared subject); a topic-poisoned one doesn't.
# Below this cosine similarity, the rewrite is treated as suspicious and
# discarded in favor of the original question.
REWRITE_MIN_SIMILARITY = 0.30

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

# --- DeepEval answer evaluation (reference-free, computed live per answer) ---
# After each grounded answer we score it on three DeepEval metrics — WITHOUT
# any ground-truth labels — using a small, fast judge model over Groq (see
# rag/deepeval_judge.py; DeepEval has no native Groq provider):
#   * faithfulness         — is every claim in the answer grounded in the context?
#   * answer_relevancy     — does the answer actually address the question?
#   * context_relevancy    — of all retrieved text, what fraction is actually relevant
#                           (signal-to-noise), regardless of chunk ranking?
# Two more (context_precision, context_recall — both need a ground-truth answer,
# per DeepEval's own required_params) plus answer_correctness (via GEval, since
# DeepEval ships no built-in correctness metric) only run in
# evaluate_with_ground_truth(), gated on a golden-set match — see below.
# Each metric is an extra Groq call, so this adds latency. Set
# DEEPEVAL_ENABLED = False to turn evaluation off entirely (no extra API calls).
DEEPEVAL_ENABLED = True
# Separate TPD bucket from DEFAULT_MODEL (llama-3.3-70b-versatile), so judge
# calls no longer compete with generation for the same daily token quota.
DEEPEVAL_JUDGE_MODEL = "llama-3.1-8b-instant"
DEEPEVAL_TIMEOUT_S = 15.0  # per-call timeout on the judge model; a hang must not stall a request
DEEPEVAL_METRIC_THRESHOLD = 0.7  # pass/fail cutoff DeepEval uses for metric.success
# Rate-limit handling: metrics run several-at-once (see evaluate()) and each
# carries the full retrieved context, so a burst can trip the judge model's TPM
# limit. Retry transient 429s using the API's own suggested wait; give up if it
# asks for longer than this (a real per-day quota exhaustion, not a blip).
DEEPEVAL_MAX_RETRIES = 4
DEEPEVAL_MAX_RETRY_WAIT = 65.0  # s; covers a full per-minute (TPM) reset window
# Ceiling for the token-budget-doubling retry (see rag/deepeval_judge.py): a
# long answer can need more than the default max_tokens to finish its JSON
# reply without getting cut off mid-object.
# Was 2000, then 4000 — each still too low, same failure recurring one step
# further out each time: 1500 -> 3000 -> 4000(capped) gives only two real
# escalation steps before the retry loop gives up (`current_max_tokens < CAP`
# goes false at the cap) even with retries left. Observed hit again on
# AnswerRelevancyMetric's statement-extraction step, which — like
# FaithfulnessMetric's truth-extraction — can genuinely need more tokens than
# a small, verbose judge model (llama-3.1-8b-instant) budgets by default when
# breaking a longer answer into many granular JSON list items. Raised so the
# doubling sequence (1500 -> 3000 -> 6000) gets three real escalation steps.
DEEPEVAL_MAX_TOKENS_CAP = 6000
# Contextual metrics (faithfulness, context_relevancy, context_precision,
# context_recall) feed the judge every retrieved chunk at once. TOP_K=10 full
# chunks from a large document can push a single judge call's prompt past
# DEEPEVAL_JUDGE_MODEL's per-minute token ceiling, or make the small model
# lose track of the required JSON shape entirely. Cap what the judge (not the
# generator) sees, independent of TOP_K.
DEEPEVAL_MAX_CONTEXT_CHUNKS = 5   # at most this many retrieved chunks per judge call
DEEPEVAL_MAX_CHUNK_CHARS = 500    # each chunk truncated to this many chars first

# --- Golden-set lookup (live chat) ---
# context_precision/context_recall/answer_correctness need a ground-truth
# answer, which a real user's question never has. If a live question closely
# matches one of the curated questions in GOLDEN_SET_PATH (cosine similarity
# via the same local embedding model), RagService.evaluate_answer() reuses
# that row's ground_truth to score those metrics too. Anything below the
# threshold still gets only the three reference-free metrics.
GOLDEN_SET_PATH = os.path.join(BASE_DIR, "backend", "scripts", "eval_dataset_osw.json")
# Was 0.92 (near-paraphrase only) — loosened so more live traffic actually
# lands a match, since context_precision/context_recall/answer_correctness
# are otherwise averaged over a tiny, coincidental sliver of questions.
GOLDEN_SET_MATCH_THRESHOLD = 0.85

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
# A page with more embedded images than this is almost always one complex
# diagram sliced into many raster tiles by whatever exported the PDF, not N
# independently meaningful photos — describing each fragment individually is
# both slow (every image is its own throttled vision call, VISION_MIN_INTERVAL
# apart) and produces noisy, disjointed descriptions. Confirmed on a real
# 516-page doc with 1,370 embedded images: several pages had 150+ fragments
# each (one diagram apiece); this cap collapsed total vision calls 1370 -> 491
# by rendering those 16 pages whole instead, cutting the vision-throttling
# floor from ~46 minutes to ~16 minutes for that document.
MAX_EMBEDDED_IMAGES_PER_PAGE = 20

# --- Repeated header/footer suppression (opt-in) ---
# A line (title banner, copyright notice, running header) that appears
# byte-for-byte identical on many pages carries no page-specific meaning —
# it's boilerplate, not content, and diluting real chunks with it doesn't
# help retrieval. Confirmed live on a real document: a divider banner
# ("The 5 Essential Smart Skills for Success...") recurred identically on
# 3 separate pages; retrieval-time exact dedup (vectorstore's dedupe_mask)
# already prevents it from filling more than one result slot, so this is a
# secondary, ingestion-time improvement — stripping it before it's ever
# chunked/embedded at all, saving that storage/embedding cost and freeing
# the candidate-pool/index slot it would otherwise occupy on every page.
# Off by default: the right frequency threshold is document-dependent, and
# too aggressive a setting on a document with a genuinely reused
# instructional template (the same worksheet text across several exercises)
# could strip real content instead of boilerplate. Turn on and tune
# HEADER_FOOTER_MIN_PAGES for your own documents before relying on it.
STRIP_REPEATED_HEADERS_FOOTERS = False
# A line must appear identically (whitespace/case-normalized) on at least
# this many DISTINCT pages before it's treated as boilerplate. Deliberately
# an ABSOLUTE count, not a percentage of total pages — a banner recurring
# once per chapter (e.g. every ~100th page in a 500-page document) is just
# as much boilerplate as one on every single page, but a percentage
# threshold tuned for "most pages" would miss that pattern entirely (it's
# only ~1% of pages, confirmed on the real 3-of-~500 case above).
HEADER_FOOTER_MIN_PAGES = 5
# Ignore short lines even if they repeat often (a bare page number, a
# single word) — too generic to safely treat as boilerplate on their own;
# real boilerplate (a title banner, a copyright line) is usually a full
# phrase.
HEADER_FOOTER_MIN_LINE_CHARS = 12
