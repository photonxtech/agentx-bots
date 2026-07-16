"""Central configuration for the Multi-RAG app.

Everything tweakable lives here so you can tune the pipeline without hunting
through the code.
"""

# --- Embedding model (runs locally via sentence-transformers) ---
# all-MiniLM-L6-v2: 384-dim, ~80MB, fast, strong quality/speed trade-off.
# Swap for "BAAI/bge-small-en-v1.5" for a small quality bump at similar cost.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

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
INDEX_DIR = "index_store"
# Per-file ingestion cache, keyed by content hash: re-uploading the same file
# skips OCR + vision entirely.
CACHE_DIR = "index_store/cache"

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
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
VISION_MAX_TOKENS = 500
VISION_MAX_DIM = 1280   # px; images are downscaled to this before upload

# --- OCR ---
OCR_LANGUAGES = ["en"]   # add e.g. "fr", "de" — see EasyOCR supported languages

# --- PDF handling ---
# Pages with fewer than this many extracted characters are treated as scanned
# and OCR'd from a rendered page image.
PDF_OCR_MIN_CHARS = 20
# Embedded images pulled out of PDFs are saved here (for future vision-model
# use). Ignore tiny images below this pixel area — usually logos/icons/noise.
EXTRACTED_IMAGES_DIR = "extracted_images"
MIN_EMBEDDED_IMAGE_AREA = 100 * 100
