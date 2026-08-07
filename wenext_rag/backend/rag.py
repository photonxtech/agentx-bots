import json
import os
import glob
import math
import re
import time
from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from rank_bm25 import BM25Okapi

# --- SWAPPED: Groq & HuggingFace Imports ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- 1. API KEY UPDATE ---
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is not set in the environment or .env file.")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")
INGEST_MANIFEST_PATH = os.path.join(DB_PATH, "ingest_manifest.json")
EVAL_DATASET_PATH = os.path.join(DATA_DIR, "eval_dataset.json")
COLLECTION_NAME = "rag_documents"
PIPELINE_VERSION = "langchain-groq-v1"

# Distance metrics differ by embedding model. L2 distance threshold for BGE is usually around 0.8 - 1.2
RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "1.0"))
TOP_K = int(os.getenv("RAG_TOP_K", "8"))

# --- Hybrid search (dense + BM25) config ---
# Each retriever pulls a wider candidate pool than TOP_K; results are fused
# with Reciprocal Rank Fusion (RRF) and truncated to TOP_K afterwards.
DENSE_CANDIDATE_K = int(os.getenv("RAG_DENSE_CANDIDATE_K", "25"))
BM25_CANDIDATE_K = int(os.getenv("RAG_BM25_CANDIDATE_K", "25"))
RRF_K = int(os.getenv("RAG_RRF_K", "60"))
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Model configuration
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "llama-3.3-70b-versatile") # Popular fast Groq model

# --- 2. EMBEDDINGS UPDATE ---
# Runs locally on CPU/GPU — eliminates rate limits and API costs for ingestion
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# --- 3. LLM UPDATE ---
llm = ChatGroq(
    model=CHAT_MODEL,
    temperature=0.1,
    groq_api_key=api_key,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ---------------------------------------------------------------------------
# Answer-generation chain (LCEL)
# ---------------------------------------------------------------------------
ANSWER_SYSTEM_PROMPT = """You are an assistant that answers questions using the provided Document Context.

DOCUMENT CONTEXT:
{context}

CRITICAL RULES:
1. Answer the User's latest question using ONLY information from the DOCUMENT CONTEXT.
2. Treat related wording as equivalent (e.g. "WhatsApp automation", "WhatsApp setup", "WhatsApp Business API onboarding" may refer to the same steps in the docs).
3. Refer to the conversation history to resolve context references (like "what is the first one" or "explain that more").
4. Do NOT use outside knowledge beyond what appears in the DOCUMENT CONTEXT.
5. If the DOCUMENT CONTEXT contains relevant steps or guidance for the question, summarize them clearly even if the exact phrase from the question is not present verbatim.
6. Only if the DOCUMENT CONTEXT has no relevant information at all, reply EXACTLY with:
"I am sorry, but I can only answer questions related to the provided documents."
Do not add anything else to the refusal.
7. You MUST cite the source of your facts by appending the citation index (e.g., [1] or [2]) to the sentences. At the very end of your response, list the source file name and page number corresponding to the citations you used (e.g., "[1] WeNext_Meta_WhatsApp_FAQ.pdf, Page 2").
"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ANSWER_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

answer_chain = ANSWER_PROMPT | llm | StrOutputParser()

# ---------------------------------------------------------------------------
# Query-reformulation chain (LCEL)
# ---------------------------------------------------------------------------
REFORMULATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the chat history and latest user question, rewrite it as a standalone "
        "search query for a document knowledge base. Return only the rewritten question.",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

reformulation_chain = REFORMULATION_PROMPT | llm | StrOutputParser()

_vectorstore: Optional[Chroma] = None

# --- BM25 (sparse/keyword) index state ---
# Chroma has no native full-text index, so the sparse side of hybrid search
# is built and held in memory, kept in sync with the Chroma collection.
_bm25_index: Optional[BM25Okapi] = None
_bm25_corpus: list[dict] = []
_bm25_built: bool = False

def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=DB_PATH,
        )
    return _vectorstore

def _reset_vectorstore() -> None:
    global _vectorstore
    _vectorstore = None

def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer for BM25 (no stopword removal - BM25's IDF
    term already down-weights very common words)."""
    return _TOKEN_RE.findall(text.lower())

def _build_bm25_index() -> None:
    """(Re)build the in-memory BM25 index from whatever is currently in Chroma."""
    global _bm25_index, _bm25_corpus, _bm25_built
    _bm25_built = True

    if get_chunk_count() == 0:
        _bm25_index = None
        _bm25_corpus = []
        return

    vs = _get_vectorstore()
    result = vs.get(include=["documents", "metadatas"])
    texts = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    _bm25_corpus = [
        {
            "text": text,
            "source": (meta or {}).get("source", "unknown"),
            "page": (meta or {}).get("page", (meta or {}).get("page_number", "?")),
        }
        for text, meta in zip(texts, metadatas)
    ]

    tokenized_corpus = [_tokenize(chunk["text"]) for chunk in _bm25_corpus]
    _bm25_index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    print(f"BM25 index built over {len(_bm25_corpus)} chunks.")

def _get_bm25_index() -> Optional[BM25Okapi]:
    """Lazily build the BM25 index on first use (e.g. startup skipped ingestion
    because Chroma already had chunks from a previous run)."""
    if not _bm25_built:
        _build_bm25_index()
    return _bm25_index

def _reset_bm25_index() -> None:
    global _bm25_index, _bm25_corpus, _bm25_built
    _bm25_index = None
    _bm25_corpus = []
    _bm25_built = False

def get_chunk_count() -> int:
    vs = _get_vectorstore()
    return len(vs.get()["ids"])

def _pdf_inventory() -> dict[str, float]:
    inventory: dict[str, float] = {}
    for pdf_path in sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf"))):
        inventory[os.path.basename(pdf_path)] = os.path.getmtime(pdf_path)
    return inventory

def _load_ingest_manifest() -> dict:
    if not os.path.exists(INGEST_MANIFEST_PATH):
        return {}
    try:
        with open(INGEST_MANIFEST_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}

def _save_ingest_manifest(pdf_inventory: dict[str, float], chunk_count: int) -> None:
    os.makedirs(os.path.dirname(INGEST_MANIFEST_PATH), exist_ok=True)
    with open(INGEST_MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "pdfs": pdf_inventory,
                "chunk_count": chunk_count,
                "pipeline": PIPELINE_VERSION,
                "embedding_model": EMBEDDING_MODEL,
            },
            handle,
            indent=2,
        )

# --- Ground-truth lookup (data/eval_dataset.json) ---
# Feeds the metrics that need a reference answer (context_recall,
# context_entity_recall, context_precision, answer_correctness) so they
# compare against a real answer instead of falling back to the question text.
# Tries an exact (normalized) question match first; if that misses, falls
# back to embedding cosine similarity against the eval questions, so
# paraphrased user questions can still pick up a known ground truth.
_eval_entries: list[dict] = []  # [{"normalized": str, "ground_truth": str, "embedding": list[float] | None}, ...]
_eval_ground_truths: dict[str, str] = {}
_eval_dataset_mtime: Optional[float] = None
EVAL_SIMILARITY_THRESHOLD = float(os.getenv("RAG_EVAL_SIMILARITY_THRESHOLD", "0.92"))

def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def _load_eval_dataset() -> None:
    """Load/refresh eval_dataset.json into _eval_entries + _eval_ground_truths.
    Re-reads (and re-embeds) only if the file's mtime changed, so this is
    cheap to call on every request."""
    global _eval_entries, _eval_ground_truths, _eval_dataset_mtime

    if not os.path.exists(EVAL_DATASET_PATH):
        if _eval_entries:
            print("eval_dataset.json no longer found; clearing ground-truth lookup.")
        _eval_entries = []
        _eval_ground_truths = {}
        _eval_dataset_mtime = None
        return

    mtime = os.path.getmtime(EVAL_DATASET_PATH)
    if mtime == _eval_dataset_mtime:
        return

    try:
        with open(EVAL_DATASET_PATH, encoding="utf-8") as handle:
            raw_entries = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Failed to load eval_dataset.json: {exc}")
        return

    questions, normalized_list, ground_truths_list = [], [], []
    for entry in raw_entries:
        question = (entry or {}).get("question")
        ground_truth = (entry or {}).get("ground_truth")
        if question and ground_truth:
            questions.append(question)
            normalized_list.append(_normalize_question(question))
            ground_truths_list.append(ground_truth)

    embeddings_list: list[Optional[list[float]]] = [None] * len(questions)
    if questions:
        try:
            embeddings_list = embeddings.embed_documents(questions)
        except Exception as exc:
            print(f"Failed to embed eval_dataset.json questions, fuzzy matching disabled: {exc}")

    _eval_entries = [
        {"normalized": norm, "ground_truth": gt, "embedding": emb}
        for norm, gt, emb in zip(normalized_list, ground_truths_list, embeddings_list)
    ]
    _eval_ground_truths = {entry["normalized"]: entry["ground_truth"] for entry in _eval_entries}
    _eval_dataset_mtime = mtime
    print(f"Loaded {len(_eval_entries)} ground truths from eval_dataset.json.")

def _find_similar_ground_truth(question: str) -> Optional[str]:
    """Embedding-similarity fallback for paraphrased questions that don't
    exact-match an eval_dataset.json entry."""
    candidates = [e for e in _eval_entries if e.get("embedding") is not None]
    if not candidates:
        return None

    query_vec = embeddings.embed_query(question.strip())
    best_entry = max(candidates, key=lambda e: _cosine(query_vec, e["embedding"]))
    best_score = _cosine(query_vec, best_entry["embedding"])

    if best_score >= EVAL_SIMILARITY_THRESHOLD:
        print(f"Fuzzy-matched eval ground truth (similarity={best_score:.3f}) for: {question!r}")
        return best_entry["ground_truth"]
    return None

def get_ground_truth(question: str) -> Optional[str]:
    """Return a known ground truth for this question from data/eval_dataset.json:
    exact (normalized) match first, then embedding-similarity fallback.
    Returns None if neither finds a confident match."""
    _load_eval_dataset()

    exact = _eval_ground_truths.get(_normalize_question(question))
    if exact:
        return exact

    return _find_similar_ground_truth(question)

def _needs_reingestion() -> bool:
    current = _pdf_inventory()
    if not current:
        return False

    if get_chunk_count() == 0:
        return True

    manifest = _load_ingest_manifest()
    if manifest.get("pipeline") != PIPELINE_VERSION:
        return True
    if manifest.get("embedding_model") != EMBEDDING_MODEL:
        return True
    if not manifest.get("pdfs"):
        _save_ingest_manifest(current, get_chunk_count())
        return False
    return manifest.get("pdfs") != current

def _clear_collection() -> None:
    vs = _get_vectorstore()
    try:
        vs.delete_collection()
    except Exception:
        pass
    _reset_vectorstore()
    _reset_bm25_index()

def _load_pdf_documents() -> list[Document]:
    documents: list[Document] = []
    for pdf_path in sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf"))):
        print(f"Parsing PDF: {pdf_path}")
        try:
            pages = PyPDFLoader(pdf_path).load()
            source_name = os.path.basename(pdf_path)
            for page in pages:
                page.metadata["source"] = source_name
                if "page" in page.metadata:
                    page.metadata["page"] = int(page.metadata["page"]) + 1
            documents.extend(pages)
        except Exception as exc:
            print(f"Error reading {pdf_path}: {exc}")
    return documents

def get_ingested_sources() -> list[str]:
    vs = _get_vectorstore()
    if get_chunk_count() == 0:
        return []
    result = vs.get(include=["metadatas"])
    sources = {meta.get("source") for meta in result["metadatas"] if meta.get("source")}
    return sorted(sources)

def get_rag_status() -> dict:
    pdf_inventory = _pdf_inventory()
    _load_eval_dataset()
    return {
        "chunk_count": get_chunk_count(),
        "indexed_sources": get_ingested_sources(),
        "pdf_files_on_disk": sorted(pdf_inventory.keys()),
        "needs_reingestion": _needs_reingestion(),
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "pipeline": PIPELINE_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "chat_model": CHAT_MODEL,
        "eval_ground_truths_loaded": len(_eval_ground_truths),
    }

def ingest_documents(force: bool = False) -> None:
    pdf_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}. Please place files there.")
        return

    if not force and not _needs_reingestion():
        print(f"Chroma DB already contains {get_chunk_count()} chunks. Skipping ingestion.")
        return

    print("Building LangChain vector index...")
    _clear_collection()

    raw_documents = _load_pdf_documents()
    if not raw_documents:
        print("No readable text found in the PDFs.")
        return

    splits = text_splitter.split_documents(raw_documents)
    if not splits:
        print("No chunks produced after splitting.")
        return

    print(f"Total chunks: {len(splits)}. Embedding locally and storing in Chroma...")
    vectorstore = _get_vectorstore()

    # Local embedding doesn't hit external API rate limits, so batching can be larger and faster
    batch_size = 100
    total_batches = (len(splits) + batch_size - 1) // batch_size
    for batch_index in range(total_batches):
        batch = splits[batch_index * batch_size:(batch_index + 1) * batch_size]
        vectorstore.add_documents(batch)
        print(f"Ingested batch {batch_index + 1}/{total_batches}")

    _save_ingest_manifest(_pdf_inventory(), len(splits))
    print(f"Successfully ingested {len(splits)} chunks into Chroma DB.")

    _build_bm25_index()

def is_greeting(query: str) -> bool:
    greetings = [
        "hi", "hello", "hey", "hola", "greetings",
        "good morning", "good afternoon", "who are you", "what is your name",
    ]
    query_clean = query.lower().strip().replace("?", "").replace("!", "")
    return any(query_clean == greeting or query_clean.startswith(greeting + " ") for greeting in greetings)

def _to_langchain_messages(history: list[dict]) -> list:
    messages = []
    for msg in history[-10:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages

def _reformulate_query(query: str, history: list[dict]) -> str:
    if not history:
        return query

    try:
        rewritten = reformulation_chain.invoke({
            "input": query,
            "chat_history": _to_langchain_messages(history),
        })
        print(f"Query reformulation: {query!r} -> {rewritten!r}")
        return rewritten.strip() or query
    except Exception as exc:
        print(f"Query reformulation failed, using original query: {exc}")
        return query

def _dense_search(query: str, top_k: int) -> list[dict]:
    """Dense (embedding) side of hybrid retrieval - unchanged Chroma similarity search,
    just pulling a wider candidate pool than TOP_K for fusion."""
    vs = _get_vectorstore()
    try:
        results = vs.similarity_search_with_score(query, k=top_k)
    except Exception as exc:
        print(f"Dense retrieval error: {exc}")
        return []

    return [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", doc.metadata.get("page_number", "?")),
            "distance": distance,
        }
        for doc, distance in results
    ]

def _bm25_search(query: str, top_k: int) -> list[dict]:
    """Sparse (keyword) side of hybrid retrieval - exact term matches (acronyms,
    IDs, numbers) that embeddings can miss."""
    index = _get_bm25_index()
    if index is None or not _bm25_corpus:
        return []

    scores = index.get_scores(_tokenize(query))
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [_bm25_corpus[i] for i in ranked_idx[:top_k] if scores[i] > 0]

def _chunk_key(chunk: dict) -> tuple:
    """Identity for merging a chunk seen by both retrievers. Chroma doesn't
    expose ids on similarity_search results, so (source, page, text) stands
    in as a stable key - chunks are split deterministically, so duplicates
    are effectively identical content anyway."""
    return (chunk["source"], chunk["page"], chunk["text"])

def _reciprocal_rank_fusion(
    dense_chunks: list[dict],
    bm25_chunks: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """Merge two ranked lists using RRF: score(doc) = sum(1 / (k + rank)).
    Rank-based (not raw-score-based) fusion sidesteps the fact that Chroma's
    L2 distance and BM25's score are on incomparable scales."""
    scores: dict[tuple, float] = {}
    chunk_lookup: dict[tuple, dict] = {}

    for rank, chunk in enumerate(dense_chunks):
        key = _chunk_key(chunk)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        chunk_lookup.setdefault(key, chunk)

    for rank, chunk in enumerate(bm25_chunks):
        key = _chunk_key(chunk)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        chunk_lookup.setdefault(key, chunk)

    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [chunk_lookup[key] for key in ranked_keys]

def get_relevant_context(query: str, top_k: int = TOP_K) -> tuple[list[dict], float]:
    """Hybrid retrieval: dense (Chroma/BGE) + sparse (BM25) candidates fused
    with RRF, truncated to top_k. Returns the fused chunks plus the dense
    side's min distance, which still drives the RELEVANCE_THRESHOLD gate."""
    dense_chunks = _dense_search(query, DENSE_CANDIDATE_K)
    bm25_chunks = _bm25_search(query, BM25_CANDIDATE_K)

    if not dense_chunks and not bm25_chunks:
        return [], 1.0

    # min_distance still comes from the dense side only, since BM25 scores
    # aren't on the same scale as RELEVANCE_THRESHOLD was tuned against.
    min_distance = min((c["distance"] for c in dense_chunks), default=1.0)

    fused_chunks = _reciprocal_rank_fusion(dense_chunks, bm25_chunks)
    retrieved_chunks = fused_chunks[:top_k]

    print(
        f"Hybrid retrieval for query={query!r}: dense_candidates={len(dense_chunks)}, "
        f"bm25_candidates={len(bm25_chunks)}, fused_top_k={len(retrieved_chunks)}, "
        f"min_dense_distance={min_distance:.4f}, threshold={RELEVANCE_THRESHOLD}, "
        f"top_source={retrieved_chunks[0]['source'] if retrieved_chunks else 'none'}"
    )
    return retrieved_chunks, min_distance

def _format_context(chunks: list[dict]) -> str:
    """Group chunks by (source, page) before numbering, so two chunks pulled
    from the same page (e.g. one via dense, one via BM25) collapse into a
    single citation index instead of being cited twice."""
    grouped: list[dict] = []
    seen: dict[tuple, int] = {}

    for chunk in chunks:
        key = (chunk["source"], chunk["page"])
        if key in seen:
            grouped[seen[key]]["text"] += "\n" + chunk["text"]
        else:
            seen[key] = len(grouped)
            grouped.append({"source": chunk["source"], "page": chunk["page"], "text": chunk["text"]})

    parts = []
    for idx, chunk in enumerate(grouped):
        parts.append(
            f"[{idx + 1}] Source: {chunk['source']} (Page {chunk['page']})\n"
            f"Content: {chunk['text']}\n"
        )
    return "\n".join(parts)

def _invoke_chain_with_retries(chain: Runnable, inputs: dict) -> str:
    """Invoke an LCEL chain with retry/backoff for transient upstream errors."""
    for attempt in range(4):
        try:
            return chain.invoke(inputs).strip()
        except Exception as exc:
            message = str(exc)
            if ("429" in message or "RATE_LIMIT" in message) and attempt < 3:
                wait_seconds = 4 * (attempt + 1)
                print(f"Groq API rate limit hit. Retrying in {wait_seconds}s...")
                time.sleep(wait_seconds)
                continue
            print(f"Error in Groq text generation: {exc}")
            return "I am sorry, but I encountered an error processing your request. Please try again."
    return "I am sorry, but I encountered an error processing your request. Please try again."

def generate_rag_response(query: str, history: list[dict]) -> dict:
    """Run RAG and return reply plus retrieved context texts for metrics."""
    if is_greeting(query):
        return {
            "reply": "Hello! I am your WeNext Document Assistant. How can I help you today?",
            "contexts": [],
        }

    search_query = _reformulate_query(query, history)
    context_chunks, min_distance = get_relevant_context(search_query)

    if not context_chunks or min_distance > RELEVANCE_THRESHOLD:
        print(f"Query rejected by relevance gate: min_distance={min_distance:.4f}")
        return {
            "reply": "I am sorry, but I can only answer questions related to the provided documents.",
            "contexts": [],
        }

    context_str = _format_context(context_chunks)
    contexts = [chunk["text"] for chunk in context_chunks]

    reply = _invoke_chain_with_retries(
        answer_chain,
        {
            "context": context_str,
            "chat_history": _to_langchain_messages(history),
            "input": query,
        },
    )
    return {"reply": reply, "contexts": contexts}