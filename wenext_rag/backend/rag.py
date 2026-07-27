import json
import os
import glob
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
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment or .env file.")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")
INGEST_MANIFEST_PATH = os.path.join(DB_PATH, "ingest_manifest.json")
COLLECTION_NAME = "rag_documents"
PIPELINE_VERSION = "langchain-v1"

RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.85"))
TOP_K = int(os.getenv("RAG_TOP_K", "8"))
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-2-preview")
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "gemini-2.5-flash")

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=api_key,
)

llm = ChatGoogleGenerativeAI(
    model=CHAT_MODEL,
    temperature=0.1,
    google_api_key=api_key,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ---------------------------------------------------------------------------
# Answer-generation chain (LCEL): prompt | llm | output parser
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
# Query-reformulation chain (LCEL): prompt | llm | output parser
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


def get_chunk_count() -> int:
    """Number of chunks currently stored, via LangChain's public Chroma.get() API."""
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
    """Delete the existing Chroma collection using LangChain's own API
    (no raw chromadb client needed)."""
    vs = _get_vectorstore()
    try:
        vs.delete_collection()
    except Exception:
        pass
    _reset_vectorstore()


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
    return {
        "chunk_count": get_chunk_count(),
        "indexed_sources": get_ingested_sources(),
        "pdf_files_on_disk": sorted(pdf_inventory.keys()),
        "needs_reingestion": _needs_reingestion(),
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "pipeline": PIPELINE_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "chat_model": CHAT_MODEL,
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

    print(f"Total chunks: {len(splits)}. Embedding and storing in Chroma...")
    vectorstore = _get_vectorstore()

    batch_size = 15
    total_batches = (len(splits) + batch_size - 1) // batch_size
    for batch_index in range(total_batches):
        batch = splits[batch_index * batch_size:(batch_index + 1) * batch_size]
        for attempt in range(5):
            try:
                vectorstore.add_documents(batch)
                print(f"Ingested batch {batch_index + 1}/{total_batches}")
                time.sleep(2)
                break
            except Exception as exc:
                message = str(exc)
                if ("429" in message or "RESOURCE_EXHAUSTED" in message) and attempt < 4:
                    wait_seconds = 50 * (attempt + 1)
                    print(f"Embedding rate limit hit. Waiting {wait_seconds}s before retry...")
                    time.sleep(wait_seconds)
                    continue
                raise

    _save_ingest_manifest(_pdf_inventory(), len(splits))
    print(f"Successfully ingested {len(splits)} chunks into Chroma DB.")


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


def get_relevant_context(query: str, top_k: int = TOP_K) -> tuple[list[dict], float]:
    vs = _get_vectorstore()
    try:
        results = vs.similarity_search_with_score(query, k=top_k)
    except Exception as exc:
        print(f"Retrieval error: {exc}")
        return [], 1.0

    if not results:
        return [], 1.0

    retrieved_chunks = []
    min_distance = 1.0

    for doc, distance in results:
        min_distance = min(min_distance, distance)
        retrieved_chunks.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", doc.metadata.get("page_number", "?")),
            "distance": distance,
        })

    print(
        f"Retrieval for query={query!r}: min_distance={min_distance:.4f}, "
        f"threshold={RELEVANCE_THRESHOLD}, top_source={retrieved_chunks[0]['source'] if retrieved_chunks else 'none'}"
    )
    return retrieved_chunks, min_distance


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for idx, chunk in enumerate(chunks):
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
            if ("503" in message or "UNAVAILABLE" in message or "429" in message) and attempt < 3:
                wait_seconds = 8 * (attempt + 1)
                print(f"Gemini generation unavailable. Retrying in {wait_seconds}s...")
                time.sleep(wait_seconds)
                continue
            print(f"Error in Gemini text generation: {exc}")
            return "I am sorry, but I encountered an error processing your request. Please try again."
    return "I am sorry, but I encountered an error processing your request. Please try again."


def generate_rag_response(query: str, history: list[dict]) -> str:
    if is_greeting(query):
        return "Hello! I am your WeNext Document Assistant. How can I help you today?"

    search_query = _reformulate_query(query, history)
    context_chunks, min_distance = get_relevant_context(search_query)

    if not context_chunks or min_distance > RELEVANCE_THRESHOLD:
        print(f"Query rejected by relevance gate: min_distance={min_distance:.4f}")
        return "I am sorry, but I can only answer questions related to the provided documents."

    context_str = _format_context(context_chunks)

    return _invoke_chain_with_retries(
        answer_chain,
        {
            "context": context_str,
            "chat_history": _to_langchain_messages(history),
            "input": query,
        },
    )