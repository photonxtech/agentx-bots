import json
import pickle #coverts python objinto 0,1s to store
import uuid #unique session ids
import shutil #recursively delets session folders
from pathlib import Path #setspath
from datetime import datetime #timestamps creation

from langchain_community.document_loaders import PyPDFLoader #load PDF -> Document objects
from langchain_text_splitters import RecursiveCharacterTextSplitter # split pages into chunks
from langchain_huggingface import HuggingFaceEmbeddings # embedding model (text -> vectors)
from langchain_community.vectorstores import Chroma  # vector DB for semantic search
from langchain_community.retrievers import BM25Retriever # keyword/lexical retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever # combine BM25+vector, then rerank
from langchain_community.document_compressors import FlashrankRerank # cross-encoder reranker
from langchain_groq import ChatGroq # Groq-hosted LLM client
from langchain_core.prompts import ChatPromptTemplate # prompt templating
from langchain_core.output_parsers import StrOutputParser # extract plain text from LLM response

# --- Paths: everything scoped PER SESSION (per chat / per uploaded PDF) ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_ROOT = BASE_DIR / "chroma_db"
CHUNKS_ROOT = BASE_DIR / "chunks"
SESSIONS_FILE = BASE_DIR / "sessions.json"

# Ensure all root folders exist on startup
for d in (DATA_DIR, CHROMA_ROOT, CHUNKS_ROOT):
    d.mkdir(parents=True, exist_ok=True)

# Prompt template — includes conversation history for multi-turn context
PROMPT_TEMPLATE = """You are a helpful assistant answering questions about the uploaded document.
Use ONLY the context below and the recent conversation history to answer.
If the answer isn't in the context, say you don't know.

Conversation history:
{history}

Context:
{context}

Question: {question}

Answer clearly and concisely:"""


# =================== SESSION PERSISTENCE (sessions.json) ===================

# Read all sessions from disk (returns {} if file doesn't exist yet)
def load_sessions():
    if SESSIONS_FILE.exists():
        with open(SESSIONS_FILE, "r") as f:
            return json.load(f)
    return {}


# Write the full sessions dict back to disk
def save_sessions(sessions):
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2)


# Create a new empty session with a unique UUID and return its id
def create_session():
    session_id = str(uuid.uuid4())
    sessions = load_sessions()
    sessions[session_id] = {
        "pdf_name": None,
        "created_at": datetime.now().isoformat(),
        "chat_history": []
    }
    save_sessions(sessions)
    return session_id


# Update arbitrary fields (e.g. pdf_name) on an existing session
def update_session(session_id, **kwargs):
    sessions = load_sessions()
    sessions.setdefault(session_id, {
        "pdf_name": None,
        "created_at": datetime.now().isoformat(),
        "chat_history": []
    })
    sessions[session_id].update(kwargs)
    save_sessions(sessions)


# Append one Q&A turn (with sources) to a session's chat history
def append_chat_turn(session_id, question, answer, sources):
    sessions = load_sessions()
    if session_id not in sessions:
        return
    sessions[session_id]["chat_history"].append({
        "question": question,
        "answer": answer,
        "sources": sources
    })
    save_sessions(sessions)


# Delete a session's JSON entry AND its on-disk Chroma DB, chunks pickle, and uploaded PDF
def delete_session(session_id):
    sessions = load_sessions()
    sessions.pop(session_id, None)
    save_sessions(sessions)

    chroma_dir = CHROMA_ROOT / session_id
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)

    chunks_path = CHUNKS_ROOT / f"{session_id}.pkl"
    if chunks_path.exists():
        chunks_path.unlink()

    data_dir = DATA_DIR / session_id
    if data_dir.exists():
        shutil.rmtree(data_dir, ignore_errors=True)


# =================== PDF PROCESSING / RETRIEVAL (per session_id) ===================

def process_pdf(pdf_path, session_id):
    """Loads PDF, splits into chunks, builds & saves this session's vector DB + chunks file"""
    # Load PDF into one Document per page
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split pages into overlapping chunks for better retrieval granularity
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # Save chunks to this session's pickle file (reused later by BM25, avoids re-parsing PDF)
    chunks_path = CHUNKS_ROOT / f"{session_id}.pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    # Embed chunks and persist them into this session's own Chroma vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_dir = CHROMA_ROOT / session_id
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir)
    )

    return len(documents), len(chunks)


def build_hybrid_retriever(session_id):
    """Loads this session's saved chunks + vector DB, builds hybrid BM25+vector retriever with reranking"""
    # Load pre-chunked documents for this session
    chunks_path = CHUNKS_ROOT / f"{session_id}.pkl"
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    # Keyword-based retriever (sparse/lexical matching)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    # Semantic retriever backed by this session's persisted Chroma vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_dir = CHROMA_ROOT / session_id
    vectordb = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 10})

    # Merge BM25 + vector results, weighted 40%/60%
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )

    # Rerank merged candidates and keep only the top 4 most relevant
    compressor = FlashrankRerank(top_n=4)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )


# Join retrieved doc chunks into one text block for the prompt's {context}
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_history(chat_history, max_turns=3):
    """Turns last few Q&A pairs into a text block for context"""
    # No prior turns -> placeholder text instead of empty string
    if not chat_history:
        return "No previous conversation."
    # Only keep the most recent N turns to bound prompt size
    recent = chat_history[-max_turns:]
    lines = []
    for turn in recent:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)


def get_answer(retriever, question, chat_history):
    """Runs retrieval + LLM call manually (not using LCEL chain, so we can inject history easily)"""
    # Retrieve relevant chunks for the question
    docs = retriever.invoke(question)
    context = format_docs(docs)
    history_text = format_history(chat_history)

    # LLM + prompt + parser set up per call (temperature=0 for deterministic, factual answers)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    parser = StrOutputParser()

    # Fill the prompt template with context, question, and history
    filled_prompt = prompt.invoke({
        "context": context,
        "question": question,
        "history": history_text
    })

    # Call the LLM directly (not piped through LCEL) so we can access response_metadata
    response = llm.invoke(filled_prompt)
    answer = parser.invoke(response)

    # Extract token usage from the raw LLM response, before it was parsed to a plain string
    token_usage = response.response_metadata.get("token_usage", {})

    # Build citation list: page number + chunk text for each retrieved doc
    sources = []
    for doc in docs:
        page = doc.metadata.get("page", "unknown")
        sources.append({
            "page": page,
            "text": doc.page_content
        })

    return answer, sources, token_usage