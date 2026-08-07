import os
import re
import tempfile
import json
import psycopg2
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_engine import DocumentParser, StructureChunker, GroqClient, VectorStore, LocalEmbeddingClient
import evaluation

# Load environment variables (supports running from root or backend directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(BASE_DIR, ".env")
if not os.path.exists(dotenv_path):
    dotenv_path = os.path.join(os.path.dirname(BASE_DIR), ".env")
load_dotenv(dotenv_path=dotenv_path)

# Define Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(lifespan=lifespan)

# Mount static directory
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Initialize DB
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
VECTOR_STORE = VectorStore(persist_directory=DB_DIR)
COLLECTION_NAME = "document_rag_collection"

groq_api_key = os.getenv("GROQ_API_KEY", "")
evaluation_api_key = os.getenv("EVALUATION_API_KEY", groq_api_key)

RAG_MODEL = os.getenv("GROQ_RAG_MODEL", "llama-3.3-70b-versatile")
EVALUATION_MODEL = os.getenv("GROQ_EVALUATION_MODEL", "llama-3.1-8b-instant")

# RAG enhancements: cache and conversation memory
SEMANTIC_CACHE = []
CONVERSATION_HISTORY = []

# Session token usage counter (resets on server restart)
SESSION_TOKEN_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "requests": 0,
    "provider": "groq",
    "model": "llama-3.3-70b-versatile"
}

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            database=os.getenv("DB_NAME", "rag_logs"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres")
        )
        return conn
    except Exception as e:
        print(f"[DATABASE ERROR] Failed to connect to PostgreSQL: {e}")
        return None

def init_db():
    db_name = os.getenv("DB_NAME", "rag_logs")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "postgres")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")

    # 1. Connect to default 'postgres' database to check/create the target database
    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database="postgres",
            user=db_user,
            password=db_password
        )
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,))
        exists = cur.fetchone()
        if not exists:
            print(f"[DB INIT] Creating database '{db_name}'...")
            cur.execute(f'CREATE DATABASE "{db_name}"')
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[DB INIT WARNING] Could not verify/create database: {e}")

    # 2. Connect to the target database and create the table
    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )
        conn.autocommit = True
        cur = conn.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS chat_logs (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(100),
            message_index INT,
            file_name VARCHAR(255),
            question TEXT,
            answer TEXT,
            faithfulness DOUBLE PRECISION,
            relevance DOUBLE PRECISION,
            context_precision DOUBLE PRECISION,
            context_relevance DOUBLE PRECISION,
            context_recall DOUBLE PRECISION,
            answer_correctness DOUBLE PRECISION,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(create_table_query)

        # Database Schema Migrations: ADD columns if missing
        # 1. Check and add answer_correctness column if missing
        cur.execute("""
            SELECT 1 FROM information_schema.columns 
            WHERE table_name='chat_logs' AND column_name='answer_correctness';
        """)
        if not cur.fetchone():
            print("[DB MIGRATE] Adding column 'answer_correctness' to 'chat_logs' table...")
            cur.execute("ALTER TABLE chat_logs ADD COLUMN answer_correctness DOUBLE PRECISION;")

        # 2. Check and add context_relevance column if missing
        cur.execute("""
            SELECT 1 FROM information_schema.columns 
            WHERE table_name='chat_logs' AND column_name='context_relevance';
        """)
        if not cur.fetchone():
            print("[DB MIGRATE] Adding column 'context_relevance' to 'chat_logs' table...")
            cur.execute("ALTER TABLE chat_logs ADD COLUMN context_relevance DOUBLE PRECISION;")

        # Database Schema Migrations: DROP columns if they exist
        for col in ("context_entities_recall", "noise_sensitivity", "explanation", "inference", "inference_time_ms"):
            cur.execute(f"""
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='chat_logs' AND column_name='{col}';
            """)
            if cur.fetchone():
                print(f"[DB MIGRATE] Dropping column '{col}' from 'chat_logs' table...")
                cur.execute(f"ALTER TABLE chat_logs DROP COLUMN IF EXISTS {col};")

        cur.close()
        conn.close()
        print(f"[DB INIT] Database '{db_name}' table 'chat_logs' is ready.")
    except Exception as e:
        print(f"[DB INIT ERROR] Failed to initialize table: {e}")

def log_qa_to_db(session_id: str | None, message_index: int | None, file_name: str | None, question: str, answer: str):
    if not session_id or message_index is None:
        return
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            insert_query = """
            INSERT INTO chat_logs (session_id, message_index, file_name, question, answer)
            VALUES (%s, %s, %s, %s, %s);
            """
            cur.execute(insert_query, (session_id, message_index, file_name, question, answer))
            conn.commit()
            cur.close()
            conn.close()
            print(f"[DATABASE] Logged question and answer for session={session_id} index={message_index}")
        except Exception as db_err:
            print(f"[DATABASE ERROR] Failed to insert log record: {db_err}")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

# Get count of document chunks in Database
@app.get("/db_count")
def db_count():
    try:
        count = VECTOR_STORE.get_collection(COLLECTION_NAME).count()
        source = None
        if count > 0:
            sample = VECTOR_STORE.get_collection(COLLECTION_NAME).get(limit=1)
            if sample and "metadatas" in sample and sample["metadatas"]:
                source = sample["metadatas"][0].get("source", None)
        return {"count": count, "source": source}
    except Exception as e:
        return {"count": 0, "source": None, "error": str(e)}

class SessionsBody(BaseModel):
    sessions: dict

SESSIONS_FILE = os.path.join(BASE_DIR, "sessions.json")

@app.get("/sessions")
def get_sessions():
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

@app.post("/sessions")
def save_sessions(body: SessionsBody):
    try:
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(body.sessions, f, indent=4, ensure_ascii=False)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Upload and index PDF/Word/Text/Image files
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed_extensions = {".pdf", ".docx", ".doc", ".txt", ".png", ".jpg", ".jpeg", ".webp"}
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(allowed_extensions))}"
        )
        
    if ext in {".png", ".jpg", ".jpeg", ".webp"} and not groq_api_key:
        raise HTTPException(
            status_code=400,
            detail="Groq API key is not configured in .env file, which is required for processing images."
        )
    
    try:
        # Read file contents and save to temporary file
        file_bytes = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name
            
        # Parse document pages
        pages = DocumentParser.parse(tmp_file_path, ext, groq_api_key=groq_api_key)
        
        # Chunk text based on document structure
        splitter = StructureChunker()
        chunks = splitter.split_documents(pages, file.filename)
        
        # Clear previous document cache so only one file is indexed at a time
        try:
            VECTOR_STORE.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        # Embed
        client = LocalEmbeddingClient()
        embeddings = client.get_embeddings_batch([c["text"] for c in chunks])
        
        # Insert into Chroma
        VECTOR_STORE.add_documents(COLLECTION_NAME, chunks, embeddings)
        
        # Cleanup temp file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            
        return {"success": True, "chunks": len(chunks)}
    except Exception as e:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

class QueryBody(BaseModel):
    query: str
    session_id: str | None = None
    message_index: int | None = None

# Execute search query and generate Groq answer
@app.post("/query")
def query_rag(body: QueryBody):
    global SEMANTIC_CACHE, CONVERSATION_HISTORY
    query_str = body.query
    
    if not groq_api_key:
        return {"answer": "⚠️ Groq API key is not configured in .env file. Please check your setup.", "sources": [], "eval_contexts": []}

    # Retrieve currently active file name
    active_file = None
    try:
        count = VECTOR_STORE.get_collection(COLLECTION_NAME).count()
        if count > 0:
            sample = VECTOR_STORE.get_collection(COLLECTION_NAME).get(limit=1)
            if sample and "metadatas" in sample and sample["metadatas"]:
                active_file = sample["metadatas"][0].get("source", None)
    except Exception:
        pass

    # Greeting bypass check
    greetings = ["hi", "hello", "hey", "hai", "howdy", "greetings", "good morning", "good afternoon", "good evening", "how are you", "who are you"]
    clean_query = query_str.strip().lower().rstrip("?").rstrip("!").strip()
    if clean_query in greetings:
        try:
            prompt = f"The user says: '{query_str}'. Respond with a friendly, professional greeting as the Q&A RAG chatbot. Be brief."
            llm = GroqClient(api_key=groq_api_key)
            answer = llm.generate_answer(prompt, model=RAG_MODEL)
            log_qa_to_db(body.session_id, body.message_index, active_file, query_str, answer)
            return {"answer": answer, "sources": [], "eval_contexts": []}
        except Exception as e:
            fallback_answer = "Hello! How can I help you today?"
            log_qa_to_db(body.session_id, body.message_index, active_file, query_str, fallback_answer)
            return {"answer": fallback_answer, "sources": [], "eval_contexts": []}

    # 1. Semantic Cache check
    for cached in SEMANTIC_CACHE:
        q1 = set(re.findall(r"\w+", query_str.lower()))
        q2 = set(re.findall(r"\w+", cached["query"].lower()))
        intersection = q1.intersection(q2)
        union = q1.union(q2)
        jaccard = len(intersection) / len(union) if union else 0.0
        if jaccard >= 0.85:
            # Cache hit
            # Append conversation history for memory persistence
            CONVERSATION_HISTORY.append({"role": "user", "content": query_str})
            CONVERSATION_HISTORY.append({"role": "assistant", "content": cached["answer"]})
            if len(CONVERSATION_HISTORY) > 10:
                CONVERSATION_HISTORY.pop(0)
                CONVERSATION_HISTORY.pop(0)
            
            cache_answer = cached["answer"] + "\n\n*(Served from local semantic cache)*"
            log_qa_to_db(body.session_id, body.message_index, active_file, query_str, cache_answer)
            return {
                "answer": cache_answer,
                "sources": cached["sources"],
                "eval_contexts": cached.get("eval_contexts") or [],
                "observability": {
                    "cached": True
                }
            }

    try:
        embed_client = LocalEmbeddingClient()
        # 2. Embed query
        query_vector = embed_client.get_embedding(query_str)
        
        # 3. Hybrid search (Dense + Sparse + Reranking)
        retrieved_docs = VECTOR_STORE.query(COLLECTION_NAME, query_str, query_vector, top_k=10)
        filtered_docs = [doc for doc in retrieved_docs if doc["similarity"] >= 0.50]
        
        if not filtered_docs:
            no_match_answer = "No matching context was found in the database. Please make sure you have uploaded and indexed your document."
            log_qa_to_db(body.session_id, body.message_index, active_file, query_str, no_match_answer)
            return {
                "answer": no_match_answer,
                "sources": [],
                "eval_contexts": [],
                "observability": {
                    "cached": False
                }
            }
            
        # 4. Inject Conversational History (Memory)
        history_context = ""
        if CONVERSATION_HISTORY:
            history_context = "Recent conversation context:\n"
            for msg in CONVERSATION_HISTORY[-4:]:  # last 4 turns
                role_label = "User" if msg["role"] == "user" else "Assistant"
                history_context += f"{role_label}: {msg['content']}\n"
            history_context += "\n"

        # 5. Build prompt
        context_str = ""
        for i, doc in enumerate(filtered_docs):
            context_str += f"[{i+1}] (Source: {doc['metadata'].get('source', 'Unknown')}, Page: {doc['metadata'].get('page_number', 'N/A')}):\n{doc['text']}\n\n"
            
        prompt = f"""Use the following retrieved context chunks and conversation history to answer the user query.
{history_context}
Retrieved Contexts:
{context_str}

USER_QUERY:
{query_str}

AI Answer:
"""
        system_instruction = (
            "You are a professional assistant that answers user queries strictly based on provided text chunks. "
            "You MUST cite the relevant source number inside brackets (e.g. [1], [2]) at the end of claims. "
            "To avoid cluttering, do NOT repeat the same citation consecutively on every sentence or bullet point. If multiple consecutive sentences or bullet points in a list refer to the exact same source, write the citation once at the end of the paragraph or the end of the list group rather than after every single line. "
            "You format key terms in bold. "
            "For any lists, rules, or bullet points in the context, you MUST preserve each item as its own separate, individual bullet point in your response. Do not merge separate bullet points together. "
            "Do NOT include any meta-commentary, apologies, or explanations about what is missing, what section names do not match, or where information is located. "
            "If the information to answer the user query is present anywhere in the contexts, answer the query directly and cleanly using that information. "
            "Only refuse (starting your response with 'REFUSAL:') if the information is completely absent from all contexts."
        )
        
        # 6. LLM Generation
        llm = GroqClient(api_key=groq_api_key)
        answer = llm.generate_answer(prompt, model=RAG_MODEL, system_instruction=system_instruction)
        
        # Accumulate token usage
        if hasattr(llm, "last_usage") and llm.last_usage:
            SESSION_TOKEN_USAGE["prompt_tokens"] += llm.last_usage.get("prompt_tokens", 0)
            SESSION_TOKEN_USAGE["completion_tokens"] += llm.last_usage.get("completion_tokens", 0)
            SESSION_TOKEN_USAGE["total_tokens"] += llm.last_usage.get("total_tokens", 0)
            SESSION_TOKEN_USAGE["requests"] += 1
            SESSION_TOKEN_USAGE["model"] = llm.last_usage.get("model", SESSION_TOKEN_USAGE["model"])
        
        # 7. Handle out-of-context refusal
        if answer.strip().startswith("REFUSAL:"):
            clean_answer = answer.replace("REFUSAL:", "", 1).strip()
            log_qa_to_db(body.session_id, body.message_index, active_file, query_str, clean_answer)
            return {
                "answer": clean_answer,
                "sources": [],
                "eval_contexts": [],
                "observability": {
                    "cached": False
                }
            }
            
        # 8. Record in conversation memory
        CONVERSATION_HISTORY.append({"role": "user", "content": query_str})
        CONVERSATION_HISTORY.append({"role": "assistant", "content": answer})
        if len(CONVERSATION_HISTORY) > 10:
            CONVERSATION_HISTORY.pop(0)
            CONVERSATION_HISTORY.pop(0)

        # 9. Format sources for citation details UI
        sources = []
        for i, doc in enumerate(filtered_docs):
            source_name = doc['metadata'].get('source', 'Unknown')
            page_num = doc['metadata'].get('page_number', 'N/A')
            confidence = doc.get("confidence", "MEDIUM")
            similarity_pct = round(doc['similarity'] * 100, 1)
            
            sources.append({
                "index": i + 1,
                "source_name": source_name,
                "citation": f"[Source: {source_name}, page {page_num} — {similarity_pct}% Match ({confidence} CONFIDENCE)]",
                "text": doc['text'][:300],
                "page": page_num,
                "similarity": doc['similarity'],
                "confidence": confidence
            })
            
        # 10. Record in Semantic Cache
        eval_ctxs = [doc['text'] for doc in filtered_docs]
        SEMANTIC_CACHE.append({
            "query": query_str,
            "answer": answer,
            "sources": sources,
            "eval_contexts": eval_ctxs
        })
        if len(SEMANTIC_CACHE) > 20:
            SEMANTIC_CACHE.pop(0)
            
        # Log response in PG database
        log_qa_to_db(body.session_id, body.message_index, active_file, query_str, answer)

        return {
            "answer": answer,
            "sources": sources,
            "eval_contexts": eval_ctxs,  # FULL, untruncated contexts for evaluator
            "observability": {
                "cached": False
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Clear Chroma database collection, semantic cache, and memory
@app.post("/clear")
def clear_db():
    global SEMANTIC_CACHE, CONVERSATION_HISTORY
    try:
        VECTOR_STORE.delete_collection(COLLECTION_NAME)
        SEMANTIC_CACHE.clear()
        CONVERSATION_HISTORY.clear()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Token usage stats endpoint
@app.get("/token_usage")
def token_usage():
    daily_limit = 500_000  # Groq free tier limit
    used = SESSION_TOKEN_USAGE["total_tokens"]
    remaining = max(0, daily_limit - used)
    pct_used = round((used / daily_limit) * 100, 1) if daily_limit > 0 else 0
    return {
        "provider": "groq",
        "model": SESSION_TOKEN_USAGE["model"],
        "prompt_tokens": SESSION_TOKEN_USAGE["prompt_tokens"],
        "completion_tokens": SESSION_TOKEN_USAGE["completion_tokens"],
        "total_used": used,
        "daily_limit": daily_limit,
        "remaining": remaining,
        "pct_used": pct_used,
        "requests": SESSION_TOKEN_USAGE["requests"]
    }

class EvaluationRequest(BaseModel):
    query: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None  # optional — only needed for reference-based metrics
    session_id: str | None = None
    message_index: int | None = None


@app.post("/evaluate")
def evaluate_response(body: EvaluationRequest):
    # Handle case with empty contexts explicitly
    if not body.contexts:
        return {
            "faithfulness": None,
            "relevance": None,
            "context_precision": None,
            "context_relevance": None,
            "context_recall": None,
            "answer_correctness": None,
            "notes": [],
            "evaluation_model": EVALUATION_MODEL
        }

    has_gt = bool(body.ground_truth and body.ground_truth.strip())
    matched_from_file = False
    gt_file = None
    
    # Try looking up in ground_truth.json or ground_truths.json if not provided
    if not has_gt:
        for filename in ("ground_truth.json", "ground_truths.json"):
            path_in_backend = os.path.join(BASE_DIR, filename)
            path_in_root = os.path.join(os.path.dirname(BASE_DIR), filename)
            if os.path.exists(path_in_backend):
                gt_file = path_in_backend
                break
            elif os.path.exists(path_in_root):
                gt_file = path_in_root
                break
                
        if gt_file:
            try:
                with open(gt_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                query_key = body.query.strip().lower()
                matched_gt = None
                
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get("query", "").strip().lower() == query_key:
                            matched_gt = item.get("ground_truth") or item.get("answer")
                            break
                elif isinstance(data, dict):
                    matched_gt = data.get(query_key)
                    
                if matched_gt:
                    body.ground_truth = matched_gt
                    has_gt = True
                    matched_from_file = True
            except Exception as e:
                print(f"[EVALUATOR] Failed to read ground truth file: {e}")

    try:
        # Score query using custom multi-step evaluation script
        # Capping the contexts to top 3 elements reduces payload size and avoids TPM rate-limiting sleeps
        eval_contexts = body.contexts[:3]
        if has_gt:
            res = evaluation.evaluate_with_ground_truth(
                question=body.query,
                answer=body.answer,
                ground_truth=body.ground_truth,
                contexts=eval_contexts
            )
        else:
            res = evaluation.evaluate(
                question=body.query,
                answer=body.answer,
                contexts=eval_contexts
            )

        # Map returned score keys to matching database/UI properties
        scores = {
            "faithfulness": res.get("faithfulness"),
            "relevance": res.get("answer_relevancy"),
            "context_precision": res.get("context_precision"),
            "context_relevance": res.get("context_relevancy"),
            "context_recall": res.get("context_recall") if has_gt else None,
            "answer_correctness": res.get("answer_correctness") if has_gt else None,
        }

        notes = []
        if matched_from_file:
            # Safely truncate note display
            gt_preview = body.ground_truth[:40] + "..." if len(body.ground_truth) > 40 else body.ground_truth
            notes.append(f'Using Ground Truth from {gt_file}: "{gt_preview}"')

        # Update evaluation metrics in PostgreSQL database
        if body.session_id and body.message_index is not None:
            conn = get_db_connection()
            if conn:
                try:
                    cur = conn.cursor()
                    update_query = """
                    UPDATE chat_logs
                    SET faithfulness = %s,
                        relevance = %s,
                        context_precision = %s,
                        context_relevance = %s,
                        context_recall = %s,
                        answer_correctness = %s
                    WHERE session_id = %s AND message_index = %s;
                    """
                    cur.execute(update_query, (
                        scores["faithfulness"],
                        scores["relevance"],
                        scores["context_precision"],
                        scores["context_relevance"],
                        scores["context_recall"],
                        scores["answer_correctness"],
                        body.session_id,
                        body.message_index
                    ))
                    conn.commit()
                    cur.close()
                    conn.close()
                    print(f"[DATABASE] Updated evaluation metrics for session={body.session_id} index={body.message_index}")
                except Exception as db_err:
                    print(f"[DATABASE ERROR] Failed to update evaluation metrics: {db_err}")

        return {
            **scores,
            "notes": notes,
            "evaluation_model": EVALUATION_MODEL
        }

    except Exception as e:
        err_msg = str(e)
        # Gracefully handle API rate limits and return HTTP 429
        if "rate_limit" in err_msg.lower() or "429" in err_msg or "too many requests" in err_msg.lower():
            raise HTTPException(status_code=429, detail="Groq Rate Limit Exceeded. Please wait a few seconds before trying again.")
        raise HTTPException(status_code=500, detail=f"LLM evaluator execution failed: {err_msg}")


if __name__ == "__main__":
    import uvicorn
    # Clean shutdown & binding port recovery
    uvicorn.run(app, host="0.0.0.0", port=8502)