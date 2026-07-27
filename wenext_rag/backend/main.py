import os
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import RAG functions
from backend.rag import ingest_documents, generate_rag_response, get_rag_status

# In-memory storage for session histories
# Structure: {session_id: [{"role": "user"/"assistant", "content": str}, ...]}
sessions_db = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Triggers document ingestion on application startup."""
    print("Initializing RAG system...")
    try:
        ingest_documents()
    except Exception as e:
        print(f"CRITICAL: Failed to ingest documents on startup: {e}")
    yield
    print("Shutting down RAG system...")

app = FastAPI(
    title="WeNext RAG Chatbot API",
    description="A FastAPI backend with Chroma DB and Gemini for conversational document QA.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    reply: str
    session_id: str

@app.get("/api/status")
async def rag_status():
    """Return RAG index status and ingested document sources."""
    return get_rag_status()

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint to receive messages and send responses.
    Generates a new session_id if none is provided.
    """
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
        
    session_id = request.session_id
    if not session_id or session_id.strip() == "":
        session_id = str(uuid.uuid4())
        
    # Get or initialize history for session
    if session_id not in sessions_db:
        sessions_db[session_id] = []
        
    history = sessions_db[session_id]
    
    # Generate response
    reply = generate_rag_response(user_message, history)
    
    # Update conversation history in memory
    # We append both user message and system reply to maintain context
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})
    
    # Keep history bounded to avoid prompt bloat (e.g., last 10 turns)
    if len(history) > 20:
        sessions_db[session_id] = history[-20:]
        
    return ChatResponse(reply=reply, session_id=session_id)

@app.get("/api/sessions/{session_id}")
async def get_session_history(session_id: str):
    """Retrieve history for a specific session."""
    if session_id not in sessions_db:
        return {"history": []}
    return {"history": sessions_db[session_id]}

@app.delete("/api/sessions/{session_id}")
async def clear_session_history(session_id: str):
    """Clear history for a specific session."""
    if session_id in sessions_db:
        del sessions_db[session_id]
        return {"status": "success", "message": "Session history cleared."}
    return {"status": "error", "message": "Session not found."}

# Mount static frontend files
# Note: Define all API routes BEFORE mounting static files at root
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if not os.path.exists(frontend_dir):
    os.makedirs(frontend_dir)

app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
