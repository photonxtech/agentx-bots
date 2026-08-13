# QA Automation & DeepEval Studio

## Project Overview

This project is a **FastAPI-based QA Test Case Generator** with AI-powered evaluation capabilities. It automates the creation and validation of QA test cases from uploaded documents, featuring:

- **Document Ingestion**: Support for PDF, TXT, and Markdown files
- **Vision Analysis**: Optional vision-based document analysis for PDF pages
- **LLM Generation**: Groq-powered LLM for intelligent QA dataset generation
- **Quality Evaluation**: DeepEval-based metrics (Faithfulness, Relevancy, Precision, Recall)
- **RAG Pipeline**: Retrieval-Augmented Generation for context-aware Q&A
- **Chat Interface**: Interactive chat with generated Q&A datasets
- **Web UI**: Simple, responsive web interface
- **Persistent Storage**: SQLite database for session management

Generated sessions and QA test cases are stored locally in `qa_sessions.db`.

---

## Setup Instructions

### 1. Prerequisites

- **Python 3.10+** (3.11+ recommended)
- **pip** package manager
- Valid **Groq API key** (free tier available at https://console.groq.com/)
- Git (optional, for version control)

### 2. Clone or Extract the Repository

```powershell
cd d:\New folder\qa
```

### 3. Create and Activate a Virtual Environment

**On Windows PowerShell:**
```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

**On Windows CMD:**
```cmd
python -m venv env
env\Scripts\activate.bat
```

**On macOS/Linux:**
```bash
python3 -m venv env
source env/bin/activate
```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 5. Create Environment Configuration File

Create a `.env` file in the project root directory with the required variables (see **Environment Variables** section below):

```powershell
# Windows PowerShell
@"
GROQ_API_KEY=your_groq_api_key_here
"@ | Out-File -Encoding UTF8 .env
```

---

## Required Environment Variables

### Essential Variables

```env
# Groq API Key (REQUIRED)
# Get your free API key from https://console.groq.com/
GROQ_API_KEY=your_groq_api_key_here

# For multiple accounts (comma-separated, recommended for free tier)
GROQ_API_KEYS=key1,key2,key3,key4,key5

# Or use individual keys
GROQ_API_KEY_1=key1
GROQ_API_KEY_2=key2
```

### Optional Model Configuration

```env
# LLM Models (defaults provided)
GENERATION_MODEL=llama-3.3-70b-versatile
VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
GEVAL_JUDGE_MODEL=llama-3.3-70b-versatile
RAG_JUDGE_MODEL=llama-3.3-70b-versatile
SYNTHESIZER_MODEL=llama-3.3-70b-versatile

# Embedding Model
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Performance & Rate Limiting
DEFAULT_TPM=200000                      # Tokens per minute limit
EVAL_CONCURRENCY=3                      # Max concurrent evaluations
RAG_PIPELINE_CONCURRENCY=3              # Max concurrent RAG requests
CHROMA_EMBED_BATCH_SIZE=64              # Batch size for embedding

# ChromaDB Configuration
CHROMA_DB_DIR=./chroma_db              # Vector store location

# LangSmith Tracing (Optional)
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=DocQnA-Chat
```

---

## Running the Project Locally

### Start the Development Server

**Using Uvicorn (with auto-reload):**
```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8010
```

**Parameters:**
- `--reload`: Auto-restart on code changes (development only)
- `--host 127.0.0.1`: Listen only on localhost
- `--port 8010`: Server port (change if needed)

### Access the Application

Once the server is running:
- **Web UI**: http://localhost:8001

### Expected Console Output

```
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
INFO:     Started server process [1234]
INFO:     Application startup complete
```

---

## API Endpoints

### Session Management

#### List All Sessions
```
GET /api/sessions
```
Returns all sessions with their status and metadata.

#### Create a New Session
```
POST /api/sessions
```
Creates a new empty QA generation session.

#### Get Session Details
```
GET /api/sessions/{session_id}
```
Retrieve full session details, including generated QA, scores, and chat configuration.

#### Approve/Mark Session as Golden
```
POST /api/sessions/{session_id}/approve
```
Marks a session as a golden dataset (approved/reference data).

### QA Generation & Evaluation

#### Generate QA Test Cases
```
POST /api/sessions/{session_id}/generate
```
Upload a document and generate QA test cases from it.

**Request Parameters:**
- `file` (optional): Document file (PDF, TXT, Markdown)
- `sample_json` (required): JSON sample format or guidelines
- `test_case_count` (optional): Number of test cases to generate (10-100, default: 20)

**Response:** Generated QA pairs with initial scores

#### Update QA Test Cases
```
PUT /api/sessions/{session_id}/qa
```
Modify or refine existing QA test cases.

#### Re-evaluate QA Test Cases
```
POST /api/sessions/{session_id}/re-evaluate
```
Re-run DeepEval metrics on existing QA test cases.

### RAG Pipeline

#### Save RAG Configuration
```
POST /api/sessions/{session_id}/save-rag-config
```
Configure RAG parameters (embedding model, chunk size, retrieval strategy).

**Request Body:**
```json
{
  "chat_model": "llama-3.3-70b-versatile",
  "embedding_model": "all-MiniLM-L6-v2",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "top_k": 4,
  "search_model": "similarity",
  "temperature": 0.0
}
```

#### Run RAG Pipeline
```
POST /api/sessions/{session_id}/run-rag
```
Process QA pairs through the RAG pipeline with context retrieval and evaluation.

### Chat & Interaction

#### Send Chat Message
```
POST /api/sessions/{session_id}/chat
```
Send a message and receive AI-generated answers with context retrieval.

**Request Parameters:**
- `user_message`: The user's question
- `use_context` (optional): Whether to use document context (default: true)

#### Get Chat History
```
GET /api/sessions/{session_id}/chat-history
```
Retrieve the conversation history for a session.

#### Provide Feedback on Chat Message
```
POST /api/chat/{message_id}/feedback
```
Submit feedback (thumbs up/down) on a specific chat response.

**Request Body:**
```json
{
  "feedback": 1,
  "feedback_reason": "Optional reason for feedback"
}
```

### Data Export

#### Export Golden Dataset
```
GET /api/golden-dataset/export
```
Export all approved QA test cases as a downloadable file.

---

## Key Features & Capabilities

### 1. Document Processing
- Extracts text from PDF, TXT, Markdown files
- Handles multi-page documents with vision analysis
- Preserves document structure and formatting

### 2. AI-Powered QA Generation
- Uses Groq's fast LLM inference
- Configurable test case counts (10-100)
- Generates diverse, context-relevant questions and answers

### 3. Quality Evaluation
Automatic evaluation using DeepEval metrics:
- **Faithfulness**: Answers grounded in source document
- **Answer Relevancy**: Questions answered appropriately
- **Contextual Relevancy**: Retrieved context is relevant
- **Contextual Precision**: No false positives in retrieval
- **Contextual Recall**: All relevant context is retrieved

### 4. RAG Pipeline
- Chunks documents intelligently
- Semantic search via embeddings (ChromaDB)
- Context-aware answer generation
- Per-question evaluation results

### 5. Interactive Chat
- Ask questions about uploaded documents
- Get answers with retrieved context
- Rate and provide feedback on responses
- Persistent chat history per session

### 6. Multi-Key Rotation
- Support for multiple Groq API keys
- Automatic rate-limit handling
- Round-robin key rotation
- Seamless cooldown management

---

## Project Structure

```
qa/
├── main.py                 # FastAPI application & API endpoints
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (create this)
├── qa_sessions.db         # SQLite database (auto-created)
├── README.md              # This file
├── static/                # Web UI files
│   └── index.html         # Frontend interface
├── chroma_db/             # Vector store for embeddings
└── env/                   # Virtual environment (created during setup)
```

---

## Dependencies

Key packages and their purposes:

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework for API |
| `uvicorn` | ASGI server for running FastAPI |
| `groq` | Groq LLM API client |
| `python-dotenv` | Load environment variables from .env |
| `pymupdf` (fitz) | PDF text extraction |
| `deepeval` | QA evaluation metrics |
| `sentence-transformers` | Text embeddings for semantic search |
| `chromadb` | Vector database for document embeddings |
| `google-genai` | Google Generative AI integration |
| `langsmith` | LLM tracing & monitoring (optional) |
| `python-multipart` | File upload handling |
| `httpx` | Async HTTP client |

Install all dependencies with:
```powershell
pip install -r requirements.txt
```

---

## Troubleshooting

### Common Issues

#### 1. "No Groq API keys configured"
- **Solution**: Ensure `GROQ_API_KEY` is set in `.env` file
- Verify the API key is valid at https://console.groq.com/

#### 2. "Module not found" errors
- **Solution**: Activate the virtual environment first
- Run: `pip install -r requirements.txt` again

#### 3. Port 8001 already in use
- **Solution**: Change the port in the startup command:
  ```powershell
  uvicorn main:app --reload --port 8020
  ```

#### 4. Rate limit errors from Groq
- **Solution**: 
  - Add multiple API keys to `GROQ_API_KEYS`
  - Reduce `EVAL_CONCURRENCY` or `RAG_PIPELINE_CONCURRENCY`
  - Adjust `DEFAULT_TPM` if needed

#### 5. ChromaDB connection issues
- **Solution**: Delete the `chroma_db` folder and restart
  ```powershell
  Remove-Item -Recurse chroma_db
  ```

---

## Performance Tips

1. **Batch Processing**: Use `RAG_PIPELINE_CONCURRENCY` to optimize concurrent requests
2. **Embedding Cache**: Embeddings are persisted in ChromaDB—reuse them across sessions
3. **Chunk Optimization**: Adjust `chunk_size` and `chunk_overlap` based on document type
4. **Token Rate Limiting**: Configure `DEFAULT_TPM` per model for cost control
5. **Async Processing**: The API handles concurrent requests efficiently

---

## Development Notes

- **Database**: Uses SQLite locally; easily upgradeable to PostgreSQL
- **Async**: FastAPI uses async/await for high concurrency
- **Vector Store**: ChromaDB persists embeddings for fast retrieval
- **Tracing**: Optional LangSmith integration for debugging and monitoring
- **Rate Limiting**: Built-in token-per-minute (TPM) throttling

---

## Additional Resources

- **Groq API**: https://console.groq.com/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **DeepEval**: https://github.com/confident-ai/deepeval
- **ChromaDB**: https://docs.trychroma.com/
- **LangSmith**: https://smith.langchain.com/

---

## License

Please refer to your project's LICENSE file for licensing information.

## Support

For issues or questions, please check the troubleshooting section above or review the API documentation at http://localhost:8010/docs when the server is running.
GEVAL_JUDGE_MODEL=llama-3.3-70b-versatile
RAG_JUDGE_MODEL=openai/gpt-oss-120b
GROQ_EVAL_CONCURRENCY=2
```

## How to run the project locally

Start the FastAPI server with:

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8001         
```

Then open:

```text
http://127.0.0.1:8001/
```

The root URL serves the web interface, while the API is available under `/api`.

## API endpoints

### Session management

- `GET /api/sessions` — list all sessions
- `POST /api/sessions` — create a new session
- `GET /api/sessions/{session_id}` — get a specific session

### Generation and evaluation

- `POST /api/sessions/{session_id}/generate` — upload a document and generate QA test cases for a session
- `POST /api/sessions/{session_id}/approve` — mark a session as part of the golden dataset

### Golden dataset export

- `GET /api/golden-dataset/export` — export approved golden dataset sessions

## Additional notes and dependencies

### Supported input formats

- PDF
- TXT
- Markdown (`.md`)

### Main dependencies

- FastAPI
- Uvicorn
- python-multipart
- python-dotenv
- Groq
- PyMuPDF
- httpx
- DeepEval
- Pillow

### Notes

- The first run will create the local SQLite database file `qa_sessions.db`.
- PDF files are processed by extracting text and also rendering pages into images for vision-based analysis.
- If `GROQ_API_KEY` is missing, generation and evaluation features will fail until it is configured.
