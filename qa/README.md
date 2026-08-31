# QA Automation & DeepEval Studio

FastAPI-based QA Test Case Generator with AI-powered evaluation. Automates creation and validation of QA test cases from uploaded documents (PDF, TXT, Markdown) using Groq LLM, DeepEval metrics, and RAG (Retrieval-Augmented Generation) pipeline.

---

## Project Overview

QA Studio is an end-to-end AI-powered Quality Assurance automation system designed to:
- **Generate** comprehensive QA test cases from documents using advanced LLMs
- **Evaluate** test case quality using multiple DeepEval metrics
- **Retrieve** contextual information via RAG pipeline with semantic search
- **Chat** interactively with document context
- **Track** evaluation metrics and maintain golden datasets for benchmarking

### Key Capabilities
- Multi-format document processing (PDF, TXT, Markdown)
- Vision-based document analysis for complex layouts
- Automatic QA pair generation with configurable counts
- **Genesis Capital Field-Targeted QA Generation** — 32 testable fields for feasibility reviews
- Quality metrics: Faithfulness, Answer Relevancy, Contextual Relevancy, Precision, Recall
- Interactive chat with document context retrieval
- Multi-Groq API key rotation with automatic rate-limit handling
- Optional LangSmith integration for tracing and monitoring

---

## Setup Instructions

### Prerequisites
- **Python**: 3.10 or higher
- **pip**: Package manager
- **Groq API Key**: Get one free at https://console.groq.com/ (includes rate limits)

### 1. Virtual Environment Setup

**Windows PowerShell:**
```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
python -m venv env
env\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python3 -m venv env
source env/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables Configuration

Create a `.env` file in the project root directory with the following variables:

```env
# ========== REQUIRED ==========
GROQ_API_KEY=your_groq_api_key_here

# ========== OPTIONAL: Multiple API Keys (Recommended) ==========
# For multiple Groq accounts or managing rate limits:
GROQ_API_KEYS=key1,key2,key3,key4,key5

# Alternative format (one per environment variable):
# GROQ_API_KEY_1=your_key_1
# GROQ_API_KEY_2=your_key_2
# GROQ_API_KEY_3=your_key_3

# ========== Model Configuration ==========
# Generation model for creating QA test cases
GENERATION_MODEL=openai/gpt-oss-120b

# Vision model for document analysis (handles images in PDFs)
VISION_MODEL=qwen/qwen3.6-27b

# Judge model for DeepEval GEval metric
GEVAL_JUDGE_MODEL=openai/gpt-oss-120b

# Judge model for RAG evaluation
RAG_JUDGE_MODEL=openai/gpt-oss-120b

# Synthesizer model for generating test cases
SYNTHESIZER_MODEL=openai/gpt-oss-120b

# Embedding model for semantic search (local, no API needed)
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2

# ========== Performance & Concurrency ==========
# Maximum tokens per minute (TPM) for rate limiting
DEFAULT_TPM=200000

# Max concurrent metric evaluations
EVAL_CONCURRENCY=3

# Max concurrent RAG pipeline requests
RAG_PIPELINE_CONCURRENCY=3

# Batch size for ChromaDB embeddings
CHROMA_EMBED_BATCH_SIZE=64

# ChromaDB directory path
CHROMA_DB_DIR=./chroma_db

# ========== LangSmith Integration (Optional) ==========
# Enable LangSmith tracing for debugging
LANGSMITH_TRACING=false

# Your LangSmith API key
LANGSMITH_API_KEY=your_langsmith_key

# LangSmith project name
LANGSMITH_PROJECT=QA-Studio-Sessions
```

**Important Notes:**
- `GROQ_API_KEY` is required for all LLM operations
- Multiple API keys help distribute rate limits across different Groq accounts
- Environment variables are loaded via `python-dotenv` from the `.env` file
- Model names follow Groq's format: `provider/model-name`

---

## How to Run the Project Locally

### Start the FastAPI Server

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

**Optional parameters:**
- `--reload`: Auto-restart server on code changes (development only)
- `--host 127.0.0.1`: Bind to localhost only (use `0.0.0.0` for network access)
- `--port 8001`: Server port (change if 8001 is in use)

### Access the Application

Once the server is running:

- **Web UI / Dashboard**: http://localhost:8001/
- **Interactive API Docs (Swagger)**: http://localhost:8001/docs
- **Alternative API Docs (ReDoc)**: http://localhost:8001/redoc
- **OpenAPI Schema**: http://localhost:8001/openapi.json

---

## API Endpoints

### Session Management

#### List All Sessions
```
GET /api/sessions
```
**Description**: Retrieve all QA generation sessions with metadata and status.

**Response**: Array of session objects containing ID, creation time, documents, and QA pairs.

---

#### Create a New Session
```
POST /api/sessions
```
**Description**: Initialize a new empty QA generation session.

**Response**: New session object with unique `session_id`.

---

#### Get Session Details
```
GET /api/sessions/{session_id}
```
**Description**: Retrieve comprehensive session information including generated QA, evaluation scores, documents, and chat configuration.

**Parameters**:
- `session_id` (path): Unique session identifier

**Response**: Full session details with all metadata.

---

#### Delete Document from Session
```
DELETE /api/sessions/{session_id}/documents/{document_id}
```
**Description**: Remove a specific document and its associated QA pairs from a session.

**Parameters**:
- `session_id` (path): Session identifier
- `document_id` (path): Document identifier

---

### QA Generation & Evaluation

#### Generate QA Test Cases
```
POST /api/sessions/{session_id}/generate
```
**Description**: Upload a document and automatically generate QA test cases using AI. Supports both standard QA generation and Genesis Capital field-targeted generation.

**Parameters**:
- `session_id` (path): Target session
- `file` or `files` (form): Document file(s) (PDF, TXT, or Markdown) — optional
- `sample_json` (form): JSON format example or guidelines — required
- `test_case_count` (form): Number of test cases to generate (10, 20, or 100, default: 20) — optional (ignored if `genesis_mode=true`)
- `genesis_mode` (form): Enable Genesis Capital field-targeted mode (boolean, default: false) — optional
- `genesis_field_count` (form): Number of Genesis fields to generate QA for (10, 20, or 32, default: 32) — optional (only used if `genesis_mode=true`)

**Response**: Generated QA pairs with initial DeepEval metric scores.

**Supported Formats**:
- `.pdf` — PDF documents (text + vision analysis)
- `.txt` — Plain text files
- `.md` — Markdown files
- `.docx` — Word documents
- `.xlsx` — Excel spreadsheets

##### Standard Mode (genesis_mode=false)
Generates generic QA test cases from document content using DeepEval Synthesizer.

**Example Request**:
```json
{
  "sample_json": "{ \"question\": \"...\", \"answer\": \"...\" }",
  "test_case_count": 20,
  "genesis_mode": false
}
```

##### Genesis Capital Field-Targeted Mode (genesis_mode=true)
Generates QA pairs specifically targeted at the 32 testable fields in Genesis Capital Feasibility Review Templates. Each field has specific extraction methods, validation rules, and formatting requirements.

**Genesis Fields Overview**:
- **32 Total Fields** across multiple sections: Report Header, Executive Summary, Project Details, Financial Analysis, Timeline, and Risk Assessment
- **Field Types**: freeform text, currency, dates, percentages, dropdowns, narratives, calculated fields
- **Intelligent Extraction**: Uses document-specific extraction methods (RAG + LLM, regex patterns, vision analysis)
- **Format Validation**: Automatic formatting for dates (MM/DD/YYYY), currency ($#,###,###.##), percentages, timelines

**Example Request**:
```json
{
  "sample_json": "{ \"field_name\": \"...\", \"value\": \"...\" }",
  "genesis_mode": true,
  "genesis_field_count": 32
}
```

**Field Categories**:
1. **Report Header** — Date, Project Address, Sponsor, Borrower Entity
2. **Executive Summary** — Comments, Third-party Reviews, Genesis Agreement, Timelines
3. **Project Details** — Type, Description, Stories, Units, Square Footage
4. **Financial Analysis** — Budget, Cost Per Unit, Contingency, Financing
5. **Timeline Analysis** — Project Start Date, Elapsed Time, Projected Completion
6. **Risk Assessment** — Construction Risk, Market Risk, Sponsor Track Record

---

#### Update QA Test Cases
```
PUT /api/sessions/{session_id}/qa
```
**Description**: Modify or refine existing QA test cases in a session.

**Parameters**:
- `session_id` (path): Session identifier

**Request Body**:
```json
{
  "qa_pairs": [
    {
      "id": "qa_123",
      "question": "Updated question?",
      "expected_output": "Updated answer."
    }
  ]
}
```

---

#### Re-evaluate QA Test Cases
```
POST /api/sessions/{session_id}/re-evaluate
```
**Description**: Re-run all DeepEval metrics on existing QA test cases without regenerating them.

**Response**: Updated QA pairs with recalculated scores.

---

### RAG Pipeline

#### Save RAG Configuration
```
POST /api/sessions/{session_id}/save-rag-config
```
**Description**: Configure RAG parameters for document chunking, embedding, and retrieval.

**Request Body**:
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

**Parameters**:
- `chat_model`: LLM for generating answers
- `embedding_model`: Model for text embeddings (determines semantic search quality)
- `chunk_size`: Document chunk size in characters
- `chunk_overlap`: Overlap between chunks for context continuity
- `top_k`: Number of top-k retrieval results
- `search_model`: Search strategy ("similarity" or other)
- `temperature`: LLM generation temperature (0.0 = deterministic, 1.0 = creative)

---

#### Run RAG Pipeline
```
POST /api/sessions/{session_id}/run-rag
```
**Description**: Process all QA pairs through the RAG pipeline: retrieve context, generate answers, and evaluate results.

**Response**: Processed QA pairs with retrieved context, generated answers, and evaluation scores.

---

### Chat & Interaction

#### Send Chat Message
```
POST /api/sessions/{session_id}/chat
```
**Description**: Send a user question and receive an AI-generated answer with optional document context.

**Parameters**:
- `session_id` (path): Session identifier
- `user_message` (form): The user's question
- `use_context` (form): Whether to retrieve and use document context (boolean, default: true)

**Response**: Chat message object with generated answer, retrieved context, and message ID.

---

#### Get Chat History
```
GET /api/sessions/{session_id}/chat-history
```
**Description**: Retrieve the complete conversation history for a session.

**Parameters**:
- `session_id` (path): Session identifier

**Response**: Array of chat messages with user questions and AI responses.

---

#### Provide Chat Feedback
```
POST /api/chat/{message_id}/feedback
```
**Description**: Submit user feedback (like/dislike) on a specific chat response.

**Parameters**:
- `message_id` (path): Chat message identifier

**Request Body**:
```json
{
  "feedback": 1,
  "feedback_reason": "Answer was accurate and helpful"
}
```

**Feedback Values**:
- `1` — Thumbs up (positive)
- `-1` — Thumbs down (negative)
- `0` — Neutral

---

#### Get Message Metrics
```
POST /api/chat/{message_id}/metrics
```
**Description**: Calculate and retrieve DeepEval metrics for a specific chat message.

**Parameters**:
- `message_id` (path): Chat message identifier

**Response**: DeepEval metric scores for the message.

---

### Session Approval & Export

#### Approve/Publish Session
```
POST /api/sessions/{session_id}/approve
```
**Description**: Mark a session as a golden dataset (approved reference data for benchmarking).

**Parameters**:
- `session_id` (path): Session identifier

---

#### Export Golden Dataset
```
GET /api/sessions/{session_id}/golden-dataset/export
```
**Description**: Download all approved QA test cases from a session as a file (JSON or other format).

**Parameters**:
- `session_id` (path): Session identifier

**Response**: File download of the golden dataset.

---

## Project Structure

```
qa/
├── main.py                    # FastAPI application, all API endpoints, LLM orchestration
├── requirements.txt           # Python package dependencies
├── .env                       # Environment variables (create this, not in repo)
├── README.md                  # This documentation file
├── qa_sessions.db             # SQLite database (auto-created on first run)
├── static/
│   └── index.html             # Web UI dashboard (Tailwind CSS, JavaScript)
├── chroma_db/                 # Vector store for embeddings (auto-created)
│   └── [collection data]/
├── env/                       # Virtual environment (local development)

```

---

---

## Dependencies

All dependencies are listed in `requirements.txt`. Key packages:

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | Latest | Web framework for API endpoints |
| `uvicorn` | Latest | ASGI server for running FastAPI |
| `groq` | Latest | Groq LLM API client |
| `python-dotenv` | Latest | Load `.env` environment variables |
| `pymupdf` (fitz) | Latest | PDF text and image extraction |
| `deepeval` | Latest | QA evaluation metrics library |
| `sentence-transformers` | Latest | Text embeddings for semantic search |
| `chromadb` | Latest | Vector database for embeddings |
| `google-genai` | Latest | Google Generative AI integration |
| `langsmith` | Latest | LLM tracing and monitoring (optional) |
| `python-multipart` | Latest | File upload handling in FastAPI |
| `httpx` | Latest | Async HTTP client |
| `python-docx` | Latest | Word document processing |
| `openpyxl` | Latest | Excel file processing |
| `extract-msg` | Latest | MSG file extraction |

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Supported Input Formats

The project can process the following document types:

- **PDF** (`.pdf`) — Extracts text and performs vision analysis on document images
- **Text** (`.txt`) — Plain text files
- **Markdown** (`.md`) — Markdown-formatted documents
- **Word** (`.docx`) — Word documents
- **Excel** (`.xlsx`) — Spreadsheet files

---

## Key Features

### 1. **Document Processing**
- Multi-format document support (PDF, TXT, Markdown, DOCX, XLSX, MSG)
- Vision-based analysis for complex PDF layouts
- Automatic text extraction and structuring
- Page-level metadata preservation

### 2. **AI-Powered QA Generation**
- Uses Groq's high-performance LLM inference
- Configurable test case count (10-100 pairs per document)
- Generates diverse, context-relevant questions and answers
- Template-based generation with JSON schema validation

### 3. **Quality Evaluation (DeepEval Metrics)**
- **Faithfulness**: Ensures answers are grounded in source documents
- **Answer Relevancy**: Validates that answers properly address questions
- **Contextual Relevancy**: Checks if retrieved context is relevant
- **Contextual Precision**: Measures accuracy of retrieved context (no false positives)
- **Contextual Recall**: Verifies all relevant context is retrieved
- **GEval**: Custom metric using judge model for nuanced evaluation

### 4. **RAG (Retrieval-Augmented Generation) Pipeline**
- Intelligent document chunking with configurable overlap
- Semantic search via text embeddings (ChromaDB vector store)
- Context-aware answer generation
- Per-question metric evaluation
- Multi-document retrieval support

### 5. **Interactive Chat**
- Ask questions about uploaded documents
- Real-time answer generation with document context
- Rating and feedback on responses
- Persistent chat history per session
- Context retrieval tracking

### 6. **Multi-API Key Management**
- Support for multiple Groq API keys (different accounts or tiers)
- Automatic round-robin key rotation
- Smart rate-limit detection and rotation
- Per-key cooldown management
- Transparent failover for production reliability

### 7. **Genesis Capital Field-Targeted QA Generation**
Specialized mode for Genesis Capital feasibility review document processing:

- **32 Testable Fields**: Comprehensive coverage of Genesis Capital Feasibility Review Template fields
- **Intelligent Extraction**: Domain-specific extraction methods per field (RAG + LLM, regex patterns, vision analysis)
- **Field-Specific Validation**: Automatic validation and formatting based on field type:
  - Currency fields: `$#,###,###.##` format
  - Date fields: ISO storage (YYYY-MM-DD), displayed as MM/DD/YYYY
  - Dropdown fields: Validation against approved options (Trinity, Granite, DCMI, Northwest Monitoring)
  - Narrative fields: Zero-hallucination synthesis from source documents
  - Timeline fields: Calculated as "X days (Y months)" format
  - GFA fields: "XXX,XXX SF" format
- **Multi-Document Awareness**: Tracks which document each field is sourced from
- **Configurable Coverage**: Generate QA for 10, 20, or all 32 fields
- **Field Categories**:
  - Report Header: Date, Project Address, Sponsor, Borrower
  - Executive Summary: Comments, Third-party Reviews, Risk Assessment
  - Project Details: Type, Configuration, Units, Square Footage
  - Financial Analysis: Budget, Cost per Unit, Contingency
  - Timeline: Elapsed Time, Projected Completion
  - Risk Assessment: Construction Risk, Market Risk, Sponsor Review

### 8. **Data Management & Export**
- SQLite database for persistent storage
- Session-based organization of QA pairs
- Golden dataset approval for benchmarking
- Dataset export for downstream use
- LangSmith integration for tracing (optional)

---

## Additional Notes

### First Run
- The SQLite database (`qa_sessions.db`) is created automatically on the first run
- ChromaDB folder is initialized for vector storage
- No manual database setup is required

### PDF Processing
- PDFs are processed in two ways:
  1. Text extraction using PyMuPDF for structured data
  2. Vision analysis by rendering pages as images (for complex layouts)
- Multi-page documents are fully supported

### Genesis Capital Feasibility Reviews
**Genesis Fields** provide specialized, domain-specific QA generation for Genesis Capital project feasibility review documents:

**When to use Genesis Mode**:
- Processing Trinity Reports, SCA documents, construction budgets, or project timelines
- Generating field-specific QA pairs for Genesis Capital workflows
- Validating feasibility review completeness and accuracy
- Benchmarking against Genesis field extraction standards

**How to Enable**:
1. Set `genesis_mode=true` in your `/api/sessions/{session_id}/generate` request
2. Choose `genesis_field_count` (10, 20, or 32 fields)
3. Upload relevant Genesis Capital documents (Trinity, SCA, Budget, etc.)

**Genesis Field Extraction Sources**:
- **Trinity Report** → Project scope, risk assessment, third-party reviews
- **SCA** → Sponsor information, construction analysis, additional comments
- **Construction Budget** → Borrower entity, cost analysis, contingency
- **Project Timeline** → Dates, elapsed time, project status
- **Deal Notes** → Project timeline calculations, milestones

**Field Validation Examples**:
- Date fields auto-format as MM/DD/YYYY
- Currency fields auto-format with thousand separators ($#,###,###.##)
- Dropdown fields validate against exact Genesis-approved options
- Narrative fields synthesize multiple document sources without hallucination
- Timeline fields calculate elapsed/remaining time in "X days (Y months)" format

### Rate Limiting & Performance
- Built-in token-per-minute (TPM) throttling per model
- Automatic request queuing and backoff
- Configurable concurrency limits for evaluation and RAG operations
- Designed to handle free-tier Groq API rate limits efficiently

### Database
- Uses SQLite for local development (easily upgradeable to PostgreSQL)
- Stores sessions, QA pairs, chat history, and evaluation metrics
- Schema auto-migrates on startup

### Async Processing
- FastAPI uses async/await for high concurrency
- Multiple requests can be processed simultaneously
- Ideal for batch QA generation and evaluation

### LangSmith Integration (Optional)
- Enable tracing with `LANGSMITH_TRACING=true`
- Automatically logs all LLM calls and metrics
- Useful for debugging and monitoring production systems

---

## Common Issues & Troubleshooting

### "No Groq API keys configured"
**Solution**: 
- Ensure `GROQ_API_KEY` is set in your `.env` file
- Verify the API key is valid at https://console.groq.com/

### "ModuleNotFoundError: No module named..."
**Solution**:
- Activate the virtual environment first
- Run `pip install -r requirements.txt`
- Check that all dependencies installed correctly: `pip list`

### Port 8001 Already in Use
**Solution**:
```bash
uvicorn main:app --reload --port 8020
```
(Use any available port number)

### Rate Limit Errors (429) from Groq
**Solution**:
- Add multiple API keys: `GROQ_API_KEYS=key1,key2,key3`
- Reduce concurrency: `EVAL_CONCURRENCY=1` or `RAG_PIPELINE_CONCURRENCY=1`
- Increase `DEFAULT_TPM` if you have higher-tier API limits

### ChromaDB Connection Issues
**Solution**:
```bash
Remove-Item -Recurse chroma_db  # Windows PowerShell
# or
rm -rf chroma_db  # Linux/macOS
```
Then restart the server to reinitialize.

### "SSL: CERTIFICATE_VERIFY_FAILED" Errors
**Solution**:
- Ensure your internet connection is stable
- Update certificates: `pip install --upgrade certifi`

### Slow PDF Processing
**Solution**:
- Reduce `EVAL_CONCURRENCY` or `RAG_PIPELINE_CONCURRENCY`
- Increase `CHROMA_EMBED_BATCH_SIZE` for faster embeddings
- Consider splitting large PDFs into smaller documents

---

## Performance Optimization Tips

1. **Batch Processing**: Set `RAG_PIPELINE_CONCURRENCY` based on available resources
2. **Embedding Cache**: Embeddings are persisted in ChromaDB — reuse them across sessions
3. **Chunk Size Tuning**: Adjust `chunk_size` and `chunk_overlap` for different document types
4. **Token Budget**: Monitor `DEFAULT_TPM` and adjust based on API tier
5. **Concurrent Requests**: FastAPI handles multiple requests efficiently — scale horizontally as needed
6. **Database Optimization**: Regular cleanup of old sessions improves query performance

---

## Development Notes

- **Language**: Python 3.10+
- **Framework**: FastAPI with async support
- **Database**: SQLite (local) → PostgreSQL (production-ready)
- **Vector Store**: ChromaDB for semantic search
- **LLM Provider**: Groq (free tier available)
- **Evaluation**: DeepEval framework
- **Frontend**: HTML + Tailwind CSS + Vanilla JavaScript

---

## Additional Resources

- **Groq Console**: https://console.groq.com/
- **Groq API Documentation**: https://console.groq.com/docs
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **DeepEval GitHub**: https://github.com/confident-ai/deepeval
- **ChromaDB Documentation**: https://docs.trychroma.com/
- **LangSmith Documentation**: https://smith.langchain.com/
- **Sentence Transformers**: https://www.sbert.net/

---

## Support & Feedback

For issues or questions:
1. Check the **Troubleshooting** section above
2. Review **API Documentation** at http://localhost:8001/docs (when server is running)
3. Check Groq API status and rate limits at https://console.groq.com/
4. Review LangSmith traces (if enabled) for debugging LLM calls
