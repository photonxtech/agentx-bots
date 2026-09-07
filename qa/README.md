# QA Automation & DeepEval Studio

FastAPI application for generating, evaluating, and testing document-based QA datasets with Groq models, DeepEval, ChromaDB, and optional LangSmith tracing.

## Setup Instructions

### Prerequisites

- Python 3.10 or newer
- pip
- At least one Groq API key
- Internet access for Groq API requests and the first download of embedding models

### Create and activate a virtual environment

From this directory (`qa/`):

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

For Windows Command Prompt:

```cmd
python -m venv env
env\Scripts\activate.bat
```

For macOS/Linux:

```bash
python3 -m venv env
source env/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

The application imports the `groq` and `Pillow` packages directly. If they are not already available in the environment, install them as well:

```bash
pip install groq pillow numpy
```

## Required Environment Variables

Create a `.env` file in the `qa/` directory. `python-dotenv` loads it when the application starts.

### Required

```env
GROQ_API_KEY=your_groq_api_key
```

The application also supports key rotation. Keys are collected in this order, with duplicates removed:

```env
GROQ_API_KEYS=key1,key2,key3
GROQ_API_KEY_1=another_key
GROQ_API_KEY_2=another_key
```

At least one value from `GROQ_API_KEY`, `GROQ_API_KEYS`, or `GROQ_API_KEY_1` through `GROQ_API_KEY_10` is required for generation, evaluation, RAG, and chat.

### Optional model settings

```env
GENERATION_MODEL=openai/gpt-oss-120b
VISION_MODEL=qwen/qwen3.6-27b
GEVAL_JUDGE_MODEL=openai/gpt-oss-120b
RAG_JUDGE_MODEL=openai/gpt-oss-120b
SYNTHESIZER_MODEL=openai/gpt-oss-120b
```

### Optional performance settings

```env
DEFAULT_TPM=200000
EVAL_CONCURRENCY=3
RAG_PIPELINE_CONCURRENCY=3
CHROMA_EMBED_BATCH_SIZE=64
CHROMA_DB_DIR=./chroma_db
```

The rate limiter also accepts a model-specific TPM variable. The model name is converted to uppercase and non-alphanumeric characters become underscores. For example:

```env
TPM_OPENAI_GPT_OSS_120B=200000
```

### Optional LangSmith tracing

```env
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=DocQnA-Chat
```

Tracing is enabled only when `LANGSMITH_TRACING` is truthy and `LANGSMITH_API_KEY` is present. LangSmith failures do not stop normal generation, chat, or feedback processing.

## How to Run Locally

Run the command from the `qa/` directory so that `main.py`, `static/`, `qa_sessions.db`, and `chroma_db/` resolve correctly:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

Open:

- Web UI: http://127.0.0.1:8001/
- Swagger API documentation: http://127.0.0.1:8001/docs
- ReDoc: http://127.0.0.1:8001/redoc
- OpenAPI schema: http://127.0.0.1:8001/openapi.json

Use another port if `8001` is already in use, for example `--port 8020`.

## API Endpoints

All application endpoints use the `/api` prefix. The exact request and response schemas are also available at `/docs`.

### Sessions and documents

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/sessions` | List sessions with ID, title, status, and golden-dataset status. |
| `POST` | `/api/sessions` | Create an empty session. |
| `GET` | `/api/sessions/{session_id}` | Get session metadata, documents, generated QA, scores, RAG configuration, and golden data. |
| `DELETE` | `/api/sessions/{session_id}/documents/{document_id}` | Remove one document from a session. QA data is not regenerated automatically. |

### QA generation and evaluation

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/sessions/{session_id}/generate` | Multipart upload and QA generation. |
| `PUT` | `/api/sessions/{session_id}/qa` | Save edited generated QA. Body must contain `generated_qa`; it may be a JSON value or a JSON string. |
| `POST` | `/api/sessions/{session_id}/re-evaluate` | Re-run aggregate DeepEval evaluation for saved generated QA. |

The generate endpoint accepts these multipart form fields:

- `file`: One optional uploaded file.
- `files`: Optional list of uploaded files.
- `sample_json`: Required target schema or sample JSON string.
- `test_case_count`: Standard-mode count. Allowed values are `10`, `20`, or `30`; default is `20`.
- `genesis_mode`: Boolean, default `false`.
- `genesis_field_count`: Genesis-mode field count, default `32`; the implementation samples up to the 32 available fields and is intended for `10`, `20`, or `32`.

### Genesis Capital field mode

Set `genesis_mode=true` to generate field-targeted QA for the Genesis Capital feasibility review template. The code defines 32 document-answerable or calculated fields across these sections:

- Report Header: approved date, project address/title, sponsor, borrower entity.
- Executive Summary: comments, third-party review, reviewer, review outcome, Genesis agreement, and timelines.
- Loan Summary: project type, rehab and holdback amounts, calculated costs, contingency, budget review, plan status, plan review status, and permit status.
- Finished Product Details: property type, region, units, stories, structures, and gross buildable square footage.

Field rules include currency, percentage, date, timeline, GFA, dropdown, narrative, and calculated formats. Seven purely analyst-entered fields are excluded from automatic Genesis field generation: Project Status, Report Created by, timeline appropriateness, Draw Hold, Specify Draw Hold Items, Other Special Conditions, and Permits Post Funding.

### RAG and chat

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/sessions/{session_id}/save-rag-config` | Save RAG settings and enable document chat. Accepts a JSON `RAGConfigRequest`. |
| `POST` | `/api/sessions/{session_id}/run-rag` | Run retrieval, answer generation, and per-question DeepEval evaluation. An optional JSON RAG config may be supplied. |
| `POST` | `/api/sessions/{session_id}/chat` | Ask a question using the session's saved RAG configuration. JSON body requires `question`. |
| `POST` | `/api/chat/{message_id}/metrics` | Evaluate one stored chat answer. |
| `GET` | `/api/sessions/{session_id}/chat-history` | Return stored chat messages and feedback. |
| `POST` | `/api/chat/{message_id}/feedback` | Save `yes` or `no` feedback. JSON body uses `feedback` and optional `reason`. |

RAG configuration defaults and limits:

```json
{
  "chat_model": "openai/gpt-oss-20b",
  "embedding_model": "all-MiniLM-L6-v2",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "top_k": 4,
  "search_model": "similarity",
  "temperature": 0.0
}
```

`chunk_size` is limited to 50-8000, `chunk_overlap` to 0-4000, `top_k` to 1-20, and `temperature` to 0.0-1.0.

### Approval and export

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/sessions/{session_id}/approve` | Mark a session as approved and golden. |
| `GET` | `/api/sessions/{session_id}/golden-dataset/export` | Download that session's golden dataset as JSON. |

## Supported Documents and Processing Notes

- `.pdf`: PyMuPDF text extraction plus page rendering for vision analysis. Page markers are retained in the extracted context.
- Other uploaded files are decoded as UTF-8 by the current extractor. The UI and dependencies include support-related packages for DOCX, XLSX, and MSG workflows, but the current `extract_document_text` implementation does not have separate parsers for those formats.
- Files whose names contain `Plans` are stored in the session but are intentionally excluded from QA generation and RAG processing.
- At least one processable document is required before generation or saving RAG configuration.
- PDF vision analysis and all LLM-backed operations require a configured Groq key.

## Dependencies

The declared dependencies in `requirements.txt` are:

- FastAPI and Uvicorn for the web API and server.
- `python-multipart` for file uploads and form fields.
- `python-dotenv` for `.env` loading.
- `google-genai`, `httpx`, and `langsmith` for model/API integrations and optional tracing.
- `pymupdf` for PDF processing.
- `deepeval` for QA and RAG evaluation metrics.
- `sentence-transformers` for embeddings.
- `chromadb` for persistent vector storage.
- `python-docx`, `openpyxl`, and `extract-msg` for document-related dependencies.

The current `main.py` also imports `groq`, `numpy`, and `Pillow`; make sure these packages are installed in the active environment if they are not provided by another dependency.

## Local Storage

- `qa_sessions.db` is a local SQLite database created and migrated by the application.
- `chroma_db/` stores persistent ChromaDB collections. Its location can be changed with `CHROMA_DB_DIR`.
- Do not commit `.env`, API keys, SQLite data, or generated vector-store files.