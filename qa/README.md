# QA Automation & DeepEval Studio

## Project Overview

This project is a FastAPI-based application for generating QA test cases from uploaded source documents such as PDF, TXT, and Markdown files. It provides:

- document ingestion and text extraction
- optional vision-based PDF page analysis
- Groq-backed LLM generation for QA datasets
- DeepEval-based quality evaluation and scoring
- a simple web UI served from the `static` directory

Generated sessions are stored locally in a SQLite database file named `qa_sessions.db`.

## Setup Instructions

### 1. Prerequisites

- Python 3.10+
- pip
- A valid Groq API key(s)

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Create a local `.env` file

Create a `.env` file in this folder with the required environment variables (see below).

## Required Environment Variables

The application reads the following environment variables:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Optional overrides:

```env
VISION_MODEL=qwen/qwen3.6-27b
GEVAL_JUDGE_MODEL=llama-3.3-70b-versatile
RAG_JUDGE_MODEL=openai/gpt-oss-120b
GROQ_EVAL_CONCURRENCY=2
```

## How to Run the Project Locally

With the virtual environment activated, start the FastAPI server:

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

Then open:

```text
http://127.0.0.1:8001/
```

The root URL serves the web interface, and the API is available under `/api`.

## API Endpoints

### Session management

- `GET /api/sessions` — list all sessions
- `POST /api/sessions` — create a new session
- `GET /api/sessions/{session_id}` — retrieve a specific session

### Generation and evaluation

- `POST /api/sessions/{session_id}/generate` — upload a document and generate QA test cases for a session
- `POST /api/sessions/{session_id}/approve` — mark a session as approved for the golden dataset

### Golden dataset export

- `GET /api/golden-dataset/export` — export approved golden dataset sessions

## Additional Notes and Dependencies

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
