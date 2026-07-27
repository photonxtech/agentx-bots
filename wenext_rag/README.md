# WeNext RAG

## Project overview
WeNext RAG is a local document-question answering application built with FastAPI, LangChain, Chroma DB, and Google Gemini. It ingests PDF documents placed in the data folder, creates embeddings, and serves answers through a simple chat API and web frontend.

## Setup instructions
1. Clone or open the repository locally.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place your PDF documents in the data folder.
5. Create a .env file in the project root and add the required environment variables below.

## Required environment variables
Create a .env file with at least:

```env
GEMINI_API_KEY=your_google_gemini_api_key
```

Optional configuration values:

```env
RAG_RELEVANCE_THRESHOLD=0.85
RAG_TOP_K=8
RAG_EMBEDDING_MODEL=gemini-embedding-2-preview
RAG_CHAT_MODEL=gemini-2.5-flash
```

## How to run the project locally
Start the backend server:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000/
```

On startup, the app will automatically ingest the PDFs from the data folder into the Chroma database stored in the chroma_db folder.

## API endpoints
The backend exposes the following endpoints:

- GET /api/status
  - Returns ingestion status, indexed sources, chunk count, and configuration.
- POST /api/chat
  - Sends a chat message and receives a grounded response.
  - Request body:
    ```json
    {
      "message": "What does the document say about onboarding?",
      "session_id": "optional-session-id"
    }
    ```
- GET /api/sessions/{session_id}
  - Retrieves chat history for a session.
- DELETE /api/sessions/{session_id}
  - Clears chat history for a session.

## Additional notes and dependencies
- Python 3.10+ is recommended.
- PDF files should be placed in the data folder before startup.
- The Chroma vector database is stored locally in the chroma_db folder.
- Main dependencies include FastAPI, Uvicorn, LangChain, Chroma, PyPDF, and Google Gemini libraries.
