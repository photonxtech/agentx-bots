# WeNext RAG

## Project overview
WeNext RAG is a local document question-answering application built with FastAPI, LangChain, Chroma, and Groq. It ingests PDF files from the data folder, stores embeddings locally in the chroma_db folder, and serves grounded answers through a simple chat API and web frontend.

## Setup instructions
1. Clone or open the repository locally.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a .env file in the project root and add the required environment variables listed below.
5. Place your PDF documents in the data folder before starting the app.

## Required environment variables
Create a .env file with at least the following value:

```env
GROQ_API_KEY=your_groq_api_key
```

Optional configuration values you can add:

```env
RAG_RELEVANCE_THRESHOLD=1.0
RAG_TOP_K=8
RAG_DENSE_CANDIDATE_K=25
RAG_BM25_CANDIDATE_K=25
RAG_RRF_K=60
RAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RAG_CHAT_MODEL=llama-3.3-70b-versatile
```

## How to run the project locally
Start the backend server with:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Then open the app in your browser at:

```text
http://127.0.0.1:8000/
```

On startup, the app will automatically ingest the PDFs from the data folder into the local Chroma database stored in the chroma_db folder.

## API endpoints
The backend exposes the following endpoints:

- GET /api/status
  - Returns ingestion status, indexed sources, chunk count, and current configuration.
- POST /api/chat
  - Sends a user message and receives a grounded response.
  - Example request body:
    ```json
    {
      "message": "What does the document say about onboarding?",
      "session_id": "optional-session-id"
    }
    ```
- GET /api/sessions/{session_id}
  - Retrieves chat history for a specific session.
- DELETE /api/sessions/{session_id}
  - Clears chat history for a specific session.

## Additional notes and dependencies
- Python 3.10+ is recommended.
- PDF files should be placed in the data folder before startup.
- The Chroma vector database is stored locally in the chroma_db folder.
- Main dependencies include FastAPI, Uvicorn, LangChain, Chroma, PyPDF, Groq, Hugging Face embeddings, and related LangChain integrations.
