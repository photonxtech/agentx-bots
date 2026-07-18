# Multimodal DocuBot

## Project overview

This project is a RAG-style multimodal document assistant. It ingests PDFs and images, stores searchable text and image captions in a local Chroma vector store, and uses Google Gemini models via LangChain to answer user questions with retrieved document context.

The backend is built with FastAPI and exposes endpoints for file ingestion, querying, and store management. The frontend is served from the `frontend/` directory.

## Setup instructions

1. Clone or open the repository.
2. Create a Python virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Create a `.env` file at the repository root if it does not already exist. Add your Google API key:

   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. Ensure the following folders exist or are created automatically at runtime:
   - `data/pdfs`
   - `data/images`
   - `chroma_db`
   - `docstore`

## Required environment variables

The backend loads environment variables using `python-dotenv`.

- `GOOGLE_API_KEY` — required for Google Gemini / Google Generative AI access.

## How to run the project locally

From the repository root, activate the virtual environment and run:

```powershell
uvicorn backend.api:app --reload
```

Then open your browser at `http://127.0.0.1:8000/` to access the frontend served by the FastAPI app.

## API endpoints

The backend exposes the following endpoints:

- `GET /`
  - Returns the frontend `index.html` page.

- `GET /health`
  - Returns a health status JSON response.

- `GET /sources`
  - Returns a list of unique ingested document source names from the vector store.

- `POST /ingest`
  - Accepts multipart form data with one or more files.
  - Supported file types: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.webp`.
  - PDF files are processed for text and images.
  - Image files are captioned and added to the store.
  - Example request body: `files` as uploaded files.

- `POST /query`
  - Accepts JSON with:
    - `question` (string, required)
    - `chat_history` (optional list of role/content objects)
  - Returns an AI answer along with retrieved text passages and image data.

- `POST /clear`
  - Clears the local vector store and docstore.

## Additional notes

- The project uses local storage directories:
  - `data/pdfs` for uploaded PDFs
  - `data/images` for saved extracted images and uploaded images
  - `chroma_db` for Chroma persistence
  - `docstore` for raw document data storage

- The core model configuration is defined in `backend/store_utils.py`:
  - Caption model: `gemini-3.1-flash-lite`
  - Chat model: `gemini-3.5-flash`
  - Embedding model: `models/gemini-embedding-001`

- If you want to ingest PDFs from the command line, use the script in `backend/ingest.py`.

- Dependencies are listed in `requirements.txt`.

## Dependencies

- Python
- FastAPI
- Uvicorn
- python-dotenv
- ChromaDB
- LangChain
- Google Gemini integration via `langchain-google-genai`
- PyMuPDF
- Pillow
- python-multipart
- requests
