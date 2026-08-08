# WeNext AI Features Guide — RAG Demo

A local RAG app that answers questions about the WeNext platform using ONLY
the content of `WeNext_AI_Features_Guide.docx`, via a 4-stage pipeline:

1. **Multi-Query Rewrite** — expands your question into 3 phrasings
2. **Retrieve** — embeds each phrasing locally and pulls matching chunks from ChromaDB
3. **Choice Select** — LLM re-ranks/filters which chunks are actually relevant
4. **Tree Summarize** — LLM writes the final answer from the selected chunks only

## Project layout

```
backend/    FastAPI app, RAG pipeline, ingestion, eval — run all commands from here
frontend/   Static single-page chat UI, served by the backend
```

## 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt --break-system-packages
```

(drop `--break-system-packages` if you're in a virtualenv)

Copy `.env.example` to `.env.local` and adjust if needed (defaults already match a local Ollama setup).

## 2. Install Ollama and pull the model

Download from https://ollama.com, then:

```bash
ollama pull llama3.1:8b
```

Make sure the Ollama server is running (it usually starts automatically; otherwise run `ollama serve`).
No API key needed — inference runs entirely on your machine.

## 3. Ingest the document (run once)

```bash
python3 ingest.py
```

This parses the docx into ~35 chunks, embeds them with a local
`sentence-transformers` model (no API cost, downloads once on first run),
and stores them in a local ChromaDB folder (`backend/chroma_db/`).

Re-run this any time you replace `data/WeNext_AI_Features_Guide.docx` with an updated version.

## 4. Start the server

```bash
uvicorn main:app --reload --port 8000
```

(run from inside `backend/`; it serves the UI from `../frontend`)

## 5. Open the app

http://localhost:8000

Ask things like:
- "How does the WhatsApp AI agent decide what to say?"
- "What models power WeNext's AI?"
- "How does the AI score leads?"

## Files

| File | Purpose |
|---|---|
| `backend/chunker.py` | Parses the docx by heading hierarchy into chunks |
| `backend/ingest.py` | Embeds chunks + writes them into ChromaDB |
| `backend/rag_pipeline.py` | The 4-stage RAG pipeline (Ollama + ChromaDB) |
| `backend/main.py` | FastAPI backend, exposes `POST /query` |
| `backend/eval/` | Golden dataset + DeepEval scoring scripts |
| `backend/data/` | Source docx lives here |
| `backend/chroma_db/` | Local vector store (created by `ingest.py`, git-ignored) |
| `frontend/index.html` | Frontend — single-file, shows the pipeline stages live |

## Troubleshooting

- **"Could not reach Ollama"** on startup — make sure the Ollama app/service is running (`ollama serve`) and `llama3.1:8b` has been pulled.
- **First run is slow** — `sentence-transformers` downloads the embedding model (~90MB) once.
- **Answer says "context doesn't contain enough information"** — the prompt deliberately
  refuses to guess outside the doc's content; try rephrasing or check the doc actually covers it.
- **Changed the docx?** — re-run `python3 ingest.py` before restarting the server.
