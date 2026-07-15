# PhotonX Documentation Assistant (RAG)

A production-grade **Retrieval-Augmented Generation** service that answers
questions **only** from PhotonX documentation. It reads PDFs, chunks them,
embeds them, stores the vectors in ChromaDB, retrieves the most relevant
passages for a question, and asks OpenAI to synthesize a grounded answer with
**citations**. If the answer is not in the docs, it responds with exactly:

> I couldn't find that information in the PhotonX documentation.

The RAG pipeline is implemented **manually** — no LangChain, no LlamaIndex — so
every component is small, readable, and swappable.

---

## Table of Contents
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the API](#running-the-api)
- [Ingesting Documents](#ingesting-documents)
- [Asking Questions](#asking-questions)
- [API Reference](#api-reference)
- [How It Works (Design Notes)](#how-it-works-design-notes)
- [Future Improvements](#future-improvements)

---

## Architecture

```
                          ┌─────────────────────────────────────────────┐
                          │                 FastAPI (app/main.py)        │
                          │   POST /ingest   POST /ask   GET /health     │
                          │                 GET /stats                   │
                          └───────────────┬─────────────────┬───────────┘
                                          │                 │
                        ┌─────────────────▼──┐   ┌──────────▼───────────┐
                        │  IngestionService  │   │      QAService       │
                        │  (load→chunk→embed │   │ (retrieve→guard→LLM  │
                        │      →store)       │   │      →cite)          │
                        └───────┬────────────┘   └───────┬──────────────┘
                                │                         │
        ┌────────────┬─────────┼──────────┬────────┐     │
        ▼            ▼         ▼           ▼        ▼     ▼
   PDFLoader     Chunker    Embedder   VectorStore  Retriever   LLMClient
   (pypdf)    (headings+   (OpenAI    (ChromaDB,   (embed q +  (OpenAI
              overlap)     embeddings) cosine)     search)     chat)
                                          │
                                          ▼
                                   chroma_db/  (persistent)

   Everything is wired once in app/services/container.py (composition root).
```

**Request flow — `/ask`:**
`question` → embed query → cosine search (top-k) in ChromaDB → filter by score →
build grounded prompt → OpenAI chat completion → answer + deduplicated citations.

**Anti-hallucination layers:**
1. Strict system prompt: answer *only* from context, else emit the exact refusal.
2. If no chunk clears the relevance threshold, we refuse **without calling the LLM**.
3. Citations are built from the *retrieved* chunks, never parsed from model text,
   so a source can never be invented.

---

## Folder Structure

```
photonx-rag/
├── app/
│   ├── api/                 # HTTP layer (thin routers)
│   │   ├── ask.py           #   POST /ask, GET /health, GET /stats
│   │   └── ingest.py        #   POST /ingest
│   ├── core/                # Cross-cutting concerns
│   │   ├── config.py        #   Typed settings from .env (single source of truth)
│   │   ├── logger.py        #   Central logging setup
│   │   └── exceptions.py    #   Typed domain exceptions
│   ├── rag/                 # RAG building blocks (one responsibility each)
│   │   ├── pdf_loader.py    #   STEP 1: extract text per page
│   │   ├── chunker.py       #   STEP 2: heading-aware overlapping chunks
│   │   ├── embedding.py     #   STEP 3: batched OpenAI embeddings
│   │   ├── vector_store.py  #   STEP 4: ChromaDB (cosine) wrapper
│   │   ├── retriever.py     #   STEP 5: query → top-k chunks
│   │   ├── prompt.py        #   STEP 6: system prompt + context builder
│   │   └── llm.py           #   STEP 7: chat completion
│   ├── schemas/             # Pydantic models
│   │   ├── documents.py     #   Internal domain models
│   │   └── api.py           #   Request/response models
│   ├── services/            # Orchestration + dependency injection
│   │   ├── container.py     #   Composition root (wires everything)
│   │   ├── ingestion_service.py
│   │   └── qa_service.py
│   └── main.py              # FastAPI app + lifecycle + error handlers
├── docs/                    # Put your PDF files here
├── chroma_db/               # Persistent vector store (created at runtime)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Requirements

- **Python 3.12**
- An **OpenAI API key**

---

## Installation

```bash
# 1. Create and activate a virtual environment (Python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Copy the example env file and fill in your key:

```bash
cp .env.example .env
```

| Variable            | Default                    | Description                                   |
|---------------------|----------------------------|-----------------------------------------------|
| `OPENAI_API_KEY`    | *(required)*               | Your OpenAI API key.                          |
| `OPENAI_MODEL`      | `gpt-4.1`                  | Chat model that writes the final answer.      |
| `EMBEDDING_MODEL`   | `text-embedding-3-small`   | Model used to embed chunks and queries.       |
| `CHROMA_DB`         | `./chroma_db`              | On-disk location for ChromaDB.                |
| `CHROMA_COLLECTION` | `photonx_docs`             | Collection name.                              |
| `DOCS_DIR`          | `./docs`                   | Folder scanned by `/ingest`.                  |
| `CHUNK_SIZE`        | `800`                      | Chunk size in characters.                     |
| `CHUNK_OVERLAP`     | `150`                      | Overlap between consecutive chunks.           |
| `TOP_K`             | `5`                        | Chunks retrieved per query.                   |
| `LLM_TEMPERATURE`   | `0.0`                      | Kept at 0 for factual, deterministic answers. |
| `LLM_MAX_TOKENS`    | `800`                      | Max tokens in the answer.                     |
| `LOG_LEVEL`         | `INFO`                     | `DEBUG`/`INFO`/`WARNING`/`ERROR`.             |

Configuration is validated on startup (e.g. overlap must be smaller than chunk
size), so misconfiguration fails fast rather than at request time.

---

## Running the API

```bash
uvicorn app.main:app --reload
```

Then open the interactive docs at **http://localhost:8000/docs**.

---

## Ingesting Documents

1. Drop one or more `.pdf` files into the `docs/` folder.
2. Trigger indexing:

```bash
curl -X POST http://localhost:8000/ingest
```

```json
{ "status": "success", "documents": 1, "chunks": 42 }
```

Re-running `/ingest` **upserts** by chunk id, so it is safe to run repeatedly.

---

## Asking Questions

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What services does PhotonX provide?"}'
```

```json
{
  "answer": "PhotonX provides photonic chip design, AI model optimization, and enterprise RAG deployment consulting. [1]",
  "sources": [
    { "document": "PhotonX Company Profile.pdf", "page": 2, "section": "Services" }
  ]
}
```

When the answer isn't in the docs:

```json
{
  "answer": "I couldn't find that information in the PhotonX documentation.",
  "sources": []
}
```

---

## API Reference

| Method | Path       | Description                                            |
|--------|------------|--------------------------------------------------------|
| POST   | `/ingest`  | Index every PDF in `docs/`. Returns doc & chunk counts.|
| POST   | `/ask`     | Answer a question from the indexed docs, with sources. |
| GET    | `/health`  | Liveness probe. Returns `{"status": "ok"}`.            |
| GET    | `/stats`   | Index stats: collection, #docs, #chunks, models.       |

**Error responses** use a uniform `{"detail": "..."}` envelope:

| Status | Meaning                                             |
|--------|-----------------------------------------------------|
| 400    | No/empty/unreadable PDFs at ingest time.            |
| 422    | Invalid request body (e.g. empty question).         |
| 500    | Vector store failure.                               |
| 502    | Upstream OpenAI (embedding or LLM) failure.         |

---

## How It Works (Design Notes)

- **Clean architecture & SOLID.** Each RAG concern is a single-responsibility
  class depending on abstractions (e.g. a `DocumentLoader` protocol), injected
  via constructors. The API layer is thin; orchestration lives in services; all
  concrete wiring happens once in the `Container` composition root.
- **Configuration.** `pydantic-settings` gives typed, validated settings loaded
  from `.env` exactly once (`get_settings()` is cached). No module reads
  `os.environ` directly.
- **Chunking.** Text is first split on detected headings (so a chunk rarely
  mixes topics and each carries its `section`), then packed into
  `800`-char windows with `150`-char overlap, preferring sentence boundaries.
- **Cosine similarity.** The ChromaDB collection is created with
  `hnsw:space = cosine`; distances are converted to a `[0, 1]` similarity score.
- **Robustness.** Embeddings are batched with exponential-backoff retries;
  one unreadable PDF is skipped rather than aborting a batch ingest.
- **Logging.** PDF loading, chunk creation, embedding generation, vector
  insertion, retrieval, and LLM response time are all logged.

---

## Future Improvements

The pipeline is intentionally decomposed so these extensions are additive:

- **More formats** — add `DocxLoader` / `MarkdownLoader` / `HTMLLoader` that
  yield the same `PageContent`; nothing downstream changes.
- **Website crawling** — a crawler feeding the same loader interface.
- **Conversation memory** — carry prior turns into the prompt / session store.
- **Hybrid search** — combine BM25/keyword scores with vector similarity in the
  retriever.
- **Re-ranking** — insert a cross-encoder re-ranker between retrieval and the LLM.
- **Streaming responses** — stream `/ask` tokens via server-sent events.
- **Auth & rate limiting** — API keys, quotas, per-tenant collections.
- **Evaluation harness** — automated groundedness / answer-quality tests.
```
