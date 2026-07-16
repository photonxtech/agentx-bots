# 📚 Multi-RAG — Chat with your documents (PDF · Word · PowerPoint · Text · Images)

A **Streamlit** app that ingests a document, builds a searchable knowledge base
from it, and answers your questions using **Groq** — grounded strictly in that
document. Every chat owns exactly **one file**, and each chat can only ever
answer from its own document (per-chat isolation).

The project is deliberately written so the **RAG mechanics are visible**: each
pipeline stage is its own small, readable module instead of being hidden behind
a heavy framework.

---

## 1. Project overview

**RAG = Retrieval-Augmented Generation.** An LLM doesn't know about *your*
files. Instead of fine-tuning, we *retrieve* the relevant pieces of your data at
question time and *augment* the prompt with them, so the model answers from
facts you supplied — not from memory. This kills most hallucination and lets the
app cite sources.

### Pipeline, stage by stage (and the file that implements each)

| # | Stage | What happens | File |
|---|-------|--------------|------|
| 1 | **Ingestion** | PDF (text layer + OCR of embedded/scanned pages) · Word · PowerPoint · text · images → `Document`s. Results cached by content hash. | [rag/ingestion.py](rag/ingestion.py) |
| 2 | **Vision** | Every image (standalone, embedded, or a rendered page) is described by a Groq multimodal model, so charts/tables/photos become searchable — not just their OCR text. | [rag/vision.py](rag/vision.py) |
| 3 | **Chunking** | Split long text into sentence-aware, overlapping windows. | [rag/chunking.py](rag/chunking.py) |
| 4 | **Embedding** | Each chunk → a vector via a **local** model (free, offline). | [rag/embeddings.py](rag/embeddings.py) |
| 5 | **Indexing / Retrieval** | Store vectors in **Weaviate** (Chroma fallback); **hybrid** search = semantic (cosine) + keyword (BM25). | [rag/vectorstore.py](rag/vectorstore.py) |
| 6 | **Generation** | Stuff top-k chunks into a grounded prompt → Groq (streamed). Follow-up questions are first rewritten into standalone queries. | [rag/generator.py](rag/generator.py) |
| — | **UI + chat store** | Upload, index, chat, per-chat history, live metrics. | [app.py](app.py), [chat_store.py](chat_store.py) |

> **"Multi" here = multi-format, multi-modal ingestion.** Text, tables, and
> images (via OCR **and** a vision model) all flow into one unified index.

### Key features

- **Multi-format ingestion:** PDF, Word (`.docx`), PowerPoint (`.pptx`), text
  (`.txt`/`.md`), and images (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`).
- **OCR + vision:** EasyOCR reads printed text; a Groq vision model *describes*
  what images actually show. Vision failures degrade gracefully to OCR-only.
- **Hybrid retrieval:** blends embedding similarity with BM25 keyword scoring
  (`HYBRID_ALPHA`), so both meaning (*"who earns the most"*) and exact tokens
  (*"ZEBRA-42"*) are found.
- **Persistent index:** documents survive restarts (Weaviate volume, or
  Chroma under `index_store/`). Per-file cache skips repeat OCR/vision.
- **Per-chat isolation:** each chat has one file; retrieval is scoped so a chat
  never sees another chat's document.
- **Chat memory:** follow-up questions (*"and when is it due?"*) are rewritten
  into standalone search queries using recent history.
- **Live metrics:** rewrite / embed+search / time-to-first-token / generation
  timings shown per answer.

---

## 2. Frameworks used — and *why*

| Concern | Choice | Why this one |
|---|---|---|
| **Frontend** | **Streamlit** | Pure-Python UI, zero JS; file uploader + chat widgets built in. |
| **LLM generation & vision** | **Groq** (`groq` SDK) | Very fast inference on open models; generous free tier; OpenAI-compatible; multimodal models for image understanding. |
| **Embeddings** | **sentence-transformers** (`all-MiniLM-L6-v2`), local | Groq has **no** embedding endpoint, so we embed locally: free, private, offline. |
| **Vector store** | **Weaviate** (self-hosted via Docker) + **Chroma** fallback | Weaviate persists in a Docker volume and survives restarts; if it's unreachable the app automatically falls back to embedded Chroma so nothing breaks. |
| **Keyword search** | **rank-bm25** | Adds exact-token matching alongside embeddings (hybrid search). |
| **PDF parsing** | **PyMuPDF (fitz)** | Extracts the text layer *and* renders scanned pages / pulls embedded images for OCR — no external system install. |
| **Word / PowerPoint** | **python-docx / python-pptx** | Paragraphs, tables, and embedded images from Office files. |
| **OCR** | **EasyOCR** | Pure-Python, offline, no system install (unlike Tesseract). |

### Why *no* LangChain / LlamaIndex?
Both are excellent, but they **hide** the RAG mechanics behind abstractions.
Since a goal here is to *understand RAG deeply*, the pipeline is hand-written and
transparent — each stage is isolated, so swapping in a framework later is easy.

---

## 3. Groq model comparison

Groq serves **open-weight models**. The generator can fetch the **live list**
from Groq at runtime; a curated fallback list lives in [config.py](config.py).
Defaults used by the app:

| Role | Model (config key) | Notes |
|---|---|---|
| **Answers** | `llama-3.3-70b-versatile` (`DEFAULT_MODEL`) | Best all-round quality for grounded RAG answers. |
| **Query rewrite** | `llama-3.1-8b-instant` (`REWRITE_MODEL`) | Small + fast; only rewrites follow-ups into standalone queries. |
| **Vision** | `meta-llama/llama-4-scout-17b-16e-instruct` (`VISION_MODEL`) | Multimodal; describes charts, tables, photos, layouts. |

Other good answer models (in the fallback list): `llama-3.1-8b-instant`
(fastest), `deepseek-r1-distill-llama-70b` (reasoning-heavy), `gemma2-9b-it`
(lightweight). See <https://console.groq.com/docs/models> for the current list.

> ⚠️ **Groq notes:** no embedding models (handled locally), and model IDs change
> over time. To switch providers you'd only need to edit
> [rag/generator.py](rag/generator.py) and [rag/vision.py](rag/vision.py).

---

## 4. Setup

### Prerequisites
- **Python 3.10+**
- **Docker** (optional but recommended, for the default Weaviate backend)
- A **Groq API key** — get a free one at <https://console.groq.com/keys>

### Install

```bash
# 1. create + activate a virtual environment
python -m venv .venv
.venv\Scripts\activate           # Windows (PowerShell/cmd)
# source .venv/bin/activate      # macOS/Linux

# 2. install dependencies
#    (first install is large: pulls PyTorch for OCR + embeddings)
pip install -r requirements.txt

# 3. add your Groq key (see next section)
copy .env.example .env           # Windows   (cp .env.example .env on macOS/Linux)
# then edit .env and paste YOUR key
```

---

## 5. Required environment variables

Configuration comes from two places:

- **Secrets** live in a `.env` file (loaded via `python-dotenv`).
- **Tunables** (models, chunk size, top-k, backend, ports) live in
  [config.py](config.py) — edit that file directly, no env var needed.

`.env` requires a single variable:

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Your Groq API key (starts with `gsk_...`). Get it at <https://console.groq.com/keys>. Used for answers, query rewriting, and vision. |

Example `.env`:

```dotenv
GROQ_API_KEY=gsk_your_key_here
```

> 🔐 **Security:** `.env` is git-ignored — never commit real keys. If a key is
> ever exposed, **rotate it** in the Groq console. See the "Security note"
> at the bottom of this file.

---

## 6. Running the project locally

### Option A — with Weaviate (recommended, persistent)

```bash
# 1. start Weaviate (persists data in a Docker volume)
docker compose up -d

# 2. run the app
streamlit run app.py
```

Weaviate management:
```bash
docker compose down       # stop, keep data
docker compose down -v    # stop and WIPE the index volume
```

### Option B — without Docker (automatic Chroma fallback)

Just run the app — if Weaviate isn't reachable it automatically falls back to
embedded Chroma (data stored under `index_store/chroma`):

```bash
streamlit run app.py
```

To force Chroma even when Weaviate is available, set
`VECTOR_BACKEND = "chroma"` in [config.py](config.py).

The app opens at **<http://localhost:8501>**. The first question is slower:
EasyOCR and the embedding model download their weights once, then cache.

### Optional — sanity-check the pipeline from the CLI

```bash
python scripts/test_pipeline.py    # ingest → index → retrieve → (generate)
```
> Note: this script references sample files under `samples/`; provide your own
> or adjust the paths at the top of the script.

---

## 7. API endpoints

**This app has no custom REST/HTTP API of its own** — it is an interactive
Streamlit UI served on **`http://localhost:8501`**. It talks to two services:

| Service | Where | Port(s) | Purpose |
|---|---|---|---|
| **Streamlit UI** | local | `8501` (HTTP) | The app you interact with. |
| **Weaviate** | local Docker | `8085` → container `8080` (REST), `50051` (gRPC) | Vector store. Ports set in [docker-compose.yml](docker-compose.yml) / [config.py](config.py). Anonymous access is enabled for local dev. |
| **Groq API** | external | HTTPS | LLM generation, query rewrite, and vision (requires `GROQ_API_KEY`). |

---

## 8. Project layout

```
mutlirag/
├── app.py               # Streamlit UI: upload → index → chat, per-chat history
├── chat_store.py        # JSON-backed chat sessions + recency grouping
├── config.py            # all tunables (models, chunking, retrieval, backend, ports)
├── requirements.txt
├── docker-compose.yml   # self-hosted Weaviate
├── .env.example         # template for GROQ_API_KEY
├── rag/
│   ├── ingestion.py     # PDF / DOCX / PPTX / text / image → Documents (+ cache)
│   ├── vision.py        # Groq multimodal image descriptions
│   ├── chunking.py      # sentence-aware overlapping chunker
│   ├── embeddings.py    # local sentence-transformers
│   ├── vectorstore.py   # Weaviate (+ Chroma fallback) + hybrid BM25 search
│   └── generator.py     # Groq call, live model list, follow-up rewriting
├── scripts/
│   ├── make_fixtures.py # generate sample files
│   └── test_pipeline.py # end-to-end CLI smoke test
├── extracted_images/    # (generated) images pulled from PDFs/DOCX/PPTX
└── index_store/         # (generated) Chroma data + chat_history.json + cache
```

---

## 9. Dependencies & additional notes

Full pinned list in [requirements.txt](requirements.txt). Highlights:

- **Heavy first install:** `sentence-transformers` and `easyocr` pull in
  **PyTorch** (hundreds of MB). Model weights download once, then cache.
- **CPU by default:** OCR/embeddings run on CPU (`gpu=False`). A GPU will speed
  up OCR and embedding but isn't required.
- **`.venv/` is committed here** but git-ignored by `.gitignore`; recreate it
  from `requirements.txt` rather than relying on it.
- `pandas` / `openpyxl` are listed in requirements, but **Excel/CSV ingestion is
  not currently wired into `ingest()`** — supported upload types are PDF, DOCX,
  PPTX, TXT/MD, and images (see `UPLOAD_TYPES` in [app.py](app.py)).

### Tuning (in [config.py](config.py))
| Setting | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Chunk window & overlap (chars). |
| `TOP_K` | `10` | Base chunks retrieved (auto-scales with #sources). |
| `HYBRID_ALPHA` | `0.65` | Blend: `1.0` = pure semantic, `0.0` = pure BM25. |
| `VECTOR_BACKEND` | `"weaviate"` | `"weaviate"` or `"chroma"`. |
| `VISION_ENABLED` | `True` | Set `False` for OCR-only (faster, no vision API calls). |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Swap for e.g. `BAAI/bge-small-en-v1.5`. |

### Ideas to extend
- **Reranking:** add a cross-encoder to reorder top-k before generation.
- **Excel/CSV:** re-add a loader in `ingestion.py` — the rest is unchanged.
- **Multi-file chats:** relax the one-file-per-chat constraint.

---

## 🔐 Security note

`.env` is git-ignored so real keys stay out of version control. **`.env.example`
is meant to be committed and must contain only a placeholder** — if a real key
was ever committed to it (or to `.env`), treat that key as compromised and
**rotate it immediately** at <https://console.groq.com/keys>. Never paste live
keys into example/template files.
</content>
</invoke>
