# PhotonX RAG — Engineering Documentation

**Two ways to build the same Q&A engine.**

A complete, stage-by-stage guide to the PhotonX documentation assistant — built
once **by hand** (no framework), then rebuilt on **LangChain**. Every pipeline
stage is explained once, then compared side by side so you can see exactly what
the framework replaces, and what it costs.

> A styled, browser-ready version of this document lives at
> [`docs-site/PhotonX_RAG_Documentation.html`](docs-site/PhotonX_RAG_Documentation.html) —
> open it in any browser, or email/share the file directly.

**Legend:** 🟦 = Manual build (no framework) · 🟪 = LangChain build (framework)

---

## Table of Contents

- [00 · What the application does](#00--what-the-application-does)
- [01 · Architecture & folder layout](#01--architecture--folder-layout)
- [02 · PDF Loading](#02--pdf-loading)
- [03 · Chunking](#03--chunking)
- [04 · Embeddings](#04--embeddings)
- [05 · Vector Store](#05--vector-store)
- [06 · Retrieval](#06--retrieval)
- [07 · Prompt & LLM](#07--prompt--llm)
- [08 · How it never hallucinates](#08--how-it-never-hallucinates)
- [09 · The verdict](#09--the-verdict)
- [10 · Running & testing both](#10--running--testing-both)

---

## 00 · What the application does

Both applications answer plain-English questions using **only** the content of
the PhotonX documentation PDF, return **citations** to the exact page, and
**never invent answers**. If the answer isn't in the docs, they reply with a
fixed sentence:

```
"I couldn't find that information in the PhotonX documentation."
```

This pattern is called **RAG — Retrieval-Augmented Generation**. Instead of
asking the model to answer from memory (which causes hallucination), we first
*retrieve* the relevant passages from our own documents, then ask the model to
answer using *only* those passages.

```
Question → Find relevant text → Give it to the model → Grounded answer + citations
```

### The two builds

| | 🟦 Manual | 🟪 LangChain |
|---|---|---|
| **Approach** | Built from scratch, no framework | Rebuilt on the framework |
| **Trade** | Max control & understanding; more code | Far less code; less control over internals |

---

## 01 · Architecture & folder layout

Both projects share the **same clean architecture** — the framework only changes
the RAG internals, not how the app is organized. Keeping the structure identical
isolates the one variable we're studying: the pipeline.

| Folder | Responsibility |
|---|---|
| `app/api/` | Thin FastAPI routes — `/ingest`, `/ask`, `/health`, `/stats` |
| `app/core/` | Config (from `.env`), logging, typed exceptions |
| `app/rag/` | **The pipeline** — this is where the two builds differ |
| `app/schemas/` | Pydantic request/response models |
| `app/services/` | Orchestration + dependency-injection container |

The difference lives entirely in `app/rag/`:

**🟦 Manual `app/rag/`** — seven modules, one per stage:
`pdf_loader.py` · `chunker.py` · `embedding.py` · `vector_store.py` ·
`retriever.py` · `prompt.py` · `llm.py`

**🟪 LangChain `app/rag/`** — two modules; the framework supplies the stages:
`pipeline.py` · `prompt.py`

| Metric | Value |
|---|---|
| 🟦 RAG code (manual, 7 files) | **732 lines** |
| 🟪 RAG code (LangChain, 1 file) | **260 lines** |
| Reduction with the framework | **~64% less** |

---

## 02 · PDF Loading

**Job:** read the PDF, pull out the text page by page, and remember which page
each piece came from — that page number becomes the citation later.

| | Approach |
|---|---|
| 🟦 **Manual** | Custom `PDFLoader` (pypdf). Loops over pages, extracts text, skips blank pages, raises a clear error on a scanned/empty PDF. Wrapped behind a `DocumentLoader` protocol so Word/HTML loaders can be added later without touching anything downstream. |
| 🟪 **LangChain** | `PyPDFLoader`. One import — `PyPDFLoader(path).load()` returns a list of `Document` objects, one per page, with the page number already in `metadata`. |

```python
# LangChain — the whole loader
from langchain_community.document_loaders import PyPDFLoader
pages = PyPDFLoader(str(path)).load()   # one Document per page
```

**Alternatives:** `pdfplumber` (complex tables/layout, slower) · `PyMuPDF`
(speed at volume) · OCR/Tesseract (scanned PDFs).

**Why pypdf:** lightweight and reliable for text-based PDFs. The PhotonX profile
is real text, not scanned, so OCR isn't needed.

---

## 03 · Chunking

**Job:** a whole document is too big to feed the model at once, so we split it
into small pieces. Both builds use **800 characters per chunk with 150
characters of overlap** — the *size* is identical. The *strategy* differs.

| | Approach |
|---|---|
| 🟦 **Manual** | Heading-aware + footer-stripping. Strips repeated page boilerplate (company name, URL, "Page 3"), detects real section headings, and tags each chunk with a `section` label. Breaks on sentence boundaries. |
| 🟪 **LangChain** | `RecursiveCharacterTextSplitter`. Splits on a priority list of separators — paragraphs → lines → sentences → words — using the biggest one that keeps chunks under the size limit. Structure-agnostic; no section labels. |

```python
# LangChain — the whole chunker
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(pages)
```

> ⚠️ **This stage caused a real bug in the manual build.** Repeated page footers
> were being treated as content, so valid questions returned "not found". The fix
> — footer-stripping + heading awareness — cut **62 junk chunks down to 14 clean
> ones**. LangChain's generic splitter doesn't do this out of the box; matching
> it means writing a custom splitter, which starts undoing the code savings.

| Chunking type | Trade-off |
|---|---|
| Fixed-size | Simplest; cuts mid-sentence |
| Recursive character 🟪 | Respects paragraph/sentence boundaries |
| Heading-aware 🟦 | Best metadata; most code |
| Semantic | Splits by meaning shift; smart but costly |
| Token-based | Fits model token limits exactly |

---

## 04 · Embeddings

**Job:** convert each chunk into a **vector** — a list of 1536 numbers that
captures its meaning. Chunks about similar topics get similar numbers, which is
what makes *search by meaning* possible. Both builds use OpenAI's
`text-embedding-3-small`.

| | Approach |
|---|---|
| 🟦 **Manual** | Hand-written `Embedder`. Batches up to 128 chunks per call, retries with exponential backoff, preserves input order. ~90 lines of reliability code. |
| 🟪 **LangChain** | `OpenAIEmbeddings`. Batching, retries, and ordering are handled inside it. One line. |

```python
# LangChain — batching & retries included
self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

> ✓ The **same model embeds both the chunks (at ingest) and the question (at
> query time)**, so their vectors live in the same space and can be compared
> directly. That comparison is how "Who started the company?" matches a chunk
> saying "co-founders" — no shared keyword required.

**Why 3-small over 3-large:** best cost/quality balance. Accurate enough for
document Q&A and much cheaper — `large` would be overkill for a company profile.

---

## 05 · Vector Store

**Job:** save all the vectors and, given a new query vector, find the most
similar ones fast. Both builds use **ChromaDB** with **cosine similarity**,
persisted to disk so ingestion happens once.

| | Approach |
|---|---|
| 🟦 **Manual** | Custom `VectorStore` wrapper. Opens a persistent Chroma client, upserts chunk text + embeddings + metadata, converts cosine distance into a clean 0–1 similarity score. |
| 🟪 **LangChain** | `Chroma` (langchain-chroma). Store and embedding function wired together in the constructor. `add_documents()` embeds and stores in a single call. |

```python
# LangChain — store wired to embeddings
self._store = Chroma(
    collection_name="photonx_docs_lc",
    embedding_function=self._embeddings,
    persist_directory="./chroma_db",
    collection_metadata={"hnsw:space": "cosine"},
)
```

**Alternatives:** FAISS (fast in-memory, no metadata filtering) · Pinecone /
Weaviate (managed, cloud-scale, paid) · pgvector (if you already run PostgreSQL).

**Why Chroma:** runs locally, persists to disk, zero infrastructure setup, and
stores metadata (page, section) needed for citations — ideal for a
self-contained app.

---

## 06 · Retrieval

**Job:** embed the incoming question, then pull the **top 5** chunks whose
vectors are closest to it by cosine similarity.

| | Approach |
|---|---|
| 🟦 **Manual** | Custom `Retriever`. Composes the embedder + store, sorts by score, returns text/page/section/score. A clean seam where a re-ranker or hybrid search would slot in later. |
| 🟪 **LangChain** | `store.as_retriever(...)`. One method turns the store into a configured retriever. |

```python
# LangChain — the whole retriever
self._retriever = self._store.as_retriever(
    search_type="similarity", search_kwargs={"k": 5},
)
```

| Retrieval approach | Trade-off |
|---|---|
| Vector / cosine *(both use this)* | Matches meaning, misses exact rare terms |
| Keyword (BM25) | Exact word match, misses synonyms |
| Hybrid | Combines both — best coverage |
| + Re-ranking | Second model reorders for precision |

---

## 07 · Prompt & LLM

**Job:** hand the retrieved chunks to the model with strict instructions, then
generate the answer. Both use `gpt-4.1` at **temperature 0** (factual,
deterministic, no creative drift).

| | Approach |
|---|---|
| 🟦 **Manual** | Hand-built prompt + `qa_service.py`. String-built system/user prompts, then a service that orchestrates retrieve → prompt → call → parse → cite, step by step in plain Python. |
| 🟪 **LangChain** | `ChatPromptTemplate` + LCEL chain. The whole orchestration collapses into one composed chain using the `\|` pipe operator. |

```python
# LangChain — LCEL: the entire orchestration
chain = (
    {"context": retrieve_and_format, "question": RunnablePassthrough()}
    | prompt          # ChatPromptTemplate
    | llm             # ChatOpenAI
    | StrOutputParser()
)
answer = chain.invoke("What services does PhotonX provide?")
```

> ✓ **LCEL** (LangChain Expression Language) is the `a | b | c` syntax that wires
> steps into one runnable. That single `chain.invoke(...)` runs retrieve → format
> → prompt → model → parse. In the manual build, that flow *was* the entire
> `qa_service.py`.

---

## 08 · How it never hallucinates

Both builds enforce the **same three-layer defense** — the framework did not
weaken the safety contract, because we kept this logic ourselves in both.

1. **Strict prompt.** The system prompt forbids outside knowledge and mandates
   the exact refusal sentence when the answer isn't in context.
2. **Empty-retrieval short-circuit.** If nothing relevant is retrieved, we
   return the refusal *without even calling the model*.
3. **Citations from retrieval, never from text.** Sources are built from the
   retrieved chunks — so a citation can never be fabricated by the model.

---

## 09 · The verdict

Both apps were run against the same PDF and the same questions. Both answer
correctly and both refuse out-of-scope questions safely. Here's where they
genuinely differ.

| Dimension | 🟦 Manual (no framework) | 🟪 LangChain |
|---|---|---|
| RAG pipeline code | 732 lines · 7 files | 260 lines · 1 file |
| Embedding batching / retries | Hand-written | ✅ Built in |
| Chunking control | ✅ Full — footer-strip, heading-aware | Generic, size-based |
| Citation metadata | ✅ document · page · **section** | document · page · *section = N/A* |
| Swap model / provider | Rewrite a class | ✅ Swap one class |
| Dependencies | ✅ Lean | Heavy tree |
| Debuggability | ✅ See every step | More indirection |
| Time to build | Slower | ✅ Fast |

> ℹ️ **An honest surprise from testing:** on "What services does PhotonX
> provide?", the two chunkers split the PDF differently and *LangChain happened
> to retrieve a better answer* (the full page-2 service list, where the manual
> build surfaced a page-1 summary). Neither chunker is universally right —
> **chunking strategy is a tuning lever, not a solved problem.**

**Choose 🟦 manual when** retrieval *quality* on your specific documents is the
priority: you need footer control, heading-aware chunking, rich metadata, and
full visibility to debug why a result appeared.

**Choose 🟪 LangChain when** speed and breadth matter: rapid prototyping, easy
provider/store swaps, and built-in features (streaming, memory, agents) you'd
otherwise write yourself.

---

## 10 · Running & testing both

They run side by side — manual on port **8000**, LangChain on **8001**, each with
its own Chroma collection so the indexes never collide.

```bash
# 1 — index the PDFs (run once; re-run after adding documents to docs/)
curl -X POST http://localhost:8000/ingest

# 2 — ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What services does PhotonX provide?"}'
```

### Questions that demo each behavior

| Ask this | Shows |
|---|---|
| What services does PhotonX provide? | ✅ Answers + cites a page |
| Who founded PhotonX? | ✅ Precise page-6 citation |
| What do clients say about PhotonX? | ✅ Testimonials, cited |
| What is the capital of France? | Refuses — proves no hallucination |

**Best demo sequence:** ask services → ask founders → ask capital of France.
That proves the three things that matter — it *answers*, it *cites*, and it
*refuses* rather than making something up.

### Adding more documents

Drop more `.pdf` files into `docs/`, then re-run `POST /ingest`. Both apps scan
the whole folder, so answers will span all documents and each citation names its
source document + page. (Re-ingest rebuilds the index from whatever is currently
in the folder — adding *and* removing files both work.)

---

*PhotonX Technologies · RAG engineering reference · Manual build vs. LangChain
build · Both apps verified end-to-end against the PhotonX company profile PDF.*
