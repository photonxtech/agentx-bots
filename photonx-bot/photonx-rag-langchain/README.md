# PhotonX Documentation Assistant — LangChain Edition

The **same** RAG Q&A service as the manual project ([`../photonx-rag`](../photonx-rag)),
rebuilt on **LangChain** to see how a framework abstracts each component we
wrote by hand. Same PDF, same API, same anti-hallucination guarantee — but the
pipeline is now assembled from LangChain building blocks instead of custom code.

---

## The whole point: what replaces what

Every hand-written file in the manual project maps to a single LangChain
abstraction here. This is the comparison to internalize:

| Manual project (`../photonx-rag`) | LangChain edition (this project) | Where |
|---|---|---|
| `rag/pdf_loader.py` (~90 lines) | `PyPDFLoader` | `rag/pipeline.py` |
| `rag/chunker.py` (~230 lines) | `RecursiveCharacterTextSplitter` | `rag/pipeline.py` |
| `rag/embedding.py` (batching, retries) | `OpenAIEmbeddings` | `rag/pipeline.py` |
| `rag/vector_store.py` | `Chroma` (`langchain-chroma`) | `rag/pipeline.py` |
| `rag/retriever.py` | `store.as_retriever(...)` | `rag/pipeline.py` |
| `rag/prompt.py` | `ChatPromptTemplate` | `rag/prompt.py` |
| `rag/llm.py` | `ChatOpenAI` | `rag/pipeline.py` |
| `services/qa_service.py` (orchestration) | an **LCEL chain** (`a \| b \| c`) | `rag/pipeline.py` |

**~640 lines of RAG logic → ~1 file of framework calls.**

### The LCEL chain (the "magic")

LangChain Expression Language uses the `|` operator to compose steps into one
runnable:

```python
chain = (
    {"context": retrieve_and_format, "question": RunnablePassthrough()}
    | prompt          # ChatPromptTemplate
    | llm             # ChatOpenAI
    | StrOutputParser()
)
answer = chain.invoke("What services does PhotonX provide?")
```

That single `chain.invoke(...)` runs: retrieve → format context → fill prompt →
call the model → parse to a string. In the manual project, that flow was the
whole of `qa_service.py`.

---

## What stayed the same (and why)

These were kept identical on purpose, to isolate the variable (the RAG pipeline)
and prove the framework didn't force us to give anything up:

- **Clean architecture** — `api/` (thin routes), `services/` (container),
  `core/` (config, logging, exceptions), `schemas/` (Pydantic models).
- **Typed config** from `.env` via `pydantic-settings`.
- **Anti-hallucination** — strict prompt, refuse on empty retrieval, and
  **citations built from retrieved docs** (never parsed as fact from the model).
- **Same REST API** — `POST /ingest`, `POST /ask`, `GET /health`, `GET /stats`.

---

## What LangChain gave us — and what it cost

**Gained**
- **Far less code.** No hand-written batching/retry/ordering for embeddings; no
  manual Chroma wiring; no manual chain orchestration.
- **Swappability.** Change `ChatOpenAI` → another provider, or `Chroma` →
  another store, by swapping one class.
- **Built-in features.** Splitters, retrievers, output parsers, streaming,
  memory, agents — all one import away.

**Cost / trade-offs (observed while building this)**
- **Less control over chunking.** The manual project's chunker strips repeated
  page footers and is heading-aware — the fixes that solved our real retrieval
  bugs. `RecursiveCharacterTextSplitter` is generic; to replicate those fixes
  we'd pre-clean text or write a custom splitter (which starts undoing the
  savings).
- **Weaker metadata.** `PyPDFLoader` gives `page` but not our `section` label,
  so sources here show `section: "N/A"`. Richer metadata means custom loaders.
- **Heavier dependency tree** and faster-moving APIs than the manual stack.
- **More indirection.** When retrieval returns junk, it's harder to see *why*
  inside the framework than in code you wrote.

**Takeaway:** LangChain is the right call for speed, breadth, and easy provider
swaps. The manual approach wins when you need tight control over
chunking/metadata/retrieval quality — exactly the levers we had to tune on the
real PhotonX PDF.

---

## Setup & Run

> This project shares the manual project's virtualenv to save disk space; it
> only adds the `langchain*` packages on top. If you prefer an isolated env,
> create your own `.venv` and `pip install -r requirements.txt`.

```bash
# From this folder, using the shared interpreter:
cp .env.example .env            # add your OPENAI_API_KEY (collection is photonx_docs_lc)
# PDFs already in docs/

uvicorn app.main:app --reload --port 8001
```

Then:

```bash
curl -X POST http://localhost:8001/ingest
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What services does PhotonX provide?"}'
```

> Runs on **port 8001** so it can run alongside the manual project (port 8000).
> It also uses a separate Chroma collection (`photonx_docs_lc`), so the two
> indexes never collide.

---

## Verified behavior (matches the manual project)

| Question | Result |
|---|---|
| "What services does PhotonX provide?" | Full, grounded service list, cited `[5]`, source = page 2 |
| "Who founded PhotonX?" | "Prathik Gadde and Chaitanya Y…", cited `[2]`, source = page 6 |
| "What is the capital of France?" | Exact refusal, no sources |

---

## Folder Structure

```
photonx-rag-langchain/
├── app/
│   ├── api/            # thin FastAPI routes (ingest, ask, health, stats)
│   ├── core/           # config, logger, exceptions (reused, framework-agnostic)
│   ├── rag/
│   │   ├── pipeline.py # ← the whole RAG pipeline, in LangChain
│   │   └── prompt.py   # ChatPromptTemplate
│   ├── schemas/        # Pydantic request/response models (reused)
│   ├── services/       # container (composition root)
│   └── main.py
├── docs/               # the PhotonX PDF
├── chroma_db/          # persistent vectors (collection: photonx_docs_lc)
├── requirements.txt
└── .env.example
```
