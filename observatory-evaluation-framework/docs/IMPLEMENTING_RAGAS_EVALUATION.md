# Implementing Real RAGAS Evaluation (Step 2: Configure → Evaluate)

A self-contained implementation guide for the evaluation half of a RAG-evaluation
tool: take a pipeline configuration, run a real RAG pipeline with it, score the
output with the genuine `ragas` library, log to LangSmith, persist, and display.

Upload this file into a chat session and ask for it to be implemented. It is
written to be followed without access to the original codebase.

---

## 0. What you are building

```
config form → RAG pipeline → ragas scoring → LangSmith → Postgres → UI
              (local CPU)     (LLM judge)     (optional)
```

Four metrics, scored over the whole test set as one batch:

| Metric | Measures | Needs ground truth? |
|---|---|---|
| `faithfulness` | Fraction of answer claims supported by retrieved context (hallucination) | no |
| `answer_relevancy` | Whether the answer addresses the question | no |
| `context_precision` | **Rank-aware** — is the relevant chunk ranked near the top? | yes |
| `context_recall` | Fraction of the reference answer covered by retrieved context | yes |

Inputs required per test case: `question`, `retrieved_contexts`, `generated_answer`,
`expected_answer` (the reference / ground truth).

---

## 1. Environment — read this before installing anything

**Python 3.12 is required.** Not 3.13, not 3.14. `ragas` needs the langchain
0.3.x line, which uses the pydantic-v1 compatibility shim that breaks on
Python ≥ 3.14.

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux
```

### requirements.txt — these pins are load-bearing

```
# Web / API
fastapi==0.115.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9
aiofiles==24.1.0

# Database
sqlalchemy==2.0.35
psycopg[binary]==3.3.4

# Document parsing
PyPDF2==3.0.1
python-docx==1.1.2

# RAG pipeline — fastembed supplies BOTH embeddings and ONNX rerankers,
# so no torch is needed anywhere.
fastembed==0.8.0
chromadb==1.5.9
rank-bm25==0.2.2

# RAGAS + LangSmith
# DO NOT bump the langchain pins without testing. langchain-community >= 0.4
# removed chat_models.vertexai, which ragas 0.4.3 imports unconditionally at
# package import time -> ModuleNotFoundError before any of your code runs.
ragas==0.4.3
# Needed by ragas.testset (string-distance node filtering). NOT a base ragas
# dependency -- without it testset generation raises ImportError before any
# LLM call is made.
rapidfuzz==3.14.5
langchain==0.3.27
langchain-core==0.3.86
langchain-community==0.3.31
langchain-groq==0.3.4
langchain-openai==0.3.35
langsmith==0.10.17

# Misc — httpx 0.28.x is required by the ragas/langchain stack
httpx==0.28.1
python-dotenv==1.0.1
pandas==3.0.5
```

Installed footprint: **~0.8 GB** (`pyarrow` 90 MB, `scipy` 114 MB, `pandas`
63 MB are the bulk). Model weights download at runtime on top of that
(67 MB – 2.29 GB each, cached after first use).

> **Deployment warning.** This will fail on small/free-tier containers — a known
> failure mode is pip dying mid-install while the app still boots, producing
> `ModuleNotFoundError: ragas` on every request. Use a container with real disk
> and bake the model weights into the image rather than downloading at cold start.

### .env

```env
DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/rag_eval
UPLOAD_DIR=./uploads
RAG_INDEX_CACHE_DIR=./.rag_index_cache

# Required — drives generation and judging
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile

# The RAGAS judge. Kept distinct from the generator automatically.
RAGAS_JUDGE_MODEL=openai/gpt-oss-120b

# Optional — tracing only, computes nothing
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=rag-eval-ui
```

---

## 2. The four gotchas — each of these costs hours to diagnose

Fix these up front; they are not optional.

### 2.1 `ragas` import fails before your code runs

```
ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
```

`ragas` declares `langchain-community` with no upper bound, so a fresh install
pulls 0.4.x, which deleted that module. **Fix:** pin `langchain-community==0.3.31`.

### 2.2 Groq rejects `n > 1`, silently NaN-ing answer_relevancy

```
BadRequestError: 400 - {'error': {'message': "'n' : number must be at most 1"}}
```

`ResponseRelevancy` defaults to `strictness=3`, which requests 3 completions.
Groq caps at 1. **Fix:** `ResponseRelevancy(..., strictness=1)`. Note this makes
the metric noisier than standard RAGAS — document it wherever you report scores.

### 2.3 ragas crashes on langchain's FastEmbedEmbeddings

```
ValidationError: 1 validation error for EmbeddingUsageEvent
model  Input should be a valid string [input_type=TextEmbedding]
```

ragas 0.4.3 copies `embeddings.model` into a pydantic telemetry event typed
`str`; `langchain_community.embeddings.FastEmbedEmbeddings` stores the model
*object* there. Fails mid-run, NaNs the metric. **Fix:** use the thin wrapper in
§4.1 that keeps `model` a plain string.

### 2.4 ragas drives its own asyncio loop

`ragas.evaluate()` is synchronous and starts its own event loop, which clashes
with FastAPI's. **Fix:** call it via `asyncio.to_thread(...)` so it runs on a
thread with no loop attached.

---

## 3. Stage 1 — the RAG pipeline

### 3.1 Shared document reader

Put text extraction in **one module** used by both QA generation and indexing.
If they disagree, the ground truth can reference text the retriever never saw,
and every metric measures that mismatch instead of the pipeline.

```python
# services/document_reader.py
import os

def _iter_docx_blocks(document):
    """Yield paragraphs and tables in true document order.

    python-docx exposes .paragraphs and .tables as separate flat lists, so the
    obvious "\n".join(p.text for p in doc.paragraphs) SILENTLY DROPS EVERY
    TABLE. Report-style documents keep most of their substance in tables --
    measured losses of 20% and 95% on real files.
    """
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    from docx.oxml.ns import qn
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _read_docx(path: str) -> str:
    import docx
    from docx.table import Table
    document = docx.Document(path)
    parts = []
    for block in _iter_docx_blocks(document):
        if isinstance(block, Table):
            for row in block.rows:
                cells = [c.text.strip() for c in row.cells]
                # merged cells repeat their text across the span
                deduped = [c for i, c in enumerate(cells) if i == 0 or c != cells[i - 1]]
                line = " | ".join(c for c in deduped if c)
                if line:
                    parts.append(line)
        else:
            if block.text.strip():
                parts.append(block.text.strip())
    return "\n".join(parts)


def read_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        import PyPDF2
        with open(path, "rb") as f:
            r = PyPDF2.PdfReader(f)
            return "\n".join(p.extract_text() for p in r.pages if p.extract_text()).strip()
    if ext == ".docx":
        return _read_docx(path).strip()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()
```

**Verify immediately:** print `len(read_document(path))` before and after. On a
table-heavy `.docx` the fix took one file from 306 → 6,460 characters (1 chunk → 14).

### 3.2 Chunk → embed → index

```python
def _chunk_document(text, chunk_size, chunk_overlap):
    """Sliding window snapped to paragraph/sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    chunks, start, n = [], 0, len(text)
    step = max(chunk_size - chunk_overlap, 1)
    while start < n:
        end = min(start + chunk_size, n)
        b = text.rfind("\n\n", start, end)
        if b == -1 or b <= start + chunk_size // 2:
            b = text.rfind(". ", start, end)
        if b == -1 or b <= start:
            b = end
        else:
            b += 1
        if text[start:b].strip():
            chunks.append(text[start:b].strip())
        if b >= n:
            break
        start += step
    return chunks
```

Build both a dense and a sparse index, and cache the bundle:

```python
_INDEX_CACHE: dict[str, dict] = {}

def _index_cache_key(path, config) -> str:
    # ONLY fields that change the index. Including vectordb/index_type here
    # would force pointless rebuilds, since they do not affect retrieval.
    raw = f"{path}|{config.chunk_size}|{config.chunk_overlap}|{resolve_embedding(config)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def _build_or_get_index(path, config):
    key = _index_cache_key(path, config)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]

    from fastembed import TextEmbedding
    from rank_bm25 import BM25Okapi
    import chromadb

    chunks = _chunk_document(read_document(path), config.chunk_size, config.chunk_overlap)
    if not chunks:
        raise RuntimeError("Document produced no chunks.")

    embedder = TextEmbedding(model_name=resolve_embedding(config))
    embeddings = list(embedder.embed(chunks))

    client = chromadb.PersistentClient(path=INDEX_CACHE_DIR)
    try:
        client.delete_collection(f"idx_{key}")
    except Exception:
        pass
    col = client.create_collection(f"idx_{key}")
    col.add(ids=[str(i) for i in range(len(chunks))],
            embeddings=[e.tolist() for e in embeddings], documents=chunks)

    bundle = {"chunks": chunks, "embedder": embedder, "collection": col,
              "bm25": BM25Okapi([_tokenize(c) for c in chunks])}
    _INDEX_CACHE[key] = bundle
    return bundle
```

### 3.3 Retrieve — keyword / semantic / hybrid

```python
def _reciprocal_rank_fusion(rank_lists, k=60):
    scores = {}
    for ranked in rank_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def _retrieve(question, index, config, notes):
    chunks, top_k = index["chunks"], config.top_k
    candidate_k = min(max(top_k * 3, top_k), len(chunks))
    dense, sparse = [], []

    if config.search_type in ("semantic", "hybrid"):
        q = list(index["embedder"].embed([question]))[0].tolist()
        dense = [int(i) for i in index["collection"].query(
            query_embeddings=[q], n_results=candidate_k)["ids"][0]]

    if config.search_type in ("keyword", "hybrid"):
        s = index["bm25"].get_scores(_tokenize(question))
        sparse = [i for i, _ in sorted(enumerate(s), key=lambda x: x[1],
                                       reverse=True)[:candidate_k]]

    if config.search_type == "semantic":
        ids = dense[:top_k]
    elif config.search_type == "keyword":
        ids = sparse[:top_k]
    else:
        ids = _reciprocal_rank_fusion([dense, sparse])[:top_k]

    retrieved = [chunks[i] for i in ids]

    # Optional rerank. Use fastembed's ONNX cross-encoders -- NOT FlagEmbedding,
    # which pulls ~2.5 GB of torch. Reranking reorders the chunks already
    # retrieved, so it can raise context_precision but by construction
    # CANNOT change context_recall.
    if config.reranker_model and retrieved:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            rr = _get_cached_reranker(config.reranker_model)   # cache per model!
            scores = list(rr.rerank(question, retrieved))
            retrieved = [c for _, c in sorted(zip(scores, retrieved),
                                              key=lambda x: x[0], reverse=True)]
        except Exception as e:
            notes.append(f"Reranker failed ({e}) — results NOT reranked.")

    return retrieved
```

### 3.4 Generate the answer

```python
PROMPT = """Answer the question using ONLY the provided context. Be concise and factual.
If the context does not contain the answer, say so explicitly.

Context:
{context}

Question: {question}

Answer:"""

async def _generate_answer(question, contexts, config, model):
    async with httpx.AsyncClient(timeout=90.0) as client:
        res = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": model,
                  "messages": [{"role": "user", "content": PROMPT.format(
                      context="\n---\n".join(contexts) or "(no relevant context found)",
                      question=question)}],
                  "temperature": config.temperature},
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()
```

Assemble one dict per test case — these keys are what ragas consumes:

```python
results.append({
    "question": qa["question"],
    "generated_answer": answer,
    "expected_answer": qa["answer"],   # ground truth
    "contexts": contexts,
})
```

---

## 4. Stage 2 — RAGAS scoring

### 4.1 The embeddings wrapper (works around gotcha 2.3)

```python
class _FastEmbedLangchain:
    """Minimal LangChain-compatible Embeddings over fastembed.

    `model` MUST be a plain string: ragas 0.4.3 copies this attribute into a
    pydantic telemetry event declared as str. langchain_community's
    FastEmbedEmbeddings stores the TextEmbedding object there, which raises
    ValidationError part-way through evaluation.
    """

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding
        self.model = model_name                 # str, for ragas telemetry
        self._embedder = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts):
        return [e.tolist() for e in self._embedder.embed(list(texts))]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts):
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text):
        return await asyncio.to_thread(self.embed_query, text)
```

### 4.2 Judge must never be the generator

A model grading its own output inflates faithfulness and relevancy. Enforce it
structurally and report the substitution.

```python
def _resolve_models(config, notes):
    generator = config.chat_model if config.chat_model in GROQ_CHAT_MODELS else GROQ_MODEL
    judge = RAGAS_JUDGE_MODEL
    if judge == generator:
        judge = "llama-3.3-70b-versatile" if generator != "llama-3.3-70b-versatile" \
                else "openai/gpt-oss-120b"
        notes.append(f"Judge matched the generator — judged with '{judge}' "
                     f"instead to avoid self-evaluation bias.")
    return generator, judge
```

> Do **not** offer agentic models with built-in web search (e.g. `groq/compound`)
> as the generator. They can answer from the internet instead of the retrieved
> context, silently invalidating `faithfulness` and `context_recall`.

### 4.3 The scoring call

```python
RAGAS_METRIC_COLUMNS = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
}


def _clean_score(value):
    """ragas returns NaN when a judge call fails. Keep those as None so they are
    EXCLUDED from the average rather than counted as zero."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)


def _run_ragas_sync(results, judge_model, embedding_model_name):
    """Blocking — call via asyncio.to_thread (gotcha 2.4)."""
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import (Faithfulness, ResponseRelevancy,
                               LLMContextPrecisionWithReference, LLMContextRecall)
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig as RagasRunConfig
    from langchain_groq import ChatGroq

    llm = LangchainLLMWrapper(ChatGroq(model=judge_model,
                                       api_key=GROQ_API_KEY, temperature=0.0))
    embeddings = LangchainEmbeddingsWrapper(_FastEmbedLangchain(embedding_model_name))

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=r.get("contexts") or [],
            response=r.get("generated_answer") or "",
            reference=r.get("expected_answer") or "",
        )
        for r in results
    ]

    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings, strictness=1),  # gotcha 2.2
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]

    # Groq free tier is ~30 req/min and ragas issues several calls per sample
    # per metric. Stay serial and let it retry rather than stampede.
    run_config = RagasRunConfig(max_workers=1, timeout=180, max_retries=5, max_wait=90)

    df = evaluate(dataset=EvaluationDataset(samples=samples), metrics=metrics,
                  llm=llm, embeddings=embeddings, run_config=run_config,
                  show_progress=False).to_pandas()

    per_sample = []
    for i in range(len(results)):
        per_sample.append({name: _clean_score(df[col].iloc[i])
                           for col, name in RAGAS_METRIC_COLUMNS.items()
                           if col in df.columns})

    aggregate, scored_counts = {}, {}
    for _, name in RAGAS_METRIC_COLUMNS.items():
        vals = [r[name] for r in per_sample if r.get(name) is not None]
        scored_counts[name] = len(vals)
        aggregate[name] = round(sum(vals) / len(vals), 4) if vals else None

    return {"aggregate": aggregate, "per_sample": per_sample,
            "scored_counts": scored_counts}
```

Which fields each metric reads — useful for knowing what a low score implicates:

| Metric | question | contexts | answer | reference |
|---|:--:|:--:|:--:|:--:|
| `Faithfulness` | ✓ | ✓ | ✓ | — |
| `ResponseRelevancy` | ✓ | — | ✓ | — |
| `LLMContextPrecisionWithReference` | ✓ | ✓ | — | ✓ |
| `LLMContextRecall` | ✓ | ✓ | — | ✓ |

### 4.4 Never hide partial coverage

```python
for name, count in scored_counts.items():
    if count and count < len(results):
        notes.append(f"{name}: only {count} of {len(results)} test cases scored "
                     f"(judge rate-limit) — the average covers those {count}.")
```

Attach provenance so any reported number is reproducible:

```python
batch_metrics = {
    **aggregate,
    "_meta": {
        "num_test_cases": len(results),
        "scored_counts": scored_counts,
        "generator_model": generator_model,
        "judge_model": judge_model,
        "embedding_model": embedding_model_name,
        "framework": "ragas",
        "notes": notes,
    },
}
```

---

## 5. Stage 3 — LangSmith (records, does not compute)

LangSmith has **no RAGAS implementation**. It stores runs and feedback. Skip
gracefully when no key is set.

```python
def log_to_langsmith(session_id, run_config_id, results, qa_pairs, aggregate):
    if not LANGCHAIN_API_KEY:
        return None
    from langsmith import Client
    client = Client(api_key=LANGCHAIN_API_KEY)

    dataset_name = f"rag-eval-session-{str(session_id)[:8]}"
    try:
        ds = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(dataset_id=ds.id, examples=[
            {"inputs": {"question": qa["question"]},
             "outputs": {"expected_answer": qa["answer"]}} for qa in qa_pairs])
    except Exception:
        pass  # already exists

    project_name = f"rag-eval-run-{str(run_config_id)[:8]}"
    now = datetime.now(timezone.utc)

    for r in results:
        rid = str(uuid.uuid4())
        client.create_run(id=rid, name="rag-pipeline", run_type="chain",
                          project_name=project_name,
                          inputs={"question": r["question"]},
                          outputs={"answer": r["generated_answer"],
                                   "contexts": r["contexts"]},
                          start_time=now, end_time=now)
        for k, v in (r.get("metrics") or {}).items():
            if isinstance(v, (int, float)):
                # NOTE: do NOT pass project_id alongside run_id -- the API
                # rejects it: "project_id cannot be provided if run_id ... is provided"
                client.create_feedback(run_id=rid, key=k, score=float(v))

    # one summary run carrying the batch aggregate
    sid = str(uuid.uuid4())
    client.create_run(id=sid, name="ragas-batch-summary", run_type="chain",
                      project_name=project_name,
                      inputs={"num_samples": len(results)}, outputs=aggregate,
                      start_time=now, end_time=now)
    for k, v in aggregate.items():
        if isinstance(v, (int, float)):
            client.create_feedback(run_id=sid, key=f"batch_{k}", score=float(v))

    try:
        return str(client.read_project(project_name=project_name).url)
    except Exception:
        return "https://smith.langchain.com"
```

---

## 6. Persistence

```python
# Batch metrics belong on the run config, not on individual results.
class RunConfig(Base):
    ...
    langsmith_experiment_url = Column(String(1000), nullable=True)
    metrics = Column(JSONB, nullable=True)   # aggregate + _meta
```

Idempotent migration on startup:

```python
migrations = [
    "ALTER TABLE run_configs ADD COLUMN IF NOT EXISTS metrics JSONB",
    "ALTER TABLE run_configs ADD COLUMN IF NOT EXISTS langsmith_experiment_url VARCHAR(1000)",
]
```

---

## 7. The endpoint

Keep the router thin — orchestrate and persist, never compute.

```python
@router.post("/evaluate", response_model=RunConfigOut)
async def run_evaluation(session_id: UUID, payload: EvaluateRequest,
                         db: DBSession = Depends(get_db)):
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(404, "Session not found")
    if not session.qa_json:
        raise HTTPException(400, "No QA data generated yet — complete Step 1 first")

    run_config = RunConfig(session_id=session_id, **payload.model_dump())
    db.add(run_config)
    db.flush()                      # need the id before evaluating

    try:
        results, batch_metrics, ls_url = await call_rag_api(
            config=payload, qa_pairs=session.qa_json,
            document_path=session.document_path,
            session_id=str(session_id), run_config_id=str(run_config.id))
    except Exception as e:
        db.rollback()
        raise HTTPException(502, f"Evaluation pipeline failed: {e}")

    run_config.metrics = batch_metrics
    if ls_url:
        run_config.langsmith_experiment_url = ls_url
    for r in results:
        db.add(RunResult(run_config_id=run_config.id, question=r["question"],
                         generated_answer=r["generated_answer"],
                         expected_answer=r["expected_answer"],
                         metrics=r.get("metrics")))
    db.commit()

    return db.query(RunConfig).options(joinedload(RunConfig.results)) \
             .filter(RunConfig.id == run_config.id).first()
```

> **Production note:** this is synchronous. A 50-case run holds the HTTP
> connection for many minutes. Move it behind a job queue before deploying.

---

## 8. Display rules

Three rules that keep the UI honest:

1. **Show the batch aggregate as the headline.** Per-config comparison is the point.
2. **Show coverage when it is short.** Render `4 of 10 cases` in a warning colour
   whenever `scored_counts[metric] < num_test_cases`. An average over a subset
   must never look like an average over everything.
3. **Show provenance.** Generator, judge, and embedding model, so a number can
   be reproduced.

Filter `null` metrics out rather than rendering them as `0%`.

---

## 9. Verification checklist

Run these in order. Do not skip #2 — it is the only real proof.

**1. Import check**

```python
from ragas.metrics import Faithfulness, ResponseRelevancy, \
    LLMContextPrecisionWithReference, LLMContextRecall
print(LLMContextPrecisionWithReference().get_required_columns())
# expect: {'SINGLE_TURN': {'user_input', 'reference', 'retrieved_contexts'}}
```

**2. Discrimination test** — one grounded sample, one deliberately hallucinated:

```python
samples = [
    SingleTurnSample(
        user_input="What embedding model does the system default to?",
        retrieved_contexts=["The pipeline defaults to the BAAI/bge-small-en-v1.5 embedding model."],
        response="It defaults to BAAI/bge-small-en-v1.5.",
        reference="The default embedding model is BAAI/bge-small-en-v1.5."),
    SingleTurnSample(
        user_input="What embedding model does the system default to?",
        retrieved_contexts=["Postgres stores run configurations and results."],
        response="It defaults to OpenAI text-embedding-3-large with 3072 dimensions.",
        reference="The default embedding model is BAAI/bge-small-en-v1.5."),
]
```

Expected: sample 1 scores **1.0** on faithfulness / precision / recall;
sample 2 scores **0.0**. If both score the same, scoring is not wired correctly.

Note `answer_relevancy` will be *high on both* — it measures whether the answer
addresses the question, not whether it is true. That is correct behaviour.

**3. Config sensitivity** — same test set, two configs:

| Config | context_precision | context_recall |
|---|---|---|
| `keyword`, `top_k=1` | ~0.50 | ~0.50 |
| `hybrid`, `top_k=5` | ~1.00 | ~1.00 |

If the numbers do not move, the config is not reaching the pipeline.

**4. Reranker check** — retrieve with and without a reranker; the *order* must
change while the *set* stays identical. That confirms precision can move and
recall cannot.

**5. Document extraction check** — on a table-heavy `.docx`, confirm character
count and chunk count jump versus paragraph-only extraction.

---

## 10. Cost and rate limits

API calls ≈ `test_cases × metrics × ~3`. Per-metric cost differs by algorithm:

| Metric | Calls per sample | Why |
|---|---|---|
| `context_recall` | ~1 | one classification pass |
| `answer_relevancy` | ~1 | one reverse-question, then local cosine |
| `faithfulness` | ~2+ | decomposes the answer into claims — scales with answer length |
| `context_precision` | ~`top_k` | one verdict **per retrieved chunk** |

So raising `top_k` costs API budget as well as precision. Embeddings and
reranking are free — they run locally on CPU.

On Groq's free tier (~30 req/min) expect throttling above ~5 test cases.
Rate-limit failures surface as reduced `scored_counts`, not as crashes — alert
on that, or a throttled run quietly reports on fewer cases.

---

## 11. Known limitations to state wherever scores are reported

- **Ground truth is LLM-generated**, so `context_precision` and `context_recall`
  measure against synthetic references. Human-review the QA set to fix this —
  it also unlocks `AnswerCorrectness`, which is otherwise meaningless.
- **`answer_relevancy` runs at `strictness=1`** (Groq's `n=1` cap), making it
  noisier than standard RAGAS.
- **LLM judges are stochastic** even at `temperature=0`; small test sets wobble.
- **`faithfulness` asks "is the answer supported by the context?", not "is it
  true?"** An answer can be faithful to a retrieved-but-wrong chunk.
- Storage/index-backend settings (`vectordb`, `index_type`) do not affect scores
  — with the same embedding model and `top_k`, every backend returns the same
  neighbours. Label them as informational or omit them.

---

## 12. Test set generation with ragas (replaces flat-prompt Q&A generation)

If you also want ragas to build the test set rather than asking an LLM for
"N question/answer pairs", use `TestsetGenerator`. It builds a knowledge graph
(headlines, summaries, themes, NER, then similarity/entity-overlap
relationships), synthesises personas, and emits single-hop and multi-hop queries
in varied styles.

**Beware: the widely-circulated v0.1.x snippets do not work on 0.4.x.**

| v0.1.x (docs you will find first) | v0.4.3 (actual) |
|---|---|
| `from ragas.testset.generator import TestsetGenerator` | `from ragas.testset import TestsetGenerator` |
| `from ragas.testset.evolutions import simple, reasoning, multi_context` | **removed** — use `ragas.testset.synthesizers` |
| `TestsetGenerator.from_langchain(gen_llm, critic_llm, emb)` | `TestsetGenerator(llm=..., embedding_model=...)` — no separate critic |
| `test_size=10` | `testset_size=10` |
| `distributions={simple: 0.5, ...}` | `query_distribution=[(Synthesizer(llm=llm), 0.5), ...]` |

```python
def _generate_sync(document_path, testset_size, notes):
    from langchain_core.documents import Document as LCDocument
    from ragas.testset import TestsetGenerator
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig as RagasRunConfig
    from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
    from langchain_groq import ChatGroq

    llm = LangchainLLMWrapper(ChatGroq(model=TESTSET_MODEL,
                                       api_key=GROQ_API_KEY, temperature=0.0))
    embeddings = LangchainEmbeddingsWrapper(_FastEmbedLangchain(EMB_MODEL))
    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)
    docs = [LCDocument(page_content=read_document(document_path))]
    run_config = RagasRunConfig(max_workers=1, timeout=180, max_retries=5, max_wait=90)

    def _run(qd):
        return generator.generate_with_langchain_docs(
            documents=docs, testset_size=testset_size, query_distribution=qd,
            run_config=run_config, raise_exceptions=True)

    try:
        testset = _run(None)          # ragas picks + filters synthesizers
    except Exception as e:
        # Multi-hop needs cross-chunk relationships a short document cannot
        # supply -> "No relationships match the provided condition".
        notes.append("Multi-hop generation not possible — single-hop only.")
        testset = _run([(SingleHopSpecificQuerySynthesizer(llm=llm), 1.0)])

    return testset.to_pandas()
```

Output columns: `user_input`, `reference_contexts`, `reference`,
`persona_name`, `query_style`, `query_length`, `synthesizer_name`.

Map `user_input → question` and `reference → answer` to feed §4, and keep
`reference_contexts` — it makes the ground truth traceable to a source chunk,
which the flat-prompt approach cannot do.

### Gotchas specific to testset generation

**12.1 — `rapidfuzz` is missing.**
```
ImportError: rapidfuzz is required for string distance.
```
Not a base ragas dependency. `pip install rapidfuzz`.

**12.2 — the model must be strict about JSON.**
```
OutputParserException: Invalid json output: ... This is because the role
description of the Natural Language Processing Engineer mentions ...
```
`llama-3.3-70b-versatile` appends explanatory prose after the JSON and generation
dies at scenario generation. `openai/gpt-oss-120b` holds the format. Use a
strict-JSON model and make it configurable.

**12.3 — pandas NaN becomes the string `"nan"`.**
Columns a synthesizer did not populate come back as `NaN`/`pd.NA`, and `str()`
on those yields `'nan'` / `'<NA>'`, which will be displayed to users. Normalise:

```python
def _text(row, key):
    import pandas as pd
    v = row.get(key)
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass                      # arrays/lists are never NA
    t = str(v).strip()
    return "" if t.lower() in ("nan", "<na>", "none") else t
```

**12.4 — same asyncio problem as §2.4.** Wrap in `asyncio.to_thread`.

### Cost — measured

**~17 Groq calls per question** (knowledge graph + personas + scenarios +
queries), versus ~4 calls for a whole test set under the flat-prompt approach.
A 4-question set took **3m31s** end to end. Default the endpoint to a small
number (6, not 15), and put it behind a job queue before deploying.

### Report which generator ran

A ragas test set and a fallback test set are not interchangeable. If you keep a
fallback path, return metadata (`generator`, `model`, `requested`, `produced`,
`notes`) and surface it in the UI. Silent substitution turns "these are ragas
metrics against a ragas test set" into a claim you cannot support.
