# RAG Evaluation UI

A session-based tool for evaluating RAG (Retrieval-Augmented Generation) pipeline
configurations. Upload a document, auto-generate a QA test set from it, pick a
pipeline configuration (chunking, embedding model, search type, reranker, top_k,
temperature), run it, and get back **real DeepEval metrics** — computed by the
`deepeval` library, scored over the whole test set as one batch, and logged to
LangSmith.

## What the metrics actually are

Four metrics, computed by DeepEval's own implementations — not hand-written
judge prompts:

| Metric | What it measures | Uses ground truth? |
|---|---|---|
| `faithfulness` | Share of claims in the answer that the retrieved context supports. This is the hallucination measure. | no |
| `answer_relevancy` | How directly the answer addresses the question. A confidently wrong answer can still score high; a "the context does not contain the answer" reply scores 0. | no |
| `context_precision` | **Rank-aware** precision@k — are the chunks relevant to the ground truth ranked near the top? | yes |
| `context_recall` | Share of the ground-truth answer that the retrieved context actually covers. | yes |

Scores are aggregated across the whole test set and shown as one overall set of
numbers per run, since batch-level figures are what you compare configs on.

### Division of labour — deepeval vs LangSmith

These are easy to get backwards, so to be explicit:

- **`deepeval` computes the metrics.** It is the only thing here producing scores.
- **LangSmith records them.** It is a tracing / dataset / experiment store. It
  has no metric implementation and does not calculate anything. Each run uploads
  a dataset of QA pairs, one run per test case with its per-case scores as
  feedback, plus one `eval-batch-summary` run carrying the aggregate.

### Guardrails on the numbers

- **The judge is never the generator.** The generator is whatever `chat_model`
  you pick; the judge defaults to `openai/gpt-oss-120b` (override with
  `DEEPEVAL_MODEL`). If the two would collide, the judge is switched
  automatically and a note is attached to the run — a model grading its own
  output inflates faithfulness and relevancy.
- **Partial coverage is always visible.** If a judge call fails (usually a Groq
  rate limit) that sample is excluded rather than counted as zero, and the UI
  shows `n of N cases` on the affected tile. An average over a subset can never
  be mistaken for an average over everything.
- **Ground truth is LLM-generated** from the same document, so
  `context_precision` and `context_recall` are measured against synthetic
  references, not human labels. Review the QA set in Step 1 if the numbers
  matter.

## DeepEval cost controls

| Control | Default | Effect |
|---|---|---|
| `DEEPEVAL_TRUTHS_LIMIT` | 15 | Caps facts extracted from context before claim verification — the most expensive prompt, and re-sent inside the verdict call. |
| `DEEPEVAL_INCLUDE_REASON` | `false` | Skips one call per metric per sample. The UI shows batch aggregates only, so per-sample prose was unread tokens. |
| `DEEPEVAL_ENABLE_CTX_RELEVANCY` | `false` | A 5th metric whose cost scales with `top_k` while largely duplicating context_precision. Opt-in. |
| `DEEPEVAL_MAX_QUALITY_RETRIES` | 1 | Test set generation: each critic retry re-sends the full context. |

**The trade-off, stated plainly:** capping truths at 15 means claims beyond that
are not checked, so faithfulness becomes an approximation over a subset —
cheaper and less complete.

### One DeepEval default you must not keep

`FaithfulnessMetric` counts any verdict that is not `"no"` as faithful, so a
claim the context merely *does not mention* (`"idk"`) passes. A deliberately
hallucinated answer scored **faithfulness 1.0** with the default; a strict scorer
gave the identical sample **0.0**. This code sets
`penalize_ambiguous_claims=True`, which makes unsupported claims subtract —
"supported by the context" rather than merely "not contradicted by it".
Verified: the same sample then scores 0.0.

## Groq multi-account failover

Rate limits are handled on two independent axes, both at the httpx transport
layer so deepeval and raw calls all inherit it:

1. **Key rotation** — `GROQ_API_KEY`, `GROQ_API_KEY_2`, … On 429/401/403 the
   request retries on the next organisation's key.
2. **Model fallback** — Groq's tokens-per-day quota is scoped *per model per
   organisation*, so a different model is a separate daily budget.
   `DEEPEVAL_MODEL_FALLBACKS` is tried once every key is exhausted.

Both are reported in run notes; neither is silent. The server prints its key
pool at startup — `.env` is read at import time, so a key added while the server
was running is invisible until restart.

## Which config knobs actually change the results

Verified by running the same test set through each variant and diffing both the
retrieved chunks and the resulting scores:

| Knob | Wired? |
|---|---|
| `chunk_size`, `chunk_overlap` | ✅ changes chunking, so changes retrieval |
| `top_k` | ✅ changes how many chunks are retrieved |
| `search_type` (keyword / semantic / hybrid) | ✅ different retrievers and rankings |
| `embedding_model` | ✅ selects the FastEmbed model used to embed and retrieve |
| `temperature` | ✅ passed to the generator |
| `chat_model` | ✅ selects the Groq model that generates answers |
| `reranker_model` | ✅ real — raises `context_precision`, but by construction **cannot** change `context_recall` |
| `vectordb` | ❌ **informational only** — retrieval always runs on Chroma |
| `index_type` | ❌ **informational only** — Chroma's index settings are not driven by it |

The two decorative knobs live in a collapsed **Infrastructure** section of the
config form, kept out of the main grid and labelled *recorded only — does not
affect scores*. They are still stored on the run so you can tag an experiment,
but two runs differing only in those settings are the *same* pipeline, so any
score difference between them is judge noise, not a real comparison.

They are decorative for a structural reason, not an unfinished one: with the
same embedding model and `top_k`, every vector backend returns the same
neighbours, and index type is a speed/recall tradeoff that only bites at
100k+ vectors. Wiring them fully would buy scale and ops flexibility — not
different scores.

## Diagnostics

Each run turns its own scores into concrete next actions, naming the knob *and*
its current value (`Raise Top K (now 2)`, `Lower Temperature (now 1.2)`).
The rules encode how the metrics interact — for example a low `faithfulness`
alongside a healthy `context_recall` is reported as a generation problem rather
than a retrieval one, and a low `answer_relevancy` under weak recall points you
at recall first. When everything is healthy it suggests trimming `top_k` or
model size to cut cost while the scores hold.

Suggestions are only ever drawn from knobs that genuinely affect the pipeline —
`vectordb` and `index_type` are never recommended.

Note the judge never sees the config — it only sees the contexts and answer the
config produced. That is the correct arrangement: you measure the artifacts, not
the settings.

## Multi-document sessions

A session holds **any number of documents**. Upload several at once or add them
incrementally — repeated uploads append rather than replace.

| Behaviour | Detail |
|---|---|
| Chunking | **Per document.** No chunk ever spans two files — one straddling unrelated documents would be incoherent to retrieve and impossible to attribute. |
| Attribution | Every chunk records its source filename in Chroma metadata. The filename never enters the chunk *text*, which would put it into the context the judge scores. |
| Index cache | Keyed on all document paths, sorted. Adding or removing a file rebuilds; reordering does not. |
| Test set | Contexts grouped within a single file, then spread across the corpus so questions do not all come from the first document. |
| Corpus change | Uploading or removing a document **clears the test set** — it no longer matches the corpus. |
| Duplicates / bad types | Skipped with a reason, reported in the response rather than silently dropped. |
| Legacy sessions | Sessions predating this still resolve via `document_path`, and migrate to `documents` on first upload. |

Endpoints:

| Method | Path | Notes |
|---|---|---|
| POST | `/api/sessions/{id}/upload` | Field name is `files`, repeatable. Appends. |
| DELETE | `/api/sessions/{id}/documents/{filename}` | Removes one document and its file on disk. |

Source filenames are stored as `{session_uuid}_{original}` on disk to stop
sessions colliding; `display_name()` strips that prefix so the UUID never
reaches the UI or LangSmith.

## Test set generation

Step 1 uses **DeepEval's `Synthesizer`**, not a flat "give me N Q&A pairs"
prompt. A base question is drafted from sampled chunks, then hardened along one
of seven **evolution** axes (reasoning, multi-context, concretizing, constrained,
comparative, hypothetical, in-breadth — the first five are enabled), while a
critic model re-rolls anything below a 0.5 quality threshold.

Details: That matters because the generated questions are harder and more
realistic: single-hop *and* multi-hop queries, drawn from a knowledge graph, in
varied query styles including deliberately imperfect phrasing that mirrors how
people actually search.

Each case stores provenance alongside the pair:

| Field | Meaning |
|---|---|
| `question` / `answer` | The contract the evaluator consumes |
| `reference_contexts` | Chunks the reference answer was derived from — makes ground truth traceable |
| `synthesizer` | Always `deepeval` |
| `evolutions` | Which axes hardened the question, e.g. `Reasoning`, `Multi-context` |
| `synthetic_input_quality` | The critic's score for the question |

### Cost — measured, not estimated

**~17 Groq calls per question**, versus ~4 calls for an entire test set under
the old approach. A 4-question set took **3m31s**. The endpoint therefore
defaults to **6** questions, not 15. Scale up only on a paid tier, and move the
call behind a job queue before deploying — it is synchronous today.

### Two constraints worth knowing

- **The generation model must be strict about JSON.** Every step parses against
  a schema, and one malformed reply discards the whole run. Default is
  `openai/gpt-oss-120b` (override with `DEEPEVAL_MODEL`). Note the adapter
  enforces the schema itself — json_mode plus Pydantic validation plus one
  repair retry — so weaker models fare better here than in libraries that only
  ask nicely and parse.
- **`MULTICONTEXT` evolutions need more than one chunk per context.**
  `DEEPEVAL_CHUNKS_PER_CONTEXT` defaults to 2 for exactly that reason.

If generation fails entirely it falls back to the old single-prompt generator —
but the substitution is reported in the API response and shown as a warning
banner in the UI, because the two are **not** equivalent.

## Available models

All model lists are validated against the provider's own registry, so no id is
hand-typed and the options cannot drift out of date.

**Embedding models — 28**, every text model FastEmbed serves, grouped in the form
by English small/balanced/large, long-context (8k tokens), multilingual, and
code. Ranges from `bge-small-en-v1.5` (384d, 67 MB) to `jina-embeddings-v3`
(1024d, ~100 languages). The two CLIP models FastEmbed also ships are excluded:
they truncate at 77 tokens and are built for image-text matching, not document
retrieval.

**Rerankers — 6**, FastEmbed's ONNX cross-encoders, from
`ms-marco-MiniLM-L-6-v2` (80 MB) to `jina-reranker-v2-base-multilingual`.

**Chat models — 6** general-purpose models Groq serves. Four are deliberately
withheld:

| Excluded | Why |
|---|---|
| `groq/compound`, `groq/compound-mini` | Agentic systems with built-in web search. They can answer from the internet instead of the retrieved context, silently invalidating `faithfulness` and `context_recall`. |
| `openai/gpt-oss-safeguard-20b` | Tuned for content moderation, not question answering. |
| `whisper-*`, `canopylabs/orpheus-*`, `llama-prompt-guard-*` | Speech models and 512-token classifiers. |

Groq is the only chat provider wired up, so the chat list is bounded by what
Groq hosts — adding GPT-4o, Claude, or Gemini would mean adding those providers.

## Architecture

```
Frontend (vanilla JS)  ──HTTP──>  FastAPI backend  ──>  Postgres
                                        │
                                        ├─ QA generation     (Groq)
                                        ├─ RAG pipeline      (FastEmbed + ChromaDB + BM25)
                                        ├─ Metric scoring    (deepeval, judged by Groq)
                                        └─ Experiment logging (LangSmith, optional)
```

### Evaluation flow

1. **Create session, upload document** — stored in Postgres + `UPLOAD_DIR`.
2. **Generate the test set with `deepeval`** — the `Synthesizer` groups chunks
   into contexts, drafts questions, writes reference answers, hardens each along
   an evolution axis, and has a critic model re-roll anything below the quality
   threshold. Each case carries `reference_contexts`, so the ground truth is
   traceable to its source chunk.
3. **Submit a run config** and save the `RunConfig` row.
4. **Run the RAG pipeline** — for the given config:
   - chunk the document (`chunk_size` / `chunk_overlap`)
   - embed chunks with the selected FastEmbed model and index them in ChromaDB
   - build a BM25 index for keyword search
   - for each question, retrieve `top_k` chunks using `search_type`
     (`keyword` = BM25, `semantic` = dense, `hybrid` = reciprocal rank fusion
     of both), reranking with a FastEmbed cross-encoder if `reranker_model` is set
   - generate the answer with `chat_model` at `temperature`, given that context
   - the index is cached per `(document, chunk_size, chunk_overlap, embedding_model)`
     so re-running the same chunking config doesn't re-embed every time
5. **Score the batch with `deepeval`** — all four metrics over the whole test set.
6. **Log to LangSmith** — dataset + per-case runs + batch summary.
7. **Save** batch metrics on the `RunConfig` and per-case rows as `RunResult`.

## Tech stack

| Layer | Tech |
|---|---|
| Backend | FastAPI, SQLAlchemy, Postgres |
| Retrieval | FastEmbed (28 embedding models + 6 ONNX rerankers), ChromaDB, rank_bm25 |
| Test set generation | **deepeval 4.1.7** `Synthesizer` (evolutions + critic filtration) |
| Evaluation | **deepeval 4.1.7**, judged via Groq |
| LLMs | Groq — generation and judging |
| Observability | LangSmith (optional) |
| Frontend | Vanilla HTML/CSS/JS, served as static files by FastAPI |

## Project structure

```
RAGAS/
├── backend/
│   ├── main.py                  # FastAPI app, mounts routers + frontend
│   ├── database.py              # SQLAlchemy engine/session
│   ├── models.py                # Session, RunConfig, RunResult ORM models
│   ├── schemas.py               # Pydantic request/response schemas
│   ├── routers/
│   │   ├── sessions.py          # session CRUD
│   │   ├── qa.py                # upload, QA generation, QA download
│   │   ├── evaluate.py          # run config submission + evaluation pipeline
│   │   └── feedback.py          # human thumbs up/down + comments
│   └── services/
│       ├── document_reader.py   # shared pdf/docx/txt extraction
│       ├── deepeval_llm.py      # Groq-backed DeepEvalBaseLLM adapter
│       ├── deepeval_testset_generator.py  # deepeval Synthesizer
│       ├── deepeval_evaluator.py  # deepeval batch metrics (cost-capped)
│       ├── groq_keys.py           # multi-account key pool + failover
│       ├── qa_generator.py      # single-prompt fallback only
│       └── rag_evaluator.py     # RAG pipeline orchestration + LangSmith
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── uploads/                     # uploaded documents (gitignored in practice)
├── requirements.txt
└── .env
```

## Setup

### Environment

**Python 3.12 is what this is built and tested on.** The pins in
`requirements.txt` are a verified working set. (The original hard 3.12
requirement came from `ragas` needing the langchain 0.3.x line, whose pydantic-v1
shim breaks on Python ≥ 3.14 — that constraint is gone now that ragas is no
longer used, but the stack has only been verified on 3.12.)

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux
```

Everything needed is in that one install — including reranking, which runs on
FastEmbed's ONNX cross-encoders rather than torch. Model weights download on
first use and are cached, so the first run with a newly selected embedding or
reranker model is slower.

### Prerequisites

- Python 3.12
- A running Postgres instance

### Configure environment

Create a `.env` file in the project root:

```env
# Postgres
DATABASE_URL=postgresql+psycopg://<user>:<password>@localhost:5432/rag_eval

# File uploads
UPLOAD_DIR=./uploads

# Required — Groq drives test set generation, RAG answers, and judging
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b

# Optional — the judge model. A single constant on purpose: scores only compare
# across runs when the judge is held fixed. Forced to differ from the generator
# so nothing ever grades its own output.
DEEPEVAL_MODEL=openai/gpt-oss-120b

# Optional — OpenAI-compatible gateway (e.g. OmniRoute) for answer generation.
# Blank = talk to Groq directly. Health-probed, so an unreachable gateway falls
# back to Groq rather than failing the run.
LLM_GATEWAY_URL=
LLM_GATEWAY_TOKEN=

# Optional — last-resort judge on that gateway, used only once every Groq model
# is exhausted. Works with no provider key, but must be an auto/* alias: pinning
# a specific free model disables the fallback routing and gets rate-limited at
# once. See _gateway_judge_model() in backend/services/deepeval_llm.py.
GATEWAY_JUDGE_MODEL=auto/best-free

# Optional — experiment tracing
LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=rag-eval-ui

# Optional — local cache dir for built retrieval indexes
RAG_INDEX_CACHE_DIR=./.rag_index_cache
```

Never commit real API keys or database credentials — use a `.env.example` with
blank values for version control.

### Run

```bash
.venv/Scripts/python -m uvicorn backend.main:app --reload --port 8000
```

Tables are created and migrated automatically on startup. The frontend is served
at `http://localhost:8000/`.

## API overview

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/sessions` | Create a session |
| GET | `/api/sessions` | List sessions |
| GET | `/api/sessions/{id}` | Get session detail (incl. run history) |
| DELETE | `/api/sessions/{id}` | Delete a session |
| POST | `/api/sessions/{id}/upload` | Upload a document |
| POST | `/api/sessions/{id}/generate-qa` | Generate QA pairs from the document |
| GET | `/api/sessions/{id}/download-qa` | Download QA pairs as JSON |
| POST | `/api/sessions/{id}/evaluate` | Run a config through the RAG + scoring pipeline |
| GET | `/api/sessions/{id}/runs` | List all past run configs + results |

## Known limitations

- `vectordb` and `index_type` are informational only — see the knob table above.
- Groq's free tier is rate-limited (~30 req/min) and DeepEval issues several
  calls per sample per metric, so scoring runs serially with retries. Large test
  sets take a while.
- Ground truth is LLM-generated, not human-labelled.
- 
## Roadmap ideas

- Side-by-side run comparison view, so metric deltas between configs are visible
  at a glance instead of by scrolling between run cards
- Batch evaluation across multiple configs in one submission
- Human review/editing of the generated ground truth before scoring
- Real multi-backend `vectordb` support (Pinecone, pgvector, Qdrant) — worth it
  for scale and ops, but see above: it would not change the scores
