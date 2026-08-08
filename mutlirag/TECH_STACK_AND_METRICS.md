# Multi-RAG — Tech Stack & Evaluation Metrics

## 1. Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| Backend / API | FastAPI + Uvicorn | Serves REST API and the HTML/JS frontend from one process |
| Frontend | Vanilla HTML/JS (`/ui`) | No build step, calls the API directly |
| LLM generation & vision | Groq API | Fast inference on open-weight models |
| Embeddings | `sentence-transformers` (local) | Free, offline, no API key |
| Reranking | `sentence-transformers` CrossEncoder (local) | Second-stage relevance scoring |
| Vector DB | Weaviate (default) + ChromaDB (fallback) | Auto-falls back to Chroma if Weaviate is unreachable |
| Keyword search | `rank-bm25` | Combined with embeddings for hybrid search |
| PDF parsing | PyMuPDF (fitz) | Text layer + render-to-image for scanned pages |
| Word / PowerPoint | `python-docx` / `python-pptx` | Text, tables, embedded images |
| OCR | EasyOCR | Offline, pure-Python |
| Evaluation logging | Postgres (`psycopg2`) + LangSmith | Q&A + metrics history, tracing/experiments |
| Live/benchmark eval judge | Groq via `deepeval` (`rag/evaluation.py`, `rag/deepeval_judge.py`) | See §3 |
| LangSmith offline eval judge | Groq via `openevals` (`scripts/run_langsmith_eval.py`) | Separate, independent judge — see §3 |

---

## 2. Pipeline Configuration (`backend/config.py`)

### Chunking
| Setting | Value | Purpose |
|---|---|---|
| `CHUNK_SIZE` | `800` chars | Size of each text chunk |
| `CHUNK_OVERLAP` | `120` chars | Overlap between consecutive chunks, so sentences straddling a boundary stay retrievable from both sides |

### Retrieval
| Setting | Value | Purpose |
|---|---|---|
| `TOP_K` | `10` | Base number of chunks retrieved per query (auto-scales up with more indexed documents) |
| `HYBRID_ALPHA` | `0.65` | Hybrid score = `0.65 * semantic (embedding cosine) + 0.35 * keyword (BM25)` — favors meaning while still catching exact tokens (IDs, invoice numbers, etc.) |
| `VECTOR_BACKEND` | `weaviate` | Self-hosted via Docker; auto-falls back to Chroma if unreachable |

### Reranker
| Setting | Value | Purpose |
|---|---|---|
| `RERANK_ENABLED` | `True` | Second-stage cross-encoder re-scoring, on by default |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M-param cross-encoder; jointly scores (query, chunk) pairs — more accurate than the independent hybrid signals, but only run over the small candidate pool hybrid search already returned, not the whole index |

### Models
| Role | Model (config key) | Notes |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (`EMBEDDING_MODEL`) | 384-dim, ~130MB, MTEB-leading accuracy for its size; runs locally |
| Answer generation | `llama-3.3-70b-versatile` (`DEFAULT_MODEL`) | Best all-round quality for grounded RAG answers |
| Query rewrite | `llama-3.1-8b-instant` (`REWRITE_MODEL`) | Small/fast; rewrites follow-up questions into standalone search queries using the last `REWRITE_HISTORY_TURNS` (6) messages |
| Vision (image understanding) | `qwen/qwen3.6-27b` (`VISION_MODEL`) | Multimodal; describes charts, tables, photos, layouts. Reasoning model, so vision calls pass `reasoning_effort="none"` |
| DeepEval judge (live + benchmark) | `llama-3.1-8b-instant` (`DEEPEVAL_JUDGE_MODEL`) | Separate TPD bucket from `DEFAULT_MODEL`; DeepEval has no native Groq provider, so `rag/deepeval_judge.py` wraps the Groq client in a `DeepEvalBaseLLM` |
| LangSmith offline eval judge | `openai/gpt-oss-120b` via `openevals`, served over Groq | One of the few Groq models supporting strict JSON-schema structured output required by `openevals`; independent of the DeepEval judge above |

---

## 3. Evaluation Metrics

Two independent judge pipelines exist:

- **Live chat / benchmark** (`rag/evaluation.py`) — DeepEval's native metric classes (+ one `GEval` judge for answer correctness, which DeepEval has no built-in metric for), run after every answer via `rag/deepeval_judge.py`'s Groq wrapper (DeepEval has no native Groq provider).
- **Offline batch eval** (`scripts/run_langsmith_eval.py`) — `openevals` prebuilt RAG judge prompts, run against the curated golden set, reported as LangSmith experiments. Scored by a completely separate judge setup, so numbers aren't directly comparable across the two.

### Reference-free metrics (run on every live chat answer)
| Metric | What it measures |
|---|---|
| **Faithfulness** | Fraction of the answer's atomic claims that are supported by the retrieved context (catches hallucination) — `FaithfulnessMetric` |
| **Answer relevancy** | How well the answer addresses the question — `AnswerRelevancyMetric` |
| **Context relevancy** | Signal-to-noise of retrieved context — of all retrieved sentences, what fraction are actually relevant (ignores ranking) — `ContextualRelevancyMetric` |

### Ground-truth metrics (need a curated/matched answer)
| Metric | What it measures |
|---|---|
| **Context precision** | Are the relevant retrieved chunks ranked near the top? — `ContextualPrecisionMetric`. Unlike the old hand-written version, DeepEval's implementation *requires* `expected_output`, so this moved out of the live/reference-free set. |
| **Context recall** | Did retrieval pull back everything needed to produce the ground-truth answer? — `ContextualRecallMetric` |
| **Answer correctness** | Does the generated answer match the ground-truth answer, factually and semantically? — DeepEval has no built-in metric for this, so it's a `GEval` judge (a weaker, judged tier than the decompose-and-count metrics above) |

On live chat, all three ground-truth metrics only get computed when the user's question closely matches (cosine similarity ≥ `GOLDEN_SET_MATCH_THRESHOLD` = `0.92`) a curated question in the golden set (`rag/golden_set.py`) — sourced from the LangSmith dataset if configured, else the local `scripts/eval_dataset_osw.json`. Otherwise, only the three reference-free metrics are scored.

### Suggested additions (not currently implemented)
| Metric | Why it's useful |
|---|---|
| Hit rate / Recall@K | Did the right chunk get retrieved at all, independent of the judge-based metrics above |
| MRR / NDCG | Rank-quality metric, complements context precision |
| Golden-set match rate | % of live questions that clear the 0.92 threshold — shows how often the full 6-metric evaluation applies |
| Rerank lift | Score before vs. after cross-encoder reranking — quantifies the value `RERANK_ENABLED` adds |
| Judge JSON-parse failure rate | Reliability of the judge model call (tune `DEEPEVAL_JUDGE_MODEL` in `config.py` if this is high) |
| End-to-end latency breakdown | Retrieval + rerank + generation + judge-eval time, per chat turn |
| Tokens / cost per query | Generation + judge calls (up to 5 extra Groq calls per turn for the 4 live metrics) |

---

## 4. Logging & Observability

- **Postgres** — every chat turn's question/answer/sources + all 6 DeepEval-scored metrics, additive to the existing JSON `chat_store.py` (never the source of truth, purely for history/analytics).
- **LangSmith** — live chat turns logged as runs with DeepEval scores attached as feedback (`rag/langsmith_logging.py`); offline batch experiments via `scripts/run_langsmith_eval.py`. Both no-op safely if `LANGSMITH_TRACING`/`LANGSMITH_API_KEY` aren't set.
