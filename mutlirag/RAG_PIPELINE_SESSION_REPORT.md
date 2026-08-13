# Multi-RAG Pipeline — Point-to-Point Observation Report

**Scope:** Retrieval pipeline hardening on the production Multi-RAG backend (`mutlirag/backend`).
**Status at time of writing:** 106 tests passing, 0 failing, 0 skipped.

---

## 1. Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| API server | FastAPI + Uvicorn | Serves both the JSON API and the static frontend |
| LLM generation | Groq (`llama-3.3-70b-versatile`, fallback list in `config.py`) | `REWRITE_MODEL = llama-3.1-8b-instant` rewrites follow-ups into standalone queries |
| Embeddings | `sentence-transformers`, `BAAI/bge-small-en-v1.5` (384-dim) | Runs fully local, no API key/network call at inference time |
| Reranker | Cross-encoder `BAAI/bge-reranker-base` (278M params) | Second-stage joint (query, chunk) scoring over the candidate pool |
| Keyword search | `rank-bm25` | Fused with dense cosine similarity — see Hybrid Search below |
| Vector DB | Weaviate (self-hosted, Docker) with automatic fallback to Chroma | `WEAVIATE_COLLECTION = MultiRagChunk` |
| OCR | EasyOCR + Pillow | Scanned PDF pages (`PDF_OCR_MIN_CHARS` threshold) |
| Vision | Groq multimodal (`qwen/qwen3.6-27b`) | Describes embedded images/diagrams/photos so they become searchable |
| Document parsing | PyMuPDF (PDF), python-docx, python-pptx, pandas/openpyxl (Excel/CSV) | |
| Answer evaluation | DeepEval (reference-free: faithfulness, answer_relevancy, context_relevancy) + golden-set-gated GEval correctness | Judge model: `llama-3.1-8b-instant`, separate token bucket from generation |
| Metrics/history logging | PostgreSQL (`psycopg2-binary`) | Additive — `chat_store.py` JSON remains source of truth for chat state |
| Tracing/offline eval | LangSmith + `openevals` | `scripts/run_langsmith_eval.py`, `scripts/run_ragas_eval.py` (out of scope this round) |
| Tests | pytest | 106 tests, no live network/DB/model dependency (fakes/mocks throughout) |

### Retrieval architecture (as of this session)
```
query
  -> intent routing (NORMAL_QUERY / TOC_QUERY / PAGE_QUERY)
  -> [NORMAL_QUERY only]
       query rewrite (Groq, uses chat history)
       hybrid search: HYBRID_ALPHA * dense_cosine + (1-HYBRID_ALPHA) * BM25   [ALPHA=0.65]
       -> candidate pool (RETRIEVAL_CANDIDATE_K = 40), TOC pages excluded
       -> neighbor/parent expansion (bounded, protected from later filters)
       -> cross-encoder rerank (BAAI/bge-reranker-base)
       -> diversity-aware final selection (MMR, MMR_LAMBDA = 0.7)
       -> FINAL_CONTEXT_K = 10 chunks -> LLM
  -> [TOC_QUERY / PAGE_QUERY] deterministic metadata lookup, bypasses all of the above
```

---

## 2. Problems Raised, Pin-to-Pin

### Problem 1 — TOC entries not fully detected
**Raised:** TOC-shaped rows using no-space dotted numbering (e.g. `"4.4Need to Know"`) were slipping past the TOC heuristic and being treated as normal content.
**Root cause:** `_TOC_ENTRY_RE` in `ingestion.py` required a space or digit immediately after the dotted number; PDFs that lost the space during text extraction didn't match.
**Fix:** Regex widened to `r"^\d+\.\d+(?:\.\d+)*(?:\s+\S|[A-Z])|^\d+\s+\S"` — now matches a capital letter directly after the number too.
**Verified:** New regression tests in `test_toc_heuristic.py` covering the no-space case; confirmed against real document text.
**Operational impact:** Ingestion-time fix — only takes effect on **re-ingested** documents (or after clearing the ingestion cache and re-uploading). Already-indexed documents keep their old tagging until re-uploaded.

### Problem 2 — Duplicate/triplicate chunks in retrieval results
**Raised:** User pasted real retrieved-chunk JSON showing the same chunk appearing 2–3 times in one result set.
**Root cause (two layers):**
1. Re-uploading the same source file repeatedly accumulated new copies of its chunks in the vector store instead of replacing the old ones.
2. Even for a single ingestion, near-identical text (e.g. overlapping child chunks) could independently score well enough to occupy multiple result slots.
**Fix:**
1. `RagService._replace_existing_file()` — on re-upload, existing chunks for that file are removed before the new ones are added (one file per chat, no accumulation). Wired into both `ingest_file` and `ingest_file_stream`.
2. Exact-text deduplication mask added directly inside `vectorstore.search()`, `get_toc_chunks()`, and `get_by_pages()`, scoped per chat — a safety net independent of the upload path.
**Verified:** `test_service_upload_replace.py`, `test_vectorstore.py` dedup tests. Confirmed against the user's own reported JSON pattern.
**What failed first:** Initially treated this as a single-cause bug; the user's second JSON paste showed duplicates persisting after the first fix landed, which is what surfaced the second (retrieval-time) layer. Fixed by adding the dedup mask as a second, independent line of defense rather than assuming the upload fix alone was sufficient.

### Problem 3 — TOC chunks still appearing in normal query results
**Raised:** Repeatedly ("check it once", "u didnt understand tell me") — user showed live JSON where the *last* chunk in a NORMAL_QUERY result set was clearly a table-of-contents page, despite the exclusion logic already existing.
**Investigation:** Rather than re-theorizing, ran a live diagnostic script directly against the running Weaviate-backed store to inspect the actual `is_index` metadata on the offending chunk.
**Root cause:** The specific TOC page in question used the no-space dotted-number format from Problem 1 — it had never been tagged `is_index=True` in the first place, so the exclusion mask correctly left it in (there was nothing to exclude). This was a downstream symptom of Problem 1, not a separate exclusion-logic bug.
**Fix:** Same regex fix as Problem 1. No changes needed to the exclusion mask itself (`config.EXCLUDE_INDEX_PAGES`, `vectorstore._is_toc`), which was already correct.
**Verified:** Re-ran the diagnostic script against the corrected tagging; confirmed the page now carries `is_index=True` and is excluded.
**Lesson:** This is the clearest example in the session of "check it once" being the right call — the bug was one layer removed from where the visible symptom (TOC chunk in results) suggested it lived.

### Problem 4 — "Semantic boundary chunking" ambiguity
**Raised:** User asked for "structure aware parent child semantic boundary chunking."
**Clarification:** Structural (heading/font-size-driven) parent-child chunking already existed. What was actually missing was **true semantic** chunking — splitting on detected topic shifts within a section, not just structure.
**Resolved via explicit choice (AskUserQuestion):** User selected embedding-based topic-shift detection, applied only at the parent tier.
**Implementation:** `_split_text_semantic()` / `_consecutive_cosine_distances()` in `chunking.py`. Adaptive per-section percentile threshold (`SEMANTIC_BREAKPOINT_PERCENTILE = 95`) instead of a fixed cosine cutoff, since "a big jump" isn't comparable across documents/embedding models. Guarded by `SEMANTIC_MIN_CHUNK_CHARS`, `SEMANTIC_CHUNKING_MIN_UNITS/MAX_UNITS` so short or pathologically long sections fall back to plain splitting.
**Verified:** `test_semantic_chunking.py`.
**Operational impact:** Ingestion-time only — requires re-ingestion (re-upload) to take effect on any document; does not touch already-indexed chunks.

### Problem 5 — Missing evidence chunk in a multi-part answer (the "three guidelines" bug)
**Raised in full detail:** For the query *"What are the three key guidelines for smart problem solving?"*, Guidelines #1 and #2 came through in the answer but Guideline #3 was consistently missing, even though its source chunk existed in the index.
**Investigation (live, not theoretical):** A diagnostic script confirmed the Guideline #3 chunk ranked **~140th out of ~500 candidates** for its own query — far below any reasonable candidate-pool width. Its own wording (starting mid-list, low keyword/semantic overlap with the query) simply didn't score well standalone, even though it was structurally necessary.
**Root cause:** Two compounding gaps:
1. `retrieve()` used a single number for both the search candidate pool size *and* the final chunk count sent to the LLM — a ~15-chunk pool was reranked and truncated straight to 5, giving the reranker very little to work with.
2. There was no mechanism to recover a low-scoring chunk based on **structural adjacency** to a chunk that *did* score well — widening the candidate pool alone can't fix a chunk that scores badly on its own merits.
**Fix (four coordinated pieces):**
- **Candidate/final split:** `RETRIEVAL_CANDIDATE_K = 40` (fed to reranker) vs. `FINAL_CONTEXT_K = 10` (sent to LLM), replacing the old single derived number.
- **Neighbor/parent expansion:** new `vectorstore.get_neighbors(doc, window=1)` — for each of the top `NEIGHBOR_EXPANSION_MAX_ANCHORS` (8) candidates, pulls the immediately adjacent chunk(s) in the same page's reading order (positional adjacency, not numeric page proximity), scoped to the same chat/source, transparently skipping TOC pages. Capped globally at `NEIGHBOR_EXPANSION_MAX_ADDED = 4` total across all anchors.
- **Protection from filters:** expanded neighbors are marked and exempted from `RERANK_MIN_SCORE`'s floor (`reranker.rerank(..., protect=...)`) and from MMR diversity trimming (`reranker.diversity_select(..., protect=...)`) — they were added for structural completeness, not because they scored well, so score-based filters must not remove them.
- **MMR diversity selection:** `diversity_select()`/`_mmr()` added so near-duplicate top chunks don't crowd out a genuinely distinct piece of evidence within `FINAL_CONTEXT_K`.
**Verified:**
- `test_neighbor_expansion.py` — dedicated regression suite modeled on the real chunk text: proves Guideline #3 is unretrievable without expansion and recovered with it; Recall@5 measured at 1.0 with expansion vs. 0.5 without; chat-isolation and TOC-skip behavior both covered.
- **Live verification against real production data** (no synthetic fixtures) — ran the fixed pipeline against the actual already-indexed chat, confirmed Guideline #3 now reaches `generator.context_texts()`, the exact function the API uses to build the LLM prompt.
**What failed along the way (test-fixture bugs, not production bugs, but worth recording honestly):**
1. First embedding fixture used a 2D vector where the "topic affinity" dimension was negligible after L2 normalization — every fake embedding collapsed to nearly the same direction, silently defeating the test's own ability to distinguish anything. Caught by printing actual computed scores instead of hand-calculating expected ones. Fixed with proper one-hot-per-topic-word embeddings.
2. Sequential distractor page numbers made distractors structurally adjacent to *each other*, letting expansion pull in irrelevant neighbors instead of the real one. Fixed by scattering distractor page numbers non-sequentially.
3. `window=1` returns **both** the previous and next neighbor; with a too-tight expansion cap, the "previous" (irrelevant) neighbor consumed the only available slot before the real "next" neighbor was reached. Fixed by correcting the cap in the affected tests — this also surfaced the real global-cap bug described next.
4. **Real production bug found via building this test, not by the original report:** the unbounded-per-anchor reservation in `diversity_select` meant several anchors each contributing neighbors could together exceed the diversity budget, starving the best *regular* candidate of a slot. Fixed by adding the global `NEIGHBOR_EXPANSION_MAX_ADDED` cap.
5. A verification script initially reported a false negative — it checked the expanded neighbor's narrow `.text` (child-chunk slice) for the guideline text, not `meta["parent_text"]` (what the LLM actually receives). Corrected the check before concluding the fix worked.
**Explicitly deferred, by design:**
- Embedding model (`BAAI/bge-small-en-v1.5`) and `HYBRID_ALPHA` (0.65) were **not** changed — user explicitly ruled this out until diagnostics proved it necessary, and the diagnostics pointed to a structural gap, not a model-quality gap.
- `chunk_type` ("content"/"toc") metadata was added for diagnostics/readability but is not load-bearing in any exclusion or expansion logic yet — old chunks without it are unaffected.
- Repeated header/footer stripping (`STRIP_REPEATED_HEADERS_FOOTERS`) shipped **disabled by default** — the right recurrence threshold is document-dependent, and turning it on without tuning risks stripping legitimate repeated content (e.g. a worksheet template reused across exercises).
- `scripts/run_ragas_eval.py` / `run_langsmith_eval.py` were left untouched — they still reference the old `config.TOP_K` pattern and were out of scope for this pass.

### Problem 6 — Repeated PDF headers/footers diluting chunks
**Raised as part of the larger spec:** repeated banners/headers should ideally be removed at ingestion, not just deduplicated at retrieval time.
**Fix:** `_repeated_boilerplate_lines()` (cheap text-only pre-pass, absolute page-count threshold rather than a percentage — a banner recurring once every ~100 pages in a 500-page document is still boilerplate) and `_strip_boilerplate_lines()`, wired into `load_pdf()` **after** the OCR-fallback decision (which must see un-stripped text) but before chunking.
**Verified:** `test_ingestion_boilerplate.py`.
**Shipped disabled by default** (`STRIP_REPEATED_HEADERS_FOOTERS = False`) pending per-document tuning — an explicit, deliberate scope-down, not an oversight.

---

## 3. How Things Were Checked (Methodology)

- **No guessing.** Every root-cause claim in this report was confirmed by either (a) a passing/failing test built from real chunk text, or (b) a throwaway diagnostic script run directly against the live Weaviate-backed store, per explicit user instruction ("do not guess", "check it once").
- **Correct field, not convenient field.** Verification checked `meta["parent_text"]` (what the LLM actually sees via `generator.context_texts()`), not just `.text` (the narrow retrieval-time child slice) — an earlier check against the wrong field produced a false negative before this was caught.
- **Regression tests before closing any bug.** Each fix has a dedicated test file; the neighbor-expansion fix specifically includes a test that fails without the fix (`test_without_neighbor_expansion_guideline_3_is_missing`) to prove causality, not just that the new code runs.
- **Recall@k as a concrete metric**, not just "it looks right" — Recall@5 measured at 1.0 vs. 0.5 for the guideline-evidence scenario.
- **Full suite run after every change**, currently 106 passed / 0 failed / 0 skipped.

---

## 4. Operational Notes (restart / re-ingest)

| Change | Needs server restart? | Needs re-upload/re-ingest? |
|---|---|---|
| TOC regex fix (#1) | Yes (code change) | Yes — old tagging persists until re-ingested |
| Duplicate fix — upload-replace (#2) | Yes | No — fixes future uploads; existing dupes need one-time cleanup |
| Duplicate fix — retrieval dedup mask (#2) | Yes | No — operates on existing data immediately |
| Semantic chunking (#4) | Yes | Yes — ingestion-time only |
| Candidate/final-k split, neighbor expansion, MMR (#5) | Yes | **No** — operates on `page`/`chunk`/`parent_text` metadata already present on existing chunks; verified live against already-indexed data |
| `chunk_type` metadata | Yes | Only if you want it populated on old chunks (cosmetic only) |
| Header/footer stripping (#6) | Yes | Yes, and only if explicitly enabled (`STRIP_REPEATED_HEADERS_FOOTERS = True`) |

A restart is required for every item because `config.py`/`service.py`/`vectorstore.py`/`reranker.py` are loaded once at process startup (unless running with `--reload`).
