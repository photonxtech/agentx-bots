# Multi-Website AI Chatbot Platform

Admins register website URLs; the platform crawls each site, extracts and chunks its content,
embeds it into a per-website ChromaDB collection, and serves a floating chat widget that answers
questions using only that website's indexed content — with sources, a confidence score, and a
fixed "I couldn't find that information on this website." fallback when nothing relevant is found.
A daily scheduler keeps every site's index in sync.

Retrieval is a hybrid pipeline (semantic search + BM25 keyword search + exact keyword match,
fused with Reciprocal Rank Fusion and reranked with a cross-encoder) rather than plain vector
similarity — see [Retrieval pipeline](#retrieval-pipeline-in-detail) below.

## Stack

- **Backend:** Python, FastAPI, SQLAlchemy + SQLite, ChromaDB, OpenAI (`gpt-4o-mini` via
  Structured Outputs), HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (via
  `langchain-huggingface`), a `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker, `rank-bm25`,
  Playwright, Trafilatura, BeautifulSoup4, APScheduler, `pyspellchecker`.
- **Frontend:** React, Vite, TypeScript, MUI, TanStack Query, Axios, Framer Motion, React Markdown.

No Docker — everything runs from a local Python venv and Node/npm.

## Project layout

```
Ai/
├── backend/       FastAPI app (crawler, extractor, chunker, embeddings, vectordb, llm, scheduler, services, api)
├── frontend/      React admin dashboard + chat widget demo page
├── setup.sh       one-time setup for both
└── README.md
```

Backend internals:

```
backend/app/
├── main.py            FastAPI app, CORS, lifespan (DB init, admin seed, scheduler), /health, /metrics
├── config.py          Pydantic Settings (.env)
├── database/          SQLAlchemy models (Website, Page, Conversation, Message, CrawlJob, AdminUser, EvalRun) + session
├── api/routes/        auth, websites, crawl, pages, chat, conversations, analytics, evaluate
├── crawler/           sitemap discovery + recursive same-domain crawl (both run, merged) + URL filtering
├── extractor/         Playwright render + Trafilatura + BeautifulSoup fallback
├── chunker/           RecursiveCharacterTextSplitter (1000 chars / 200 overlap)
├── embeddings/        HuggingFace all-MiniLM-L6-v2 singleton (query embeddings LRU-cached)
├── vectordb/          ChromaDB, one collection per website (website_{id}), cosine + tuned HNSW params
├── llm/               query spelling/typo correction, hybrid retrieval + RRF + reranking, prompts, OpenAI client
├── scheduler/         APScheduler daily sync job
├── services/          crawl/sync/reindex orchestration, chat streaming, caching, eval harness, analytics, auth
└── utils/             logging, retry, hashing, TTL cache, text dedup, URL filtering, JWT/password hashing, metrics
```

## Requirements

- Python 3.11+ (this build was set up with Python 3.12 via Homebrew — the system default
  Python 3.14 was too new for some pinned dependency wheels at build time)
- Node.js 18+ / npm

## Setup

```bash
cd Ai
./setup.sh
```

This creates the backend venv, installs Python deps + the Playwright Chromium browser, installs
frontend npm deps, and copies both `.env.example` files to `.env`.

Then edit `backend/.env` and set at minimum:

```
OPENAI_API_KEY=...         # from platform.openai.com
JWT_SECRET=...             # any long random string
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=...
```

### Manual setup (equivalent to `setup.sh`, if you'd rather run it yourself)

```bash
# Backend
cd Ai/backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
playwright install chromium
cp .env.example .env   # then edit it
deactivate

# Frontend
cd Ai/frontend
npm install
cp .env.example .env
```

## Running

```bash
./backend/run.sh     # http://localhost:8001  (terminal 1)
./frontend/run.sh    # http://localhost:5173   (terminal 2)
```

- Admin dashboard: `http://localhost:5173/admin` (log in with `ADMIN_EMAIL`/`ADMIN_PASSWORD`)
- Demo widget page: `http://localhost:5173/`
- If port `5173` is already taken by something else on your machine, Vite silently picks the
  next free port (commonly `5174`) — check your terminal output for the actual URL, and make
  sure that origin is listed in `backend/.env`'s `CORS_ORIGINS` (both are included by default).

**Important:** don't run the backend with `uvicorn --reload`, and don't run a second backend
process against the same `data/chroma` directory — see [Notes](#notes) below for why.

## Using it

1. In the admin dashboard, click **Add Website**, enter a URL and display name.
2. Click the crawl icon on that row — it discovers pages via `sitemap.xml` **and** a recursive
   same-domain crawl (both run and get merged, since either source alone can miss pages),
   extracts clean text with Playwright + Trafilatura (falling back to BeautifulSoup), chunks it,
   embeds it, and stores it in that site's ChromaDB collection (`website_{id}`).
3. Click **Details** to see indexed pages and crawl job history.
4. Open `http://localhost:5173/`, pick the website from the dropdown, and chat with it via the
   floating button — answers are grounded only in that site's indexed content.
5. **Sync** re-crawls and indexes only new/changed pages, deleting vectors for pages no longer
   found. **Reindex** force re-embeds every currently-discovered page. A daily background job
   (default 03:00, `DAILY_SYNC_HOUR` in `.env`) runs the same sync automatically for every website.
   A crawl/sync/reindex already running for a website blocks a second one from starting
   concurrently (409 response) — running two writers against the same vector collection at once
   corrupts it.

## Retrieval pipeline (in detail)

For every chat question:

1. **Spelling correction** (`llm/query_normalizer.py`) — fuzzy-matches typos against the site's
   own crawled vocabulary first (catches things a generic dictionary misses, e.g. brand names),
   falls back to `pyspellchecker`, and detects/merges accidentally split words ("plan form" →
   "platform"). Only affects retrieval/generation input, not what's stored in conversation history.
2. **Hybrid candidate search** — semantic (ChromaDB ANN on the MiniLM embedding), BM25 keyword
   search (`rank-bm25`), and exact substring matching for distinctive terms — three independent
   signals with different blind spots.
3. **Reciprocal Rank Fusion** — combines the three ranked lists by rank position rather than raw
   score (their scales aren't comparable).
4. **Cross-encoder reranking** (`llm/reranker.py`) — scores each (question, chunk) pair directly,
   against **both** the bare question and a version with the website's name folded in
   ("PhotonX: how to contact"), keeping whichever scoring is higher per chunk — some short/vague
   questions need that site-name context to score well, others score dramatically *worse* with it,
   so neither framing alone is reliable.
5. **Diversity cap** — at most 2 chunks per source page make the final list, so one page repeating
   a theme can't crowd out a different page holding the actual answer.
6. **Threshold + top-k** — keeps chunks scoring above `RELEVANCE_THRESHOLD`, capped at
   `TOP_K_CHUNKS`.
7. **Generation** — `gpt-4o-mini` via OpenAI Structured Outputs (strict `{answer, grounded}`
   JSON schema, temperature 0.2). The prompt first decides small-talk vs. real question, then
   only answers real questions from the provided context, using the fixed fallback line otherwise.
   `sources`/`confidence` shown to the user are suppressed whenever the model reports
   `grounded: false`.
8. **Caching** — retrieval results and full responses are cached in-process (TTL from
   `RESPONSE_CACHE_TTL_SECONDS`), invalidated by a per-website generation counter bumped on every
   crawl/sync/reindex/delete — never served stale after content changes.

## Configuration reference (`backend/.env`)

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key (required) |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model used for answering questions |
| `JWT_SECRET` | — | Signing secret for admin JWTs (required) |
| `JWT_EXPIRE_MINUTES` | 480 | Admin session length before re-login is required |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | — | Seeded admin login (required) |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:5174` | Comma-separated allowed frontend origins |
| `DATABASE_URL` | `sqlite:///./data/app.db` | Relational metadata store |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Vector store location |
| `MAX_PAGES` | 500 | Max pages crawled per website |
| `MAX_CRAWL_DEPTH` | 5 | Max recursive-crawl link depth |
| `CRAWLER_CONCURRENCY` | 2 | Concurrent page fetches/renders |
| `CRAWLER_DELAY_SECONDS` | 1 | Delay between requests to the same domain |
| `RELEVANCE_THRESHOLD` | 0.01 | Minimum post-rerank relevance score for a chunk to be used as context |
| `TOP_K_CHUNKS` | 25 | Chunks retrieved per question |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model (documents + queries) |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker model |
| `RESPONSE_CACHE_TTL_SECONDS` | 600 | TTL for cached retrieval results/responses |
| `DAILY_SYNC_HOUR` | 3 | Hour (0-23, server local time) the daily sync job runs |
| `LOG_LEVEL` | INFO | Logging verbosity |

## API summary

`POST /auth/login` · `POST /websites` · `GET /websites` · `PATCH /websites/{id}` ·
`DELETE /website/{id}` · `POST /crawl/{id}` · `POST /sync/{id}` · `POST /reindex/{id}` ·
`GET /websites/{id}/crawl-jobs` · `GET /pages/{website_id}` · `POST /chat` (SSE) ·
`GET /conversations` · `GET /conversations/{id}` · `DELETE /conversations/{id}` ·
`POST /messages/{id}/feedback` · `GET /analytics/{website_id}` ·
`POST /evaluate/{website_id}` · `GET /evaluate/{website_id}` · `GET /health` · `GET /metrics` (admin)

All admin-mutating routes require `Authorization: Bearer <token>` from `/auth/login`.

## Notes

- **This backend has no automated test suite.** All behavior described here — including the
  retrieval pipeline tuning — was verified by hand against a live site and live OpenAI calls
  during development. If you add new logic, verify it the same way (live crawl + live chat
  question) before trusting it.
- `passlib[bcrypt]` requires `bcrypt==4.0.1` specifically — newer bcrypt releases removed an
  attribute passlib 1.7.4 depends on for its self-test and will raise `ValueError` on every hash call.
- Sitemap XML is parsed with `defusedxml` (not the stdlib `xml.etree`) since it comes from
  arbitrary admin-submitted third-party sites — this prevents XXE/billion-laughs attacks.
- `backend/run.sh` intentionally does **not** use `uvicorn --reload`. ChromaDB's embedded,
  file-based persistence is not safe under process restarts or concurrent access from more than
  one process — using `--reload`, running a second backend instance, or querying the same
  `data/chroma` directory from a side script while the server is running has repeatedly been
  observed to silently corrupt or empty a website's collection. Only ever run one backend process
  against a given `data/chroma` directory, and restart it manually (not via file-watch) after
  changing backend code. If a website's chat answers suddenly go blank/wrong, `POST
  /reindex/{id}` rebuilds its vectors from the pages already stored in SQLite — nothing is
  actually lost, since crawled text/hashes live there, not in Chroma.
- ChromaDB's HNSW `ef_search` defaults to 10, which is lower than the number of neighbors this
  app's retrieval requests — left at the default, larger collections eventually crash queries
  with `RuntimeError: Cannot return the results in a contiguous 2D array`. This is fixed in
  `vectordb/collection.py` by setting `ef_search=200` explicitly; if you change `TOP_K_CHUNKS`
  to something much larger, raise this too.
- `requirements.txt` intentionally does **not** include bare `langchain`/`langchain-community` —
  only `langchain-text-splitters` and `langchain-huggingface` are actually imported anywhere.
