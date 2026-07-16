# Multi-Website AI Chatbot Platform

Admins register website URLs; the platform crawls each site, extracts and chunks its content,
embeds it into a per-website ChromaDB collection, and serves a floating chat widget that answers
questions using only that website's indexed content — with sources, a confidence score, and a
fixed "I couldn't find that information on this website." fallback when nothing relevant is found.
A daily scheduler keeps every site's index in sync.

## Stack

- **Backend:** Python, FastAPI, SQLAlchemy + SQLite, ChromaDB, LangChain, OpenAI (`gpt-4o-mini`),
  HuggingFace `sentence-transformers/all-MiniLM-L6-v2`, Playwright, Trafilatura, BeautifulSoup4, APScheduler.
- **Frontend:** React, Vite, TypeScript, MUI, React Query, Axios, Framer Motion, React Markdown.

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
├── main.py            FastAPI app, CORS, lifespan (DB init, admin seed, scheduler)
├── config.py          Pydantic Settings (.env)
├── database/          SQLAlchemy models + session
├── api/routes/         auth, websites, crawl, pages, chat, conversations, analytics
├── crawler/            sitemap discovery + recursive fallback + URL filtering
├── extractor/          Playwright render + Trafilatura + BeautifulSoup fallback
├── chunker/            RecursiveCharacterTextSplitter (1000/200)
├── embeddings/         HuggingFace all-MiniLM-L6-v2 singleton
├── vectordb/            ChromaDB, one collection per website (website_{id})
├── llm/                 OpenAI client, prompt, retrieval + similarity/confidence
├── scheduler/           APScheduler daily sync job
├── services/            crawl/sync/reindex orchestration, chat, analytics, auth
└── utils/               logging, retry, hashing, URL filtering, JWT/password hashing
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

## Running

```bash
./backend/run.sh     # http://localhost:8001  (terminal 1)
./frontend/run.sh    # http://localhost:5173   (terminal 2)
```

- Admin dashboard: `http://localhost:5173/admin` (log in with `ADMIN_EMAIL`/`ADMIN_PASSWORD`)
- Demo widget page: `http://localhost:5173/`

## Using it

1. In the admin dashboard, click **Add Website**, enter a URL and display name.
2. Click the crawl icon on that row — it discovers pages via `sitemap.xml` (or a recursive
   same-domain crawl if there's no sitemap), extracts clean text with Playwright + Trafilatura
   (falling back to BeautifulSoup), chunks it, embeds it, and stores it in that site's ChromaDB
   collection (`website_{id}`).
3. Click **Details** to see indexed pages and crawl job history.
4. Open `http://localhost:5173/`, pick the website from the dropdown, and chat with it via the
   floating button — answers are grounded only in that site's indexed content.
5. **Sync** re-crawls and indexes only new/changed pages, deleting vectors for pages no longer
   found. **Reindex** force re-embeds every currently-discovered page. A daily background job
   (default 03:00, `DAILY_SYNC_HOUR` in `.env`) runs the same sync automatically for every website.

## Configuration reference (`backend/.env`)

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key (required) |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model used for answering questions |
| `JWT_SECRET` | — | Signing secret for admin JWTs (required) |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | — | Seeded admin login (required) |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed frontend origins |
| `MAX_PAGES` | 500 | Max pages crawled per website |
| `MAX_CRAWL_DEPTH` | 5 | Max recursive-crawl link depth (ignored when a sitemap is used) |
| `CRAWLER_CONCURRENCY` | 2 | Concurrent page fetches/renders |
| `CRAWLER_DELAY_SECONDS` | 1 | Delay between requests to the same domain |
| `SIMILARITY_THRESHOLD` | 0.3 | Minimum cosine similarity for a chunk to be used as context |
| `TOP_K_CHUNKS` | 5 | Chunks retrieved per question |
| `DAILY_SYNC_HOUR` | 3 | Hour (0-23, server local time) the daily sync job runs |

## API summary

`POST /auth/login` · `POST /websites` · `GET /websites` · `PATCH /websites/{id}` · `DELETE /website/{id}` ·
`POST /crawl/{id}` · `POST /sync/{id}` · `POST /reindex/{id}` · `GET /websites/{id}/crawl-jobs` ·
`GET /pages/{website_id}` · `POST /chat` (SSE) · `GET /conversations` · `GET /conversations/{id}` ·
`DELETE /conversations/{id}` · `POST /messages/{id}/feedback` · `GET /analytics/{website_id}`

All admin-mutating routes require `Authorization: Bearer <token>` from `/auth/login`.

## Tests

```bash
cd backend && source venv/bin/activate && pytest -v
```

74 tests covering pure logic (URL filtering, hashing, chunking, JWT, similarity/confidence, HTML
cleaning), the crawl/sync/reindex orchestration (with the crawler/extractor/embedder mocked), and
every API route's contract (with crawling/LLM calls mocked). Genuinely I/O-bound integration points
— live Playwright rendering and real OpenAI streaming — were verified manually against a real site
and a live OpenAI call during development rather than mocked in the automated suite.

The frontend has no test runner in this stack; `npm run build` type-checks and bundles as the
automated gate for each change, verified manually in the browser alongside it.

## Notes

- `passlib[bcrypt]` requires `bcrypt==4.0.1` specifically — newer bcrypt releases removed an
  attribute passlib 1.7.4 depends on for its self-test and will raise `ValueError` on every hash call.
- Sitemap XML is parsed with `defusedxml` (not the stdlib `xml.etree`) since it comes from
  arbitrary admin-submitted third-party sites — this prevents XXE/billion-laughs attacks.
- `backend/run.sh` intentionally does **not** use `uvicorn --reload`. ChromaDB's default
  (embedded SQLite) persistence isn't safe under process restarts or concurrent access from
  more than one process — using `--reload`, or querying the same `data/chroma` directory from
  a second script while the server is running, has been observed to silently empty a website's
  collection. Restart the server manually after changing backend code. If a website's chat
  answers suddenly go blank, `POST /reindex/{id}` rebuilds its vectors from the pages already
  stored in SQLite — nothing is actually lost, since crawled text/hashes live there, not in Chroma.
