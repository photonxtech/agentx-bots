# Multi-Website AI Chatbot Platform — Design Spec

Date: 2026-07-09

## Overview

A platform where an admin registers website URLs. The system crawls each site, extracts clean textual content, chunks and embeds it, and stores vectors in a per-website ChromaDB collection. End users chat with a floating widget that answers strictly from that website's indexed content, citing sources and a confidence score, with a fixed fallback response when no relevant content is found. A daily scheduler keeps each site's index in sync (new/updated/deleted/unchanged pages) via content-hash diffing.

Everything lives under `Ai/` (no Docker). Backend: Python/FastAPI in a local venv. Frontend: React/Vite/TypeScript/MUI.

## Decisions locked in during brainstorming

- **Frontend**: one React app, two areas — `/admin/*` dashboard and `/` demo page hosting the chat widget (not two separate apps).
- **Admin auth**: simple JWT login (single admin user via env-configured credentials), protecting all admin-mutating endpoints. Chat and public read endpoints stay open (end users don't log in).
- **Streaming**: real token-level streaming via Server-Sent Events (SSE) from the Groq API through FastAPI to the frontend.
- **Run workflow**: convenience shell scripts (`setup.sh`, `backend/run.sh`, `frontend/run.sh`) on top of manual venv/uvicorn/npm commands, documented in the README. No Docker.
- **Crawl limits**: configurable defaults — max 500 pages/site, max crawl depth 5, 2 concurrent Playwright pages, 1s delay per domain between requests.
- **Confidence score**: derived from vector-retrieval similarity (normalized cosine similarity of the best surviving chunk(s)), not LLM self-rating.

## Project structure

```
Ai/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── database/{models.py, session.py}
│   │   ├── api/
│   │   │   ├── deps.py
│   │   │   └── routes/{auth.py, websites.py, crawl.py, pages.py, chat.py, analytics.py}
│   │   ├── crawler/       # sitemap discovery, recursive fallback, URL filtering, politeness
│   │   ├── extractor/     # Playwright render + Trafilatura + BS4 fallback + cleaning
│   │   ├── chunker/       # RecursiveCharacterTextSplitter wrapper
│   │   ├── embeddings/    # HuggingFaceEmbeddings singleton
│   │   ├── vectordb/      # Chroma client + per-website collection helpers
│   │   ├── llm/           # Groq client, prompt templates, RAG chain, SSE streaming
│   │   ├── scheduler/     # APScheduler daily sync job
│   │   ├── services/      # crawl_service, chat_service, sync_service — orchestration layer
│   │   └── utils/         # logging, retry decorator, hashing, url utils
│   ├── data/               # sqlite db + chroma persistent dir (gitignored)
│   ├── requirements.txt
│   ├── .env.example
│   └── run.sh
├── frontend/
│   ├── src/
│   │   ├── admin/          # dashboard pages/components
│   │   ├── widget/         # floating button, chat window, sidebar
│   │   ├── api/             # axios client + React Query hooks
│   │   ├── context/         # theme + auth context
│   │   ├── components/      # shared UI
│   │   └── App.tsx / router
│   ├── package.json
│   └── run.sh
├── setup.sh
├── README.md
└── .gitignore
```

Layering rule: API routes validate input/auth and delegate to `services/`; only `services/` calls into crawler/extractor/chunker/embeddings/vectordb/llm directly. Each module is independently testable.

## Data model (SQLite via SQLAlchemy)

- **Website**: id, url, name, logo_url, status (active/crawling/error), sitemap_url, crawl_depth_limit, max_pages, created_at, updated_at, last_synced_at
- **Page**: id, website_id (FK), url, title, content_hash (SHA256), status (indexed/failed/skipped/deleted), last_crawled_at, indexed_at
- **Conversation**: id, website_id (FK), session_id, title, created_at
- **Message**: id, conversation_id (FK), role (user/assistant), content, sources (JSON), confidence, feedback (up/down/null), created_at
- **AdminUser**: id, email, hashed_password
- **CrawlJob**: id, website_id (FK), status, pages_found, pages_indexed, pages_failed, started_at, finished_at, error_log

ChromaDB holds the vectors (one persistent collection per website, `website_{id}`), with metadata: website_id, page_id, title, url, chunk_id, hash. SQLite is the source of truth for pages/hashes used in diffing; embeddings never live in SQLite.

## Pipeline: crawl → extract → chunk → embed → store

1. **Crawl**: try `/sitemap.xml` (and sitemap index files) first, extracting all `<url>` entries. If missing/empty, fall back to recursive BFS from the homepage following same-domain links, bounded by `max_pages` (default 500) and `crawl_depth_limit` (default 5). Filter out non-http(s) schemes (`mailto:`, `tel:`), external domains, and image/video/pdf/zip/css/js URLs. Politeness: 1s delay per domain, max 2 concurrent Playwright pages (all configurable via env/website settings). Retry transient failures with backoff.
2. **Extract**: Playwright loads each URL headless, waits for network-idle, grabs rendered HTML. Trafilatura extracts main content; if the result is below a minimum length threshold, fall back to BeautifulSoup, stripping `<nav>`, `<footer>`, `<script>`, `<style>`, and common cookie-banner/ad/social-icon selectors, keeping only visible textual content.
3. **Hash & diff**: SHA256 of extracted text, compared against the stored `Page.content_hash` to classify new/updated/unchanged/deleted — used on both initial crawl and daily sync.
4. **Chunk**: `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` per page.
5. **Embed**: `sentence-transformers/all-MiniLM-L6-v2` via LangChain `HuggingFaceEmbeddings`, loaded once as a process-wide singleton.
6. **Store**: upsert chunks into the website's Chroma collection with the metadata schema above. On "updated," delete existing vectors for that `page_id` first, then insert fresh ones. On "deleted" page, delete its vectors and mark the `Page` row deleted.

## Chat / RAG flow

- Embed the user's question; query only the target website's Chroma collection for top-5 chunks.
- Similarity filter: drop chunks below a configurable cosine-similarity threshold (default ~0.3). If none survive, skip the Groq call and return the fixed fallback: *"I couldn't find that information on this website."*
- Confidence score: normalized similarity of the best surviving chunk(s), surfaced as a 0–100% value.
- Prompt construction: system prompt restricts Groq's `llama-3.3-70b-versatile` to answer only from the provided context and to say it doesn't know when context is insufficient (never hallucinate).
- `POST /chat` streams the Groq token stream to the client via SSE; a final SSE event carries `sources` (page ids/urls/titles) and `confidence`. Completed messages persist to `Conversation`/`Message`.

## Scheduler (daily sync)

APScheduler background job runs once daily per website: re-discover the current URL set (sitemap/crawl), hash-diff against stored pages — new → index, changed → delete old vectors + reindex, missing → delete vectors + mark page deleted, unchanged → skip. Manual `Sync` (same diff logic, on demand) and `Reindex` (force full re-embed of all current pages) admin actions reuse these same service functions.

## API surface

- `POST /auth/login` → JWT (protects all admin-mutating routes below)
- `POST /websites`, `GET /websites`, `PATCH /websites/{id}`, `DELETE /website/{id}` (admin)
- `POST /crawl/{id}`, `POST /sync/{id}`, `POST /reindex/{id}` (admin)
- `GET /pages/{id}` (public — pages for a given website)
- `GET /analytics/{id}` (admin — chat volume, feedback, top questions)
- `POST /chat` (public, SSE streaming)
- `GET /conversations`, `DELETE /conversations/{id}` (public, scoped by session)
- `POST /messages/{id}/feedback` (public — thumbs up/down)

## Frontend

- **Router**: `/admin/login`, `/admin` (website CRUD, crawl status, indexed pages, analytics), `/` (demo page hosting the widget).
- **Widget**: `FloatingButton` (bottom-right) → Framer-Motion-animated `ChatWindow` — header (logo, name, "AI Assistant", online dot), message list (Markdown + code blocks, typing/loading animation, autoscroll), input (multiline, Enter to send / Shift+Enter for newline, suggested questions, clear chat), plus per-message copy/thumbs-up/down/regenerate. `Sidebar`: recent conversations, new chat, delete conversation.
- **State**: React Query for all server data; a small custom hook consumes the chat SSE stream into local component state; Context API for theme (dark/light) and auth token.
- **Styling**: MUI theme, glassmorphism panels (backdrop-blur, translucent surfaces) for the widget, rounded corners, responsive down to mobile widths.

## Config, logging, error handling

- All secrets/config via `.env` (Groq API key, JWT secret, admin credentials, crawl limits, similarity threshold, CORS origins) — never hardcoded.
- Structured logging (Python `logging`, configurable level) across crawler/extractor/llm/scheduler.
- Retry-with-backoff decorator applied to network calls (Playwright navigation, Groq requests, embedding calls).
- FastAPI global exception handlers return consistent JSON error shapes; frontend surfaces failures via MUI snackbars/toasts without crashing the widget.

## Out of scope for this build

- Multi-tenant/multi-admin-user management (single admin login only).
- True multi-domain embeddable widget bundle (widget lives inside the same React app, not a separate distributable script).
- Horizontal scaling / distributed crawling (single-process APScheduler + sequential-ish crawl with bounded concurrency).
