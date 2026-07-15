# PhotonX Chat — React Frontend

A branded chat UI for the PhotonX Documentation Assistant. It talks to the
FastAPI RAG backend(s), shows grounded answers with **page-level citations**,
and lets you **switch between the two backends** (Manual / LangChain) live from
the header — handy for demos and comparisons.

Built with **Vite + React + TypeScript**. No UI framework, no icon library —
lean by design, matching the backend's philosophy.

## Features

- **Chat interface** with user/assistant bubbles and a typing indicator
- **Citations as source cards** — document · page · section for every answer
- **Backend toggle** — flip between Manual (`:8000`) and LangChain (`:8001`);
  the accent color follows the active backend (teal / violet)
- **Live status strip** — health dot, indexed docs/chunks, and model, pulled
  from `/stats`
- **Re-index button** — triggers `/ingest` and toasts the result
- **Refusal styling** — "couldn't find that…" answers render distinctly, with
  no fake sources
- **Light / dark theme** — follows the OS, with a manual toggle (persisted)
- **Robust errors** — backend-down, timeouts, and API errors surface clearly

## Prerequisites

At least one backend must be running:

```bash
# Manual backend  → http://localhost:8000
cd ../photonx-rag && uvicorn app.main:app --port 8000

# LangChain backend → http://localhost:8001   (optional, for the toggle)
cd ../photonx-rag-langchain && uvicorn app.main:app --port 8001
```

> The backends now include **CORS** allowing `http://localhost:5173` (the Vite
> dev origin) by default. Change it with the `CORS_ORIGINS` env var on the
> backend if you serve the UI from elsewhere.

Make sure you've run `/ingest` at least once so there's something to answer from.

## Run the frontend

```bash
npm install
npm run dev          # → http://localhost:5173
```

To point at different backend URLs, copy `.env.example` to `.env` and edit:

```bash
VITE_MANUAL_URL=http://localhost:8000
VITE_LANGCHAIN_URL=http://localhost:8001
```

## Build for production

```bash
npm run build        # outputs to dist/
npm run preview      # serve the build locally
```

## Project structure

```
src/
├── api/
│   ├── types.ts       # TypeScript mirror of the backend's Pydantic models
│   ├── backends.ts    # the two switchable backends + their URLs
│   └── client.ts      # typed fetch wrapper (errors, timeouts, /ask /ingest /stats /health)
├── components/
│   ├── Header.tsx     # brand, backend toggle, theme toggle
│   ├── StatusStrip.tsx# health + stats + re-index
│   ├── Message.tsx    # chat bubbles + source cards + typing indicator
│   ├── Composer.tsx   # auto-growing input (Enter to send, Shift+Enter newline)
│   └── icons.tsx      # inline SVG icons
├── App.tsx            # state, wiring, theme hook
├── App.css            # component styles
└── index.css          # design tokens (colors, type) + light/dark themes
```

## How it maps to the backend

| UI action | Backend call |
|---|---|
| Send a question | `POST /ask` → `{ answer, sources[] }` |
| Re-index docs | `POST /ingest` → `{ status, documents, chunks }` |
| Status strip | `GET /stats` + `GET /health` |

The answer's `sources` (document, page, section) render as the source cards
under each assistant reply.
