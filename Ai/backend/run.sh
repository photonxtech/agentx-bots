#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source venv/bin/activate
# No --reload: ChromaDB's default (embedded SQLite) persistence isn't safe under process
# restarts or concurrent access — using --reload here has been observed to silently wipe
# collections when the file watcher restarts the worker. Restart manually after code changes.
# --host 0.0.0.0: uvicorn defaults to 127.0.0.1 (localhost-only), which a host like Render
# can't reach — its port scanner checks 0.0.0.0 and times out the deploy otherwise.
# $PORT: hosts like Render assign the port dynamically via this env var; 8001 is only the
# local-dev fallback.
uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8001}"
