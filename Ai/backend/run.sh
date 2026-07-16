#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source venv/bin/activate
# No --reload: ChromaDB's default (embedded SQLite) persistence isn't safe under process
# restarts or concurrent access — using --reload here has been observed to silently wipe
# collections when the file watcher restarts the worker. Restart manually after code changes.
uvicorn app.main:app --port 8001
