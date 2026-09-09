"""
FastAPI application — entry point.
Mounts API routes and serves the frontend as static files.
"""
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from backend.database import engine, Base
from backend.routers import sessions, qa, evaluate, feedback

from sqlalchemy import text

# Create all tables on startup (safe — skips existing tables)
Base.metadata.create_all(bind=engine)

# Safe column migrations — add any new columns that don't exist yet
def safe_migrate():
    """Add new columns to existing tables without losing data."""
    migrations = [
        "ALTER TABLE run_configs ADD COLUMN IF NOT EXISTS langsmith_experiment_url VARCHAR(1000)",
        "ALTER TABLE run_configs ADD COLUMN IF NOT EXISTS metrics JSONB",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS qa_meta JSONB",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS documents JSONB",
        "ALTER TABLE run_configs ADD COLUMN IF NOT EXISTS langsmith_summary_run_id VARCHAR(64)",
        # Seed Q&A: user-uploaded manually-created Q&A pairs, kept separate
        # from the system-generated set so re-generation never overwrites them.
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS seed_qa_json JSONB",
    ]
    with engine.connect() as conn:
        for stmt in migrations:
            try:
                conn.execute(text(stmt))
            except Exception as e:
                print(f"Migration skipped ({e})")
        conn.commit()

safe_migrate()


# Report the Groq key pool at startup. .env is read at import time, so a key
# added while the server was running is invisible until restart — this line is
# how you confirm the pool actually picked it up.
def _report_key_pool() -> None:
    from backend.services.groq_keys import available_keys
    keys = available_keys()
    if not keys:
        print("[groq] WARNING: no usable GROQ API key found — generation and scoring will fail")
    else:
        tails = ", ".join(f"...{k[-6:]}" for k in keys)
        suffix = "" if len(keys) > 1 else "  (no fallback — add GROQ_API_KEY_2 in .env)"
        print(f"[groq] key pool: {len(keys)} key(s) [{tails}]{suffix}")

_report_key_pool()

app = FastAPI(
    title="RAG Evaluation UI",
    description="Session-based RAG pipeline evaluation tool",
    version="1.0.0",
    redirect_slashes=False,
)

# CORS — allow all for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(sessions.router)
app.include_router(qa.router)
app.include_router(evaluate.router)
app.include_router(feedback.router)

@app.get("/healthz", include_in_schema=False)
def healthz():
    """Liveness probe for the load balancer.

    Deliberately does NOT touch the database. A brief RDS blip should not cause
    the balancer to kill every healthy task and take the whole service down;
    readiness is a separate question, answered below.
    """
    return {"status": "ok"}


@app.get("/readyz", include_in_schema=False)
def readyz():
    """Readiness probe — reports whether dependencies are actually usable."""
    from fastapi.responses import JSONResponse
    from backend.services.groq_keys import key_pool_size

    checks = {"database": False, "groq_keys": key_pool_size()}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception as e:
        checks["database_error"] = str(e)[:200]

    ready = checks["database"] and checks["groq_keys"] > 0
    return JSONResponse(checks, status_code=200 if ready else 503)


# Serve frontend static files (must be last so API routes take priority)
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
