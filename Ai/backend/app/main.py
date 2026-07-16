from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.deps import get_current_admin
from app.api.routes import analytics, auth, chat, conversations, crawl, evaluate, pages, websites
from app.config import settings
from app.database.models import AdminUser
from app.database.session import SessionLocal, init_db
from app.scheduler.setup import shutdown_scheduler, start_scheduler
from app.services.auth_service import seed_admin_user
from app.utils.logging import get_logger
from app.utils.metrics import chat_metrics

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    db = SessionLocal()
    try:
        seed_admin_user(db)
    finally:
        db.close()
    start_scheduler()
    logger.info("Application startup complete")
    yield
    shutdown_scheduler()
    logger.info("Application shutdown complete")


app = FastAPI(title="Multi-Website AI Chatbot Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(auth.router)
app.include_router(websites.router)
app.include_router(crawl.router)
app.include_router(pages.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(analytics.router)
app.include_router(evaluate.router)


@app.get("/health")
def health() -> dict:
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        database_status = "ok"
    except Exception:
        logger.exception("Health check database probe failed")
        database_status = "error"
    finally:
        db.close()
    return {"status": "ok", "database": database_status}


@app.get("/metrics")
def metrics(_admin: AdminUser = Depends(get_current_admin)) -> dict:
    # Lightweight in-process observability (request counts, cache hit rate, average
    # latency/tokens) — not a Prometheus-format exporter, but enough to see system health
    # and cost at a glance without standing up an external metrics stack.
    return chat_metrics.snapshot()
