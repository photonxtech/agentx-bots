"""FastAPI application entrypoint.

Wires routers, global exception handling, and the app lifecycle. Run with:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import ask, ingest
from app.core.config import get_settings
from app.core.exceptions import PhotonXError
from app.core.logger import get_logger
from app.services.container import get_container

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the container on startup so config/DB errors surface immediately."""
    logger.info("Starting PhotonX RAG API...")
    get_container()  # fail fast on misconfiguration
    yield
    logger.info("Shutting down PhotonX RAG API.")


app = FastAPI(
    title="Buddy — PhotonX Documentation Assistant",
    description=(
        "A retrieval-augmented Q&A API that answers strictly from PhotonX "
        "documentation and never hallucinates."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the browser-based React frontend to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(ask.router)


@app.exception_handler(PhotonXError)
async def photonx_error_handler(_: Request, exc: PhotonXError) -> JSONResponse:
    """Catch any domain error that escaped a route and return a clean 500."""
    logger.error("Unhandled domain error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"service": "Buddy — PhotonX Documentation Assistant", "docs": "/docs"}
