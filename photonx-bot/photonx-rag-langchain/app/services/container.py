"""Composition root.

Builds the single ``RAGPipeline`` and exposes it as a cached singleton. Far
smaller than the manual project's container because LangChain already composes
the internal components for us — we only wire the pipeline into the app.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.core.logger import get_logger
from app.rag.pipeline import RAGPipeline

logger = get_logger(__name__)


class Container:
    """Holds the application's long-lived pipeline."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipeline = RAGPipeline(settings)
        logger.info("Application container initialized (LangChain).")


@lru_cache(maxsize=1)
def get_container() -> Container:
    """Return the process-wide container singleton."""
    return Container(get_settings())
