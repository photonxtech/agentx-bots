"""Centralized logging configuration.

Provides a single ``get_logger`` factory so every module logs through a
consistently formatted, correctly leveled logger. Configuration is applied
once, idempotently, on first use.
"""

from __future__ import annotations

import logging
import sys

from app.core.config import get_settings

_CONFIGURED = False

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _configure_root_logger() -> None:
    """Attach a single stdout handler to the root logger, once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers when reloaded (e.g. uvicorn --reload).
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)

    # Tame noisy third-party loggers.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    # This telemetry logger is noisy and buggy in some Chroma builds even with
    # telemetry disabled; silence it entirely.
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
    logging.getLogger("openai").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name`` (typically ``__name__``)."""
    _configure_root_logger()
    return logging.getLogger(name)
