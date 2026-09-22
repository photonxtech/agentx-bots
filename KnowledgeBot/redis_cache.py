"""Caches the expensive, content-derived output of ingestion — extracted
text, title, and the generated summary — keyed by a hash of the source
(a URL, or a file's actual bytes). This is purely a speed/cost
optimization: if the same link or file shows up again (same user or a
different one), skip straight past fetching/extraction/OCR/vision/
summarization and reuse what was already computed.

Deliberately fails open: if Redis isn't configured or isn't reachable,
every function here just returns None / does nothing, and the ingestion
pipeline falls back to doing the full work, exactly as if this module
didn't exist. Caching should never be a hard dependency.
"""

import hashlib
import json
import logging
import os

import redis

log = logging.getLogger("knowledgebot")

_DEFAULT_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days — long enough to help with
                                          # repeat shares, short enough that
                                          # stale content doesn't linger forever

_client = None
_connection_failed = False  # tried once and failed -> stop retrying every call


def _get_client():
    global _client, _connection_failed
    if _connection_failed:
        return None
    if _client is not None:
        return _client

    redis_url = os.environ.get("REDIS_URL", "")
    if not redis_url:
        log.info(
            "REDIS_URL not set — content caching is disabled (optional; "
            "see .env.example). Every link/file will be freshly processed, "
            "even repeats."
        )
        _connection_failed = True  # not configured — same effect as unreachable
        return None

    try:
        client = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=3)
        client.ping()
        log.info("Connected to Redis content cache")
        _client = client
        return _client
    except Exception as e:
        log.warning(f"Redis unavailable, content caching disabled: {e}")
        _connection_failed = True
        return None


def is_available() -> bool:
    """Checked once at startup so the terminal log makes it obvious whether
    caching is actually active for this run, rather than only surfacing on
    the first ingestion — which is easy to misread as a genuine cache miss.
    """
    return _get_client() is not None


def _key_for_url(url: str) -> str:
    return "content_cache:url:" + hashlib.sha256(url.encode("utf-8")).hexdigest()


def _key_for_file_bytes(file_bytes: bytes) -> str:
    return "content_cache:file:" + hashlib.sha256(file_bytes).hexdigest()


def get_cached_url(url: str) -> dict | None:
    client = _get_client()
    if not client:
        return None
    try:
        raw = client.get(_key_for_url(url))
        return json.loads(raw) if raw else None
    except Exception as e:
        log.warning(f"Redis get failed (url): {e}")
        return None


def set_cached_url(url: str, title: str, text: str, summary: str, ttl: int = _DEFAULT_TTL_SECONDS):
    client = _get_client()
    if not client:
        return
    try:
        payload = json.dumps({"title": title, "text": text, "summary": summary})
        client.setex(_key_for_url(url), ttl, payload)
    except Exception as e:
        log.warning(f"Redis set failed (url): {e}")


def get_cached_file(file_bytes: bytes) -> dict | None:
    client = _get_client()
    if not client:
        return None
    try:
        raw = client.get(_key_for_file_bytes(file_bytes))
        return json.loads(raw) if raw else None
    except Exception as e:
        log.warning(f"Redis get failed (file): {e}")
        return None


def set_cached_file(file_bytes: bytes, text: str, source_type: str, summary: str, ttl: int = _DEFAULT_TTL_SECONDS):
    client = _get_client()
    if not client:
        return
    try:
        payload = json.dumps({"text": text, "source_type": source_type, "summary": summary})
        client.setex(_key_for_file_bytes(file_bytes), ttl, payload)
    except Exception as e:
        log.warning(f"Redis set failed (file): {e}")