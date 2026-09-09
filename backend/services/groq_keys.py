"""
Groq API key pool with automatic failover.

Multiple keys from separate Groq organisation accounts are treated as one pool.
When a key is rate-limited (429) or rejected (401/403), the request is retried
on the next key. Two keys on separate accounts therefore roughly double the
effective requests-per-minute.

Rotation happens at the **httpx transport** layer rather than in application
code. That is the only place it can work universally: DeepEval reaches Groq
through the OpenAI SDK while answer generation uses raw httpx — different
libraries, but both ultimately send via httpx and both accept an injected
client. Retrying inside them would mean patching their internals.

Configure in .env, any of these forms:

    GROQ_API_KEY=gsk_primary
    GROQ_API_KEY_2=gsk_second_account
    GROQ_API_KEY_3=gsk_third_account

or a single comma-separated list, which takes precedence:

    GROQ_API_KEYS=gsk_primary,gsk_second_account
"""
import os
import time
import logging
import threading
import itertools

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Status codes worth trying another account for. 429 is the rate limit; 401/403
# mean that key is dead, in which case the pool should carry on without it.
_ROTATE_ON = {401, 403, 429}

_lock = threading.Lock()
_rotations = 0          # how many times a request had to move to another key
_active_index = 0       # index of the key currently preferred


def _placeholder(value: str) -> bool:
    return not value or value.startswith("your_")


def _is_groq(request: httpx.Request) -> bool:
    """Only Groq requests may have their Authorization header rewritten.

    These clients are also used for an OpenAI-compatible gateway (see
    `llm_endpoint`), which carries its own token. Rotating a `gsk_...` key over
    that header would authenticate every gateway request as nobody.
    """
    return request.url.host == "api.groq.com"


def available_keys() -> list[str]:
    """Ordered, de-duplicated key pool."""
    listed = os.getenv("GROQ_API_KEYS", "")
    if listed.strip():
        candidates = [k.strip() for k in listed.split(",")]
    else:
        candidates = [os.getenv("GROQ_API_KEY", "")]
        # GROQ_API_KEY_2 .. _9, plus an explicitly named FALLBACK
        candidates += [os.getenv(f"GROQ_API_KEY_{i}", "") for i in range(2, 10)]
        candidates.append(os.getenv("GROQ_API_KEY_FALLBACK", ""))

    keys, seen = [], set()
    for k in candidates:
        k = (k or "").strip()
        if k and not _placeholder(k) and k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def key_pool_size() -> int:
    return len(available_keys())


def rotation_stats() -> dict:
    with _lock:
        return {"keys": key_pool_size(), "rotations": _rotations, "active_key_index": _active_index}


def reset_rotation_stats() -> None:
    global _rotations
    with _lock:
        _rotations = 0


def _note_rotation(new_index: int) -> None:
    global _rotations, _active_index
    with _lock:
        _rotations += 1
        _active_index = new_index


def _ordered_attempts(keys: list[str]) -> list[int]:
    """Indices to try, starting from the currently-active key so a rotation
    sticks instead of hammering an exhausted account on every request."""
    with _lock:
        start = _active_index if _active_index < len(keys) else 0
    return [(start + off) % len(keys) for off in range(len(keys))]


class RotatingKeyTransport(httpx.HTTPTransport):
    """Sync transport that retries the request on the next key after a 429/401/403."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        keys = available_keys()
        if len(keys) <= 1 or not _is_groq(request):
            return super().handle_request(request)

        # Materialise the body once so it can be replayed on retry. A streaming
        # body could not be resent, so leave those alone.
        try:
            body = request.read()
        except Exception:
            return super().handle_request(request)

        last: httpx.Response | None = None
        for attempt, idx in enumerate(_ordered_attempts(keys)):
            retry = httpx.Request(
                method=request.method, url=request.url,
                headers=request.headers, content=body,
                extensions=request.extensions,
            )
            retry.headers["Authorization"] = f"Bearer {keys[idx]}"
            response = super().handle_request(retry)

            if response.status_code not in _ROTATE_ON:
                if attempt:
                    _note_rotation(idx)
                    logger.info("Groq request succeeded on key #%d after rotation", idx + 1)
                return response

            response.read()
            response.close()
            last = response
            logger.warning(
                "Groq key #%d returned %s — trying next account", idx + 1, response.status_code
            )
            # A brief pause before the next account; a burst that tripped one
            # limit will usually trip the next one immediately otherwise.
            time.sleep(0.5)

        _note_rotation(0)
        return last if last is not None else super().handle_request(request)


class AsyncRotatingKeyTransport(httpx.AsyncHTTPTransport):
    """Async counterpart of RotatingKeyTransport."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import asyncio

        keys = available_keys()
        if len(keys) <= 1 or not _is_groq(request):
            return await super().handle_async_request(request)

        try:
            body = await request.aread()
        except Exception:
            return await super().handle_async_request(request)

        last: httpx.Response | None = None
        for attempt, idx in enumerate(_ordered_attempts(keys)):
            retry = httpx.Request(
                method=request.method, url=request.url,
                headers=request.headers, content=body,
                extensions=request.extensions,
            )
            retry.headers["Authorization"] = f"Bearer {keys[idx]}"
            response = await super().handle_async_request(retry)

            if response.status_code not in _ROTATE_ON:
                if attempt:
                    _note_rotation(idx)
                    logger.info("Groq request succeeded on key #%d after rotation", idx + 1)
                return response

            await response.aread()
            await response.aclose()
            last = response
            logger.warning(
                "Groq key #%d returned %s — trying next account", idx + 1, response.status_code
            )
            await asyncio.sleep(0.5)

        _note_rotation(0)
        return last if last is not None else await super().handle_async_request(request)


def primary_key() -> str:
    """First usable key — what clients are constructed with before rotation."""
    keys = available_keys()
    return keys[0] if keys else ""


def make_client(timeout: float = 90.0) -> httpx.Client:
    return httpx.Client(transport=RotatingKeyTransport(), timeout=timeout)


def make_async_client(timeout: float = 90.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=AsyncRotatingKeyTransport(), timeout=timeout)
