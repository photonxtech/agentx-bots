"""
Where chat-completion requests are sent.

The app talks to exactly one wire format — OpenAI's `/chat/completions` — so the
endpoint is a runtime choice rather than a code change. Two targets exist:

  * **Groq direct** (default). Key rotation in `groq_keys` supplies the failover.
  * **An OpenAI-compatible gateway** such as OmniRoute, set via `LLM_GATEWAY_URL`.
    The gateway adds a *provider* fallback axis underneath the two this app
    already has (key rotation, then model chaining), so a Groq outage or an
    exhausted daily quota degrades to another provider instead of failing.

The gateway is used only when it answers a health probe. A configured but dead
gateway therefore falls back to Groq direct rather than becoming a new single
point of failure — the failure mode a gateway is supposed to remove.

DELIBERATELY NOT ROUTED HERE: the DeepEval judge (`deepeval_llm.build_groq_judge`).
Metric scores are only comparable across runs when the judge is held fixed, and
a gateway that silently swaps providers would make substitutions invisible —
the exact thing `deepeval_llm._note_model_fallback` exists to report. The judge
stays on Groq with its explicit model chain.

.env:

    LLM_GATEWAY_URL=http://localhost:20128/v1   # blank/absent = Groq direct
    LLM_GATEWAY_TOKEN=                          # OmniRoute access token, if set

IMPORTANT, when pointing this at OmniRoute: turn **token compression off** in the
gateway's own config/dashboard. Its default stacked pipeline (RTK -> Caveman)
rewrites prompt text to save tokens. This is an evaluation harness — a retrieved
context that reaches the model compressed is not the context the run claims to
be scoring, and faithfulness/precision numbers computed against it are wrong.
"""
import os
import time
import logging
import threading
from typing import NamedTuple

import httpx
from dotenv import load_dotenv

from backend.services.groq_keys import primary_key

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# How long a probe result is trusted. A short TTL means starting or stopping the
# gateway is picked up without restarting the app.
_PROBE_TTL = 60.0
_PROBE_TIMEOUT = 2.0

_lock = threading.Lock()
_probe: tuple[float, bool] | None = None   # (checked_at, gateway_is_up)


class Endpoint(NamedTuple):
    base_url: str
    token: str
    via_gateway: bool

    @property
    def chat_completions(self) -> str:
        return f"{self.base_url}/chat/completions"


def gateway_url() -> str | None:
    """Configured gateway base URL, or None when the app should use Groq direct."""
    url = os.getenv("LLM_GATEWAY_URL", "").strip().rstrip("/")
    return url or None


def _cached_probe() -> bool | None:
    with _lock:
        if _probe and (time.monotonic() - _probe[0]) < _PROBE_TTL:
            return _probe[1]
    return None


def _store_probe(up: bool) -> bool:
    with _lock:
        global _probe
        was = _probe[1] if _probe else None
        _probe = (time.monotonic(), up)
    if was is not None and was != up:
        logger.warning("LLM gateway %s at %s", "came back up" if up else "went down", gateway_url())
    return up


def _endpoint(up: bool) -> Endpoint:
    """Assemble the endpoint for a known gateway state."""
    url = gateway_url()
    if up and url:
        # OmniRoute works keyless against its free tiers, so an empty token is a
        # valid configuration — send the header only when one is set.
        return Endpoint(url, os.getenv("LLM_GATEWAY_TOKEN", "").strip(), True)
    return Endpoint(GROQ_BASE_URL, primary_key(), False)


def auth_headers(ep: Endpoint) -> dict:
    headers = {"Content-Type": "application/json"}
    if ep.token:
        headers["Authorization"] = f"Bearer {ep.token}"
    return headers


async def resolve_endpoint() -> Endpoint:
    """Endpoint to use for the next request, health-probing the gateway if stale."""
    url = gateway_url()
    if not url:
        return _endpoint(False)

    cached = _cached_probe()
    if cached is not None:
        return _endpoint(cached)

    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
            res = await client.get(f"{url}/models")
        # Any HTTP answer proves something is listening and speaking the API;
        # a 401 only means the token is wrong, which is not a routing decision.
        up = res.status_code < 500
    except Exception as e:
        up = False
        if _cached_probe() is not False:
            logger.warning("LLM gateway %s unreachable (%s) — using Groq direct", url, e)

    return _endpoint(_store_probe(up))


def resolve_endpoint_sync() -> Endpoint:
    """Blocking counterpart of `resolve_endpoint`, for non-async call sites."""
    url = gateway_url()
    if not url:
        return _endpoint(False)

    cached = _cached_probe()
    if cached is not None:
        return _endpoint(cached)

    try:
        with httpx.Client(timeout=_PROBE_TIMEOUT) as client:
            up = client.get(f"{url}/models").status_code < 500
    except Exception as e:
        up = False
        if _cached_probe() is not False:
            logger.warning("LLM gateway %s unreachable (%s) — using Groq direct", url, e)

    return _endpoint(_store_probe(up))


_models_cache: tuple[float, frozenset[str]] | None = None


def gateway_models() -> frozenset[str]:
    """Model ids the gateway currently serves; empty when it is not in use.

    Lets a pinned model be checked before a run commits to it, instead of
    discovering a bad id partway through and losing the work already done.
    """
    global _models_cache
    ep = resolve_endpoint_sync()
    if not ep.via_gateway:
        return frozenset()

    with _lock:
        if _models_cache and (time.monotonic() - _models_cache[0]) < _PROBE_TTL:
            return _models_cache[1]

    try:
        with httpx.Client(timeout=_PROBE_TIMEOUT * 2) as client:
            res = client.get(f"{ep.base_url}/models", headers=auth_headers(ep))
            res.raise_for_status()
            ids = frozenset(m["id"] for m in res.json().get("data", []) if m.get("id"))
    except Exception as e:
        logger.warning("Could not list gateway models: %s", e)
        ids = frozenset()

    with _lock:
        _models_cache = (time.monotonic(), ids)
    return ids


def endpoint_stats() -> dict:
    """For a run's notes: which endpoint served it."""
    ep = _endpoint(bool(_cached_probe()))
    return {
        "gateway_configured": gateway_url() or "",
        "gateway_active": ep.via_gateway,
        "base_url": ep.base_url,
    }
