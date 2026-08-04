"""Vision: describe images with a Groq multimodal model.

OCR (ingestion.py) only reads printed text. This module asks a vision LLM
what an image actually *shows* — charts, tables, photos, layouts — and that
description is indexed alongside the OCR text, so visual content becomes
searchable and answerable.

Every call degrades gracefully: on any failure (no API key, offline, rate
limit, unsupported image) we return "" and the pipeline continues OCR-only,
exactly as before.
"""

from __future__ import annotations

import base64
import io
import re
import threading
import time

import config

# Reasoning models (e.g. qwen3.6) may wrap their chain-of-thought in <think>...</think>.
# We ask for reasoning_effort="none", but strip any leftover block defensively so the
# indexed description never contains raw reasoning.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Parse Groq's "Please try again in 7m48.72s" / "3.5s" / "120ms" hint.
_RETRY_AFTER_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)\b", re.IGNORECASE)

# The vision model is rate-limited per minute AND per day. Serialize calls and
# space them out so ingesting a multi-image file doesn't fire a burst that trips
# the per-minute limit. A module-level lock + timestamp is enough (ingestion runs
# in FastAPI's threadpool).
_throttle_lock = threading.Lock()
_last_call_ts = 0.0


def _retry_after_seconds(message: str) -> float | None:
    """Extract the suggested wait (seconds) from a 429 message, or None."""
    m = _RETRY_AFTER_RE.search(message or "")
    if not m:
        return None
    minutes = int(m.group(1)) if m.group(1) else 0
    value = float(m.group(2))
    seconds = value / 1000.0 if m.group(3).lower() == "ms" else value
    return minutes * 60 + seconds


def _throttle() -> None:
    """Block until at least VISION_MIN_INTERVAL has elapsed since the last call."""
    global _last_call_ts
    with _throttle_lock:
        wait = config.VISION_MIN_INTERVAL - (time.monotonic() - _last_call_ts)
        if wait > 0:
            time.sleep(wait)
        _last_call_ts = time.monotonic()

VISION_PROMPT = (
    "Describe this image for a document-search index. Be factual and specific:\n"
    "- If it is a chart/graph: state the chart type, axes, series, and the key "
    "numbers or trends visible.\n"
    "- If it is a table: reproduce the rows and values as text.\n"
    "- If it is a form, bill, or receipt: list the field names and their values.\n"
    "- If it is a photo/diagram/logo: say what it depicts in 1-2 sentences.\n"
    "Transcribe any visible text exactly. Do not speculate beyond what is shown."
)


def _to_jpeg_data_url(image) -> str:
    """PIL image -> base64 JPEG data URL, downscaled to keep the request small.

    Groq caps base64 image payloads (~4 MB), and detail beyond ~1.3k px on the
    long side adds cost without helping the description.
    """
    img = image.convert("RGB")
    if max(img.size) > config.VISION_MAX_DIM:
        img.thumbnail((config.VISION_MAX_DIM, config.VISION_MAX_DIM))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def describe_image(image) -> str:
    """Describe a PIL image via the Groq vision model. Returns "" on failure.

    Retries transient (per-minute) rate limits, honoring the API's suggested
    wait. Gives up immediately when the wait is long (per-day quota exhausted) or
    retries are used up, so ingestion degrades gracefully instead of hanging.
    """
    if not config.VISION_ENABLED:
        return ""
    from rag.generator import _client  # lazy: avoid import cycle at module load

    data_url = _to_jpeg_data_url(image)
    for attempt in range(config.VISION_MAX_RETRIES + 1):
        _throttle()
        try:
            resp = _client().chat.completions.create(
                model=config.VISION_MODEL,
                temperature=0.2,
                max_tokens=config.VISION_MAX_TOKENS,
                reasoning_effort="none",  # skip chain-of-thought; we only want the description
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
            )
            raw = resp.choices[0].message.content or ""
            return _THINK_RE.sub("", raw).strip()
        except Exception as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "rate_limit" in msg.lower()
            wait = _retry_after_seconds(msg)
            # Retry only transient per-minute limits with a short, bounded wait.
            if (
                is_rate_limit
                and attempt < config.VISION_MAX_RETRIES
                and wait is not None
                and wait <= config.VISION_MAX_RETRY_WAIT
            ):
                time.sleep(wait + 0.5)
                continue
            return ""  # non-retryable, quota exhausted, or out of retries
    return ""
