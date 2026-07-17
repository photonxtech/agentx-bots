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

import config

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
    """Describe a PIL image via the Groq vision model. Returns "" on failure."""
    if not config.VISION_ENABLED:
        return ""
    try:
        from rag.generator import _client  # lazy: avoid import cycle at module load

        resp = _client().chat.completions.create(
            model=config.VISION_MODEL,
            temperature=0.2,
            max_tokens=config.VISION_MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": _to_jpeg_data_url(image)}},
                    ],
                }
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""
