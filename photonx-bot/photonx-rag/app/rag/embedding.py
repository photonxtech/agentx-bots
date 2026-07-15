"""Embedding generation (STEP 3).

Wraps the OpenAI embeddings endpoint behind a small, batched, retry-aware
interface. The rest of the app depends only on ``embed_texts`` / ``embed_query``
and never on the OpenAI SDK directly, so the provider can be swapped later.
"""

from __future__ import annotations

import time

from openai import OpenAI, OpenAIError

from app.core.exceptions import EmbeddingError
from app.core.logger import get_logger

logger = get_logger(__name__)

# OpenAI accepts up to 2048 inputs per request; keep a safe margin.
_MAX_BATCH = 128
_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 2.0


class Embedder:
    """Generate embeddings for chunk texts and user queries."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, batching requests transparently.

        Args:
            texts: Non-empty strings to embed.

        Returns:
            One embedding vector per input, in the same order.

        Raises:
            EmbeddingError: If the API fails after retries.
        """
        if not texts:
            return []

        vectors: list[list[float]] = []
        total = len(texts)
        start = time.perf_counter()

        for i in range(0, total, _MAX_BATCH):
            batch = texts[i : i + _MAX_BATCH]
            vectors.extend(self._embed_batch(batch))
            logger.debug("Embedded %d/%d texts", min(i + _MAX_BATCH, total), total)

        elapsed = time.perf_counter() - start
        logger.info("Generated %d embeddings in %.2fs", total, elapsed)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._embed_batch([text])[0]

    # -- internals ----------------------------------------------------------
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Embed one batch with bounded exponential-backoff retries."""
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.embeddings.create(
                    model=self._model, input=batch
                )
                # Preserve input order (API returns items with an ``index``).
                ordered = sorted(response.data, key=lambda d: d.index)
                return [item.embedding for item in ordered]
            except OpenAIError as exc:
                last_exc = exc
                wait = _RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    "Embedding attempt %d/%d failed: %s (retrying in %.1fs)",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                    wait,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(wait)

        raise EmbeddingError(
            f"Failed to generate embeddings after {_MAX_RETRIES} attempts: "
            f"{last_exc}"
        ) from last_exc
