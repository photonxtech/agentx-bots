"""Domain-specific exceptions.

Using a small hierarchy of typed exceptions (rather than bare ``Exception``)
lets the API layer map each failure class to the right HTTP status and a clean
message, without leaking stack traces or vendor details to callers.
"""

from __future__ import annotations


class PhotonXError(Exception):
    """Base class for all application errors."""


class DocumentError(PhotonXError):
    """Problems reading or parsing source documents (missing/empty PDFs)."""


class EmbeddingError(PhotonXError):
    """Failures while generating embeddings from OpenAI."""


class VectorStoreError(PhotonXError):
    """Failures interacting with ChromaDB."""


class LLMError(PhotonXError):
    """Failures while calling the chat-completions model."""
