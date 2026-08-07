"""Embeddings: turn text into vectors with a local sentence-transformers model.

Groq has no embedding endpoint, so we embed locally. This is free, private,
and fast enough for a prototype. The model downloads once on first use.
"""

from __future__ import annotations

import threading

import numpy as np

import config

_model = None
_model_lock = threading.Lock()


def get_model():
    """Load the embedding model once and cache it.

    Double-checked locking: rag.evaluation.evaluate() fires several metrics
    concurrently in a thread pool, and on the very first request they can all
    reach here before `_model` is set, each starting its own (slow, memory-
    heavy) SentenceTransformer load. The lock makes only the first caller
    actually load it; the outer unlocked check keeps every later call
    (the common case) lock-free.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of strings -> L2-normalized float32 matrix (n, dim).

    Normalizing means a dot product equals cosine similarity, which is what
    the vector store expects.
    """
    model = get_model()
    vecs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs.astype("float32")


def embedding_dim() -> int:
    return get_model().get_sentence_embedding_dimension()
