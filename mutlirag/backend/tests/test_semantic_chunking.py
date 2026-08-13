"""Unit tests for rag.chunking's semantic boundary splitting
(_split_text_semantic + its wiring into _pack_sections/chunk_documents).

The embedding model is always mocked — these tests check the boundary-
decision LOGIC (adaptive threshold, min-chunk-chars guard, fallback
conditions), not real embedding quality, so no model download/GPU/network
is needed and results are deterministic.
"""

import numpy as np

import config
from rag.chunking import _split_text, _split_text_semantic, chunk_documents
from rag.ingestion import Document

CAT_SENTENCES = [
    "Cats are great pets.",
    "Cats like to sleep.",
    "Cats also enjoy playing with string.",
]
ROCKET_SENTENCES = [
    "Rockets use liquid fuel.",
    "Rockets reach high altitudes.",
    "Rockets are used for space travel.",
]


def _two_topic_text() -> str:
    return " ".join(CAT_SENTENCES + ROCKET_SENTENCES)


def _topic_vector_embed(texts):
    """Deterministic stand-in for the real embedding model: every "Cats..."
    sentence maps to (1, 0), every "Rockets..." sentence to (0, 1) — so the
    only similarity drop in the whole text is exactly at the topic switch."""
    vecs = np.zeros((len(texts), 2), dtype="float32")
    for i, t in enumerate(texts):
        vecs[i] = [1.0, 0.0] if t in CAT_SENTENCES else [0.0, 1.0]
    return vecs


# --------------------------------------------------------------------------- #
# Fallback conditions — must reduce to the plain, embedding-free splitter
# --------------------------------------------------------------------------- #
def test_falls_back_when_too_few_units(monkeypatch):
    def boom(texts):
        raise AssertionError("embed() must not be called below SEMANTIC_CHUNKING_MIN_UNITS")

    monkeypatch.setattr("rag.chunking.embed", boom)
    # Two short sentences (each well under `size`, so _units won't hard-split
    # either one further) but together longer than `size` -> forces entry
    # into the split path with exactly 2 units, below the default MIN of 4.
    text = "Short sentence one. Short sentence two."
    result = _split_text_semantic(text, size=25, overlap=0)
    assert result == _split_text(text, size=25, overlap=0)


def test_falls_back_when_too_many_units(monkeypatch):
    def boom(texts):
        raise AssertionError("embed() must not be called above SEMANTIC_CHUNKING_MAX_UNITS")

    monkeypatch.setattr("rag.chunking.embed", boom)
    monkeypatch.setattr(config, "SEMANTIC_CHUNKING_MAX_UNITS", 5)
    text = " ".join(f"Sentence number {i}." for i in range(20))
    result = _split_text_semantic(text, size=30, overlap=0)
    assert result == _split_text(text, size=30, overlap=0)


def test_falls_back_when_embedding_model_unavailable(monkeypatch):
    def boom(texts):
        raise RuntimeError("model not cached")

    monkeypatch.setattr("rag.chunking.embed", boom)
    text = _two_topic_text()
    result = _split_text_semantic(text, size=40, overlap=0)
    assert result == _split_text(text, size=40, overlap=0)


def test_text_fitting_in_one_chunk_is_never_split(monkeypatch):
    def boom(texts):
        raise AssertionError("embed() must not be called when text already fits")

    monkeypatch.setattr("rag.chunking.embed", boom)
    text = "Short text."
    assert _split_text_semantic(text, size=1000, overlap=0) == [text]


# --------------------------------------------------------------------------- #
# Core behavior — a real topic shift is detected and cut on
# --------------------------------------------------------------------------- #
def test_semantic_break_detected_at_topic_shift(monkeypatch):
    monkeypatch.setattr("rag.chunking.embed", _topic_vector_embed)
    text = _two_topic_text()  # 167 chars total; cat-half alone is 77

    # size=120 is deliberately picked so PLAIN char-budget packing would NOT
    # cut at the topic boundary: it fits all 3 cat sentences + rocket0 (102
    # chars) before tipping over on rocket1 — i.e. budget alone would blend
    # one rocket sentence into the "cat" chunk. Semantic detection must still
    # cut right at the true topic shift (before rocket0), earlier than the
    # budget would have forced on its own. min_chunk_chars low so the guard
    # never blocks it.
    chunks = _split_text_semantic(text, size=120, overlap=0, min_chunk_chars=5)

    assert len(chunks) == 2
    assert all(s in chunks[0] for s in CAT_SENTENCES)
    assert all(s in chunks[1] for s in ROCKET_SENTENCES)
    assert not any(s in chunks[0] for s in ROCKET_SENTENCES)
    assert not any(s in chunks[1] for s in CAT_SENTENCES)


def test_min_chunk_chars_guard_prevents_premature_break(monkeypatch):
    """Even with an aggressively low percentile (every jump looks "big"), a
    semantic cut must never fire before the accumulated chunk reaches
    min_chunk_chars — output degenerates to pure character-budget cuts,
    identical to _split_text."""
    def wild_embed(texts):
        # A different, maximally-dissimilar direction per sentence.
        vecs = np.eye(len(texts), dtype="float32")
        return vecs

    monkeypatch.setattr("rag.chunking.embed", wild_embed)
    text = " ".join(f"Sentence number {i} here." for i in range(10))

    semantic = _split_text_semantic(
        text, size=60, overlap=0, breakpoint_percentile=1, min_chunk_chars=100_000
    )
    plain = _split_text(text, size=60, overlap=0)
    assert semantic == plain


# --------------------------------------------------------------------------- #
# Wiring: config.SEMANTIC_CHUNKING_ENABLED gates whether chunk_documents uses
# the semantic splitter at all.
# --------------------------------------------------------------------------- #
def test_disabled_flag_never_calls_semantic_splitter(monkeypatch):
    monkeypatch.setattr(config, "SEMANTIC_CHUNKING_ENABLED", False)

    def boom(*a, **kw):
        raise AssertionError("_split_text_semantic must not be used when disabled")

    monkeypatch.setattr("rag.chunking._split_text_semantic", boom)

    doc = Document(
        text=_two_topic_text() * 20,  # long enough to force a parent-level split
        source="doc.txt", kind="text", meta={"page": 1},
    )
    chunks = chunk_documents([doc])
    assert len(chunks) >= 1  # just confirming it ran without hitting the spy


def test_enabled_flag_uses_semantic_splitter(monkeypatch):
    """Confirms the PARENT-level oversized-section split routes through
    _split_text_semantic when enabled. Child-level splitting still legitimately
    calls plain _split_text regardless of this flag (semantic boundaries are
    parent-tier only by design), so this spies/counts rather than forbidding
    _split_text outright."""
    import rag.chunking as chunking_mod

    monkeypatch.setattr(config, "SEMANTIC_CHUNKING_ENABLED", True)
    monkeypatch.setattr(chunking_mod, "embed", _topic_vector_embed)

    calls = {"n": 0}
    original = chunking_mod._split_text_semantic

    def spy(*a, **kw):
        calls["n"] += 1
        return original(*a, **kw)

    monkeypatch.setattr(chunking_mod, "_split_text_semantic", spy)

    long_text = " ".join(CAT_SENTENCES * 30 + ROCKET_SENTENCES * 30)
    doc = Document(text=long_text, source="doc.txt", kind="text", meta={"page": 1})
    chunks = chunk_documents([doc])
    assert len(chunks) >= 1
    assert calls["n"] >= 1, "expected the oversized section to be split via _split_text_semantic"
