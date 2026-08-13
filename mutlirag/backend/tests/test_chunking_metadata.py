"""Unit tests: is_index (and other page metadata) must survive PDF page ->
Document -> chunk_documents() -> child chunk. This is the propagation path
the TOC/page-retrieval feature depends on.

Semantic chunking (config.SEMANTIC_CHUNKING_ENABLED) embeds sentences when a
section is split at the parent tier — these fixtures are long enough to hit
that path, so the embedding model is mocked out here too (these tests check
metadata propagation, not chunk-boundary quality; see test_semantic_chunking
for that).
"""

import numpy as np
import pytest

from rag.chunking import chunk_documents
from rag.ingestion import Document


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch):
    def fake_embed(texts):
        return np.tile(np.array([1.0, 0.0], dtype="float32"), (len(texts), 1))

    monkeypatch.setattr("rag.chunking.embed", fake_embed)


def _long_text(n_sentences: int) -> str:
    """Text long enough to force chunk_documents to actually split it,
    instead of taking the "short doc, keep intact" shortcut."""
    sentence = "This is a reasonably long sentence about the document content. "
    return (sentence * n_sentences).strip()


def test_is_index_true_propagates_to_every_child_chunk():
    doc = Document(
        text=_long_text(40),
        source="doc.pdf",
        kind="pdf",
        meta={"page": 5, "ocr": True, "images": ["/tmp/img1.png"], "is_index": True},
    )
    chunks = chunk_documents([doc])
    assert len(chunks) > 1, "fixture text should have been split into multiple chunks"
    for c in chunks:
        assert c.meta["is_index"] is True
        assert c.meta["page"] == 5
        assert c.meta["ocr"] is True
        assert c.meta["images"] == ["/tmp/img1.png"]


def test_is_index_false_propagates_to_every_child_chunk():
    doc = Document(
        text=_long_text(40),
        source="doc.pdf",
        kind="pdf",
        meta={"page": 6, "ocr": False, "images": [], "is_index": False},
    )
    chunks = chunk_documents([doc])
    assert len(chunks) > 1
    for c in chunks:
        assert c.meta["is_index"] is False
        assert c.meta["page"] == 6


def test_short_page_kept_intact_still_keeps_metadata():
    """A page short enough to skip splitting entirely (the `len(doc.text) <=
    child_size` fast path in chunk_documents) must still carry its metadata
    through unchanged."""
    doc = Document(
        text="Short page.",
        source="doc.pdf",
        kind="pdf",
        meta={"page": 1, "is_index": True},
    )
    chunks = chunk_documents([doc])
    assert len(chunks) == 1
    assert chunks[0].meta["is_index"] is True
    assert chunks[0].meta["page"] == 1


def test_chat_id_tag_survives_alongside_is_index():
    """api.service.ingest_file tags chat_id onto chunks AFTER chunk_documents
    returns — confirm that assignment lands on every child, matching how
    per-chat isolation actually works in RagService.ingest_file."""
    doc = Document(
        text=_long_text(40),
        source="doc.pdf",
        kind="pdf",
        meta={"page": 3, "is_index": True},
    )
    chunks = chunk_documents([doc])
    for c in chunks:
        c.meta["chat_id"] = "chat-123"
    assert len(chunks) > 1
    for c in chunks:
        assert c.meta["chat_id"] == "chat-123"
        assert c.meta["is_index"] is True
        assert c.meta["page"] == 3


def test_multiple_pages_do_not_bleed_metadata_into_each_other():
    docs = [
        Document(text=_long_text(30), source="doc.pdf", kind="pdf",
                 meta={"page": 1, "is_index": True}),
        Document(text=_long_text(30), source="doc.pdf", kind="pdf",
                 meta={"page": 2, "is_index": False}),
    ]
    chunks = chunk_documents(docs)
    pages_seen = {c.meta["page"] for c in chunks}
    assert pages_seen == {1, 2}
    for c in chunks:
        expected_is_index = c.meta["page"] == 1
        assert c.meta["is_index"] is expected_is_index
