"""Unit tests for rag.vectorstore's TOC/page-aware retrieval:
  * search() masking (NORMAL_QUERY path) — existing behavior, pinned down.
  * get_toc_chunks() / get_by_pages() — the new deterministic, metadata-only
    retrieval paths for TOC_QUERY / PAGE_QUERY.

Uses `_BaseStore` directly with the embedding model monkeypatched out (a
fixed-size fake vector), so these run fast with no model download/GPU and no
real Weaviate/Chroma connection — they only exercise the in-memory
mask/filter/sort logic, which is identical for both real backends.
"""

import numpy as np
import pytest

import config
from rag.ingestion import Document
from rag.vectorstore import _BaseStore


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch):
    """Every doc/query embeds to the same fixed vector — good enough for
    testing mask/filter/sort logic, which doesn't depend on embedding quality.
    """
    monkeypatch.setattr("rag.vectorstore.embedding_dim", lambda: 4)

    def fake_embed(texts):
        return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"), (len(texts), 1))

    monkeypatch.setattr("rag.vectorstore.embed", fake_embed)


def _store(docs: list[Document]) -> _BaseStore:
    store = _BaseStore()
    store.documents = docs
    store._matrix = (
        np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"), (len(docs), 1))
        if docs
        else store._matrix
    )
    store._rebuild_bm25()
    return store


def _doc(text, page, is_index=False, chat_id="chat1", **extra):
    meta = {"page": page, "is_index": is_index, "chat_id": chat_id, **extra}
    return Document(text=text, source="doc.pdf", kind="pdf", meta=meta)


# --------------------------------------------------------------------------- #
# search() — TOC exclusion mask (NORMAL_QUERY)
# --------------------------------------------------------------------------- #
def test_search_excludes_toc_page_by_default(monkeypatch):
    monkeypatch.setattr(config, "EXCLUDE_INDEX_PAGES", True)
    store = _store([
        _doc("table of contents entry alpha beta", page=1, is_index=True),
        _doc("real content about widgets and gadgets", page=2, is_index=False),
    ])
    results = store.search("widgets", top_k=5, chat_id="chat1")
    assert [d.meta["page"] for d, _ in results] == [2]


def test_search_includes_toc_page_when_flag_disabled(monkeypatch):
    monkeypatch.setattr(config, "EXCLUDE_INDEX_PAGES", False)
    store = _store([
        _doc("table of contents entry alpha beta", page=1, is_index=True),
        _doc("real content about widgets and gadgets", page=2, is_index=False),
    ])
    results = store.search("widgets", top_k=5, chat_id="chat1")
    pages = {d.meta["page"] for d, _ in results}
    assert pages == {1, 2}


def test_search_treats_missing_is_index_as_not_toc(monkeypatch):
    """Old, pre-feature indexed chunks have no is_index key at all — must
    never crash, and must be treated as ordinary (included) content."""
    monkeypatch.setattr(config, "EXCLUDE_INDEX_PAGES", True)
    legacy_doc = Document(text="legacy content, no metadata tag", source="old.pdf",
                           kind="pdf", meta={"page": 1, "chat_id": "chat1"})
    store = _store([legacy_doc])
    results = store.search("legacy", top_k=5, chat_id="chat1")
    assert len(results) == 1


def test_search_is_toc_alias_also_excluded(monkeypatch):
    """A chunk tagged with a future `is_toc` key (instead of `is_index`) is
    excluded the same way — forward compatibility, see vectorstore._is_toc."""
    monkeypatch.setattr(config, "EXCLUDE_INDEX_PAGES", True)
    store = _store([
        Document(text="toc via alias key", source="doc.pdf", kind="pdf",
                  meta={"page": 1, "is_toc": True, "chat_id": "chat1"}),
        _doc("normal content here", page=2, is_index=False),
    ])
    results = store.search("content", top_k=5, chat_id="chat1")
    assert [d.meta["page"] for d, _ in results] == [2]


# --------------------------------------------------------------------------- #
# get_toc_chunks() — deterministic TOC retrieval (TOC_QUERY)
# --------------------------------------------------------------------------- #
def test_get_toc_chunks_returns_sorted_by_page():
    store = _store([
        _doc("toc page five", page=5, is_index=True),
        _doc("content page three", page=3, is_index=False),
        _doc("toc page one", page=1, is_index=True),
    ])
    toc = store.get_toc_chunks(chat_id="chat1")
    assert [d.meta["page"] for d in toc] == [1, 5]


def test_get_toc_chunks_is_chat_isolated():
    store = _store([
        _doc("toc for chat1", page=2, is_index=True, chat_id="chat1"),
        _doc("toc for chat2", page=2, is_index=True, chat_id="chat2"),
    ])
    toc_chat1 = store.get_toc_chunks(chat_id="chat1")
    assert len(toc_chat1) == 1
    assert toc_chat1[0].meta["chat_id"] == "chat1"


def test_get_toc_chunks_empty_when_none_detected():
    store = _store([_doc("only real content", page=1, is_index=False)])
    assert store.get_toc_chunks(chat_id="chat1") == []


def test_get_toc_chunks_multi_page_toc_returns_all_in_order():
    store = _store([
        _doc("toc page 7", page=7, is_index=True),
        _doc("toc page 5", page=5, is_index=True),
        _doc("toc page 6", page=6, is_index=True),
        _doc("real content", page=8, is_index=False),
    ])
    toc = store.get_toc_chunks(chat_id="chat1")
    assert [d.meta["page"] for d in toc] == [5, 6, 7]


# --------------------------------------------------------------------------- #
# get_by_pages() — deterministic page retrieval (PAGE_QUERY)
# --------------------------------------------------------------------------- #
def test_get_by_pages_single_page():
    store = _store([
        _doc("page 247 content", page=247, is_index=False),
        _doc("other page", page=10, is_index=False),
    ])
    hits = store.get_by_pages([247], chat_id="chat1")
    assert [d.meta["page"] for d in hits] == [247]


def test_get_by_pages_returns_toc_page_when_explicitly_requested():
    """A page request must retrieve the page even if it's flagged is_index —
    explicit page requests bypass the TOC-exclusion rule entirely."""
    store = _store([_doc("this happens to be a toc page", page=247, is_index=True)])
    hits = store.get_by_pages([247], chat_id="chat1")
    assert len(hits) == 1
    assert hits[0].meta["page"] == 247


def test_get_by_pages_range_sorted():
    store = _store([
        _doc("page 12", page=12, is_index=False),
        _doc("page 10", page=10, is_index=False),
        _doc("page 15", page=15, is_index=False),
        _doc("page 20", page=20, is_index=False),  # outside requested range
    ])
    hits = store.get_by_pages([10, 12, 15], chat_id="chat1")
    assert [d.meta["page"] for d in hits] == [10, 12, 15]


def test_get_by_pages_missing_page_returns_empty():
    store = _store([_doc("page 1", page=1, is_index=False)])
    assert store.get_by_pages([9999], chat_id="chat1") == []


def test_get_by_pages_is_chat_isolated():
    """The same page number in a different chat's document must never leak
    into this chat's page-retrieval result."""
    store = _store([
        _doc("chat1's page 10", page=10, is_index=False, chat_id="chat1"),
        _doc("chat2's page 10", page=10, is_index=False, chat_id="chat2"),
    ])
    hits = store.get_by_pages([10], chat_id="chat1")
    assert len(hits) == 1
    assert hits[0].meta["chat_id"] == "chat1"


def test_get_by_pages_multiple_chunks_same_page_all_returned():
    """A page split into several child chunks — all of them belong to a page
    request, sorted by their chunk/child order."""
    store = _store([
        Document(text="part b", source="doc.pdf", kind="pdf",
                  meta={"page": 5, "is_index": False, "chat_id": "chat1", "chunk": 0, "child": 1}),
        Document(text="part a", source="doc.pdf", kind="pdf",
                  meta={"page": 5, "is_index": False, "chat_id": "chat1", "chunk": 0, "child": 0}),
    ])
    hits = store.get_by_pages([5], chat_id="chat1")
    assert [d.text for d in hits] == ["part a", "part b"]


# --------------------------------------------------------------------------- #
# Dedup — a page/chunk indexed more than once (duplicate upload, retried
# request) must not occupy multiple result slots in any of the three
# retrieval paths.
# --------------------------------------------------------------------------- #
def test_search_dedupes_exact_duplicate_chunk_text():
    store = _store([
        _doc("this exact chunk got indexed three times", page=1, chat_id="chat1"),
        _doc("this exact chunk got indexed three times", page=1, chat_id="chat1"),
        _doc("this exact chunk got indexed three times", page=1, chat_id="chat1"),
        _doc("a genuinely different chunk", page=2, chat_id="chat1"),
    ])
    results = store.search("chunk", top_k=10, chat_id="chat1")
    texts = [d.text for d, _ in results]
    assert texts.count("this exact chunk got indexed three times") == 1
    assert "a genuinely different chunk" in texts


def test_search_dedup_does_not_starve_top_k_with_duplicates():
    """Regression for the exact bug reported: 3 duplicate copies of one
    chunk must not fill 3 of the top_k slots that should go to distinct
    content."""
    store = _store([
        _doc("duplicate content here", page=1, chat_id="chat1"),
        _doc("duplicate content here", page=1, chat_id="chat1"),
        _doc("duplicate content here", page=1, chat_id="chat1"),
        _doc("distinct content alpha", page=2, chat_id="chat1"),
        _doc("distinct content beta", page=3, chat_id="chat1"),
        _doc("distinct content gamma", page=4, chat_id="chat1"),
    ])
    results = store.search("content", top_k=4, chat_id="chat1")
    texts = {d.text for d, _ in results}
    # With dedup, all 4 distinct texts fit in top_k=4; without it, duplicates
    # would have crowded out at least one of the distinct ones.
    assert texts == {
        "duplicate content here",
        "distinct content alpha",
        "distinct content beta",
        "distinct content gamma",
    }


def test_search_dedup_is_scoped_per_chat():
    """Two DIFFERENT chats sharing identical boilerplate text (e.g. the same
    template document uploaded to each) is not a bug — dedup must not
    collapse across chats."""
    store = _store([
        _doc("identical boilerplate paragraph", page=1, chat_id="chat1"),
        _doc("identical boilerplate paragraph", page=1, chat_id="chat2"),
    ])
    hits_chat1 = store.search("boilerplate", top_k=10, chat_id="chat1")
    hits_chat2 = store.search("boilerplate", top_k=10, chat_id="chat2")
    assert len(hits_chat1) == 1
    assert len(hits_chat2) == 1


def test_get_toc_chunks_dedupes_duplicate_toc_pages():
    store = _store([
        _doc("table of contents entry alpha", page=1, is_index=True),
        _doc("table of contents entry alpha", page=1, is_index=True),
    ])
    toc = store.get_toc_chunks(chat_id="chat1")
    assert len(toc) == 1


def test_get_by_pages_dedupes_duplicate_page_chunks():
    store = _store([
        _doc("page 247 duplicated content", page=247),
        _doc("page 247 duplicated content", page=247),
        _doc("page 247 duplicated content", page=247),
    ])
    hits = store.get_by_pages([247], chat_id="chat1")
    assert len(hits) == 1
