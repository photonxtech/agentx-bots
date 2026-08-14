"""Integration-style tests for RagService.retrieve()'s intent routing:
NORMAL_QUERY still goes through rewrite -> hybrid search -> rerank -> filter
exactly as before; TOC_QUERY/PAGE_QUERY bypass all of that for deterministic
metadata retrieval. No real embedding model, Groq call, or vector DB is used
-— RagService is built without running __init__ (which would connect to
Weaviate/Chroma and load chat history from disk), and a minimal fake store
stands in for VectorStore.
"""

import threading

import pytest

import rag.generator as generator
import rag.reranker as reranker
from api.service import NoDocumentError, RagService
from rag.ingestion import Document


class FakeStore:
    """Just enough of the VectorStore surface for retrieve() to run."""

    def __init__(self, documents):
        self.documents = documents

    def size_for_chat(self, chat_id):
        return sum(1 for d in self.documents if d.meta.get("chat_id") == chat_id)

    def sources_for_chat(self, chat_id):
        seen = {}
        for d in self.documents:
            if d.meta.get("chat_id") == chat_id:
                seen.setdefault(d.source, None)
        return list(seen)

    def search(self, query, top_k=5, chat_id=None, explain=False):
        # NORMAL_QUERY path: real vectorstore.search excludes is_index; the
        # fake mirrors that so the test can tell NORMAL apart from PAGE/TOC.
        matched = [
            d for d in self.documents
            if (chat_id is None or d.meta.get("chat_id") == chat_id) and not d.meta.get("is_index")
        ]
        matched = matched[:top_k]
        if explain:
            return [{"doc": d, "score": 0.9, "dense": 0.9, "bm25": 0.9} for d in matched]
        return [(d, 0.9) for d in matched]

    def get_neighbors(self, doc, window=1):
        # Not under test here — see test_neighbor_expansion.py for the real
        # expansion behavior. Returning none keeps these tests focused on
        # intent routing without needing to model page/chunk adjacency.
        return []

    def get_toc_chunks(self, chat_id=None):
        matched = [
            d for d in self.documents
            if d.meta.get("is_index") and (chat_id is None or d.meta.get("chat_id") == chat_id)
        ]
        return sorted(matched, key=lambda d: d.meta.get("page") or 0)

    def get_by_pages(self, pages, chat_id=None):
        wanted = set(pages)
        matched = [
            d for d in self.documents
            if d.meta.get("page") in wanted and (chat_id is None or d.meta.get("chat_id") == chat_id)
        ]
        return sorted(matched, key=lambda d: d.meta.get("page") or 0)


def _service(documents) -> RagService:
    svc = RagService.__new__(RagService)  # skip __init__ (no real DB/disk)
    svc.store = FakeStore(documents)
    svc._lock = threading.RLock()
    return svc


def _doc(text, page, is_index=False, chat_id="chat1"):
    return Document(text=text, source="doc.pdf", kind="pdf",
                     meta={"page": page, "is_index": is_index, "chat_id": chat_id})


@pytest.fixture(autouse=True)
def _stub_llm_dependent_stages(monkeypatch):
    """NORMAL_QUERY calls generator.classify_followup (a Groq call) and
    reranker.rerank (loads a cross-encoder model) — stub both to identity so
    the NORMAL_QUERY test never needs network/model access."""
    monkeypatch.setattr(
        generator, "classify_followup",
        lambda question, history: {"mode": "standalone", "query": question},
    )
    monkeypatch.setattr(reranker, "rerank", lambda query, hits, protect=None: hits)
    monkeypatch.setattr(reranker, "diversity_select", lambda hits, k, protect=None, lambda_mult=None: hits[:k])


def test_no_document_raises():
    svc = _service([])
    with pytest.raises(NoDocumentError):
        svc.retrieve("chat1", "What is RAG?", [])


def test_normal_query_excludes_toc_and_uses_hybrid_search():
    svc = _service([
        _doc("toc entry", page=1, is_index=True),
        _doc("RAG stands for retrieval augmented generation", page=2, is_index=False),
    ])
    r = svc.retrieve("chat1", "What is RAG?", [])
    assert r["query_type"] == "NORMAL_QUERY"
    assert r["direct_answer"] is None
    assert [d.meta["page"] for d, _ in r["hits"]] == [2]


def test_toc_query_routes_to_deterministic_toc_retrieval():
    svc = _service([
        _doc("toc page 1", page=1, is_index=True),
        _doc("toc page 2", page=2, is_index=True),
        _doc("real content", page=3, is_index=False),
    ])
    r = svc.retrieve("chat1", "Give me the table of contents.", [])
    assert r["query_type"] == "TOC_QUERY"
    assert r["direct_answer"] is None
    assert [d.meta["page"] for d, _ in r["hits"]] == [1, 2]


def test_toc_query_with_no_toc_returns_direct_answer():
    svc = _service([_doc("just content, no toc", page=1, is_index=False)])
    r = svc.retrieve("chat1", "Give me the table of contents.", [])
    assert r["query_type"] == "TOC_QUERY"
    assert r["hits"] == []
    assert r["direct_answer"] is not None
    assert "table of contents" in r["direct_answer"].lower()


def test_page_query_bypasses_toc_exclusion():
    svc = _service([_doc("this page happens to be a toc page", page=247, is_index=True)])
    r = svc.retrieve("chat1", "What does page 247 contain?", [])
    assert r["query_type"] == "PAGE_QUERY"
    assert r["direct_answer"] is None
    assert [d.meta["page"] for d, _ in r["hits"]] == [247]


def test_page_query_range():
    svc = _service([_doc(f"page {p}", page=p, is_index=False) for p in range(8, 20)])
    r = svc.retrieve("chat1", "Summarize pages 10 to 15.", [])
    assert [d.meta["page"] for d, _ in r["hits"]] == [10, 11, 12, 13, 14, 15]


def test_page_query_not_found_returns_direct_answer():
    svc = _service([_doc("page 1", page=1, is_index=False)])
    r = svc.retrieve("chat1", "What is on page 9999?", [])
    assert r["query_type"] == "PAGE_QUERY"
    assert r["hits"] == []
    assert r["direct_answer"] is not None
    assert "9999" in r["direct_answer"]


def test_page_query_never_leaks_another_chats_document():
    svc = _service([
        _doc("chat1 page 10", page=10, is_index=False, chat_id="chat1"),
        _doc("chat2 page 10", page=10, is_index=False, chat_id="chat2"),
    ])
    r = svc.retrieve("chat1", "What does page 10 say?", [])
    assert len(r["hits"]) == 1
    assert r["hits"][0][0].meta["chat_id"] == "chat1"


def test_clarify_followup_reuses_previous_answer_context_without_a_new_search(monkeypatch):
    """Regression test for a real production bug: "more clearly" (a
    clarification of the answer just given, not a new document question)
    was searched fresh and retrieved unrelated content. When
    classify_followup() says "clarify", retrieve() must reuse the previous
    turn's own persisted context/sources and must NOT touch the store at all."""
    svc = _service([_doc("some unrelated document content", page=1, is_index=False)])

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("clarify mode must not run a new search")

    monkeypatch.setattr(svc.store, "search", _must_not_be_called)
    monkeypatch.setattr(
        generator, "classify_followup",
        lambda question, history: {"mode": "clarify", "query": question},
    )

    history = [
        {"role": "user", "content": "think as i am kid and explain what does red mean in this document"},
        {
            "role": "assistant",
            "content": "Red means you like making decisions quickly.",
            "contexts": ["Red is one of the four 4D-i colors, associated with fast decision-making."],
            "sources": [{"source": "osw_data.pdf", "page": 12}],
        },
    ]
    r = svc.retrieve("chat1", "more clearly", history)

    assert r["query_type"] == "CLARIFY_QUERY"
    assert r["direct_answer"] is None
    assert len(r["hits"]) == 1
    assert r["hits"][0][0].text == "Red is one of the four 4D-i colors, associated with fast decision-making."
    assert r["sources"] == [{"source": "osw_data.pdf", "page": 12}]
    assert "think as i am kid" in r["search_query"]


def test_clarify_followup_with_no_previous_answer_falls_back_to_normal_search(monkeypatch):
    """No prior assistant turn to clarify (e.g. classify_followup misfires on
    the very first message) -> must fall through to an ordinary search
    instead of returning nothing."""
    svc = _service([_doc("RAG stands for retrieval augmented generation", page=2, is_index=False)])
    monkeypatch.setattr(
        generator, "classify_followup",
        lambda question, history: {"mode": "clarify", "query": question},
    )
    r = svc.retrieve("chat1", "more clearly", [])
    assert r["query_type"] == "NORMAL_QUERY"
    assert len(r["hits"]) == 1
