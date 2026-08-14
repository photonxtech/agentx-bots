"""Regression test for the live "three key guidelines" retrieval bug.

Confirmed against real production data (see conversation/diagnosis): a real
"Guideline #3" chunk — starting mid-sentence, sharing almost no BM25-relevant
tokens with the query even though it's on-topic — ranked ~140th out of ~500
candidates for its own query, entirely below any reasonable candidate pool
(RETRIEVAL_CANDIDATE_K). Widening the candidate pool alone cannot fix this;
only structural neighbor expansion (pulling the adjacent chunk of a chunk
that DID score well) does. This test reproduces that shape with a small
synthetic fixture modeled directly on the real chunk text, and proves the
fix is what makes the difference (fails with expansion disabled, passes
with it enabled) rather than asserting a result that happened to work once.

Uses the REAL `_BaseStore` (search/get_neighbors/explain) and REAL BM25 —
only the embedding model and cross-encoder are mocked out, so this runs
fast with no model download/GPU/network, but still exercises the exact
production scoring/masking/expansion code path.
"""

import threading

import numpy as np
import pytest

import config
import rag.generator as generator
import rag.reranker as reranker_mod
from api.service import RagService
from rag.ingestion import Document
from rag.vectorstore import _BaseStore

CHAT_ID = "chat1"
QUERY = "What are the three key guidelines for smart problem solving?"

# Modeled directly on the real page-49 chunks from the live document.
GUIDELINE_12_TEXT = (
    "The smart problem-solving process is based on three simple guidelines. "
    "GUIDELINE #1: DECIDE WHAT TYPE OF OUTCOME YOU WANT. This resembles your mind's GPS. "
    "GUIDELINE #2: SELECT THE SMART TRACK PROCESS TO GET YOU THERE."
)
# Starts mid-sentence, like the real chunk — shares almost no query tokens.
GUIDELINE_3_TEXT = (
    "results, they will help you get the best results you need. "
    "GUIDELINE # 3: SHIFT YOUR THINKING AT EACH PHASE OF THE PROCESS. "
    "Once you have the smart track selected."
)

# Each mentions only ONE of the 5 query-topic words (see _fake_embed) so
# none can plausibly out-rank the real Guideline #1/2 chunk (which mentions
# all 5) on either dense or BM25 — keeping this fixture's ordering
# unambiguous, matching how clearly the real chunk led the live ranking
# (fused=0.9196 vs. the next candidate's 0.8287).
DISTRACTORS = [
    "Problem areas in the annual report include budget overruns this year.",
    "The solving of complex equations requires patience and a careful method.",
    "Smart devices are becoming increasingly common in modern households.",
    "Three candidates were shortlisted for the position after first interviews.",
    "A guideline was issued regarding office attendance policy expectations.",
    "Problem tickets from customers are reviewed every Monday morning meeting.",
    "Solving puzzles is a popular hobby among people of all ages worldwide.",
    "Smart contracts are used in some blockchain applications for automation.",
    "Three rivers converge near the old town before reaching the coastline.",
    "A guideline document was distributed to all new hires during orientation.",
    # These three deliberately share 2+ topic words (like Guideline #3's own
    # chunk does — "smart"+"guideline") plus decent length/keyword density,
    # so they reliably outscore it on fused score without being the answer.
    "Smart problem detection systems reduce downtime in manufacturing plants.",
    "Three guideline updates were announced at the smart city conference today.",
    "Solving three math problems required extra time during the exam period.",
]


_TOPIC_WORDS = ("guideline", "problem", "solving", "smart", "three")


def _fake_embed(texts):
    """One dimension per topic word (one-hot on presence), then
    L2-normalized — so cosine similarity to the query (which contains all 5
    words) scales with how many of those words a text actually shares, a
    real discriminating signal. (An earlier version of this mock used a
    single scalar "affinity" plus a small constant second dimension — after
    normalization EVERY vector collapsed to nearly the same direction
    regardless of affinity, silently making dense similarity useless for
    this test. Caught by manually inspecting the real computed scores
    when this fixture wasn't behaving as expected — a reminder that a
    mocked signal needs its own sanity check, not just an assumption that
    "it varies by keyword count" holds after normalization.)
    """
    vecs = np.zeros((len(texts), len(_TOPIC_WORDS)), dtype="float32")
    for i, t in enumerate(texts):
        tl = t.lower()
        for j, w in enumerate(_TOPIC_WORDS):
            if w in tl:
                vecs[i, j] = 1.0
        if not vecs[i].any():
            vecs[i, :] = 0.01
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


class _FakeCrossEncoder:
    """Rough stand-in for the real reranker: scores anything mentioning a
    specific guideline highest, generic "guideline" mentions moderately,
    unrelated distractor text lowest. Exact values don't matter — what
    matters is that Guideline #3's own score is unremarkable, so it only
    survives via `protect`, not because it happened to score well.
    """

    def predict(self, pairs):
        scores = []
        for _, text in pairs:
            tl = text.lower()
            if "guideline #1" in tl:
                scores.append(2.5)
            elif "guideline # 3" in tl or "guideline #3" in tl:
                scores.append(0.0)
            else:
                scores.append(-1.5)
        return np.array(scores, dtype="float32")


@pytest.fixture(autouse=True)
def _mocks(monkeypatch):
    monkeypatch.setattr("rag.vectorstore.embedding_dim", lambda: len(_TOPIC_WORDS))
    monkeypatch.setattr("rag.vectorstore.embed", _fake_embed)
    monkeypatch.setattr(reranker_mod, "embed", _fake_embed)
    monkeypatch.setattr(reranker_mod, "get_model", lambda: _FakeCrossEncoder())
    monkeypatch.setattr(
        generator, "classify_followup",
        lambda question, history: {"mode": "standalone", "query": question},
    )


def _doc(text, page, chunk, child=0, chat_id=CHAT_ID):
    return Document(
        text=text, source="osw_data.pdf", kind="pdf",
        meta={"page": page, "chunk": chunk, "child": child, "chat_id": chat_id, "is_index": False},
    )


# Real, widely-scattered page numbers from the live document (none adjacent
# to each other or to page 49) — deliberately NOT sequential. Sequential
# page numbers here would make each distractor structurally "adjacent" to
# the next, so expanding neighbors around ANY distractor would pull in
# OTHER distractors as "protected" too, an artifact of the fixture, not a
# real-world scenario (a real document's distractor pages are scattered
# across genuinely different content, not placeholder filler back-to-back).
DISTRACTOR_PAGES = [7, 44, 51, 60, 68, 72, 77, 90, 300, 475, 120, 210, 390]


def _build_store() -> _BaseStore:
    docs = [
        _doc(GUIDELINE_12_TEXT, page=49, chunk=1),
        _doc(GUIDELINE_3_TEXT, page=49, chunk=2),
    ]
    for text, page in zip(DISTRACTORS, DISTRACTOR_PAGES):
        docs.append(_doc(text, page=page, chunk=0))

    store = _BaseStore()
    store.documents = docs
    store._matrix = _fake_embed([d.text for d in docs])
    store._rebuild_bm25()
    return store


def _service(store) -> RagService:
    svc = RagService.__new__(RagService)
    svc.store = store
    svc._lock = threading.RLock()
    return svc


def test_guideline_3_ranks_outside_a_small_candidate_pool_on_its_own(monkeypatch):
    """Setup sanity check: confirms this fixture actually reproduces the bug
    shape (Guideline #3 too low-ranked to enter a modest candidate pool on
    its own wording), matching what was found live — not asserting
    something that was never actually true of the fixture."""
    store = _build_store()
    hits = store.search(QUERY, top_k=5, chat_id=CHAT_ID)
    pages_chunks = [(d.meta["page"], d.meta["chunk"]) for d, _ in hits]
    assert (49, 2) not in pages_chunks, (
        "fixture is broken: Guideline #3's chunk should NOT be in a top-5 "
        "hybrid search result on its own wording, same as the live bug"
    )
    # Guideline #1/2's chunk, by contrast, should be a strong hit.
    assert (49, 1) in pages_chunks


def test_neighbor_expansion_recovers_guideline_3(monkeypatch):
    """The actual fix: with neighbor expansion enabled (the default),
    Guideline #3's chunk reaches the final context even though it never
    entered the candidate pool on its own."""
    monkeypatch.setattr(config, "RETRIEVAL_CANDIDATE_K", 5)
    monkeypatch.setattr(config, "FINAL_CONTEXT_K", 5)
    monkeypatch.setattr(config, "NEIGHBOR_EXPANSION_WINDOW", 1)
    # Small on purpose (but >=2): window=1 means "1 before AND 1 after", and
    # in this flat fixture the chunk immediately BEFORE (49,1) in global
    # reading order is an unrelated distractor (page 44) rather than a real
    # page-0 chunk of page 49 (which doesn't exist, same as the live doc) —
    # so recovering the real target (49,2), which is the AFTER neighbor,
    # needs room for both. Without a cap at all, several distractor anchors
    # being each other's POSITIONAL neighbors too could consume the whole
    # final budget and crowd out the genuinely best-scoring regular
    # candidate — see config.NEIGHBOR_EXPANSION_MAX_ADDED.
    monkeypatch.setattr(config, "NEIGHBOR_EXPANSION_MAX_ADDED", 2)

    svc = _service(_build_store())
    result = svc.retrieve(CHAT_ID, QUERY, [])

    pages_chunks = [(d.meta["page"], d.meta["chunk"]) for d, _ in result["hits"]]
    assert (49, 1) in pages_chunks, "Guideline #1/2's chunk (the real hit) should still be present"
    assert (49, 2) in pages_chunks, "Guideline #3's chunk must be recovered via neighbor expansion"


def test_without_neighbor_expansion_guideline_3_is_missing(monkeypatch):
    """Proves the PREVIOUS test's pass is actually due to neighbor
    expansion, not some other accident of the fixture: disabling it
    (NEIGHBOR_EXPANSION_WINDOW=0, the old behavior) reproduces the exact
    live failure — Guideline #3 silently missing from the final context."""
    monkeypatch.setattr(config, "RETRIEVAL_CANDIDATE_K", 5)
    monkeypatch.setattr(config, "FINAL_CONTEXT_K", 5)
    monkeypatch.setattr(config, "NEIGHBOR_EXPANSION_WINDOW", 0)

    svc = _service(_build_store())
    result = svc.retrieve(CHAT_ID, QUERY, [])

    pages_chunks = [(d.meta["page"], d.meta["chunk"]) for d, _ in result["hits"]]
    assert (49, 2) not in pages_chunks


def test_recall_at_k_for_guideline_evidence(monkeypatch):
    """Recall@k for the required evidence set {(49,1), (49,2)} — the goal
    is retrieving ALL required evidence before generation, not just
    answering-looking-plausible. Recall@5 must be 1.0 with expansion on;
    without it, recall@5 is only 0.5 (only half the evidence)."""
    monkeypatch.setattr(config, "RETRIEVAL_CANDIDATE_K", 5)
    monkeypatch.setattr(config, "FINAL_CONTEXT_K", 5)
    monkeypatch.setattr(config, "NEIGHBOR_EXPANSION_MAX_ADDED", 2)
    required = {(49, 1), (49, 2)}

    def recall_at(window):
        monkeypatch.setattr(config, "NEIGHBOR_EXPANSION_WINDOW", window)
        svc = _service(_build_store())
        result = svc.retrieve(CHAT_ID, QUERY, [])
        found = {(d.meta["page"], d.meta["chunk"]) for d, _ in result["hits"]}
        return len(found & required) / len(required)

    assert recall_at(window=1) == 1.0
    assert recall_at(window=0) == 0.5


def test_neighbor_expansion_never_crosses_chat_boundary(monkeypatch):
    """A chunk in another chat, even on the "same" page/chunk index, must
    never be pulled in as a neighbor — chat isolation holds even for the
    new expansion path."""
    store = _build_store()
    store.documents.append(_doc("other chat's completely unrelated content", page=49, chunk=2, chat_id="other-chat"))
    store._matrix = _fake_embed([d.text for d in store.documents])
    store._rebuild_bm25()

    anchor = next(d for d in store.documents if d.meta["page"] == 49 and d.meta["chunk"] == 1)
    neighbors = store.get_neighbors(anchor, window=1)
    assert all(n.meta.get("chat_id") == CHAT_ID for n in neighbors)


def test_neighbor_expansion_skips_toc_pages_transparently():
    """A whole TOC page sitting between two content pages (is_index is set
    per-page, at ingestion — see chunking.chunk_documents) must not break
    their adjacency, and must never itself be returned as a "neighbor".
    """
    docs = [
        _doc("content on page 10", page=10, chunk=0),
        _doc("toc-like filler content", page=11, chunk=0),
        _doc("content on page 12", page=12, chunk=0),
    ]
    docs[1].meta["is_index"] = True
    store = _BaseStore()
    store.documents = docs
    store._matrix = _fake_embed([d.text for d in docs])
    store._rebuild_bm25()

    neighbors = store.get_neighbors(docs[0], window=1)
    assert len(neighbors) == 1
    assert neighbors[0].meta["page"] == 12
    assert not any(n.meta.get("is_index") for n in neighbors)
