"""Unit tests for rag.reranker's new pieces: the `protect` exemption on
rerank()'s score floor, and diversity_select()/_mmr() for the final
diversity-aware selection. The cross-encoder and embedding model are
mocked — these test the SELECTION LOGIC, not real relevance quality.
"""

import numpy as np
import pytest

import config
import rag.reranker as reranker_mod
from rag.ingestion import Document
from rag.reranker import diversity_select, rerank


def _doc(text):
    return Document(text=text, source="doc.pdf", kind="pdf", meta={})


class _FakeCrossEncoder:
    """logit = score encoded in the text itself, e.g. "SCORE:2.0 ...".
    Lets tests control exactly what each chunk's rerank score will be."""

    def predict(self, pairs):
        logits = []
        for _, text in pairs:
            marker = text.split()[0]
            logits.append(float(marker.split(":")[1]))
        return np.array(logits, dtype="float32")


def _patch_cross_encoder(monkeypatch):
    monkeypatch.setattr(reranker_mod, "get_model", lambda: _FakeCrossEncoder())


def test_rerank_drops_low_scores_without_protect(monkeypatch):
    _patch_cross_encoder(monkeypatch)
    monkeypatch.setattr(config, "RERANK_MIN_SCORE", 0.5)
    hits = [(_doc("SCORE:5.0 relevant"), 0.0), (_doc("SCORE:-5.0 irrelevant"), 0.0)]
    result = rerank("query", hits)
    texts = [d.text for d, _ in result]
    assert "SCORE:5.0 relevant" in texts
    assert "SCORE:-5.0 irrelevant" not in texts


def test_rerank_protect_exempts_from_score_floor(monkeypatch):
    _patch_cross_encoder(monkeypatch)
    monkeypatch.setattr(config, "RERANK_MIN_SCORE", 0.5)
    low_doc = _doc("SCORE:-5.0 irrelevant but structurally required")
    hits = [(_doc("SCORE:5.0 relevant"), 0.0), (low_doc, 0.0)]
    result = rerank("query", hits, protect={id(low_doc)})
    texts = [d.text for d, _ in result]
    assert low_doc.text in texts, "protected hit must survive despite scoring below the floor"


def test_rerank_still_scores_protected_hits_normally(monkeypatch):
    """Protection exempts a hit from being DROPPED, but its score is still
    the real cross-encoder score, not inflated/faked."""
    _patch_cross_encoder(monkeypatch)
    doc = _doc("SCORE:-2.0 low but kept")
    result = rerank("query", [(doc, 0.0)], protect={id(doc)})
    assert result[0][1] == pytest.approx(1.0 / (1.0 + np.exp(2.0)), abs=1e-6)  # sigmoid(-2.0)


def test_rerank_empty_hits_returns_empty():
    assert rerank("query", []) == []


def _fake_embed_for_mmr(texts):
    """2D embedding: "A" texts point one way, "B" texts point another —
    lets tests control redundancy precisely."""
    vecs = []
    for t in texts:
        vecs.append([1.0, 0.0] if t.startswith("A") else [0.0, 1.0])
    return np.array(vecs, dtype="float32")


def test_diversity_select_prefers_covering_distinct_clusters(monkeypatch):
    """Two near-duplicate "A" hits (only one is needed for that idea) plus a
    lower-scoring but distinct "B" hit — MMR should still make room for B
    rather than picking both redundant A's."""
    monkeypatch.setattr(reranker_mod, "embed", _fake_embed_for_mmr)
    hits = [
        (_doc("A1 first restatement"), 0.95),
        (_doc("A2 second restatement of the same idea"), 0.90),
        (_doc("B1 a genuinely different idea"), 0.60),
    ]
    selected = diversity_select(hits, k=2, lambda_mult=0.5)
    texts = {d.text for d, _ in selected}
    assert len(selected) == 2
    assert any(t.startswith("B") for t in texts), "the distinct idea must not be crowded out"


def test_diversity_select_protected_hits_always_kept(monkeypatch):
    """A protected hit is kept regardless of how redundant it looks with
    something else already selected — being similar to its anchor is
    expected (same section), not disqualifying."""
    monkeypatch.setattr(reranker_mod, "embed", _fake_embed_for_mmr)
    protected_doc = _doc("A3 redundant with A1 but structurally required")
    hits = [
        (_doc("A1 first restatement"), 0.95),
        (protected_doc, 0.10),  # low score AND redundant — still must survive
    ]
    selected = diversity_select(hits, k=2, protect={id(protected_doc)}, lambda_mult=0.9)
    texts = [d.text for d, _ in selected]
    assert protected_doc.text in texts


def test_diversity_select_respects_k():
    hits = [(_doc(f"doc {i}"), 1.0 - i * 0.01) for i in range(10)]
    selected = diversity_select(hits, k=3)
    assert len(selected) == 3


def test_diversity_select_empty_hits_returns_empty():
    assert diversity_select([], k=5) == []
