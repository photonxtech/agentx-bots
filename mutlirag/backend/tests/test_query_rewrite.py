"""Regression tests for generator.classify_followup()/rewrite_query(), which
route a follow-up question into one of three modes before retrieval:

  standalone  — searchable as-is.
  contextual  — needs history to resolve a reference, then searched.
  clarify     — not a new document question at all; the user wants the
                PREVIOUS answer rephrased/simplified/elaborated (see
                RagService._retrieve_clarification in test_service_retrieve.py
                for the retrieval-side half of this fix).

Two real production bugs are covered here:
  1. A fully standalone question ("What is the difference between analytical
     thinking and critical thinking?") got rewritten into an unrelated
     earlier topic from history ("Cool Red vs Warm Red mindsets in the 4D-i
     framework") — the classifier said "contextual" when it should have said
     "standalone", and the rewrite drifted onto a stale topic.
  2. A pure clarification follow-up ("more clearly") was earlier being routed
     through a brand-new document search on its own (near-empty) text,
     retrieving unrelated content and producing a wrong answer — it needs to
     be classified "clarify", not "standalone"/"contextual", so the retrieval
     layer can reuse the previous turn's own context instead of searching.

Both the classifier LLM and the embedding model are faked — these tests
cover CONTROL FLOW (which mode wins and what happens next), not real model
quality.
"""

import numpy as np

import config
import rag.generator as generator


HISTORY = [
    {"role": "user", "content": "What is the difference between Cool Red and Warm Red mindsets in the 4D-i framework?"},
    {"role": "assistant", "content": "Cool Red is calm and controlling; Warm Red is enthusiastic and persuasive."},
]


def _fake_client(json_content):
    class _Message:
        content = json_content

    class _Choice:
        message = _Message()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    return lambda: _Client()


def test_classifier_says_standalone_uses_original_question(monkeypatch):
    question = "What is the difference between analytical thinking and critical thinking?"
    monkeypatch.setattr(
        generator, "_client",
        _fake_client('{"mode": "standalone", "query": %r}' % question),
    )
    result = generator.classify_followup(question, HISTORY)
    assert result == {"mode": "standalone", "query": question}
    assert generator.rewrite_query(question, HISTORY) == question


def test_classifier_says_clarify_returns_clarify_mode_with_original_text(monkeypatch):
    monkeypatch.setattr(
        generator, "_client",
        _fake_client('{"mode": "clarify", "query": "more clearly"}'),
    )
    result = generator.classify_followup("more clearly", HISTORY)
    assert result == {"mode": "clarify", "query": "more clearly"}
    # rewrite_query() (the search-string-only wrapper) must NOT turn this
    # into a search query — a caller using only that API must still get the
    # original text back untouched, never a fabricated search string.
    assert generator.rewrite_query("more clearly", HISTORY) == "more clearly"


def test_first_turn_short_circuits_without_any_llm_call(monkeypatch):
    def _must_not_be_called():
        raise AssertionError("no history -> classify_followup must not call the LLM at all")

    monkeypatch.setattr(generator, "_client", _must_not_be_called)
    assert generator.classify_followup("Any question at all", []) == {
        "mode": "standalone", "query": "Any question at all",
    }


def test_contextual_rewrite_close_in_meaning_is_used(monkeypatch):
    monkeypatch.setattr(
        generator, "_client",
        _fake_client('{"mode": "contextual", "query": "when is the electricity bill due"}'),
    )
    # Same-topic embeddings -> high similarity -> validation passes.
    monkeypatch.setattr(generator, "embed", lambda texts: np.array([[1.0, 0.0], [0.9, 0.436]], dtype="float32"))
    result = generator.classify_followup("and when is it due?", HISTORY)
    assert result == {"mode": "contextual", "query": "when is the electricity bill due"}


def test_topic_poisoned_rewrite_is_downgraded_to_standalone(monkeypatch):
    """The exact bug: classifier wrongly says 'contextual' and drags in an
    unrelated earlier topic. The embedding-similarity safety net must catch
    this independently of the classifier's own (wrong) judgment, falling
    back to the original question rather than the poisoned rewrite."""
    monkeypatch.setattr(
        generator, "_client",
        _fake_client(
            '{"mode": "contextual", "query": '
            '"What is the difference between Cool Red and Warm Red mindsets in the 4D-i framework?"}'
        ),
    )
    # Orthogonal embeddings -> zero similarity -> validation must reject.
    monkeypatch.setattr(generator, "embed", lambda texts: np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    question = "What is the difference between analytical thinking and critical thinking?"
    result = generator.classify_followup(question, HISTORY)
    assert result == {"mode": "standalone", "query": question}


def test_query_understanding_call_failure_falls_back_to_standalone(monkeypatch):
    def _raise():
        raise RuntimeError("network error")

    monkeypatch.setattr(generator, "_client", _raise)
    question = "Some question"
    assert generator.classify_followup(question, HISTORY) == {"mode": "standalone", "query": question}


def test_malformed_json_falls_back_to_standalone(monkeypatch):
    monkeypatch.setattr(generator, "_client", _fake_client("not valid json"))
    question = "Some question"
    assert generator.classify_followup(question, HISTORY) == {"mode": "standalone", "query": question}


def test_unknown_mode_value_falls_back_to_standalone(monkeypatch):
    monkeypatch.setattr(
        generator, "_client",
        _fake_client('{"mode": "something_else", "query": "whatever"}'),
    )
    question = "Some question"
    result = generator.classify_followup(question, HISTORY)
    assert result == {"mode": "standalone", "query": question}


def test_rewrite_is_valid_treats_identical_text_as_valid():
    assert generator._rewrite_is_valid("same text", "same text") is True


def test_rewrite_is_valid_rejects_overlong_rewrite(monkeypatch):
    monkeypatch.setattr(config, "REWRITE_MIN_SIMILARITY", 0.0)
    long_rewrite = "x" * 500
    assert generator._rewrite_is_valid("short question", long_rewrite) is False


def test_rewrite_is_valid_uses_similarity_threshold(monkeypatch):
    monkeypatch.setattr(config, "REWRITE_MIN_SIMILARITY", 0.5)
    monkeypatch.setattr(generator, "embed", lambda texts: np.array([[1.0, 0.0], [0.6, 0.8]], dtype="float32"))
    assert generator._rewrite_is_valid("q", "close rewrite") is True

    monkeypatch.setattr(generator, "embed", lambda texts: np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    assert generator._rewrite_is_valid("q", "unrelated rewrite") is False
