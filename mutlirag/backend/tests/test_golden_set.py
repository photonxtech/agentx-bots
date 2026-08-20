"""Unit tests for rag.golden_set — matching a live question against
user-selected LangSmith dataset(s), replacing the old single-fixed-dataset
design (which couldn't say which document's golden set applies once a chat
holds more than one document).

The LangSmith client is faked throughout; the embedding model is a small
deterministic fake — these test the SELECTION/CACHING logic, not real
semantic matching quality.
"""

import numpy as np

import rag.golden_set as golden_set


def _fake_embed(texts):
    """One-hot-per-question embedding: identical text -> identical vector,
    everything else orthogonal. Lets tests control similarity exactly."""
    vocab = ["capital of france", "capital of germany", "unrelated question"]
    vecs = []
    for t in texts:
        v = np.zeros(len(vocab), dtype="float32")
        key = t.strip().lower()
        if key in vocab:
            v[vocab.index(key)] = 1.0
        vecs.append(v)
    return np.array(vecs, dtype="float32")


def _example(question, ground_truth):
    class _Ex:
        inputs = {"question": question}
        outputs = {"ground_truth": ground_truth}
    return _Ex()


class _FakeDataset:
    id = "ds-id"


class _FakeClient:
    """Records how many times list_examples/list_datasets are called, so
    caching can be verified."""
    calls = {"list_examples": 0, "list_datasets": 0}
    datasets = {}  # name -> list of _example()

    def __init__(self, *a, **kw):
        pass

    def read_dataset(self, dataset_name):
        return _FakeDataset()

    def list_examples(self, dataset_id):
        _FakeClient.calls["list_examples"] += 1
        return _FakeClient.datasets.get("current", [])

    def list_datasets(self):
        _FakeClient.calls["list_datasets"] += 1

        class _DS:
            def __init__(self, name):
                self.name = name
        return [_DS(n) for n in _FakeClient.datasets.get("names", [])]


def _configure_langsmith(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key")


def _reset_fake_client():
    _FakeClient.calls = {"list_examples": 0, "list_datasets": 0}
    _FakeClient.datasets = {}


def setup_function(_):
    golden_set._dataset_cache.clear()
    _reset_fake_client()


def test_list_available_datasets_empty_when_langsmith_not_configured(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert golden_set.list_available_datasets() == []


def test_list_available_datasets_returns_sorted_names(monkeypatch):
    _configure_langsmith(monkeypatch)
    monkeypatch.setattr("langsmith.Client", _FakeClient)
    _FakeClient.datasets["names"] = ["wenext-golden", "osw-golden"]
    assert golden_set.list_available_datasets() == ["osw-golden", "wenext-golden"]


def test_list_available_datasets_returns_empty_on_failure(monkeypatch):
    _configure_langsmith(monkeypatch)

    class _Raising:
        def __init__(self, *a, **kw):
            raise RuntimeError("LangSmith unreachable")

    monkeypatch.setattr("langsmith.Client", _Raising)
    assert golden_set.list_available_datasets() == []


def test_lookup_returns_none_with_no_datasets_selected():
    """Short-circuits before ever calling the embedding model or LangSmith —
    no dataset selected means nothing to check against."""
    assert golden_set.lookup("What is the capital of France?", None) is None
    assert golden_set.lookup("What is the capital of France?", []) is None


def test_lookup_finds_ground_truth_in_selected_dataset(monkeypatch):
    monkeypatch.setattr(golden_set.embeddings, "embed", _fake_embed)
    monkeypatch.setattr("langsmith.Client", _FakeClient)
    _FakeClient.datasets["current"] = [_example("capital of france", "Paris")]

    result = golden_set.lookup("capital of france", ["osw-golden"])
    assert result == "Paris"


def test_lookup_below_threshold_returns_none(monkeypatch):
    monkeypatch.setattr(golden_set.embeddings, "embed", _fake_embed)
    monkeypatch.setattr("langsmith.Client", _FakeClient)
    _FakeClient.datasets["current"] = [_example("capital of germany", "Berlin")]

    # "unrelated question" is orthogonal to every curated question -> similarity 0.
    result = golden_set.lookup("unrelated question", ["osw-golden"])
    assert result is None


def test_lookup_picks_best_match_across_multiple_datasets(monkeypatch):
    monkeypatch.setattr(golden_set.embeddings, "embed", _fake_embed)
    monkeypatch.setattr("langsmith.Client", _FakeClient)

    # Both datasets happen to share the same fake backing list here (the fake
    # client keys on "current" regardless of name) — what matters is that a
    # question matching one of the two curated questions in it resolves to
    # the right ground_truth, proving the pooled-match logic works across
    # more than one requested dataset name.
    _FakeClient.datasets["current"] = [
        _example("capital of france", "Paris"),
        _example("capital of germany", "Berlin"),
    ]

    assert golden_set.lookup("capital of germany", ["osw-golden", "wenext-golden"]) == "Berlin"


def test_lookup_skips_a_dataset_that_fails_to_load(monkeypatch):
    monkeypatch.setattr(golden_set.embeddings, "embed", _fake_embed)

    class _FlakyClient(_FakeClient):
        def read_dataset(self, dataset_name):
            if dataset_name == "broken-dataset":
                raise RuntimeError("dataset not found")
            return super().read_dataset(dataset_name)

    monkeypatch.setattr("langsmith.Client", _FlakyClient)
    _FakeClient.datasets["current"] = [_example("capital of france", "Paris")]

    result = golden_set.lookup("capital of france", ["broken-dataset", "osw-golden"])
    assert result == "Paris"


def test_dataset_is_fetched_once_and_cached(monkeypatch):
    monkeypatch.setattr(golden_set.embeddings, "embed", _fake_embed)
    monkeypatch.setattr("langsmith.Client", _FakeClient)
    _FakeClient.datasets["current"] = [_example("capital of france", "Paris")]

    golden_set.lookup("capital of france", ["osw-golden"])
    golden_set.lookup("capital of france", ["osw-golden"])

    assert _FakeClient.calls["list_examples"] == 1
