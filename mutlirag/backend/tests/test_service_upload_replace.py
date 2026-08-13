"""Unit tests for RagService's auto-replace-on-upload behavior
(RagService._replace_existing_file, wired into ingest_file/ingest_file_stream).

A chat holds exactly one file at a time (see chat_file/_annotate) — before
this fix, uploading a new file only ever ADDED chunks, so a re-upload (same
file, or a revised version of it) left the old chunks in the index forever:
the same content ended up indexed multiple times, and duplicate chunks
competed for retrieval's top_k slots. This is a real production bug (see
also test_vectorstore.py's dedup tests, the retrieval-time safety net for
whatever duplicates already exist).

Real ingestion (PDF/OCR parsing) is mocked out — these tests check the
replace-before-add WIRING, not file parsing.
"""

import threading

import rag.ingestion as ingestion
from api.service import RagService
from rag.ingestion import Document


class RecordingStore:
    """Just enough of the VectorStore surface for ingest_file/ingest_file_stream
    to run, while recording what gets added/removed and in what order."""

    def __init__(self):
        self.documents: list[Document] = []
        self.calls: list[tuple[str, object]] = []  # ("add", [sources]) / ("remove", (source, chat_id))
        self.backend = "fake"

    def sources_for_chat(self, chat_id):
        seen = {}
        for d in self.documents:
            if d.meta.get("chat_id") == chat_id:
                seen.setdefault(d.source, None)
        return list(seen)

    def size_for_chat(self, chat_id):
        return sum(1 for d in self.documents if d.meta.get("chat_id") == chat_id)

    def add(self, docs, progress_callback=None):
        self.calls.append(("add", [d.source for d in docs]))
        self.documents.extend(docs)

    def remove_source(self, source, chat_id=None):
        self.calls.append(("remove", (source, chat_id)))
        before = len(self.documents)
        self.documents = [
            d for d in self.documents
            if not (d.source == source and (chat_id is None or d.meta.get("chat_id") == chat_id))
        ]
        return before - len(self.documents)

    def save(self, dirpath=None):
        pass


def _service(chat_id="chat1") -> tuple[RagService, RecordingStore]:
    svc = RagService.__new__(RagService)  # skip __init__ (no real DB/disk)
    store = RecordingStore()
    svc.store = store
    svc._lock = threading.RLock()
    svc.chats = {chat_id: {"id": chat_id, "messages": []}}
    return svc, store


def _fake_ingest_cached(file_bytes, filename, progress_callback=None):
    return [Document(text="Short test content.", source=filename, kind="text", meta={})]


def test_first_upload_does_not_remove_anything(monkeypatch):
    monkeypatch.setattr(ingestion, "ingest_cached", _fake_ingest_cached)
    svc, store = _service()

    result = svc.ingest_file("chat1", b"bytes", "doc.txt")

    assert result["chunks_indexed"] == 1
    assert [c for c in store.calls if c[0] == "remove"] == []
    assert store.sources_for_chat("chat1") == ["doc.txt"]


def test_second_upload_replaces_the_first(monkeypatch):
    monkeypatch.setattr(ingestion, "ingest_cached", _fake_ingest_cached)
    svc, store = _service()

    svc.ingest_file("chat1", b"bytes", "doc.txt")
    result = svc.ingest_file("chat1", b"other bytes", "doc.txt")

    assert "Replaced" in result["message"]
    # remove happened, and happened BEFORE the second add
    call_kinds = [c[0] for c in store.calls]
    assert call_kinds == ["add", "remove", "add"]
    assert store.size_for_chat("chat1") == 1  # not 2 — old chunks are gone


def test_uploading_a_different_filename_still_replaces(monkeypatch):
    """A chat holds ONE file — uploading revised_doc.pdf into a chat that
    already has doc.txt must remove doc.txt, not keep both."""
    monkeypatch.setattr(ingestion, "ingest_cached", _fake_ingest_cached)
    svc, store = _service()
    svc.ingest_file("chat1", b"bytes", "doc.txt")
    result = svc.ingest_file("chat1", b"other bytes", "revised_doc.pdf")

    assert result["message"].startswith("Replaced 'doc.txt'")
    assert store.sources_for_chat("chat1") == ["revised_doc.pdf"]


def test_failed_second_upload_does_not_delete_the_existing_file(monkeypatch):
    """An upload that extracts no content must never wipe out a good,
    already-indexed file — replacement only happens once we know the new
    upload actually produced chunks."""
    monkeypatch.setattr(ingestion, "ingest_cached", _fake_ingest_cached)
    svc, store = _service()
    svc.ingest_file("chat1", b"bytes", "doc.txt")

    def empty_ingest(file_bytes, filename, progress_callback=None):
        return []

    monkeypatch.setattr(ingestion, "ingest_cached", empty_ingest)
    result = svc.ingest_file("chat1", b"empty", "empty.txt")

    assert result["chunks_indexed"] == 0
    assert [c for c in store.calls if c[0] == "remove"] == []
    assert store.sources_for_chat("chat1") == ["doc.txt"]
