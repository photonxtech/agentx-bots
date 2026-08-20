"""Unit tests for RagService's upload behavior:
  * RagService._replace_matching_file — re-uploading a file with the SAME
    name replaces its old chunks, wired into ingest_file/ingest_file_stream.
  * RagService._check_document_cap — a chat can hold several distinct
    documents (config.MAX_DOCUMENTS_PER_CHAT) at once; a genuinely new
    filename is added ALONGSIDE existing ones, up to that cap, past which
    TooManyDocumentsError is raised before any parsing/OCR cost is spent.

Before the same-filename fix, uploading a new file only ever ADDED chunks, so
a re-upload (same file, or a revised version of it) left the old chunks in
the index forever: the same content ended up indexed multiple times, and
duplicate chunks competed for retrieval's top_k slots. This is a real
production bug (see also test_vectorstore.py's dedup tests, the
retrieval-time safety net for whatever duplicates already exist).

Real ingestion (PDF/OCR parsing) is mocked out — these tests check the
replace/add/cap WIRING, not file parsing.
"""

import threading

import pytest

import config
import rag.ingestion as ingestion
from api.service import RagService, TooManyDocumentsError
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


def test_uploading_a_different_filename_adds_alongside(monkeypatch):
    """A chat can hold several distinct documents at once (up to
    config.MAX_DOCUMENTS_PER_CHAT) — uploading other_doc.pdf into a chat that
    already has doc.txt must ADD it, not remove doc.txt."""
    monkeypatch.setattr(ingestion, "ingest_cached", _fake_ingest_cached)
    svc, store = _service()
    svc.ingest_file("chat1", b"bytes", "doc.txt")
    result = svc.ingest_file("chat1", b"other bytes", "other_doc.pdf")

    assert "Replaced" not in result["message"]
    assert [c for c in store.calls if c[0] == "remove"] == []
    assert sorted(store.sources_for_chat("chat1")) == ["doc.txt", "other_doc.pdf"]


def test_document_cap_blocks_a_new_filename_past_the_limit(monkeypatch):
    """Past config.MAX_DOCUMENTS_PER_CHAT distinct documents, a new filename
    is rejected with TooManyDocumentsError — raised before ingest_cached()
    (OCR/vision) is ever called, since ingestion cost shouldn't be spent on
    an upload that's going to be rejected anyway."""
    monkeypatch.setattr(config, "MAX_DOCUMENTS_PER_CHAT", 2)
    calls = []

    def _tracked_ingest(file_bytes, filename, progress_callback=None):
        calls.append(filename)
        return _fake_ingest_cached(file_bytes, filename, progress_callback)

    monkeypatch.setattr(ingestion, "ingest_cached", _tracked_ingest)
    svc, store = _service()
    svc.ingest_file("chat1", b"bytes", "doc1.txt")
    svc.ingest_file("chat1", b"bytes", "doc2.txt")

    with pytest.raises(TooManyDocumentsError):
        svc.ingest_file("chat1", b"bytes", "doc3.txt")

    assert calls == ["doc1.txt", "doc2.txt"]  # doc3.txt never got parsed
    assert sorted(store.sources_for_chat("chat1")) == ["doc1.txt", "doc2.txt"]


def test_document_cap_does_not_block_replacing_a_same_named_file(monkeypatch):
    """At the cap, re-uploading a file that's ALREADY one of the existing
    documents must still work (it replaces, not adds) — the cap only blocks
    genuinely new filenames."""
    monkeypatch.setattr(config, "MAX_DOCUMENTS_PER_CHAT", 2)
    monkeypatch.setattr(ingestion, "ingest_cached", _fake_ingest_cached)
    svc, store = _service()
    svc.ingest_file("chat1", b"bytes", "doc1.txt")
    svc.ingest_file("chat1", b"bytes", "doc2.txt")

    result = svc.ingest_file("chat1", b"revised bytes", "doc1.txt")

    assert "Replaced 'doc1.txt'" in result["message"]
    assert sorted(store.sources_for_chat("chat1")) == ["doc1.txt", "doc2.txt"]


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
