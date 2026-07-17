"""Chunking: split long text into overlapping, sentence-aware windows.

Why chunk at all? Embedding models have a limited context and retrieval is
sharper when each vector represents a focused idea. Overlap prevents a fact
that sits on a boundary from being lost between two chunks.

Instead of cutting every N characters (which slices sentences and even words
in half), we split the text into natural units — lines and sentences — and
greedily pack whole units into each chunk. Overlap is built from the last few
units of the previous chunk, so context carries over intact.

Excel/CSV rows are already small and self-contained, so we leave them whole.
"""

from __future__ import annotations

import re

import config
from rag.ingestion import Document

# Kinds whose documents are already one small self-contained unit each.
_ATOMIC_KINDS = {"excel", "csv"}

# Sentence enders followed by whitespace, or any run of newlines.
_UNIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _units(text: str, size: int) -> list[str]:
    """Split text into sentences/lines; hard-split any unit longer than size
    so packing below always terminates."""
    units: list[str] = []
    for u in _UNIT_RE.split(text):
        u = u.strip()
        if not u:
            continue
        while len(u) > size:
            units.append(u[:size])
            u = u[size:]
        units.append(u)
    return units


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    for unit in _units(text, size):
        if cur and cur_len + 1 + len(unit) > size:
            chunks.append("\n".join(cur))
            # Seed the next chunk with the tail of this one (up to `overlap`
            # characters of whole units) so boundary facts appear in both.
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(cur):
                if tail_len + len(prev) + 1 > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            cur, cur_len = tail, tail_len
        cur.append(unit)
        cur_len += len(unit) + 1

    if cur:
        chunks.append("\n".join(cur))
    return chunks


def chunk_documents(
    docs: list[Document],
    size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """Return a new list of Documents where oversized text has been split."""
    out: list[Document] = []
    for doc in docs:
        # Keep short docs (like spreadsheet rows) intact.
        if doc.kind in _ATOMIC_KINDS or len(doc.text) <= size:
            out.append(doc)
            continue
        for i, piece in enumerate(_split_text(doc.text, size, overlap)):
            out.append(
                Document(
                    text=piece,
                    source=doc.source,
                    kind=doc.kind,
                    meta={**doc.meta, "chunk": i},
                )
            )
    return out
