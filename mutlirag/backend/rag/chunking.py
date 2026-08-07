"""Chunking: split long text into overlapping, sentence-aware windows.

Why chunk at all? Embedding models have a limited context and retrieval is
sharper when each vector represents a focused idea. Overlap prevents a fact
that sits on a boundary from being lost between two chunks.

Instead of cutting every N characters (which slices sentences and even words
in half), we split the text into natural units — lines and sentences — and
greedily pack whole units into each chunk. Overlap is built from the last few
units of the previous chunk, so context carries over intact.

Structure first, sentences second: ingestion.py marks real headings/titles
(DOCX heading styles, PPTX slide titles, PDF lines with a much larger font)
with ingestion.HEADING_MARK. Before packing, text is split into sections at
those markers so a chunk never blends two unrelated headings' content
together — only once a section is small enough to fit within `size` do
several get packed into one chunk, and only when a section is itself too big
does it fall through to the sentence-level packing below (tagged with that
section's heading). Text with no headings at all (plain .txt/.md, OCR output)
behaves exactly as before: one big section, sentence-packed top to bottom.

Excel/CSV rows are already small and self-contained, so we leave them whole.
"""

from __future__ import annotations

import re

import config
from rag.ingestion import HEADING_MARK, Document

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


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split text into (heading, body) pairs at ingestion.HEADING_MARK lines.

    `heading` is None for any text appearing before the first heading (or for
    the whole document, if it has no headings at all). Sections with no body
    (e.g. two headings back to back) are dropped, but a heading with no body
    yet still opens the next section correctly.
    """
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    buf: list[str] = []
    for line in text.split("\n"):
        if line.startswith(HEADING_MARK):
            body = "\n".join(buf).strip()
            if body or heading is not None:
                sections.append((heading, body))
            heading = line[len(HEADING_MARK):].strip()
            buf = []
        else:
            buf.append(line)
    body = "\n".join(buf).strip()
    if body or heading is not None:
        sections.append((heading, body))
    return [(h, b) for h, b in sections if b]


def _pack_sections(
    sections: list[tuple[str | None, str]], size: int, overlap: int
) -> list[tuple[str, list[str]]]:
    """Pack (heading, body) sections into chunks, never blending two sections
    unless both fit comfortably in one chunk together.

    A section that fits within `size` on its own is packed greedily alongside
    its neighbors (like `_split_text`'s sentence units, but at section
    granularity) so a document with many short sections doesn't explode into
    one tiny chunk per heading. A section bigger than `size` is flushed on its
    own and split internally by `_split_text`, tagged with just that heading.
    Returns (text, headings_covered) pairs.
    """
    chunks: list[tuple[str, list[str]]] = []
    cur_parts: list[str] = []
    cur_headings: list[str] = []
    cur_len = 0

    def flush():
        if cur_parts:
            # Preserve order, drop duplicates (a chunk can span 2+ sections
            # under the same heading if it was hit and re-hit, though that's rare).
            seen = dict.fromkeys(cur_headings)
            chunks.append(("\n".join(cur_parts), list(seen)))

    for heading, body in sections:
        if len(body) > size:
            flush()
            cur_parts, cur_headings, cur_len = [], [], 0
            for piece in _split_text(body, size, overlap):
                chunks.append((piece, [heading] if heading else []))
            continue

        if cur_parts and cur_len + len(body) + 1 > size:
            flush()
            cur_parts, cur_headings, cur_len = [], [], 0
        cur_parts.append(body)
        if heading:
            cur_headings.append(heading)
        cur_len += len(body) + 1

    flush()
    return chunks


def chunk_documents(
    docs: list[Document],
    size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """Return a new list of Documents where oversized text has been split.

    Structure-aware: text is split into sections at any headings ingestion.py
    marked (DOCX heading styles, PPTX slide titles, large-font PDF lines)
    before sentence-packing, so a chunk never mixes two unrelated headings'
    content. Documents with no headings behave exactly as before — one
    section, sentence-packed top to bottom.
    """
    out: list[Document] = []
    for doc in docs:
        # Keep short docs (like spreadsheet rows) intact.
        if doc.kind in _ATOMIC_KINDS or len(doc.text) <= size:
            out.append(doc)
            continue
        sections = _split_sections(doc.text)
        for i, (piece, headings) in enumerate(_pack_sections(sections, size, overlap)):
            meta = {**doc.meta, "chunk": i}
            if headings:
                meta["section"] = " / ".join(headings)
            out.append(Document(text=piece, source=doc.source, kind=doc.kind, meta=meta))
    return out
