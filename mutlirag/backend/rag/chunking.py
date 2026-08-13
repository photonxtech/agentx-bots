"""Chunking: parent-child, sentence-aware windows over long text.

Why chunk at all? Embedding models have a limited context and retrieval is
sharper when each vector represents a focused idea. Overlap prevents a fact
that sits on a boundary from being lost between two chunks.

Two tiers, not one: a small "child" chunk is what's actually embedded/BM25-
indexed/reranked, since retrieval wants a narrow, precise match. But a chunk
narrow enough for precise retrieval is often too thin to answer from on its
own, so each child also carries the full text of its containing "parent"
chunk — larger, same structure-aware packing — and generation/evaluation use
that parent text instead (see rag.generator.context_texts()). Both tiers use
the same packing logic below, just called twice at different sizes.

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

Semantic boundaries, third (parent tier only, config.SEMANTIC_CHUNKING_ENABLED):
headings mark where a document's AUTHOR drew a boundary, but a long section
with no sub-heading can still drift across several ideas — the character
budget alone would blend them into one parent chunk just because they fit.
_split_text_semantic embeds that section's sentences and cuts a new chunk
wherever the topic actually shifts (an unusually large drop in similarity
relative to the rest of THAT section's own drops), in addition to, never
instead of, the character budget. See its docstring for the adaptive-
threshold reasoning and fallback conditions.

Excel/CSV rows are already small and self-contained, so we leave them whole.
"""

from __future__ import annotations

import re

import numpy as np

import config
from rag.embeddings import embed
from rag.ingestion import HEADING_MARK, Document
from rag.ingestion import chunk_type as _chunk_type

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


def _consecutive_cosine_distances(vecs: np.ndarray) -> np.ndarray:
    """1 - cosine similarity between each consecutive pair of rows.

    `vecs` comes from rag.embeddings.embed(), which L2-normalizes, so a dot
    product between two rows already IS their cosine similarity.
    """
    sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
    return 1.0 - sims


def _split_text_semantic(
    text: str,
    size: int,
    overlap: int,
    breakpoint_percentile: float = config.SEMANTIC_BREAKPOINT_PERCENTILE,
    min_chunk_chars: int = config.SEMANTIC_MIN_CHUNK_CHARS,
) -> list[str]:
    """Like _split_text, but also cuts a new chunk where the topic actually
    shifts, not only where the character budget runs out.

    Embeds every sentence/line unit in `text`, then treats a jump between
    consecutive units as a real topic boundary when it scores at or above
    `breakpoint_percentile` of THAT TEXT's OWN distribution of jumps — the
    same adaptive-threshold idea LlamaIndex's semantic chunker uses. A fixed
    cosine cutoff doesn't generalize across documents or embedding models;
    comparing a document's jumps only to itself does. The character budget
    (`size`) stays a hard cap either way — a semantic boundary can only cut
    EARLIER than that budget would have, never later — and a cut is never
    taken before the accumulated chunk reaches `min_chunk_chars`, so a few
    genuinely unrelated short sentences don't fragment into one-sentence
    chunks off a noisy percentile.

    Falls back to the plain, embedding-free `_split_text` (identical output
    to before this feature existed) when: the text still fits in one chunk;
    there aren't enough units for a percentile to mean anything
    (config.SEMANTIC_CHUNKING_MIN_UNITS); there are so many that embedding
    each one individually would stop paying for itself
    (config.SEMANTIC_CHUNKING_MAX_UNITS); or the embedding model can't be
    loaded for any reason — this is a context-quality improvement, not a
    step ingestion may ever hard-fail on.
    """
    if len(text) <= size:
        return [text]

    units = _units(text, size)
    if not (config.SEMANTIC_CHUNKING_MIN_UNITS <= len(units) <= config.SEMANTIC_CHUNKING_MAX_UNITS):
        return _split_text(text, size, overlap)

    try:
        vecs = embed(units)
    except Exception:
        return _split_text(text, size, overlap)

    distances = _consecutive_cosine_distances(vecs)
    threshold = float(np.percentile(distances, breakpoint_percentile))

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    for i, unit in enumerate(units):
        over_budget = bool(cur) and cur_len + 1 + len(unit) > size
        is_semantic_break = (
            i > 0 and cur_len >= min_chunk_chars and distances[i - 1] >= threshold
        )
        if cur and (over_budget or is_semantic_break):
            chunks.append("\n".join(cur))
            # Same tail-overlap seeding as _split_text, regardless of which
            # of the two cut reasons fired, so boundary-straddling sentences
            # stay retrievable from both sides either way.
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
    own and split internally — by `_split_text_semantic` (topic-shift-aware,
    see its docstring) when config.SEMANTIC_CHUNKING_ENABLED, else the plain
    character-budget `_split_text` — tagged with just that heading. This is
    the ONLY place semantic boundaries apply: once a section already fits in
    one chunk, there's no drift within it left to detect. Returns
    (text, headings_covered) pairs.
    """
    chunks: list[tuple[str, list[str]]] = []
    cur_parts: list[str] = []
    cur_headings: list[str] = []
    cur_len = 0
    split_oversized = _split_text_semantic if config.SEMANTIC_CHUNKING_ENABLED else _split_text

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
            for piece in split_oversized(body, size, overlap):
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
    parent_size: int = config.PARENT_CHUNK_SIZE,
    parent_overlap: int = config.PARENT_CHUNK_OVERLAP,
    child_size: int = config.CHILD_CHUNK_SIZE,
    child_overlap: int = config.CHILD_CHUNK_OVERLAP,
) -> list[Document]:
    """Return a new list of CHILD Documents, each carrying its PARENT chunk's
    full text in meta["parent_text"].

    Parent-child chunking: structure-aware PARENT chunks are packed first
    (never blending two headings, exactly as before, just at parent_size), then
    each parent is subdivided into small CHILD chunks by the same sentence-
    aware packing at child_size. Children are what's embedded/indexed/reranked
    for retrieval precision; rag.generator.context_texts() resolves each
    retrieved child back to its parent text (deduplicated) for generation and
    evaluation, so a narrow child match still hands the LLM its full
    surrounding context.
    """
    out: list[Document] = []
    for doc in docs:
        # Keep short docs (like spreadsheet rows) intact — nothing to split.
        if doc.kind in _ATOMIC_KINDS or len(doc.text) <= child_size:
            meta = {**doc.meta, "chunk_type": _chunk_type(doc.meta)}
            out.append(Document(text=doc.text, source=doc.source, kind=doc.kind, meta=meta))
            continue
        sections = _split_sections(doc.text)
        for p, (parent_text, headings) in enumerate(_pack_sections(sections, parent_size, parent_overlap)):
            meta_base = {**doc.meta, "chunk": p, "chunk_type": _chunk_type(doc.meta)}
            if headings:
                meta_base["section"] = " / ".join(headings)
            for c, child_text in enumerate(_split_text(parent_text, child_size, child_overlap)):
                meta = {**meta_base, "child": c, "parent_text": parent_text}
                out.append(Document(text=child_text, source=doc.source, kind=doc.kind, meta=meta))
    return out
