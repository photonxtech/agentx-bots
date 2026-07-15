"""Chunking (STEP 2).

Splits page text into overlapping, retrieval-sized chunks. The strategy is:

1. Strip repeated page boilerplate (headers/footers such as the company name,
   URL, and "Page N" lines that appear on every page). Left in, these create
   dozens of near-identical low-value chunks that pollute retrieval.
2. Detect *real* section headings and split each page into logical sections, so
   a chunk rarely straddles two unrelated topics and can be tagged with the
   heading it belongs to. Heading detection is deliberately strict: short
   in-body labels (e.g. "Cloud Architecture") must NOT be treated as headings,
   or the prose that explains them gets orphaned into tiny fragments.
3. Merge sections whose bodies are too short to stand alone, then pack the text
   into ``chunk_size``-character windows with ``chunk_overlap`` characters of
   overlap, preferring sentence boundaries so we don't cut mid-sentence.

Every emitted chunk carries ``{id, text, page, section, document}`` metadata.
"""

from __future__ import annotations

import re
import uuid
from collections import Counter

from app.core.logger import get_logger
from app.schemas.documents import Chunk, PageContent

logger = get_logger(__name__)

# Sentence-boundary splitter used to find clean break points.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

# Lines that are page boilerplate regardless of the specific document.
_BOILERPLATE_RE = re.compile(
    r"^\s*(?:"
    r"page\s+\d+"                       # "Page 3"
    r"|\d+\s*/\s*\d+"                   # "3 / 7"
    r"|(?:https?://)?[\w.-]+\.[a-z]{2,}(?:\s*\|\s*\S+@\S+)?"  # url / url | email
    r"|\S+@\S+\.\S+"                    # bare email
    r")\s*$",
    re.IGNORECASE,
)

# A section heading: an ALL-CAPS title or a numbered heading. We intentionally
# do NOT treat ordinary Title-Case lines as headings — in real brochures those
# are content labels, and splitting on them shreds the surrounding prose.
_HEADING_RE = re.compile(
    r"^\s*("
    r"\d+(?:\.\d+)*\.?\s+[A-Z][\w &/\-]{2,60}"  # "2.1 Services"
    r"|[A-Z][A-Z0-9 &/\-]{3,60}"                 # "OUR SERVICES" / "SERVICES"
    r")\s*$"
)

# A section body shorter than this is merged with its neighbour rather than
# becoming a standalone (low-context) chunk.
_MIN_SECTION_CHARS = 200


class Chunker:
    """Heading-aware, overlap-preserving text chunker."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: list[PageContent]) -> list[Chunk]:
        """Chunk many pages.

        A heading is carried forward only *within* a page, to unlabeled content
        that immediately follows it. It is deliberately **not** propagated
        across page boundaries: a heading from one page (especially a numbered
        process step like "3. Develop & Deploy") is a local label, and letting
        it persist would mislabel unrelated content on later pages.
        """
        repeated = self._repeated_lines(pages)
        all_chunks: list[Chunk] = []

        for page in pages:
            cleaned = self._strip_boilerplate(page.text, repeated)
            sections = self._merge_short_sections(
                self._split_into_sections(cleaned)
            )
            # Reset per page: leading unlabeled content on a new page is
            # "General", not whatever heading appeared on a previous page.
            last_section = "General"
            for section, body in sections:
                if section == "General":
                    section = last_section
                elif not self._is_numbered_step(section):
                    # Only non-step headings persist to following content.
                    last_section = section
                for text in self._window(body):
                    all_chunks.append(
                        Chunk(
                            id=str(uuid.uuid4()),
                            text=text,
                            document=page.document,
                            page=page.page,
                            section=section,
                        )
                    )

        logger.info("Created %d chunks from %d pages", len(all_chunks), len(pages))
        return all_chunks

    @staticmethod
    def _is_numbered_step(section: str) -> bool:
        """True for headings like "3. Develop & Deploy" (process-step labels)."""
        return bool(re.match(r"^\d+[.)]\s", section.strip()))

    def chunk_page(self, page: PageContent) -> list[Chunk]:
        """Chunk a single page (convenience wrapper over :meth:`chunk_pages`)."""
        return self.chunk_pages([page])

    # -- boilerplate removal ------------------------------------------------
    @staticmethod
    def _repeated_lines(pages: list[PageContent]) -> set[str]:
        """Find non-trivial lines that appear on most pages (headers/footers)."""
        if len(pages) < 3:
            return set()
        counter: Counter[str] = Counter()
        for page in pages:
            seen_on_page = {
                ln.strip()
                for ln in page.text.splitlines()
                if 0 < len(ln.strip()) <= 60
            }
            counter.update(seen_on_page)
        threshold = max(2, int(len(pages) * 0.6))
        return {line for line, count in counter.items() if count >= threshold}

    def _strip_boilerplate(self, text: str, repeated: set[str]) -> str:
        """Drop boilerplate and repeated header/footer lines from a page."""
        kept: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                kept.append(line)
                continue
            if stripped in repeated or _BOILERPLATE_RE.match(stripped):
                continue
            kept.append(line)
        return "\n".join(kept)

    # -- sectioning ---------------------------------------------------------
    def _split_into_sections(self, text: str) -> list[tuple[str, str]]:
        """Split page text into ``(heading, body)`` pairs."""
        sections: list[tuple[str, str]] = []
        current_heading = "General"
        buffer: list[str] = []

        for line in text.splitlines():
            if self._is_heading(line):
                if buffer:
                    sections.append((current_heading, "\n".join(buffer).strip()))
                    buffer = []
                current_heading = line.strip()
            else:
                buffer.append(line)

        if buffer:
            sections.append((current_heading, "\n".join(buffer).strip()))

        return [(h, b) for h, b in sections if b]

    @staticmethod
    def _is_heading(line: str) -> bool:
        stripped = line.strip()
        if not stripped or len(stripped) > 60:
            return False
        # Require at least two words OR a numbered prefix, to avoid matching
        # single-word body labels sitting on their own line.
        if not re.match(r"^\d", stripped) and len(stripped.split()) < 2:
            return False
        return bool(_HEADING_RE.match(stripped))

    def _merge_short_sections(
        self, sections: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Merge consecutive sections until each body has enough context.

        Short sections keep their *first* heading as the section label so the
        chunk is still attributed to a meaningful part of the document.
        """
        if not sections:
            return []

        merged: list[tuple[str, str]] = []
        head, body = sections[0]
        for next_head, next_body in sections[1:]:
            if len(body) < _MIN_SECTION_CHARS:
                # Absorb the next section into the current one.
                body = f"{body}\n{next_body}".strip()
                if head == "General":
                    head = next_head
            else:
                merged.append((head, body))
                head, body = next_head, next_body
        merged.append((head, body))
        return merged

    # -- windowing ----------------------------------------------------------
    def _window(self, text: str) -> list[str]:
        """Pack text into overlapping windows, preferring sentence breaks."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self._chunk_size:
            return [text]

        windows: list[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + self._chunk_size, n)
            if end < n:
                boundary = self._last_sentence_boundary(text, start, end)
                if boundary > start + self._chunk_size // 2:
                    end = boundary
            chunk = text[start:end].strip()
            if chunk:
                windows.append(chunk)
            if end >= n:
                break
            start = max(end - self._chunk_overlap, start + 1)
        return windows

    @staticmethod
    def _last_sentence_boundary(text: str, start: int, end: int) -> int:
        """Return the index just after the last sentence end within the window."""
        window = text[start:end]
        matches = list(_SENTENCE_END_RE.finditer(window))
        if not matches:
            return end
        return start + matches[-1].end()
