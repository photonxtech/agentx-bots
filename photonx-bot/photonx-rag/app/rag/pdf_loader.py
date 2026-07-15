"""PDF loading (STEP 1).

Reads PDF files and extracts text page-by-page, preserving the 1-based page
number for every page so that citations can point users to an exact location
in the source documentation.

Design note — the ``DocumentLoader`` protocol below is what makes the pipeline
"future ready": adding Word / Markdown / HTML support later means writing a new
loader that yields the same ``PageContent`` objects, with zero changes
downstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.exceptions import DocumentError
from app.core.logger import get_logger
from app.schemas.documents import PageContent

logger = get_logger(__name__)


class DocumentLoader(Protocol):
    """A source-format loader that yields normalized pages."""

    def load(self, path: Path) -> list[PageContent]:  # pragma: no cover
        ...


class PDFLoader:
    """Extract text from PDF files using :mod:`pypdf`."""

    def load(self, path: Path) -> list[PageContent]:
        """Extract every page of a single PDF.

        Args:
            path: Path to a ``.pdf`` file.

        Returns:
            One :class:`PageContent` per page that yielded non-empty text.

        Raises:
            DocumentError: If the file is missing, unreadable, or contains no
                extractable text at all.
        """
        if not path.exists():
            raise DocumentError(f"PDF not found: {path}")

        logger.info("Loading PDF: %s", path.name)
        try:
            reader = PdfReader(str(path))
        except (PdfReadError, OSError) as exc:
            raise DocumentError(f"Failed to read PDF '{path.name}': {exc}") from exc

        pages: list[PageContent] = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                logger.debug("Skipping empty page %d in %s", index, path.name)
                continue
            pages.append(
                PageContent(document=path.name, page=index, text=text)
            )

        if not pages:
            raise DocumentError(
                f"No extractable text found in '{path.name}' "
                "(it may be a scanned/image-only PDF)."
            )

        logger.info("Extracted %d non-empty pages from %s", len(pages), path.name)
        return pages

    def load_dir(self, directory: Path) -> Iterable[PageContent]:
        """Load every ``*.pdf`` in ``directory``, sorted for determinism.

        Individual unreadable files are logged and skipped so one bad PDF does
        not abort a whole batch ingest.

        Raises:
            DocumentError: If the directory contains no PDF files at all.
        """
        if not directory.exists():
            raise DocumentError(f"Docs directory does not exist: {directory}")

        pdf_paths = sorted(directory.glob("*.pdf"))
        if not pdf_paths:
            raise DocumentError(f"No PDF files found in {directory}")

        for path in pdf_paths:
            try:
                yield from self.load(path)
            except DocumentError as exc:
                logger.error("Skipping '%s': %s", path.name, exc)
