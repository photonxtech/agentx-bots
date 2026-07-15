"""DOCX loading (STEP 1, Word variant).

Reads ``.docx`` files and extracts their text as normalized :class:`PageContent`
objects — the same shape :class:`~app.rag.pdf_loader.PDFLoader` yields, so
everything downstream (chunker → embedder → store) is unchanged.

Word documents have no reliable notion of "pages" (pagination is decided by the
renderer, not stored in the file), so a ``.docx`` is emitted as a single logical
page numbered ``1``. Paragraph structure is preserved as newlines, which lets
the heading-aware chunker still detect sections within the document.
"""

from __future__ import annotations

from pathlib import Path

import docx
from docx.document import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.core.exceptions import DocumentError
from app.core.logger import get_logger
from app.schemas.documents import PageContent

logger = get_logger(__name__)


def _iter_block_text(document: DocxDocument):
    """Yield text from paragraphs and tables in document order.

    ``python-docx`` exposes ``document.paragraphs`` and ``document.tables``
    separately and does NOT include table cell text in ``paragraphs`` — so any
    content stored in a table (e.g. a "Featured Projects" grid) is silently
    dropped if you only read paragraphs. We instead walk the document body's
    XML children in order, so paragraph and table content is captured together
    and keeps its original position (and therefore its section context).
    """
    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]  # strip XML namespace
        if tag == "p":
            text = Paragraph(child, document).text.strip()
            if text:
                yield text
        elif tag == "tbl":
            table = Table(child, document)
            for row in table.rows:
                # Join a row's cells so "Name | Description" stays on one line;
                # de-duplicate merged cells that repeat the same text.
                cells: list[str] = []
                for cell in row.cells:
                    value = cell.text.strip()
                    if value and value not in cells:
                        cells.append(value)
                line = " — ".join(cells)
                if line:
                    yield line


class DocxLoader:
    """Extract text from Word ``.docx`` files using :mod:`python-docx`."""

    def load(self, path: Path) -> list[PageContent]:
        """Extract the text of a single ``.docx`` file.

        Args:
            path: Path to a ``.docx`` file.

        Returns:
            A single-element list holding one :class:`PageContent` (page ``1``),
            or an empty list if the document has no extractable text.

        Raises:
            DocumentError: If the file is missing or is not a valid ``.docx``.
        """
        if not path.exists():
            raise DocumentError(f"DOCX not found: {path}")

        logger.info("Loading DOCX: %s", path.name)
        try:
            document = docx.Document(str(path))
        except (PackageNotFoundError, OSError, ValueError) as exc:
            raise DocumentError(
                f"Failed to read DOCX '{path.name}': {exc}"
            ) from exc

        # Walk paragraphs AND tables in document order, so table content (e.g.
        # a Featured Projects grid) is captured, not silently dropped. Newlines
        # preserve structure for the chunker's section detection.
        blocks = list(_iter_block_text(document))
        text = "\n".join(blocks).strip()

        if not text:
            raise DocumentError(
                f"No extractable text found in '{path.name}'."
            )

        logger.info("Extracted %d text blocks from %s", len(blocks), path.name)
        return [PageContent(document=path.name, page=1, text=text)]
