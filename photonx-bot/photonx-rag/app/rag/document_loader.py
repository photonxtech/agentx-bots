"""Composite document loader.

Dispatches each source file to the right format loader by extension, so the
ingestion pipeline can index a docs folder that mixes ``.pdf`` and ``.docx``
files without knowing which is which. Adding another format later (Markdown,
HTML, …) means registering one more loader here — nothing downstream changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from app.core.exceptions import DocumentError
from app.core.logger import get_logger
from app.rag.docx_loader import DocxLoader
from app.rag.pdf_loader import DocumentLoader, PDFLoader
from app.schemas.documents import PageContent

logger = get_logger(__name__)


class CompositeLoader:
    """Route files to a per-extension :class:`DocumentLoader`."""

    def __init__(self) -> None:
        # Extension → loader. Keys are lowercase, dot-prefixed.
        self._loaders: dict[str, DocumentLoader] = {
            ".pdf": PDFLoader(),
            ".docx": DocxLoader(),
        }

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """The file extensions this loader can handle (e.g. ``.pdf``)."""
        return tuple(self._loaders)

    def load(self, path: Path) -> list[PageContent]:
        """Load one file, choosing the loader by its extension.

        Raises:
            DocumentError: If the extension is not supported, or the underlying
                loader fails.
        """
        loader = self._loaders.get(path.suffix.lower())
        if loader is None:
            raise DocumentError(
                f"Unsupported file type '{path.suffix}' for {path.name}. "
                f"Supported: {', '.join(self.supported_extensions)}."
            )
        return loader.load(path)

    def load_dir(self, directory: Path) -> Iterable[PageContent]:
        """Load every supported file in ``directory``, sorted for determinism.

        Individual unreadable files are logged and skipped so one bad file does
        not abort a whole batch ingest.

        Raises:
            DocumentError: If the directory is missing or contains no supported
                files at all.
        """
        if not directory.exists():
            raise DocumentError(f"Docs directory does not exist: {directory}")

        paths = sorted(
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in self._loaders
        )
        if not paths:
            raise DocumentError(
                f"No supported documents "
                f"({', '.join(self.supported_extensions)}) found in {directory}"
            )

        for path in paths:
            try:
                yield from self.load(path)
            except DocumentError as exc:
                logger.error("Skipping '%s': %s", path.name, exc)
