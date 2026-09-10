"""
Shared document text extraction for pdf / docx / txt.

Lives in one place because the QA generator and the RAG indexer must see the
*same* text — if they disagree, the generated ground truth can reference
content the retriever was never given, and every metric computed on top is
measuring the mismatch rather than the pipeline.
"""
import os
import logging

logger = logging.getLogger(__name__)


def _iter_docx_blocks(document):
    """Yield paragraphs and tables in true document order.

    python-docx exposes `.paragraphs` and `.tables` as separate flat lists, so
    the obvious `"\\n".join(p.text for p in doc.paragraphs)` silently discards
    every table in the file. That is not a rare edge case — report-style
    documents keep most of their substance in tables.
    """
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    from docx.oxml.ns import qn

    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _read_docx(path: str) -> str:
    import docx
    from docx.table import Table

    document = docx.Document(path)
    parts: list[str] = []

    for block in _iter_docx_blocks(document):
        if isinstance(block, Table):
            for row in block.rows:
                cells = [c.text.strip() for c in row.cells]
                # Merged cells repeat their text across the span; collapse runs
                # of identical neighbours so the chunk text is not padded out.
                deduped = [c for i, c in enumerate(cells) if i == 0 or c != cells[i - 1]]
                line = " | ".join(c for c in deduped if c)
                if line:
                    parts.append(line)
        else:
            text = block.text.strip()
            if text:
                parts.append(text)

    return "\n".join(parts)


def _read_pdf(path: str) -> str:
    """Extract text from PDF using PyMuPDF (fitz) for accurate extraction.
    
    PyMuPDF handles complex layouts, tables, multi-column text, embedded fonts,
    and styled/formatted PDFs much better than PyPDF2.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        parts: list[str] = []
        
        for page in doc:
            # Extract text with layout preservation
            text = page.get_text("text")
            if text and text.strip():
                parts.append(text.strip())
        
        doc.close()
        return "\n\n".join(parts)
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed for {path}: {e}. Falling back to PyPDF2.")
        # Fallback to PyPDF2 if PyMuPDF fails
        try:
            import PyPDF2
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        except Exception as fallback_error:
            logger.error(f"Both PyMuPDF and PyPDF2 failed for {path}: {fallback_error}")
            return ""


def _read_xlsx(path: str) -> str:
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    parts: list[str] = []
    for sheet in wb.worksheets:
        for row in sheet.iter_rows(values_only=True):
            row_vals = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
            if row_vals:
                parts.append(" | ".join(row_vals))
    return "\n".join(parts)


def _read_xls(path: str) -> str:
    import io
    import xlrd
    import msoffcrypto

    # Try standard open first
    try:
        wb = xlrd.open_workbook(path)
    except xlrd.biffh.XLRDError:
        # If encrypted/protected, attempt standard default office password (e.g. VelvetSweatshop)
        try:
            with open(path, "rb") as f:
                office_file = msoffcrypto.OfficeFile(f)
                office_file.load_key(password="VelvetSweatshop")
                decrypted = io.BytesIO()
                office_file.decrypt(decrypted)
                decrypted.seek(0)
                wb = xlrd.open_workbook(file_contents=decrypted.getvalue())
        except Exception as e:
            logger.warning("Could not decrypt .xls file %s: %s", path, e)
            return ""
    except Exception as e:
        logger.warning("Could not parse .xls file %s: %s", path, e)
        return ""

    parts: list[str] = []
    for sheet in wb.sheets():
        for row_idx in range(sheet.nrows):
            row_vals = [
                str(sheet.cell_value(row_idx, col_idx)).strip()
                for col_idx in range(sheet.ncols)
                if str(sheet.cell_value(row_idx, col_idx)).strip()
            ]
            if row_vals:
                parts.append(" | ".join(row_vals))
    return "\n".join(parts)


def _read_msg(path: str) -> str:
    try:
        import extract_msg
        msg = extract_msg.Message(path)
        parts = []
        if msg.subject:
            parts.append(f"Subject: {msg.subject.strip()}")
        if msg.sender:
            parts.append(f"From: {msg.sender.strip()}")
        if msg.date:
            parts.append(f"Date: {msg.date}")
        if msg.body:
            parts.append(msg.body.strip())
        msg.close()
        return "\n\n".join(p for p in parts if p)
    except Exception as e:
        logger.warning("Could not parse .msg file %s: %s", path, e)
        return ""


def read_document(path: str) -> str:
    """Extract plain text from a pdf, docx, xlsx, xls, msg or plain-text file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf(path).strip()
    if ext == ".docx":
        return _read_docx(path).strip()
    if ext == ".xlsx":
        return _read_xlsx(path).strip()
    if ext == ".xls":
        return _read_xls(path).strip()
    if ext == ".msg":
        return _read_msg(path).strip()

    # Generic plaintext fallback (txt, csv, json, md, etc.)
    try:
        with open(path, "rb") as f:
            raw_bytes = f.read()
        # Binary guard: if file contains null bytes or high ratio of non-printable bytes, skip it
        if b"\x00" in raw_bytes:
            logger.warning("Skipping binary file %s that has no specific parser", path)
            return ""
        return raw_bytes.decode("utf-8", errors="ignore").strip()
    except Exception as e:
        logger.warning("Could not read plain text file %s: %s", path, e)
        return ""


def document_paths(session) -> list[str]:
    """Every readable document path on a session, in upload order.

    Bridges the multi-file `documents` column and the legacy single
    `document_path`, so sessions created before multi-upload still resolve.
    Missing files are skipped rather than raising — a deleted upload should not
    make the whole session unusable.
    """
    import os
    paths: list[str] = []
    for entry in (getattr(session, "documents", None) or []):
        path = (entry or {}).get("path")
        if path and path not in paths:
            paths.append(path)
    if not paths and getattr(session, "document_path", None):
        paths.append(session.document_path)
    return [
        p for p in paths 
        if os.path.exists(p) and "copy of plans" not in os.path.basename(p).lower()
    ]


def display_name(path: str) -> str:
    """The user's original filename, recovered from the stored path.

    Uploads are saved as `{session_uuid}_{original name}` to keep sessions from
    colliding on disk. That prefix must not leak into source attribution, where
    it would be shown to users and logged into LangSmith.
    """
    import os, re
    base = os.path.basename(path)
    return re.sub(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}_",
        "", base,
    )


def read_documents(paths: list[str]) -> list[tuple[str, str]]:
    """Read several documents into (filename, text) pairs.

    Returned per-file rather than concatenated so downstream code can chunk each
    document separately. A chunk that straddled two unrelated files would be
    incoherent to retrieve and impossible to attribute.
    """
    out: list[tuple[str, str]] = []
    for path in paths:
        try:
            text = read_document(path)
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e)
            continue
        if text:
            out.append((display_name(path), text))
    return out
