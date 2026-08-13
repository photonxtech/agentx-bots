"""Ingestion: turn uploaded Excel files and images into plain text.

Each source becomes a list of `Document` records. A Document is just a piece of
text plus metadata (where it came from) so we can cite sources in answers.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field

import config


@dataclass
class Document:
    """A unit of ingested text with provenance."""
    text: str
    source: str                       # filename
    kind: str                         # "image" | "pdf" | "docx" | "pptx" | "text"
    meta: dict = field(default_factory=dict)


# Sentinel prefixed to a heading line so chunking.py can split on real document
# structure (headings/titles) instead of blindly packing by character count.
# Null bytes never occur in extracted text, so this can't collide with content.
HEADING_MARK = "\x00H\x00"

# A bare page number on its own line ("47"), or a numbered TOC-entry line
# ("4.4 Experience Exercise: Worst and Best") — PDF text extraction commonly
# splits a dot-leader TOC entry's title and page number onto separate lines,
# so counting either shape catches both layouts.
#
# Two alternatives for the entry shape:
#   1. Dotted sub-level numbering ("4.4", "4.10", "1.2.3") followed by either
#      a space or (seen on real documents — font/kerning during PDF text
#      extraction can drop the space entirely) directly by a capital letter,
#      e.g. "4.4Need to Know: Three Typical Problem Solving Mistakes". The
#      dotted prefix is specific enough on its own that skipping the space
#      check here doesn't risk matching ordinary prose.
#   2. Bare top-level numbering ("1", "23") followed by a REQUIRED space —
#      kept strict (unlike the dotted case) so an ordinary numbered list like
#      "1. Introduction to the topic..." is never mistaken for a TOC entry;
#      the required space also excludes the "1." + space form, which is that
#      exact ordinary-list style.
_TOC_PAGENUM_RE = re.compile(r"^\d{1,4}$")
_TOC_ENTRY_RE = re.compile(r"^\d+\.\d+(?:\.\d+)*(?:\s+\S|[A-Z])|^\d+\s+\S")


def _looks_like_toc(text: str) -> bool:
    """True when a page is a table of contents / section index, not real
    content — retrieved anyway, it's dense with the same keywords/phrases as
    real content elsewhere in the document (verbatim exercise/lesson titles),
    which fools both BM25 and even a cross-encoder reranker into ranking it
    highly despite having zero explanatory value. Real content pages rarely
    have more than one or two lines matching either shape above; a TOC/index
    page is made almost entirely of them. Validated against real ingested
    pages: true positive on an actual TOC page (ratio 0.92), true negatives
    on several real content pages, including one with numbered list items
    (ratio <= 0.10 in every case) — 0.5 cleanly separates the two.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 6:
        return False
    toc_like = sum(1 for l in lines if _TOC_PAGENUM_RE.match(l) or _TOC_ENTRY_RE.match(l))
    return toc_like / len(lines) >= 0.5


def chunk_type(meta: dict) -> str:
    """"toc" for a page/chunk flagged TOC/index at ingestion (is_index, or
    the forward-compat is_toc alias — see vectorstore._is_toc), else
    "content". Same underlying signal as those boolean keys, kept alongside
    them (not instead of) as a plain string field that's easier to read in
    diagnostics/logs and to filter/group on directly.
    """
    return "toc" if bool(meta.get("is_index") or meta.get("is_toc")) else "content"


def mark_heading(text: str) -> str:
    """Wrap a heading/title line so chunking.py recognizes it as a section break."""
    return f"{HEADING_MARK}{text}"


# --------------------------------------------------------------------------- #
# Plain text / Markdown
# --------------------------------------------------------------------------- #
def load_text(file_bytes: bytes, filename: str, progress_callback=None) -> list[Document]:
    """Read a .txt/.md file as one Document; chunking will split it later."""
    if progress_callback:
        progress_callback(1, 1, "Reading text file")
    text = file_bytes.decode("utf-8", errors="replace").strip()
    if not text:
        return []
    return [
        Document(
            text=text,
            source=filename,
            kind="text",
            meta={"chars": len(text)},
        )
    ]


# --------------------------------------------------------------------------- #
# Images (OCR)
# --------------------------------------------------------------------------- #
# EasyOCR loads a model into memory; do it lazily and only once.
_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr  # imported lazily so the app starts fast

        _ocr_reader = easyocr.Reader(config.OCR_LANGUAGES, gpu=False, verbose=False)
    return _ocr_reader


def _ocr_pil_image(image) -> str:
    """OCR a PIL image -> text. The single place OCR actually runs, so images
    from files, embedded PDF images, and rendered PDF pages all share it."""
    import numpy as np

    reader = _get_ocr_reader()
    # detail=0 -> just the strings, in reading order.
    lines = reader.readtext(np.array(image.convert("RGB")), detail=0, paragraph=True)
    return "\n".join(lines).strip()


def _extract_image_text(image) -> str:
    """Everything we can learn from one PIL image: OCR text + a vision-model
    description. The single shared path for uploaded images, embedded PDF/DOCX
    images, and rendered scanned pages. Vision failures degrade to OCR-only."""
    from rag import vision

    parts = [_ocr_pil_image(image)]
    desc = vision.describe_image(image)
    if desc:
        parts.append(f"[Visual description] {desc}")
    return "\n".join(p for p in parts if p).strip()


def load_image(file_bytes: bytes, filename: str, progress_callback=None) -> list[Document]:
    """Extract OCR text + a vision description from an image as one Document."""
    if progress_callback:
        progress_callback(1, 1, "Processing image & running OCR")
    from PIL import Image

    image = Image.open(io.BytesIO(file_bytes))
    text = _extract_image_text(image)

    if not text:
        return []

    return [
        Document(
            text=text,
            source=filename,
            kind="image",
            meta={"chars": len(text)},
        )
    ]


# --------------------------------------------------------------------------- #
# PDF — hybrid: text layer + OCR of embedded images (both combined per page)
# --------------------------------------------------------------------------- #
def _safe_stem(filename: str) -> str:
    """Filesystem-safe stem for naming extracted image files."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return "".join(c if c.isalnum() else "_" for c in stem) or "pdf"


def _extract_pdf_text_with_headings(page) -> str:
    """Page text, with large/bold short lines marked as headings.

    Uses PyMuPDF's font-size info to flag lines whose text is noticeably
    bigger than the page's median font size (and short, like a real heading)
    so chunking.py can split on them. Best-effort: any failure or a page with
    no recognizable spans just falls back to the plain text layer.
    """
    try:
        page_dict = page.get_text("dict")
        sizes = [
            span["size"]
            for block in page_dict.get("blocks", [])
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if span.get("text", "").strip()
        ]
        if len(sizes) < 5:
            return page.get_text("text").strip()
        median_size = sorted(sizes)[len(sizes) // 2]

        lines_out: list[str] = []
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                max_size = max(s["size"] for s in spans)
                if max_size >= median_size * 1.15 and len(text.split()) <= 12:
                    lines_out.append(mark_heading(text))
                else:
                    lines_out.append(text)
        return "\n".join(lines_out).strip()
    except Exception:
        return page.get_text("text").strip()


def _repeated_boilerplate_lines(pdf) -> set[str]:
    """Lines (whitespace/case-normalized) appearing identically on at least
    config.HEADER_FOOTER_MIN_PAGES distinct pages of this PDF — see
    config.STRIP_REPEATED_HEADERS_FOOTERS.

    A cheap text-only pre-pass (page.get_text("text"), no heading detection,
    no image/OCR work) purely to build the frequency table before the real
    per-page extraction loop in load_pdf runs.
    """
    counts: dict[str, int] = {}
    for page in pdf:
        seen_this_page: set[str] = set()
        for line in page.get_text("text").split("\n"):
            norm = re.sub(r"\s+", " ", line.strip()).lower()
            if len(norm) < config.HEADER_FOOTER_MIN_LINE_CHARS or norm in seen_this_page:
                continue  # a line repeated within the SAME page shouldn't inflate its page count
            seen_this_page.add(norm)
            counts[norm] = counts.get(norm, 0) + 1
    return {line for line, c in counts.items() if c >= config.HEADER_FOOTER_MIN_PAGES}


def _strip_boilerplate_lines(text: str, boilerplate: set[str]) -> str:
    """Drop any line of `text` that's in `boilerplate`, ignoring the
    HEADING_MARK sentinel when comparing so a font-size-detected heading
    that happens to BE a recurring banner still gets caught.
    """
    if not boilerplate:
        return text
    kept = []
    for line in text.split("\n"):
        body = line[len(HEADING_MARK):] if line.startswith(HEADING_MARK) else line
        norm = re.sub(r"\s+", " ", body.strip()).lower()
        if norm in boilerplate:
            continue
        kept.append(line)
    return "\n".join(kept)


def load_pdf(file_bytes: bytes, filename: str, progress_callback=None) -> list[Document]:
    """Extract each PDF page as one Document, combining BOTH sources of text:

    1. The page's text layer (fast, accurate) — for normal typed paragraphs.
    2. OCR of every embedded image on the page — so words baked into charts,
       screenshots, logos, or scans aren't lost on otherwise-text pages.

    Embedded images are also saved to disk (config.EXTRACTED_IMAGES_DIR) so they
    can later be fed to a vision model. If a page has neither a usable text
    layer nor embedded images, we render the whole page and OCR that.

    One Document per page keeps citations page-precise.
    """
    import fitz  # PyMuPDF, imported lazily

    os.makedirs(config.EXTRACTED_IMAGES_DIR, exist_ok=True)
    stem = _safe_stem(filename)

    docs: list[Document] = []
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    total_pages = len(pdf)

    boilerplate_lines = (
        _repeated_boilerplate_lines(pdf) if config.STRIP_REPEATED_HEADERS_FOOTERS else set()
    )

    for page_num, page in enumerate(pdf, 1):
        if progress_callback:
            progress_callback(page_num, total_pages, f"Parsing PDF page {page_num} of {total_pages}")
        text = _extract_pdf_text_with_headings(page)

        embedded = page.get_images(full=True)
        image_texts: list[str] = []
        saved_paths: list[str] = []
        rendered_ocr = ""

        if len(embedded) > config.MAX_EMBEDDED_IMAGES_PER_PAGE:
            # A page with this many separate embedded images is almost always
            # one complex diagram/collage sliced into many raster tiles by
            # whatever exported this PDF, not N independently meaningful
            # photos — describing each fragment individually is both slow
            # (every image is its own vision call, config.VISION_MIN_INTERVAL
            # apart — hundreds of images on one page can add many minutes to
            # a single page) and produces noisy, disjointed per-fragment
            # descriptions instead of one coherent one. Render the whole page
            # once instead, same as the fully-scanned-page fallback below.
            rendered_ocr = _ocr_pdf_page(page)
        else:
            # --- OCR + save every embedded image on this page ---
            for img_i, img in enumerate(embedded, 1):
                xref = img[0]
                saved, ocr_text = _handle_embedded_image(pdf, xref, stem, page_num, img_i)
                if saved:
                    saved_paths.append(saved)
                if ocr_text:
                    image_texts.append(f"[Embedded image {img_i}] {ocr_text}")

            # --- fully-scanned page fallback: no text AND no embedded images ---
            if len(text) < config.PDF_OCR_MIN_CHARS and not image_texts:
                rendered_ocr = _ocr_pdf_page(page)

        combined = "\n".join(p for p in [text, *image_texts, rendered_ocr] if p).strip()
        if boilerplate_lines:
            # Applied to the COMBINED text (not just the text layer) so a
            # banner re-captured via OCR on a scanned page gets stripped
            # too. Deliberately after the OCR-fallback decision above,
            # which uses the un-stripped `text` — otherwise a page that's
            # ENTIRELY boilerplate would look text-sparse post-strip and
            # trigger a full-page OCR that just re-introduces the same
            # banner text via the image render.
            combined = _strip_boilerplate_lines(combined, boilerplate_lines)
        if not combined:
            # The page has image(s) but we got no text from them (OCR empty and
            # the vision model was unavailable/rate-limited). Keep a placeholder
            # so the page is never silently dropped — it stays retrievable and
            # its saved image is still referenced.
            if saved_paths:
                combined = (
                    f"[Image on page {page_num} of {filename} — no text detected; "
                    "a visual description was unavailable at ingestion time.]"
                )
            else:
                continue

        used_ocr = bool(image_texts or rendered_ocr)
        docs.append(
            Document(
                text=combined,
                source=filename,
                kind="pdf",
                meta={
                    "page": page_num,
                    "ocr": used_ocr,
                    "images": saved_paths,
                    "is_index": _looks_like_toc(text),
                },
            )
        )

    pdf.close()
    return docs


def _save_and_ocr_image(img_bytes: bytes, ext: str, basename: str) -> tuple[str | None, str]:
    """Save one embedded image to disk, then OCR + vision-describe it. Shared
    by PDF and DOCX.

    Returns (saved_path_or_None, extracted_text). Failures are swallowed so one
    bad image (exotic codec, CMYK, etc.) never aborts ingestion of the whole
    file. Tiny images (logos/icons/rules) are skipped as noise.
    """
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(img_bytes))
        if image.width * image.height < config.MIN_EMBEDDED_IMAGE_AREA:
            return None, ""

        path = os.path.join(config.EXTRACTED_IMAGES_DIR, f"{basename}.{ext}")
        with open(path, "wb") as f:
            f.write(img_bytes)

        return path, _extract_image_text(image)
    except Exception:
        return None, ""


def _handle_embedded_image(pdf, xref, stem, page_num, img_i) -> tuple[str | None, str]:
    """Pull one embedded image out of a PDF by xref, then save + OCR it."""
    try:
        base = pdf.extract_image(xref)          # raw bytes in original format
        return _save_and_ocr_image(
            base["image"], base.get("ext", "png"), f"{stem}_p{page_num}_img{img_i}"
        )
    except Exception:
        return None, ""


def _ocr_pdf_page(page) -> str:
    """Render a whole PyMuPDF page to an image, then OCR + vision-describe it
    (scanned-page path)."""
    import fitz
    from PIL import Image

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom => sharper OCR
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return _extract_image_text(image)


# --------------------------------------------------------------------------- #
# Word .docx — paragraphs + tables + OCR of embedded images
# --------------------------------------------------------------------------- #
def load_docx(file_bytes: bytes, filename: str, progress_callback=None) -> list[Document]:
    """Extract a Word document as one Document, combining:

    1. Paragraph text.
    2. Table cells (flattened row by row as "cell | cell | ...").
    3. OCR of every embedded image (charts/screenshots), which is also saved
       to disk for later vision-model use.

    Chunking splits the combined text later, so we return a single Document.
    """
    if progress_callback:
        progress_callback(1, 1, "Parsing Word document")
    import docx  # python-docx, imported lazily

    os.makedirs(config.EXTRACTED_IMAGES_DIR, exist_ok=True)
    stem = _safe_stem(filename)

    doc = docx.Document(io.BytesIO(file_bytes))
    parts: list[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        try:
            style_name = para.style.name if para.style else ""
        except Exception:
            style_name = ""
        if style_name.startswith(("Heading", "Title", "Subtitle")):
            parts.append(mark_heading(text))
        else:
            parts.append(text)

    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    # Embedded images live as related parts keyed by relationship.
    saved_paths: list[str] = []
    img_i = 0
    for rel in doc.part.rels.values():
        if "image" not in rel.reltype:
            continue
        img_i += 1
        try:
            blob = rel.target_part.blob
            ext = (rel.target_part.content_type.split("/")[-1] or "png").replace("jpeg", "jpg")
            saved, ocr_text = _save_and_ocr_image(blob, ext, f"{stem}_img{img_i}")
        except Exception:
            saved, ocr_text = None, ""
        if saved:
            saved_paths.append(saved)
        if ocr_text:
            parts.append(f"[Embedded image {img_i}] {ocr_text}")

    combined = "\n".join(parts).strip()
    if not combined:
        return []

    return [
        Document(
            text=combined,
            source=filename,
            kind="docx",
            meta={"ocr": bool(saved_paths), "images": saved_paths},
        )
    ]


# --------------------------------------------------------------------------- #
# PowerPoint .pptx — one Document per slide (text + tables + image OCR/vision)
# --------------------------------------------------------------------------- #
def load_pptx(file_bytes: bytes, filename: str, progress_callback=None) -> list[Document]:
    """Extract each slide as one Document: text frames, tables (flattened
    row by row), and embedded pictures (saved + OCR'd + vision-described)."""
    from pptx import Presentation  # python-pptx, imported lazily

    os.makedirs(config.EXTRACTED_IMAGES_DIR, exist_ok=True)
    stem = _safe_stem(filename)

    prs = Presentation(io.BytesIO(file_bytes))
    docs: list[Document] = []
    total_slides = len(prs.slides)

    for slide_num, slide in enumerate(prs.slides, 1):
        if progress_callback:
            progress_callback(slide_num, total_slides, f"Parsing slide {slide_num} of {total_slides}")
        parts: list[str] = []
        saved_paths: list[str] = []
        img_i = 0

        title_shape = slide.shapes.title  # None if this slide has no title placeholder

        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                text = shape.text_frame.text.strip()
                if title_shape is not None and shape.shape_id == title_shape.shape_id:
                    parts.append(mark_heading(text))
                else:
                    parts.append(text)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        parts.append(" | ".join(cells))
            if shape.shape_type == 13:  # MSO_SHAPE_TYPE.PICTURE
                img_i += 1
                try:
                    blob = shape.image.blob
                    ext = shape.image.ext or "png"
                    saved, img_text = _save_and_ocr_image(
                        blob, ext, f"{stem}_s{slide_num}_img{img_i}"
                    )
                except Exception:
                    saved, img_text = None, ""
                if saved:
                    saved_paths.append(saved)
                if img_text:
                    parts.append(f"[Embedded image {img_i}] {img_text}")

        combined = "\n".join(parts).strip()
        if not combined:
            continue
        docs.append(
            Document(
                text=combined,
                source=filename,
                kind="pptx",
                meta={"slide": slide_num, "images": saved_paths},
            )
        )
    return docs


def ingest(file_bytes: bytes, filename: str, progress_callback=None) -> list[Document]:
    """Dispatch a single uploaded file to the right loader by extension."""
    lower = filename.lower()
    if lower.endswith((".txt", ".md")):
        return load_text(file_bytes, filename, progress_callback=progress_callback)
    if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
        return load_image(file_bytes, filename, progress_callback=progress_callback)
    if lower.endswith(".pdf"):
        return load_pdf(file_bytes, filename, progress_callback=progress_callback)
    if lower.endswith(".docx"):
        return load_docx(file_bytes, filename, progress_callback=progress_callback)
    if lower.endswith(".pptx"):
        return load_pptx(file_bytes, filename, progress_callback=progress_callback)
    raise ValueError(f"Unsupported file type: {filename}")


# --------------------------------------------------------------------------- #
# Per-file cache — skip OCR + vision when the same content is seen again
# --------------------------------------------------------------------------- #
def ingest_cached(file_bytes: bytes, filename: str, progress_callback=None) -> list[Document]:
    """`ingest`, but memoized on the file's content hash.

    OCR and vision calls are the slow/expensive part of ingestion; the cache
    makes re-uploading a file (e.g. after an app restart or index reset)
    effectively instant. A corrupt cache entry is treated as a miss.
    """
    import hashlib
    import pickle

    # Include filename in the cache key so a PDF and a PPTX with identical
    # bytes never collide, and renamed files are treated as new entries.
    name_bytes = filename.encode("utf-8")
    digest = hashlib.sha256(file_bytes + b"\x00" + name_bytes).hexdigest()[:24]
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(config.CACHE_DIR, f"{digest}.pkl")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                res = pickle.load(f)
                if progress_callback:
                    progress_callback(len(res), max(1, len(res)), "Loaded from cache")
                return res
        except Exception:
            pass  # unreadable cache -> re-ingest below

    docs = ingest(file_bytes, filename, progress_callback=progress_callback)
    # Never cache an empty result: an image that yielded nothing is usually a
    # transient failure (vision model down/rate-limited), and caching [] would
    # freeze that failure so re-uploading after a fix still returns nothing.
    if docs:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(docs, f)
        except Exception:
            pass  # caching is best-effort; never fail ingestion over it
    return docs
