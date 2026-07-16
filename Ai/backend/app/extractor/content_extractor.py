import trafilatura
from bs4 import BeautifulSoup

from app.extractor.cleaner import extract_with_bs4
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""


def _unique_paragraphs(candidate: str, existing: str) -> str:
    """Paragraphs from `candidate` that aren't already substantially present in `existing`."""
    existing_lower = existing.lower()
    unique = []
    for para in candidate.split("\n"):
        para = para.strip()
        # A short-but-meaningful line (e.g. a person's name) shouldn't be discarded just
        # for being brief — only drop near-empty scraps and lines already covered above.
        if len(para) < 3 or para.lower() in existing_lower:
            continue
        unique.append(para)
    return "\n".join(unique)


def extract_content_from_html(html: str, url: str, min_length: int = 200) -> tuple[str, str]:
    title = _extract_title(html)

    trafilatura_text = trafilatura.extract(html, url=url, favor_precision=True) or ""

    if len(trafilatura_text) < min_length:
        logger.info(
            "Trafilatura extraction too short (%d chars) for %s, falling back to BeautifulSoup",
            len(trafilatura_text), url,
        )
        fallback_text = extract_with_bs4(html)
        return title, fallback_text or trafilatura_text

    # Trafilatura's boilerplate-detection can drop small but meaningful sections (author
    # bios, contact blurbs) even when the main article extraction otherwise succeeds.
    # Append anything BeautifulSoup found that trafilatura's text doesn't already cover,
    # so that info isn't silently missing from the index.
    bs4_text = extract_with_bs4(html)
    extra = _unique_paragraphs(bs4_text, trafilatura_text)
    combined = f"{trafilatura_text}\n\n{extra}" if extra else trafilatura_text
    return title, combined
