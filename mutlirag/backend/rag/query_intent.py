"""Lightweight, deterministic query-intent routing for retrieval.

Three intents:
  NORMAL_QUERY  - ordinary content question. TOC/index pages stay excluded
                  (see vectorstore.search + config.EXCLUDE_INDEX_PAGES).
  TOC_QUERY     - an explicit "give me the table of contents / chapters /
                  sections" request. Answered deterministically from
                  metadata (vectorstore.get_toc_chunks), not embedding search.
  PAGE_QUERY    - an explicit "page N" / "pages A-B" / "pages A, B and C"
                  request. Answered deterministically from metadata
                  (vectorstore.get_by_pages), bypassing TOC exclusion, since
                  an explicit page request always wins over the generic
                  exclusion rule.

No LLM call is used to classify — regex only, so a request like "give me the
TOC" never has to wait on (or pay for) a model round-trip just to be routed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NORMAL_QUERY = "NORMAL_QUERY"
TOC_QUERY = "TOC_QUERY"
PAGE_QUERY = "PAGE_QUERY"


@dataclass(frozen=True)
class QueryIntent:
    kind: str
    pages: list[int] | None = None


# --------------------------------------------------------------------------- #
# Page-number extraction
# --------------------------------------------------------------------------- #
# A page reference always starts with an explicit keyword ("page", "pages",
# "pg", "p.", "page no", "page number") — a bare number in the question
# ("the 5 Smart Skills") is never treated as a page request.
_PAGE_KEYWORD_RE = re.compile(
    r"\b(?:page\s*(?:no\.?|number)|pages|page|pg\.?)\b|\bp\.(?=\s*\d)",
    re.IGNORECASE,
)
# The run of digits + separators ("-", "to", ",", "and") immediately after a
# page keyword. Stops at the first token that isn't a number or separator, so
# "page 170 say about RAG" only captures "170".
_NUM_RUN_RE = re.compile(r"^\s*[:#]?\s*(\d+(?:\s*(?:-|to|,|and)\s*\d+)*)", re.IGNORECASE)
_SPLIT_RE = re.compile(r",|\band\b", re.IGNORECASE)
_RANGE_RE = re.compile(r"^(\d+)\s*(?:-|to)\s*(\d+)$", re.IGNORECASE)


def _parse_num_run(run: str) -> list[int]:
    """"10 to 15" -> [10..15]; "10, 12 and 15" -> [10, 12, 15]."""
    pages: list[int] = []
    for token in _SPLIT_RE.split(run):
        token = token.strip()
        if not token:
            continue
        m = _RANGE_RE.match(token)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi:
                lo, hi = hi, lo
            pages.extend(range(lo, hi + 1))
        elif token.isdigit():
            pages.append(int(token))
    return pages


def extract_pages(question: str) -> list[int] | None:
    """Explicit page number(s) mentioned in `question`, sorted+deduped, or
    None if there's no unambiguous page reference.

    Scans every page-keyword occurrence (not just the first) so a keyword
    with no number attached ("chapter list with page numbers") doesn't
    swallow a real page reference that appears later in the sentence.
    """
    for m in _PAGE_KEYWORD_RE.finditer(question):
        tail = question[m.end():]
        num_m = _NUM_RUN_RE.match(tail)
        if not num_m:
            continue
        pages = _parse_num_run(num_m.group(1))
        if pages:
            return sorted(set(pages))
    return None


# --------------------------------------------------------------------------- #
# TOC-request detection
# --------------------------------------------------------------------------- #
# Unambiguous on their own - no other phrasing means these.
_TOC_STRONG_RE = re.compile(
    r"table\s+of\s+contents?|\btoc\b|chapter\s+list|chapter\s+structure",
    re.IGNORECASE,
)
# Ambiguous alone ("chapters"/"sections"/"contents" can appear in a normal
# content question), so these only count as TOC intent when the question
# also opens with a listing/lookup verb - matching how every example in the
# spec is actually phrased ("What are...", "List all...", "Show...", "Give
# me...").
_TOC_WEAK_RE = re.compile(
    r"\bcontents?\b|\bchapters?\b|\bsections?\b|topics?\s+(?:are\s+)?covered",
    re.IGNORECASE,
)
_TOC_LEAD_VERB_RE = re.compile(
    r"^\s*(?:what|which|list|show|give|tell|display|provide)\b", re.IGNORECASE
)


def _looks_like_toc_request(question: str) -> bool:
    if _TOC_STRONG_RE.search(question):
        return True
    return bool(_TOC_WEAK_RE.search(question) and _TOC_LEAD_VERB_RE.match(question))


def detect_intent(question: str) -> QueryIntent:
    """Classify `question` into NORMAL_QUERY / TOC_QUERY / PAGE_QUERY.

    Page numbers are checked FIRST: "give me the contents of pages 100, 101
    and 102" mentions both "contents" (a TOC word) and explicit page
    numbers - an explicit page request always takes precedence.
    """
    pages = extract_pages(question)
    if pages:
        return QueryIntent(kind=PAGE_QUERY, pages=pages)
    if _looks_like_toc_request(question):
        return QueryIntent(kind=TOC_QUERY)
    return QueryIntent(kind=NORMAL_QUERY)
