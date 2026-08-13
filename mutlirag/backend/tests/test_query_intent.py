"""Unit tests for rag.query_intent: NORMAL/TOC/PAGE classification and page-
number extraction. Pure regex logic, no model/DB/network dependency."""

import pytest

from rag.query_intent import NORMAL_QUERY, PAGE_QUERY, TOC_QUERY, detect_intent, extract_pages


# --------------------------------------------------------------------------- #
# NORMAL_QUERY — must not be misclassified as TOC or PAGE
# --------------------------------------------------------------------------- #
NORMAL_QUESTIONS = [
    "What are the 5 Smart Skills?",
    "Explain the Smart Skills program.",
    "What is Thinking and Problem Solving?",
    "Explain Experience Exercise.",
    "What is RAG?",
    "Explain embeddings.",
]


@pytest.mark.parametrize("question", NORMAL_QUESTIONS)
def test_normal_query(question):
    intent = detect_intent(question)
    assert intent.kind == NORMAL_QUERY
    assert intent.pages is None


# --------------------------------------------------------------------------- #
# TOC_QUERY
# --------------------------------------------------------------------------- #
TOC_QUESTIONS = [
    "Give me the table of contents.",
    "Show me the table of contents.",
    "Give me the TOC.",
    "Show TOC.",
    "What are the chapters in this document?",
    "List all the chapters.",
    "List all the sections.",
    "Show me the chapter structure.",
    "Give me the chapter list with page numbers.",
    "What topics are covered in this book?",
]


@pytest.mark.parametrize("question", TOC_QUESTIONS)
def test_toc_query(question):
    intent = detect_intent(question)
    assert intent.kind == TOC_QUERY, f"{question!r} misclassified as {intent.kind}"


# --------------------------------------------------------------------------- #
# PAGE_QUERY — single, range, list, and page+topic phrasing
# --------------------------------------------------------------------------- #
PAGE_CASES = [
    ("What does page 247 contain?", [247]),
    ("Summarize page 247.", [247]),
    ("What is discussed on page 170?", [170]),
    ("Explain the content on page 50.", [50]),
    ("What does page 170 say about RAG?", [170]),
    ("Tell me about pages 100 to 105.", [100, 101, 102, 103, 104, 105]),
    ("Summarize pages 10-15.", [10, 11, 12, 13, 14, 15]),
    ("Give me the contents of pages 100, 101 and 102.", [100, 101, 102]),
    ("Give me pages 10, 12 and 15.", [10, 12, 15]),
    ("pg 247", [247]),
    ("p. 247", [247]),
    ("page no 247", [247]),
    ("What is on page 9999?", [9999]),
]


@pytest.mark.parametrize("question,expected_pages", PAGE_CASES)
def test_page_query(question, expected_pages):
    intent = detect_intent(question)
    assert intent.kind == PAGE_QUERY, f"{question!r} misclassified as {intent.kind}"
    assert intent.pages == expected_pages


def test_page_number_takes_precedence_over_toc_wording():
    """"contents of pages 100, 101 and 102" mentions a TOC word ("contents")
    AND explicit page numbers — the page request must win (spec requirement:
    page-specific retrieval takes precedence over the generic TOC rule)."""
    intent = detect_intent("Give me the contents of pages 100, 101 and 102.")
    assert intent.kind == PAGE_QUERY
    assert intent.pages == [100, 101, 102]


def test_bare_number_is_not_a_page_reference():
    """A number with no page keyword must never trigger PAGE_QUERY."""
    assert extract_pages("What are the 5 Smart Skills?") is None
    assert detect_intent("What are the 5 Smart Skills?").kind == NORMAL_QUERY


def test_page_keyword_without_number_falls_through():
    """"chapter list with page numbers" has the word "page" but no digit —
    must not become a (bogus) PAGE_QUERY; falls through to TOC detection."""
    assert extract_pages("Give me the chapter list with page numbers.") is None
    assert detect_intent("Give me the chapter list with page numbers.").kind == TOC_QUERY
