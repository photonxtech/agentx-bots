"""Unit tests for ingestion.chunk_type and the repeated header/footer
suppression helpers (_repeated_boilerplate_lines / _strip_boilerplate_lines).

The PyMuPDF-driven detection pass (_repeated_boilerplate_lines takes a real
`fitz` document) isn't exercised here — these test the pure-text logic
(_strip_boilerplate_lines) and the chunk_type helper directly, which is
where the actual decision/behavior lives.
"""

from rag.ingestion import HEADING_MARK, _strip_boilerplate_lines, chunk_type


def test_chunk_type_content_by_default():
    assert chunk_type({}) == "content"
    assert chunk_type({"is_index": False}) == "content"


def test_chunk_type_toc_for_is_index():
    assert chunk_type({"is_index": True}) == "toc"


def test_chunk_type_toc_for_is_toc_alias():
    assert chunk_type({"is_toc": True}) == "toc"


def test_strip_boilerplate_lines_removes_matching_lines():
    text = "Real content line one.\nThe 5 Essential Smart Skills for Success\nReal content line two."
    boilerplate = {"the 5 essential smart skills for success"}
    result = _strip_boilerplate_lines(text, boilerplate)
    assert "Essential Smart Skills" not in result
    assert "Real content line one." in result
    assert "Real content line two." in result


def test_strip_boilerplate_lines_no_boilerplate_is_noop():
    text = "Line one.\nLine two."
    assert _strip_boilerplate_lines(text, set()) == text


def test_strip_boilerplate_lines_normalizes_whitespace_and_case():
    text = "TITLE   BANNER\nReal content."
    boilerplate = {"title banner"}
    result = _strip_boilerplate_lines(text, boilerplate)
    assert "TITLE" not in result
    assert "Real content." in result


def test_strip_boilerplate_lines_ignores_heading_mark_when_comparing():
    """A font-size-detected heading that happens to BE a recurring banner
    must still be caught — the HEADING_MARK sentinel prefix is stripped
    before comparing against the boilerplate set."""
    text = f"{HEADING_MARK}Recurring Banner Title\nReal content line."
    boilerplate = {"recurring banner title"}
    result = _strip_boilerplate_lines(text, boilerplate)
    assert "Recurring Banner Title" not in result
    assert "Real content line." in result


def test_strip_boilerplate_lines_preserves_non_matching_headings():
    text = f"{HEADING_MARK}A Real Section Heading\nReal content line."
    boilerplate = {"some other banner"}
    result = _strip_boilerplate_lines(text, boilerplate)
    assert "A Real Section Heading" in result
