"""Unit tests for ingestion._looks_like_toc — the existing, already-validated
heuristic. Not redesigned per the task's instructions; these tests pin down
the behavior (true positive on real TOC shapes, true negative on ordinary
numbered-list content) so a future change can't silently regress it."""

from rag.ingestion import _looks_like_toc


def test_actual_toc_page_is_detected():
    """Dot-leader entries + bare trailing page numbers (how PyMuPDF commonly
    splits a TOC line's title and page number onto separate lines)."""
    text = "\n".join([
        "Table of Contents",
        "1 Introduction",
        "3",
        "2 Getting Started",
        "7",
        "3 Core Concepts",
        "12",
        "4.1 Overview",
        "15",
        "4.2 Details",
        "18",
        "4.4 Experience Exercise: Worst and Best",
        "47",
        "5 Conclusion",
        "50",
    ])
    assert _looks_like_toc(text) is True


def test_multi_page_toc_continuation_page_is_detected():
    """A second/third TOC page (e.g. pages 6-7 of a multi-page TOC) is
    mostly more of the same entry shapes — no title line needed."""
    text = "\n".join([
        "6.1 Advanced Topics",
        "55",
        "6.2 Case Studies",
        "60",
        "6.3 Summary",
        "65",
        "7 Appendix A",
        "70",
        "7 Appendix B",
        "75",
        "8 Index",
        "80",
    ])
    assert _looks_like_toc(text) is True


def test_ocr_toc_text_is_detected():
    """OCR output is noisier (stray spaces, occasional misread chars) but the
    line shapes survive — the heuristic runs on whatever text ingestion
    produces, OCR or text-layer, with no special-casing needed."""
    text = "\n".join([
        "Table  of  Contents",
        "1  Introduction",
        "4",
        "2  Smart  Skills  Overview",
        "9",
        "3  Thinking  and  Problem  Solving",
        "14",
        "4  Experience  Exercises",
        "20",
    ])
    assert _looks_like_toc(text) is True


def test_toc_entry_with_no_space_after_dotted_number_is_detected():
    """Real-world regression: some PDFs' text extraction drops the space
    between a dotted sub-level number and its title entirely (a font/kerning
    artifact), producing "4.4Need to Know: ..." instead of "4.4 Need to
    Know: ...". The original regex required that space, so a genuine TOC
    page like this scored under the 0.5 ratio and leaked into normal
    retrieval undetected."""
    text = "\n".join([
        "1.",
        "Thinking and Problem Solving",
        "© Bob Wiele, OneSmartWorld",
        "Lesson 4: Problem-Solving Skills",
        "4.1Introduction",
        "35",
        "4.2Experience Exercise: Problems, Problems Everywhere",
        "36",
        "4.3 Apply @ Work: Two Significant Problems",
        "37",
        "4.4Need to Know: Three Typical Problem Solving Mistakes",
        "39",
        "4.5Practice Skill: Hindsight is 20/20",
        "40",
        "4.6Need to Know: 3 Key Guidelines",
        "41",
        "4.7Practice Skill: How to Select Outcomes",
        "42",
    ])
    assert _looks_like_toc(text) is True


def test_dotted_entry_without_space_still_excludes_ordinary_numbered_list():
    """The relaxed no-space rule is scoped to DOTTED sub-level numbers
    directly followed by a capital letter — an ordinary numbered list
    ("1. Introduction to the topic...", period+space, no sub-level dot)
    must still not be mistaken for a TOC entry."""
    text = "\n".join([
        "Our approach follows these principles:",
        "1. Introduction to the topic and its context in the broader field.",
        "2. Background research summarizing prior work and existing literature.",
        "3. Methodology describing the experimental design and data collection.",
        "4. Results presenting the key findings and their statistical significance.",
        "This section elaborates further on each numbered point above.",
        "The numbered list itself does not indicate a table of contents.",
    ])
    assert _looks_like_toc(text) is False


def test_numbered_content_page_is_not_toc():
    """Real content with a numbered list (periods after the number, as in
    normal prose numbering) must NOT be classified as TOC even though every
    item starts with a digit."""
    text = "\n".join([
        "Our approach follows these principles:",
        "1. Introduction to the topic and its context in the broader field.",
        "2. Background research summarizing prior work and existing literature.",
        "3. Methodology describing the experimental design and data collection.",
        "4. Results presenting the key findings and their statistical significance.",
        "This section elaborates further on each numbered point above.",
        "The numbered list itself does not indicate a table of contents.",
    ])
    assert _looks_like_toc(text) is False


def test_short_page_is_never_toc():
    """Fewer than 6 lines never counts, regardless of content — a short page
    of nothing but page numbers shouldn't be flagged off tiny samples."""
    text = "1 Introduction\n3\n2 Getting Started\n7"
    assert _looks_like_toc(text) is False


def test_empty_and_whitespace_text():
    assert _looks_like_toc("") is False
    assert _looks_like_toc("   \n\n   ") is False
