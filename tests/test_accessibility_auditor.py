from agents.accessibility_auditor import run
from models.document import Element, PageInfo, ParsedDocument


def _doc(pages, elements):
    return ParsedDocument(filename="test.pdf", total_pages=len(pages), pages=pages, elements=elements)


def test_flags_scanned_page():
    pages = [PageInfo(page_number=1, has_selectable_text=False, is_scanned=True, ocr_used=True)]
    report = run(_doc(pages, []))
    assert any(i.issue_type == "scanned_page_no_text" for i in report.issues)


def test_flags_missing_headings():
    pages = [PageInfo(page_number=1, has_selectable_text=True, is_scanned=False)]
    elements = [Element(id="p1-e1", page_number=1, element_type="paragraph", text="Just a paragraph.")]
    report = run(_doc(pages, elements))
    assert any(i.issue_type == "missing_headings" for i in report.issues)


def test_no_missing_headings_when_heading_present():
    pages = [PageInfo(page_number=1, has_selectable_text=True, is_scanned=False)]
    elements = [
        Element(id="p1-e1", page_number=1, element_type="heading", text="Chapter 1"),
        Element(id="p1-e2", page_number=1, element_type="paragraph", text="Body text."),
    ]
    report = run(_doc(pages, elements))
    assert not any(i.issue_type == "missing_headings" for i in report.issues)


def test_flags_image_without_description():
    pages = [PageInfo(page_number=1, has_selectable_text=True, is_scanned=False, num_images=1)]
    elements = [Element(id="p1-e1", page_number=1, element_type="image", image_index=0)]
    report = run(_doc(pages, elements))
    issue = next(i for i in report.issues if i.issue_type == "image_no_description")
    assert issue.element_id == "p1-e1"


def test_flags_table_missing_header_row():
    pages = [PageInfo(page_number=1, has_selectable_text=True, is_scanned=False)]
    elements = [
        Element(
            id="p1-e1",
            page_number=1,
            element_type="table",
            table_rows=[["", "O(1)"], ["Pop", "O(1)"]],
        )
    ]
    report = run(_doc(pages, elements))
    assert any(i.issue_type == "table_no_headers" for i in report.issues)


def test_table_with_full_header_row_not_flagged():
    pages = [PageInfo(page_number=1, has_selectable_text=True, is_scanned=False)]
    elements = [
        Element(
            id="p1-e1",
            page_number=1,
            element_type="table",
            table_rows=[["Name", "Complexity"], ["Pop", "O(1)"]],
        )
    ]
    report = run(_doc(pages, elements))
    assert not any(i.issue_type == "table_no_headers" for i in report.issues)


def test_flags_code_without_language_hint():
    pages = [PageInfo(page_number=1, has_selectable_text=True, is_scanned=False)]
    elements = [Element(id="p1-e1", page_number=1, element_type="code", text="int x = 0;")]
    report = run(_doc(pages, elements))
    assert any(i.issue_type == "code_no_language_hint" for i in report.issues)


def test_flags_low_confidence_ocr():
    pages = [PageInfo(page_number=1, has_selectable_text=False, is_scanned=True, ocr_used=True)]
    elements = [
        Element(
            id="p1-e1",
            page_number=1,
            element_type="paragraph",
            text="blurry text",
            is_ocr=True,
            ocr_confidence=30.0,
        )
    ]
    report = run(_doc(pages, elements))
    assert any(i.issue_type == "low_confidence_ocr" for i in report.issues)


def test_high_confidence_ocr_not_flagged():
    pages = [PageInfo(page_number=1, has_selectable_text=False, is_scanned=True, ocr_used=True)]
    elements = [
        Element(
            id="p1-e1",
            page_number=1,
            element_type="paragraph",
            text="clear text",
            is_ocr=True,
            ocr_confidence=95.0,
        )
    ]
    report = run(_doc(pages, elements))
    assert not any(i.issue_type == "low_confidence_ocr" for i in report.issues)


def test_issue_counts_by_type_helper():
    pages = [PageInfo(page_number=1, has_selectable_text=True, is_scanned=False)]
    elements = [
        Element(id="p1-e0", page_number=1, element_type="heading", text="Chapter 1"),
        Element(id="p1-e1", page_number=1, element_type="code", text="x=1"),
        Element(id="p1-e2", page_number=1, element_type="code", text="y=2"),
    ]
    report = run(_doc(pages, elements))
    counts = report.issue_counts_by_type()
    assert counts.get("code_no_language_hint") == 2
    assert report.total_issues == 2
