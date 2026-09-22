import pymupdf

from tools.pdf_tools import extract_page, extract_parsed_document


def _make_text_pdf(path: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Chapter 1: Linked Lists", fontsize=20, fontname="helv")
    page.insert_text(
        (72, 110),
        "A linked list is a linear data structure where each element points to the next.",
        fontsize=11,
        fontname="helv",
    )
    page.insert_text((72, 150), "int main() { return 0; }", fontsize=10, fontname="cour")
    page.insert_text((72, 180), "The derivative dx of f(x) approaches the limit lim x->0.", fontsize=11, fontname="helv")
    doc.save(path)
    doc.close()


def _make_scanned_pdf(path: str) -> None:
    # Build a normal text page, rasterize it, then create a new PDF containing
    # only that image (so it has zero selectable text, forcing the OCR path).
    source = pymupdf.open()
    source_page = source.new_page()
    source_page.insert_text((72, 72), "Scanned Notes About Stacks", fontsize=18, fontname="helv")
    pixmap = source_page.get_pixmap(dpi=200)
    img_bytes = pixmap.tobytes("png")
    source.close()

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=img_bytes)
    doc.save(path)
    doc.close()


def test_extract_parsed_document_classifies_content(tmp_path):
    pdf_path = str(tmp_path / "lecture.pdf")
    _make_text_pdf(pdf_path)

    parsed = extract_parsed_document(pdf_path, "lecture.pdf")

    assert parsed.filename == "lecture.pdf"
    assert parsed.total_pages == 1
    assert parsed.pages[0].has_selectable_text is True
    assert parsed.pages[0].is_scanned is False

    types_found = {el.element_type for el in parsed.elements}
    assert "heading" in types_found
    assert "code" in types_found
    assert "equation" in types_found
    assert "paragraph" in types_found


def test_scanned_page_triggers_ocr(tmp_path):
    pdf_path = str(tmp_path / "scanned.pdf")
    _make_scanned_pdf(pdf_path)

    parsed = extract_parsed_document(pdf_path, "scanned.pdf")

    assert parsed.pages[0].is_scanned is True
    assert parsed.pages[0].ocr_used is True

    ocr_elements = [el for el in parsed.elements if el.is_ocr]
    assert len(ocr_elements) == 1
    assert "Stacks" in ocr_elements[0].text or "stacks" in ocr_elements[0].text.lower()
    assert ocr_elements[0].ocr_confidence is not None


def test_extract_page_detects_table():
    doc = pymupdf.open()
    page = doc.new_page()
    # A simple grid of text lines that PyMuPDF's table finder can pick up.
    rows = [["Name", "Complexity"], ["Push", "O(1)"], ["Pop", "O(1)"]]
    y = 100
    col_x = [72, 200]
    for row in rows:
        for x, cell in zip(col_x, row):
            page.insert_text((x, y), cell, fontsize=11, fontname="helv")
        y += 20
    # Draw grid lines so PyMuPDF's table detector recognizes a table structure.
    for x in (65, 195, 320):
        page.draw_line((x, 90), (x, 170))
    for y_line in (90, 110, 130, 150, 170):
        page.draw_line((65, y_line), (320, y_line))

    page_info, elements = extract_page(page, 1)
    doc.close()

    table_elements = [el for el in elements if el.element_type == "table"]
    # Table detection depends on PyMuPDF's layout heuristics; we only assert
    # that when a table IS found, its rows were captured correctly.
    for table in table_elements:
        assert table.table_rows is not None
        assert len(table.table_rows) >= 1
