"""
PDF structure extraction for Stage 1 (Document Reader).

Uses PyMuPDF (imported as `pymupdf` — the `fitz` name is the deprecated
alias) to pull text, headings, code, tables, equations, and images out of
each page, falling back to Tesseract OCR (tools/ocr_tools.py) for pages
with no selectable text layer.

Classification is heuristic (font size for headings, font family for code,
a symbol-based check for equations). This is a real limitation worth being
explicit about: it will misclassify some content, especially equations
rendered as plain text with no distinctive symbols. Stage 3 (Subject
Interpreter) refines subject/content-type on top of this using the LLM, and
Stage 4 (Portion 2) verifies visual content with Gemini. Stage 1's job is a
reasonable first pass, not a perfect one.
"""

import re
import statistics

import pymupdf

from models.document import Element, PageInfo, ParsedDocument
from tools.ocr_tools import ocr_page

MIN_SELECTABLE_CHARS = 20
HEADING_SIZE_RATIO = 1.25  # heading if span size >= body size * this ratio
CODE_FONT_HINTS = ("courier", "mono", "consolas")
MATH_SYMBOLS = set("∫∑√±≤≥≠∞πθαβγδΔ∂∈∀∃∇×÷")
MATH_KEYWORDS = ("lim", "dx", "dy", "d/dx", "sqrt", "integral", "derivative")


def has_selectable_text(page: "pymupdf.Page") -> bool:
    return len(page.get_text().strip()) >= MIN_SELECTABLE_CHARS


def _looks_like_equation(text: str) -> bool:
    if any(ch in MATH_SYMBOLS for ch in text):
        return True
    lowered = text.lower()
    return any(keyword in lowered for keyword in MATH_KEYWORDS)


def _looks_like_code(font_name: str) -> bool:
    lowered = font_name.lower()
    return any(hint in lowered for hint in CODE_FONT_HINTS)


def _classify_line(text: str, font_name: str, font_size: float, body_size: float) -> str:
    if _looks_like_equation(text):
        return "equation"
    if _looks_like_code(font_name):
        return "code"
    if font_size >= body_size * HEADING_SIZE_RATIO and len(text) <= 120:
        return "heading"
    return "paragraph"


def _dominant_body_size(spans: list[dict]) -> float:
    sizes = [round(s["size"], 1) for s in spans if s["text"].strip()]
    if not sizes:
        return 11.0
    try:
        return statistics.mode(sizes)
    except statistics.StatisticsError:
        return statistics.median(sizes)


def _bbox_center_inside(bbox: list[float], table_rect: "pymupdf.Rect") -> bool:
    x0, y0, x1, y1 = bbox
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    return table_rect.contains(pymupdf.Point(center_x, center_y))


def _extract_tables(page: "pymupdf.Page") -> list[tuple["pymupdf.Rect", list[list[str]]]]:
    tables = []
    try:
        found = page.find_tables()
    except Exception:
        # Table detection is best-effort; a malformed page should not crash
        # the whole pipeline over one page's table layout.
        return tables
    for table in found:
        try:
            rows = table.extract()
        except Exception:
            continue
        cleaned_rows = [[(cell or "").strip() for cell in row] for row in rows]
        tables.append((pymupdf.Rect(table.bbox), cleaned_rows))
    return tables


def _extract_text_elements(
    page: "pymupdf.Page", page_number: int, table_rects: list["pymupdf.Rect"]
) -> list[Element]:
    elements: list[Element] = []
    page_dict = page.get_text("dict")

    all_spans = [
        span
        for block in page_dict["blocks"]
        if "lines" in block
        for line in block["lines"]
        for span in line["spans"]
    ]
    body_size = _dominant_body_size(all_spans)

    counter = 0
    for block in page_dict["blocks"]:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            spans = line["spans"]
            if not spans:
                continue
            line_text = "".join(span["text"] for span in spans).strip()
            if not line_text:
                continue

            bbox = list(line["bbox"])
            if any(_bbox_center_inside(bbox, rect) for rect in table_rects):
                continue  # already captured as part of a table element

            representative = max(spans, key=lambda s: len(s["text"]))
            element_type = _classify_line(
                line_text, representative["font"], representative["size"], body_size
            )

            counter += 1
            elements.append(
                Element(
                    id=f"p{page_number}-e{counter}",
                    page_number=page_number,
                    element_type=element_type,
                    text=line_text,
                    font_name=representative["font"],
                    font_size=round(representative["size"], 1),
                    bbox=bbox,
                    is_ocr=False,
                )
            )
    return elements


def _extract_table_elements(
    page_number: int, tables: list[tuple["pymupdf.Rect", list[list[str]]]], start_index: int
) -> list[Element]:
    elements = []
    for offset, (rect, rows) in enumerate(tables, start=start_index):
        elements.append(
            Element(
                id=f"p{page_number}-e{offset}",
                page_number=page_number,
                element_type="table",
                text="",
                bbox=[rect.x0, rect.y0, rect.x1, rect.y1],
                table_rows=rows,
            )
        )
    return elements


def _extract_image_elements(page: "pymupdf.Page", page_number: int, start_index: int) -> list[Element]:
    elements = []
    images = page.get_images(full=True)
    for offset, image_info in enumerate(images, start=start_index):
        xref = image_info[0]
        bbox = None
        try:
            rects = page.get_image_rects(xref)
            if rects:
                bbox = [rects[0].x0, rects[0].y0, rects[0].x1, rects[0].y1]
        except Exception:
            bbox = None
        elements.append(
            Element(
                id=f"p{page_number}-e{offset}",
                page_number=page_number,
                element_type="image",
                text="",
                bbox=bbox,
                image_index=offset - start_index,
            )
        )
    return elements


def extract_page(page: "pymupdf.Page", page_number: int) -> tuple[PageInfo, list[Element]]:
    """Extract structured elements from a single page."""
    selectable = has_selectable_text(page)
    images = page.get_images(full=True)

    if not selectable:
        ocr_text, ocr_confidence = ocr_page(page)
        elements: list[Element] = []
        if ocr_text.strip():
            elements.append(
                Element(
                    id=f"p{page_number}-e1",
                    page_number=page_number,
                    element_type="paragraph",
                    text=ocr_text,
                    is_ocr=True,
                    ocr_confidence=round(ocr_confidence, 1),
                )
            )
        elements.extend(_extract_image_elements(page, page_number, start_index=len(elements) + 1))
        page_info = PageInfo(
            page_number=page_number,
            has_selectable_text=False,
            is_scanned=True,
            ocr_used=True,
            num_images=len(images),
        )
        return page_info, elements

    tables = _extract_tables(page)
    table_rects = [rect for rect, _ in tables]

    text_elements = _extract_text_elements(page, page_number, table_rects)
    table_elements = _extract_table_elements(page_number, tables, start_index=len(text_elements) + 1)
    image_elements = _extract_image_elements(
        page, page_number, start_index=len(text_elements) + len(table_elements) + 1
    )

    page_info = PageInfo(
        page_number=page_number,
        has_selectable_text=True,
        is_scanned=False,
        ocr_used=False,
        num_images=len(images),
    )
    return page_info, text_elements + table_elements + image_elements


def extract_parsed_document(pdf_path: str, filename: str) -> ParsedDocument:
    """Run Stage 1 extraction over an entire PDF file on disk."""
    doc = pymupdf.open(pdf_path)
    try:
        pages: list[PageInfo] = []
        elements: list[Element] = []
        for index in range(doc.page_count):
            page = doc[index]
            page_number = index + 1
            page_info, page_elements = extract_page(page, page_number)
            pages.append(page_info)
            elements.extend(page_elements)
        return ParsedDocument(
            filename=filename,
            total_pages=doc.page_count,
            pages=pages,
            elements=elements,
        )
    finally:
        doc.close()
