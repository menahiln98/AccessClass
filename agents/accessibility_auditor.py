"""
Stage 2: Accessibility Auditor.

Like Stage 1, this is deterministic rule-checking over already-extracted
structure, so it runs as a plain Python step rather than an LLM agent.

See models/audit.py for the note on why equation-confidence auditing is
intentionally out of scope for Portion 1.
"""

from models.audit import AccessibilityIssue, AccessibilityReport
from models.document import ParsedDocument

LOW_OCR_CONFIDENCE_THRESHOLD = 60.0


def run(parsed: ParsedDocument) -> AccessibilityReport:
    issues: list[AccessibilityIssue] = []

    for page in parsed.pages:
        if page.is_scanned:
            issues.append(
                AccessibilityIssue(
                    issue_type="scanned_page_no_text",
                    page_number=page.page_number,
                    description=(
                        "This page had no selectable text and was recovered via OCR. "
                        "OCR text can contain errors — a screen reader user relies on "
                        "its accuracy for this page."
                    ),
                    severity="high",
                )
            )

    has_any_heading = any(element.element_type == "heading" for element in parsed.elements)
    if not has_any_heading and parsed.total_pages > 0:
        issues.append(
            AccessibilityIssue(
                issue_type="missing_headings",
                page_number=1,
                description=(
                    "No heading structure was detected anywhere in this document. "
                    "Screen-reader users typically navigate long documents by heading, "
                    "so this makes the document harder to navigate."
                ),
                severity="medium",
            )
        )

    for element in parsed.elements:
        if element.element_type == "image":
            issues.append(
                AccessibilityIssue(
                    issue_type="image_no_description",
                    page_number=element.page_number,
                    element_id=element.id,
                    description="Image detected with no textual description yet.",
                    severity="medium",
                )
            )

        elif element.element_type == "table":
            header_row = element.table_rows[0] if element.table_rows else []
            missing_header = (not header_row) or any(cell.strip() == "" for cell in header_row)
            if missing_header:
                issues.append(
                    AccessibilityIssue(
                        issue_type="table_no_headers",
                        page_number=element.page_number,
                        element_id=element.id,
                        description="Table detected with missing or incomplete header row.",
                        severity="medium",
                    )
                )

        elif element.element_type == "code":
            issues.append(
                AccessibilityIssue(
                    issue_type="code_no_language_hint",
                    page_number=element.page_number,
                    element_id=element.id,
                    description="Code block detected with no programming-language label attached.",
                    severity="low",
                )
            )

        if element.is_ocr and element.ocr_confidence is not None:
            if element.ocr_confidence < LOW_OCR_CONFIDENCE_THRESHOLD:
                issues.append(
                    AccessibilityIssue(
                        issue_type="low_confidence_ocr",
                        page_number=element.page_number,
                        element_id=element.id,
                        description=(
                            f"OCR confidence was low ({element.ocr_confidence:.1f}/100) for "
                            "this element — the extracted text may be inaccurate."
                        ),
                        severity="high",
                    )
                )

    return AccessibilityReport(
        document_filename=parsed.filename,
        total_pages=parsed.total_pages,
        issues=issues,
    )
