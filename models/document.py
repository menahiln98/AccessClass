"""
Pydantic models describing the structured output of Stage 1: Document Reader.

These models are the contract between the Document Reader and every stage
that reads its output (Accessibility Auditor, Subject Interpreter, and
later the Explanation/Study-Pack/Grounded Learning agents in Portion 2).
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

ElementType = Literal["heading", "paragraph", "code", "table", "equation", "image"]


class Element(BaseModel):
    """A single extracted piece of content from one page of the PDF."""

    id: str = Field(..., description="Short unique id, e.g. 'p1-e3'")
    page_number: int = Field(..., ge=1)
    element_type: ElementType

    text: str = Field(
        default="",
        description="Extracted text content. Empty string for image elements.",
    )

    font_name: Optional[str] = Field(
        default=None, description="Dominant font name for this element, if text-based."
    )
    font_size: Optional[float] = Field(
        default=None, description="Dominant font size for this element, if text-based."
    )
    bbox: Optional[list[float]] = Field(
        default=None, description="[x0, y0, x1, y1] bounding box on the page."
    )

    is_ocr: bool = Field(
        default=False, description="True if this element's text came from OCR, not the PDF text layer."
    )
    ocr_confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Tesseract mean confidence (0-100) for this element, only set when is_ocr is True.",
    )

    table_rows: Optional[list[list[str]]] = Field(
        default=None,
        description="Row-major cell text, only set when element_type == 'table'.",
    )

    image_index: Optional[int] = Field(
        default=None,
        description="Index of this image within the page's image list, only set when element_type == 'image'.",
    )


class PageInfo(BaseModel):
    """Per-page metadata used by both the reader and the accessibility auditor."""

    page_number: int = Field(..., ge=1)
    has_selectable_text: bool
    is_scanned: bool = Field(
        description="True if the page had no usable selectable text and OCR had to be used."
    )
    ocr_used: bool = False
    num_images: int = 0


class ParsedDocument(BaseModel):
    """Full structured output of the Document Reader for one uploaded PDF."""

    filename: str
    total_pages: int = Field(..., ge=0)
    pages: list[PageInfo] = Field(default_factory=list)
    elements: list[Element] = Field(default_factory=list)

    def elements_on_page(self, page_number: int) -> list[Element]:
        return [e for e in self.elements if e.page_number == page_number]
