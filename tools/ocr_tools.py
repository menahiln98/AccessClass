"""
OCR helpers for Stage 1 (Document Reader).

Used only for pages where PyMuPDF finds no usable selectable text (i.e.
scanned / image-only pages), per the proposal's Document Reader spec.
"""

import io
import os

import pymupdf
import pytesseract
from PIL import Image
from pytesseract import Output

# Only override pytesseract's binary path if the environment says where it
# is (e.g. a non-standard Windows install). Left unset, pytesseract looks
# up "tesseract" on PATH, which is where `apt-get install tesseract-ocr`
# (Dockerfile, GitHub Actions runner) and most package managers put it —
# hardcoding a Windows path here previously broke OCR on every Linux
# deployment silently (pytesseract would just fail to find the binary).
_tesseract_cmd = os.environ.get("TESSERACT_CMD")
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd

OCR_RENDER_DPI = 300


def render_page_to_image(page: "pymupdf.Page", dpi: int = OCR_RENDER_DPI) -> Image.Image:
    """Rasterize a PDF page to a PIL image for OCR."""
    pixmap = page.get_pixmap(dpi=dpi)
    png_bytes = pixmap.tobytes("png")
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def ocr_image(image: Image.Image) -> tuple[str, float]:
    """
    Run Tesseract OCR on an image.

    Returns:
        (extracted_text, mean_confidence) where mean_confidence is 0-100,
        averaged only over words Tesseract actually returned a confidence for.
    """
    data = pytesseract.image_to_data(image, output_type=Output.DICT)

    words: list[str] = []
    confidences: list[float] = []

    for text, conf in zip(data["text"], data["conf"]):
        cleaned = text.strip()
        if not cleaned:
            continue
        words.append(cleaned)
        try:
            conf_value = float(conf)
        except (TypeError, ValueError):
            continue
        if conf_value >= 0:
            confidences.append(conf_value)

    full_text = " ".join(words)
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return full_text, mean_confidence


def ocr_page(page: "pymupdf.Page") -> tuple[str, float]:
    """Convenience wrapper: render a page then OCR it."""
    image = render_page_to_image(page)
    return ocr_image(image)
