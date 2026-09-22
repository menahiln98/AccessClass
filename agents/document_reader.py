"""
Stage 1: Document Reader.

Implementation note: this stage is deterministic (PyMuPDF extraction +
Tesseract OCR) and needs no LLM reasoning, so it is implemented as a plain
Python step rather than an LLM-backed CrewAI Agent. CrewAI Flows are
explicitly designed to mix deterministic steps with agentic ones — only
Stage 3 (Subject Interpreter) actually needs an Agent/Task/Crew, because
only it needs the LLM to reason about subject matter. Wrapping Stages 1
and 2 in LLM agents would add latency, cost, and hallucination risk for
work that is already fully solved by direct extraction/rule logic.
"""

import pymupdf

from models.document import ParsedDocument
from tools.pdf_tools import extract_parsed_document


class DocumentReadError(Exception):
    """Raised when a PDF cannot be read at all (missing, empty, or corrupted)."""


def run(pdf_path: str, filename: str) -> ParsedDocument:
    """
    Execute Stage 1 on a PDF file already saved to disk.

    Raises:
        DocumentReadError: for a missing, empty, corrupted, or otherwise
            unreadable PDF, with a message describing exactly what went wrong.
    """
    try:
        parsed = extract_parsed_document(pdf_path, filename)
    except pymupdf.FileNotFoundError as exc:
        raise DocumentReadError(f"File not found at '{pdf_path}'.") from exc
    except pymupdf.EmptyFileError as exc:
        raise DocumentReadError(f"'{filename}' is empty and contains no PDF data.") from exc
    except pymupdf.FileDataError as exc:
        raise DocumentReadError(
            f"'{filename}' could not be parsed as a PDF — it may be corrupted "
            "or not a valid PDF file."
        ) from exc

    if parsed.total_pages == 0:
        raise DocumentReadError(f"'{filename}' has no pages.")

    return parsed
