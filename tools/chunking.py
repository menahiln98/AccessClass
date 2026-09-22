"""
Two grouping strategies, each serving a different Portion 2 stage:

- build_chapters(): groups elements by heading boundaries, for Stage 5's
  chapter-based audio (so a student can jump to "Linked Lists" or
  "Derivative Rules" instead of one long recording).
- build_page_chunks(): groups elements by page, for Stage 6's retrieval —
  one chunk per page keeps every citation trivially page-accurate.
"""

from models.document import ParsedDocument
from models.qa import LectureChunk


def build_chapters(parsed: ParsedDocument) -> list[dict]:
    """Returns a list of {title, page_start, page_end, text} dicts, in document order."""
    chapters: list[dict] = []
    current: dict | None = None

    for element in parsed.elements:
        if element.element_type == "heading":
            if current is not None:
                chapters.append(current)
            current = {
                "title": element.text.strip() or f"Page {element.page_number}",
                "page_start": element.page_number,
                "page_end": element.page_number,
                "text_parts": [element.text],
            }
            continue

        if current is None:
            # Content appears before the first heading in the document.
            current = {
                "title": "Introduction",
                "page_start": element.page_number,
                "page_end": element.page_number,
                "text_parts": [],
            }

        if element.text.strip():
            current["text_parts"].append(element.text.strip())
        current["page_end"] = max(current["page_end"], element.page_number)

    if current is not None:
        chapters.append(current)

    return [
        {
            "title": ch["title"],
            "page_start": ch["page_start"],
            "page_end": ch["page_end"],
            "text": "\n".join(ch["text_parts"]).strip(),
        }
        for ch in chapters
        if ch["text_parts"] or ch["title"] != "Introduction"
    ]


def build_page_chunks(document_id: str, parsed: ParsedDocument) -> list[LectureChunk]:
    """One chunk per page that has any text content."""
    chunks: list[LectureChunk] = []
    for page in parsed.pages:
        page_text = "\n".join(
            el.text.strip()
            for el in parsed.elements_on_page(page.page_number)
            if el.text.strip()
        )
        if not page_text:
            continue
        chunks.append(
            LectureChunk(
                chunk_id=f"{document_id}-p{page.page_number}",
                document_id=document_id,
                page_start=page.page_number,
                page_end=page.page_number,
                text=page_text,
            )
        )
    return chunks
