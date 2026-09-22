from models.document import Element, PageInfo, ParsedDocument
from tools.chunking import build_chapters, build_page_chunks


def _doc():
    pages = [
        PageInfo(page_number=1, has_selectable_text=True, is_scanned=False),
        PageInfo(page_number=2, has_selectable_text=True, is_scanned=False),
    ]
    elements = [
        Element(id="p1-e1", page_number=1, element_type="heading", text="Stacks"),
        Element(id="p1-e2", page_number=1, element_type="paragraph", text="A stack is LIFO."),
        Element(id="p2-e1", page_number=2, element_type="heading", text="Queues"),
        Element(id="p2-e2", page_number=2, element_type="paragraph", text="A queue is FIFO."),
    ]
    return ParsedDocument(filename="test.pdf", total_pages=2, pages=pages, elements=elements)


def test_build_chapters_splits_on_headings():
    chapters = build_chapters(_doc())
    assert len(chapters) == 2
    assert chapters[0]["title"] == "Stacks"
    assert "LIFO" in chapters[0]["text"]
    assert chapters[1]["title"] == "Queues"
    assert "FIFO" in chapters[1]["text"]


def test_build_chapters_handles_content_before_first_heading():
    pages = [PageInfo(page_number=1, has_selectable_text=True, is_scanned=False)]
    elements = [
        Element(id="p1-e1", page_number=1, element_type="paragraph", text="Preamble text."),
        Element(id="p1-e2", page_number=1, element_type="heading", text="Real Chapter"),
    ]
    doc = ParsedDocument(filename="test.pdf", total_pages=1, pages=pages, elements=elements)
    chapters = build_chapters(doc)
    assert chapters[0]["title"] == "Introduction"
    assert "Preamble" in chapters[0]["text"]
    assert chapters[1]["title"] == "Real Chapter"


def test_build_page_chunks_one_per_page_with_text():
    chunks = build_page_chunks("doc-1", _doc())
    assert len(chunks) == 2
    assert chunks[0].page_start == 1
    assert "Stacks" in chunks[0].text and "LIFO" in chunks[0].text
    assert chunks[0].chunk_id == "doc-1-p1"


def test_build_page_chunks_skips_pages_with_no_text():
    pages = [PageInfo(page_number=1, has_selectable_text=True, is_scanned=False)]
    elements = [Element(id="p1-e1", page_number=1, element_type="image", image_index=0)]
    doc = ParsedDocument(filename="test.pdf", total_pages=1, pages=pages, elements=elements)
    chunks = build_page_chunks("doc-1", doc)
    assert chunks == []
