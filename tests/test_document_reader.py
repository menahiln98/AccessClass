import pytest

from agents.document_reader import DocumentReadError, run


def test_missing_file_raises_document_read_error():
    with pytest.raises(DocumentReadError, match="not found"):
        run("/tmp/does-not-exist-accessclass.pdf", "does-not-exist.pdf")


def test_empty_file_raises_document_read_error(tmp_path):
    empty_path = tmp_path / "empty.pdf"
    empty_path.write_bytes(b"")
    with pytest.raises(DocumentReadError, match="empty"):
        run(str(empty_path), "empty.pdf")


def test_corrupted_file_raises_document_read_error(tmp_path):
    bad_path = tmp_path / "corrupted.pdf"
    bad_path.write_bytes(b"this is not a real pdf file at all")
    with pytest.raises(DocumentReadError, match="could not be parsed"):
        run(str(bad_path), "corrupted.pdf")


def test_valid_pdf_returns_parsed_document(tmp_path):
    import pymupdf

    pdf_path = tmp_path / "valid.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello AccessClass", fontsize=12, fontname="helv")
    doc.save(str(pdf_path))
    doc.close()

    parsed = run(str(pdf_path), "valid.pdf")
    assert parsed.total_pages == 1
    assert len(parsed.elements) >= 1
