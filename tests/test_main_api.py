import os
from unittest.mock import MagicMock, patch

import pymupdf
import pytest
from fastapi.testclient import TestClient

# Portion 1 requires these env vars to exist at app startup (config.validate_portion1_env).
# Tests never make real network calls with them — every Groq/Supabase call is mocked below.
os.environ.setdefault("GROQ_API_KEY", "dummy-test-key")
os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "dummy-test-key")
os.environ.setdefault("GEMINI_API_KEY", "dummy-test-key")
os.environ.setdefault("QDRANT_URL", "https://example-qdrant.io")
os.environ.setdefault("QDRANT_API_KEY", "dummy-test-key")

from main import app  # noqa: E402  (must come after the env vars are set)
from models.explanation import ElementExplanation, ExplanationDocument  # noqa: E402
from models.study_pack import StudyPack  # noqa: E402
from models.subject import SubjectTag, SubjectTagBatch  # noqa: E402


def _make_pdf(path: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Chapter 3: Derivatives", fontsize=20, fontname="helv")
    page.insert_text((72, 110), "The derivative measures rate of change.", fontsize=11, fontname="helv")
    doc.save(path)
    doc.close()


@pytest.fixture
def client():
    return TestClient(app)


def test_index_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "AccessClass" in response.text
    assert "Process Lecture" in response.text


def test_upload_rejects_non_pdf(client):
    response = client.post(
        "/upload", files={"file": ("notes.txt", b"just some text", "text/plain")}
    )
    assert response.status_code == 200
    assert "Please upload a .pdf file" in response.text


def test_upload_rejects_empty_pdf(client):
    response = client.post("/upload", files={"file": ("empty.pdf", b"", "application/pdf")})
    assert response.status_code == 200
    assert "empty" in response.text.lower()


@patch("main._study_pack_links", return_value=None)
@patch("flow.grounded_learning_agent.index_document", return_value=0)
@patch("flow.study_pack_agent.run")
@patch("flow.explanation_agent.run")
@patch("agents.subject_interpreter.get_groq_api_key", return_value="dummy-test-key")
@patch("agents.subject_interpreter.Crew")
@patch("flow.update_document")
@patch("flow.create_document_record", return_value="fake-doc-id")
@patch("flow.upload_pdf", return_value="fake-path.pdf")
def test_upload_valid_pdf_renders_results(
    _mock_upload,
    _mock_create,
    _mock_update,
    mock_crew_cls,
    _mock_key,
    mock_explain,
    mock_study_pack,
    _mock_index,
    _mock_links,
    client,
    tmp_path,
):
    mock_crew_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.pydantic = SubjectTagBatch(
        tags=[SubjectTag(element_id="p1-e1", subject="calculus", content_type="text", confidence=0.9)]
    )
    mock_crew_instance.kickoff.return_value = mock_result
    mock_crew_cls.return_value = mock_crew_instance

    mock_explain.return_value = ExplanationDocument(document_filename="derivatives.pdf", explanations=[])
    mock_study_pack.return_value = StudyPack(
        document_filename="derivatives.pdf",
        html_storage_path="fake-doc-id/study_pack.html",
        audio_chapters=[],
        glossary=[],
    )

    pdf_path = tmp_path / "derivatives.pdf"
    _make_pdf(str(pdf_path))

    with open(pdf_path, "rb") as f:
        response = client.post(
            "/upload", files={"file": ("derivatives.pdf", f.read(), "application/pdf")}
        )

    assert response.status_code == 200
    assert "fake-doc-id" in response.text
    assert "Derivatives" in response.text or "derivative" in response.text.lower()
    assert "Subject Interpreter" in response.text


@patch("main.get_document")
def test_view_document_not_found(mock_get_document, client):
    mock_get_document.return_value = None
    response = client.get("/documents/nonexistent-id")
    assert response.status_code == 200
    assert "No document found" in response.text


@patch("main.get_document")
def test_view_document_renders_stored_record(mock_get_document, client):
    # Simulates data reloaded from Supabase: plain dicts/JSON, not Pydantic objects.
    mock_get_document.return_value = {
        "id": "stored-doc-id",
        "parsed_document": {
            "filename": "old.pdf",
            "total_pages": 1,
            "pages": [
                {"page_number": 1, "has_selectable_text": True, "is_scanned": False, "ocr_used": False, "num_images": 0}
            ],
            "elements": [
                {
                    "id": "p1-e1",
                    "page_number": 1,
                    "element_type": "heading",
                    "text": "Old Chapter",
                    "is_ocr": False,
                }
            ],
        },
        "accessibility_report": {"document_filename": "old.pdf", "total_pages": 1, "issues": []},
        "subject_tagged_document": {"document_filename": "old.pdf", "tags": []},
    }

    response = client.get("/documents/stored-doc-id")
    assert response.status_code == 200
    assert "stored-doc-id" in response.text
    assert "Old Chapter" in response.text
    assert "reloaded from Supabase" in response.text


def test_ask_form_loads(client):
    response = client.get("/documents/doc-1/ask")
    assert response.status_code == 200
    assert "Ask This Lecture" in response.text


@patch("main.ask")
def test_ask_question_renders_answer(mock_ask, client):
    from models.qa import AskAnswer, Citation

    mock_ask.return_value = AskAnswer(
        question="What is a stack?",
        answer="A stack is a LIFO data structure.",
        is_supported=True,
        citations=[Citation(page_number=3, snippet="A stack is LIFO.")],
    )
    response = client.post("/documents/doc-1/ask", data={"question": "What is a stack?"})
    assert response.status_code == 200
    assert "A stack is a LIFO data structure." in response.text
    assert "Page 3" in response.text


@patch("main.ask")
def test_ask_question_handles_grounded_learning_error(mock_ask, client):
    from agents.grounded_learning_agent import GroundedLearningError

    mock_ask.side_effect = GroundedLearningError("Retrieval failed for document 'doc-1'")
    response = client.post("/documents/doc-1/ask", data={"question": "What is a stack?"})
    assert response.status_code == 200
    assert "Retrieval failed" in response.text


@patch("main.list_revision_queue")
@patch("main.add_revision_entry", return_value="entry-1")
def test_mark_confusing_adds_entry(mock_add, mock_list, client):
    mock_list.return_value = [{"page_number": 5, "note": "Confusing diagram", "created_at": "2026-01-01"}]
    response = client.post(
        "/documents/doc-1/mark-confusing", data={"page_number": 5, "note": "Confusing diagram"}
    )
    assert response.status_code == 200
    assert "Confusing diagram" in response.text
    mock_add.assert_called_once_with("doc-1", 5, "Confusing diagram")
