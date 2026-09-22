from unittest.mock import MagicMock, patch

import pymupdf

from flow import PipelineError, run_pipeline
from models.explanation import ElementExplanation, ExplanationDocument
from models.study_pack import StudyPack
from models.subject import SubjectTag, SubjectTagBatch


def _make_pdf(path: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Chapter 2: Stacks and Queues", fontsize=20, fontname="helv")
    page.insert_text(
        (72, 110),
        "A stack follows Last-In-First-Out ordering.",
        fontsize=11,
        fontname="helv",
    )
    doc.save(path)
    doc.close()


@patch("flow.grounded_learning_agent.index_document", return_value=1)
@patch("flow.study_pack_agent.run")
@patch("flow.explanation_agent.run")
@patch("agents.subject_interpreter.get_groq_api_key", return_value="dummy-test-key")
@patch("agents.subject_interpreter.Crew")
@patch("flow.update_document")
@patch("flow.create_document_record", return_value="fake-document-id")
@patch("flow.upload_pdf", return_value="fake-storage-path.pdf")
def test_full_pipeline_success(
    _mock_upload,
    _mock_create,
    mock_update,
    mock_crew_cls,
    _mock_key,
    mock_explain,
    mock_study_pack,
    mock_index,
    tmp_path,
):
    mock_crew_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.pydantic = SubjectTagBatch(
        tags=[SubjectTag(element_id="p1-e1", subject="calculus", content_type="text", confidence=0.5)]
        # Note: only tagging one element on purpose to prove the pipeline
        # doesn't require every element to be tagged to complete successfully.
    )
    mock_crew_instance.kickoff.return_value = mock_result
    mock_crew_cls.return_value = mock_crew_instance

    mock_explain.return_value = ExplanationDocument(
        document_filename="stacks.pdf",
        explanations=[
            ElementExplanation(
                element_id="p1-e1",
                element_type="heading",
                explanation="Introduces stacks and queues.",
                confidence=0.9,
                needs_review=False,
                source_model="openai/gpt-oss-120b",
            )
        ],
    )
    mock_study_pack.return_value = StudyPack(
        document_filename="stacks.pdf",
        html_storage_path="fake-document-id/study_pack.html",
        audio_chapters=[],
        glossary=[],
    )

    pdf_path = str(tmp_path / "stacks.pdf")
    _make_pdf(pdf_path)

    flow = run_pipeline(pdf_path, "stacks.pdf")

    assert flow.state.document_id == "fake-document-id"
    assert flow.state.parsed_document is not None
    assert flow.state.parsed_document.total_pages == 1
    assert flow.state.accessibility_report is not None
    assert flow.state.subject_tagged_document is not None
    assert flow.state.explanation_document is not None
    assert flow.state.study_pack is not None
    assert flow.state.indexed_chunk_count == 1

    # update_document should have been called once per successful stage (6 times)
    statuses = [call.kwargs.get("status") for call in mock_update.call_args_list]
    assert statuses == ["parsed", "audited", "tagged", "explained", "packed", "done"]


@patch("flow.create_document_record", return_value="fake-document-id")
@patch("flow.upload_pdf", return_value="fake-storage-path.pdf")
def test_pipeline_fails_gracefully_on_unreadable_pdf(_mock_upload, _mock_create, tmp_path):
    bad_path = tmp_path / "bad.pdf"
    bad_path.write_bytes(b"not a pdf")

    with patch("flow.update_document") as mock_update:
        try:
            run_pipeline(str(bad_path), "bad.pdf")
            assert False, "expected PipelineError"
        except PipelineError as exc:
            assert "read_document" in str(exc)
        # The failure should have been persisted to the document row.
        assert mock_update.call_args.kwargs.get("status") == "error"
