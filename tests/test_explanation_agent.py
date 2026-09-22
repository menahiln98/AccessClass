from unittest.mock import MagicMock, patch

from agents.explanation_agent import run
from models.document import Element, PageInfo, ParsedDocument
from models.explanation import ExplanationBatch, ExplanationItem, VisualExplanation
from tools.gemini_tools import GeminiError


def _doc_with_image_and_code():
    pages = [PageInfo(page_number=1, has_selectable_text=True, is_scanned=False, num_images=1)]
    elements = [
        Element(id="p1-e1", page_number=1, element_type="image", bbox=[10, 10, 100, 100], image_index=0),
        Element(id="p1-e2", page_number=1, element_type="code", text="int x = 1;"),
    ]
    return ParsedDocument(filename="test.pdf", total_pages=1, pages=pages, elements=elements)


@patch("agents.explanation_agent.Crew")
@patch("agents.explanation_agent.get_groq_api_key", return_value="dummy-test-key")
@patch("agents.explanation_agent.explain_image")
@patch("agents.explanation_agent.crop_page_to_png", return_value=b"fake-png")
def test_run_explains_image_and_code(mock_crop, mock_explain_image, _mock_key, mock_crew_cls):
    mock_explain_image.return_value = VisualExplanation(
        short_alt_text="A pointer diagram",
        detailed_explanation="Shows a node pointing to the next node.",
        academic_meaning="Illustrates linked list traversal.",
        confidence=0.85,
        is_unclear=False,
    )

    mock_crew_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.pydantic = ExplanationBatch(
        items=[ExplanationItem(element_id="p1-e2", explanation="Declares an integer x set to 1.", confidence=0.95)]
    )
    mock_crew_instance.kickoff.return_value = mock_result
    mock_crew_cls.return_value = mock_crew_instance

    result = run(_doc_with_image_and_code(), "/fake/path.pdf")

    assert len(result.explanations) == 2
    image_exp = next(e for e in result.explanations if e.element_id == "p1-e1")
    assert "linked list" in image_exp.explanation.lower()
    assert image_exp.needs_review is False
    assert image_exp.source_model == "gemini-3.5-flash"

    code_exp = next(e for e in result.explanations if e.element_id == "p1-e2")
    assert "integer" in code_exp.explanation.lower()
    assert code_exp.needs_review is False


def test_run_handles_image_with_no_bbox():
    pages = [PageInfo(page_number=1, has_selectable_text=True, is_scanned=False, num_images=1)]
    elements = [Element(id="p1-e1", page_number=1, element_type="image", bbox=None, image_index=0)]
    doc = ParsedDocument(filename="test.pdf", total_pages=1, pages=pages, elements=elements)

    result = run(doc, "/fake/path.pdf")

    assert len(result.explanations) == 1
    assert result.explanations[0].needs_review is True
    assert result.explanations[0].confidence == 0.0


@patch("agents.explanation_agent.explain_image", side_effect=GeminiError("API down"))
@patch("agents.explanation_agent.crop_page_to_png", return_value=b"fake-png")
def test_run_marks_needs_review_on_gemini_failure(_mock_crop, _mock_explain):
    pages = [PageInfo(page_number=1, has_selectable_text=True, is_scanned=False, num_images=1)]
    elements = [Element(id="p1-e1", page_number=1, element_type="image", bbox=[0, 0, 10, 10], image_index=0)]
    doc = ParsedDocument(filename="test.pdf", total_pages=1, pages=pages, elements=elements)

    result = run(doc, "/fake/path.pdf")

    assert result.explanations[0].needs_review is True
    assert "API down" in result.explanations[0].explanation


@patch("agents.explanation_agent.Crew")
@patch("agents.explanation_agent.get_groq_api_key", return_value="dummy-test-key")
def test_table_flagged_for_review_when_not_real_table(_mock_key, mock_crew_cls):
    pages = [PageInfo(page_number=1, has_selectable_text=True, is_scanned=False)]
    elements = [
        Element(id="p1-e1", page_number=1, element_type="table", table_rows=[["Write", "Compile"], ["", ""]])
    ]
    doc = ParsedDocument(filename="test.pdf", total_pages=1, pages=pages, elements=elements)

    mock_crew_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.pydantic = ExplanationBatch(
        items=[
            ExplanationItem(
                element_id="p1-e1",
                explanation="This looks like a design layout, not real tabular data.",
                confidence=0.9,
                looks_like_real_table=False,
            )
        ]
    )
    mock_crew_instance.kickoff.return_value = mock_result
    mock_crew_cls.return_value = mock_crew_instance

    result = run(doc, "/fake/path.pdf")

    assert result.explanations[0].needs_review is True
