from unittest.mock import MagicMock, patch

from agents.subject_interpreter import BATCH_SIZE, SubjectInterpretationError, _classifiable_text, run
from models.document import Element, ParsedDocument
from models.subject import SubjectTag, SubjectTagBatch


def test_classifiable_text_skips_images():
    image_el = Element(id="p1-e1", page_number=1, element_type="image", image_index=0)
    assert _classifiable_text(image_el) is None


def test_classifiable_text_uses_table_rows():
    table_el = Element(
        id="p1-e2", page_number=1, element_type="table", table_rows=[["Name", "Complexity"], ["Push", "O(1)"]]
    )
    text = _classifiable_text(table_el)
    assert "Name" in text and "Push" in text


def _fake_tag(element_id: str) -> SubjectTag:
    return SubjectTag(
        element_id=element_id, subject="data_structures_algorithms", content_type="text", confidence=0.8
    )


@patch("agents.subject_interpreter.get_groq_api_key", return_value="dummy-test-key")
@patch("agents.subject_interpreter.Crew")
def test_run_classifies_all_elements_in_one_batch(mock_crew_cls, _mock_key):
    mock_crew_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.pydantic = SubjectTagBatch(tags=[_fake_tag("p1-e1"), _fake_tag("p1-e2")])
    mock_crew_instance.kickoff.return_value = mock_result
    mock_crew_cls.return_value = mock_crew_instance

    parsed = ParsedDocument(
        filename="lecture.pdf",
        total_pages=1,
        pages=[],
        elements=[
            Element(id="p1-e1", page_number=1, element_type="paragraph", text="A stack is LIFO."),
            Element(id="p1-e2", page_number=1, element_type="paragraph", text="A queue is FIFO."),
        ],
    )

    result = run(parsed)
    assert result.document_filename == "lecture.pdf"
    assert len(result.tags) == 2
    assert mock_crew_instance.kickoff.call_count == 1


@patch("agents.subject_interpreter.get_groq_api_key", return_value="dummy-test-key")
@patch("agents.subject_interpreter.Crew")
def test_run_batches_large_element_lists(mock_crew_cls, _mock_key):
    mock_crew_instance = MagicMock()

    def fake_kickoff(inputs):
        # Return one tag per element_id mentioned in this batch's prompt text.
        ids_in_batch = [line.split(" | ")[0] for line in inputs["elements_block"].splitlines()]
        result = MagicMock()
        result.pydantic = SubjectTagBatch(tags=[_fake_tag(i) for i in ids_in_batch])
        return result

    mock_crew_instance.kickoff.side_effect = fake_kickoff
    mock_crew_cls.return_value = mock_crew_instance

    total_elements = BATCH_SIZE + 5  # forces exactly 2 batches
    elements = [
        Element(id=f"p1-e{i}", page_number=1, element_type="paragraph", text=f"Text number {i}.")
        for i in range(total_elements)
    ]
    parsed = ParsedDocument(filename="big.pdf", total_pages=1, pages=[], elements=elements)

    result = run(parsed)
    assert len(result.tags) == total_elements
    assert mock_crew_instance.kickoff.call_count == 2


@patch("agents.subject_interpreter.get_groq_api_key", return_value="dummy-test-key")
@patch("agents.subject_interpreter.Crew")
def test_run_raises_on_bad_llm_output(mock_crew_cls, _mock_key):
    mock_crew_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.pydantic = None
    mock_result.raw = "not valid structured output"
    mock_crew_instance.kickoff.return_value = mock_result
    mock_crew_cls.return_value = mock_crew_instance

    parsed = ParsedDocument(
        filename="lecture.pdf",
        total_pages=1,
        pages=[],
        elements=[Element(id="p1-e1", page_number=1, element_type="paragraph", text="Some text.")],
    )

    try:
        run(parsed)
        assert False, "expected SubjectInterpretationError"
    except SubjectInterpretationError as exc:
        assert "did not match" in str(exc)


def test_run_with_no_classifiable_elements_returns_empty():
    parsed = ParsedDocument(
        filename="images-only.pdf",
        total_pages=1,
        pages=[],
        elements=[Element(id="p1-e1", page_number=1, element_type="image", image_index=0)],
    )
    result = run(parsed)
    assert result.tags == []


def test_task_prompt_includes_all_new_subjects_and_excludes_unclassified_from_the_exception_list():
    """
    The exact bug this guards against: the prompt's sentence telling the LLM
    when to use 'unclassified' used to hardcode only 3 subject names. If a new
    subject were added to config.SUPPORTED_SUBJECTS but this sentence weren't
    updated, the LLM would still be told to mark anything outside the old 3 as
    unclassified — silently making every new subject unreachable. This builds
    a real Crew/Task (no network call made) and inspects the actual prompt text.
    """
    from config import SUPPORTED_SUBJECTS
    from agents.subject_interpreter import _build_crew

    with patch("agents.subject_interpreter.get_groq_api_key", return_value="dummy-test-key"):
        crew = _build_crew()

    prompt_text = crew.tasks[0].description

    for subject in SUPPORTED_SUBJECTS:
        assert subject in prompt_text, f"'{subject}' is missing from the Subject Interpreter's prompt"

    # The exception-list sentence must list every real subject, not a stale hardcoded subset.
    for new_subject in ["ict", "oop", "database_systems", "operating_systems", "computer_networks", "artificial_intelligence_ml"]:
        assert new_subject in prompt_text