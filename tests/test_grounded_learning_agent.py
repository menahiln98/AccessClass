from unittest.mock import MagicMock, patch

import pytest

from agents.grounded_learning_agent import GroundedLearningError, ask, index_document
from models.document import Element, PageInfo, ParsedDocument
from models.qa import AskAnswer, Citation


def _doc():
    pages = [PageInfo(page_number=1, has_selectable_text=True, is_scanned=False)]
    elements = [Element(id="p1-e1", page_number=1, element_type="paragraph", text="A stack is LIFO.")]
    return ParsedDocument(filename="test.pdf", total_pages=1, pages=pages, elements=elements)


@patch("agents.grounded_learning_agent.upsert_chunks")
@patch("agents.grounded_learning_agent.ensure_collection")
@patch("agents.grounded_learning_agent.embed_text", return_value=[0.1, 0.2])
def test_index_document_embeds_and_upserts_each_chunk(mock_embed, mock_ensure, mock_upsert):
    count = index_document("doc-1", _doc())
    assert count == 1
    mock_ensure.assert_called_once()
    mock_embed.assert_called_once_with("A stack is LIFO.", task_type="RETRIEVAL_DOCUMENT")
    mock_upsert.assert_called_once()


def test_index_document_returns_zero_for_no_text():
    pages = [PageInfo(page_number=1, has_selectable_text=True, is_scanned=False)]
    elements = [Element(id="p1-e1", page_number=1, element_type="image", image_index=0)]
    doc = ParsedDocument(filename="test.pdf", total_pages=1, pages=pages, elements=elements)
    assert index_document("doc-1", doc) == 0


def test_ask_rejects_empty_question():
    with pytest.raises(GroundedLearningError, match="empty"):
        ask("doc-1", "   ")


@patch("agents.grounded_learning_agent.search_chunks")
@patch("agents.grounded_learning_agent.embed_text", return_value=[0.1, 0.2])
def test_ask_returns_unsupported_when_nothing_relevant(mock_embed, mock_search):
    mock_search.return_value = [{"page_start": 1, "page_end": 1, "text": "unrelated", "score": 0.1}]
    result = ask("doc-1", "What is a black hole?")
    assert result.is_supported is False
    assert "not appear to be covered" in result.answer


@patch("agents.grounded_learning_agent.Crew")
@patch("agents.grounded_learning_agent.get_groq_api_key", return_value="dummy-test-key")
@patch("agents.grounded_learning_agent.search_chunks")
@patch("agents.grounded_learning_agent.embed_text", return_value=[0.1, 0.2])
def test_ask_returns_grounded_answer_when_relevant(mock_embed, mock_search, _mock_key, mock_crew_cls):
    mock_search.return_value = [{"page_start": 3, "page_end": 3, "text": "A stack is LIFO.", "score": 0.9}]

    mock_crew_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.pydantic = AskAnswer(
        question="What is a stack?",
        answer="A stack is a Last-In-First-Out data structure.",
        is_supported=True,
        citations=[Citation(page_number=3, snippet="A stack is LIFO.")],
    )
    mock_crew_instance.kickoff.return_value = mock_result
    mock_crew_cls.return_value = mock_crew_instance

    result = ask("doc-1", "What is a stack?")
    assert result.is_supported is True
    assert result.citations[0].page_number == 3
