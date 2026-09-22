from unittest.mock import MagicMock, patch

from agents.study_pack_agent import build_html_study_pack, run
from models.document import Element, PageInfo, ParsedDocument
from models.explanation import ElementExplanation, ExplanationDocument
from models.study_pack import GlossaryBatch, GlossaryTerm
from tools.tts_tools import TTSError


def _doc():
    pages = [PageInfo(page_number=1, has_selectable_text=True, is_scanned=False)]
    elements = [
        Element(id="p1-e1", page_number=1, element_type="heading", text="Stacks"),
        Element(id="p1-e2", page_number=1, element_type="code", text="int x = 1;"),
        Element(id="p1-e3", page_number=1, element_type="table", table_rows=[["Op", "Cost"], ["Push", "O(1)"]]),
        Element(id="p1-e4", page_number=1, element_type="image", image_index=0),
    ]
    return ParsedDocument(filename="test.pdf", total_pages=1, pages=pages, elements=elements)


def test_build_html_study_pack_renders_each_element_type():
    explanations = ExplanationDocument(
        document_filename="test.pdf",
        explanations=[
            ElementExplanation(
                element_id="p1-e2", element_type="code", explanation="Sets x to 1.",
                confidence=0.9, needs_review=False, source_model="groq",
            ),
            ElementExplanation(
                element_id="p1-e4", element_type="image", explanation="A stack diagram.",
                confidence=0.9, needs_review=False, source_model="gemini-3.5-flash",
            ),
        ],
    )
    html_out = build_html_study_pack(_doc(), explanations)

    assert "<h2>Stacks</h2>" in html_out
    assert "<pre><code>int x = 1;</code></pre>" in html_out
    assert "Sets x to 1." in html_out
    assert "<table>" in html_out and "Push" in html_out
    assert "A stack diagram." in html_out


def test_build_html_study_pack_handles_missing_explanations():
    html_out = build_html_study_pack(_doc(), None)
    assert "No description available for this image." in html_out
    assert "<pre><code>int x = 1;</code></pre>" in html_out  # still renders, just no note


@patch("agents.study_pack_agent.Crew")
@patch("agents.study_pack_agent.get_groq_api_key", return_value="dummy-test-key")
@patch("agents.study_pack_agent.synthesize_speech")
@patch("agents.study_pack_agent.upload_file", return_value="doc-1/audio/chapter_1.mp3")
@patch("agents.study_pack_agent.upload_bytes", return_value="doc-1/study_pack.html")
def test_run_builds_full_study_pack(mock_upload_bytes, mock_upload_file, mock_synthesize, _mock_key, mock_crew_cls):
    mock_crew_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.pydantic = GlossaryBatch(terms=[GlossaryTerm(term="Stack", definition="A LIFO structure.")])
    mock_crew_instance.kickoff.return_value = mock_result
    mock_crew_cls.return_value = mock_crew_instance

    pack = run(_doc(), "doc-1", explanation_doc=None)

    assert pack.html_storage_path == "doc-1/study_pack.html"
    assert len(pack.audio_chapters) == 1
    assert pack.audio_chapters[0].title == "Stacks"
    assert pack.glossary[0].term == "Stack"
    mock_upload_bytes.assert_called_once()
    mock_synthesize.assert_called_once()


@patch("agents.study_pack_agent.Crew")
@patch("agents.study_pack_agent.get_groq_api_key", return_value="dummy-test-key")
@patch("agents.study_pack_agent.synthesize_speech", side_effect=TTSError("voice service unavailable"))
@patch("agents.study_pack_agent.upload_bytes", return_value="doc-1/study_pack.html")
def test_run_skips_chapter_when_tts_fails(mock_upload_bytes, _mock_synth, _mock_key, mock_crew_cls):
    mock_crew_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.pydantic = GlossaryBatch(terms=[])
    mock_crew_instance.kickoff.return_value = mock_result
    mock_crew_cls.return_value = mock_crew_instance

    pack = run(_doc(), "doc-1", explanation_doc=None)

    assert pack.audio_chapters == []  # the one chapter failed synthesis and was skipped, not crashed
