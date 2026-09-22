from unittest.mock import MagicMock, patch

import pymupdf
import pytest

from models.explanation import VisualExplanation
from tools.gemini_tools import GeminiError, crop_page_to_png, embed_text, explain_image


def _make_pdf(path: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "A diagram caption", fontsize=12, fontname="helv")
    doc.save(path)
    doc.close()


def test_crop_page_to_png_returns_valid_png_bytes(tmp_path):
    pdf_path = str(tmp_path / "test.pdf")
    _make_pdf(pdf_path)
    png_bytes = crop_page_to_png(pdf_path, 1, [50, 50, 300, 150])
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


@patch("tools.gemini_tools.get_gemini_api_key", return_value="dummy-test-key")
@patch("tools.gemini_tools.genai.Client")
def test_explain_image_returns_parsed_response(mock_client_cls, _mock_key):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = VisualExplanation(
        short_alt_text="A binary tree",
        detailed_explanation="A tree with a root and two children.",
        academic_meaning="Demonstrates ordering.",
        confidence=0.9,
        is_unclear=False,
    )
    mock_client.models.generate_content.return_value = mock_response
    mock_client_cls.return_value = mock_client

    # Clear the lru_cache so our mocked Client is actually used.
    from tools.gemini_tools import get_client

    get_client.cache_clear()

    result = explain_image(b"fake-png-bytes", "data structures")
    assert result.short_alt_text == "A binary tree"
    assert result.confidence == 0.9


@patch("tools.gemini_tools.get_gemini_api_key", return_value="dummy-test-key")
@patch("tools.gemini_tools.genai.Client")
def test_explain_image_raises_on_bad_response(mock_client_cls, _mock_key):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = None
    mock_response.text = "not valid json"
    mock_client.models.generate_content.return_value = mock_response
    mock_client_cls.return_value = mock_client

    from tools.gemini_tools import get_client

    get_client.cache_clear()

    with pytest.raises(GeminiError, match="did not match"):
        explain_image(b"fake-png-bytes", "calculus")


@patch("tools.gemini_tools.get_gemini_api_key", return_value="dummy-test-key")
@patch("tools.gemini_tools.genai.Client")
def test_embed_text_returns_vector(mock_client_cls, _mock_key):
    mock_client = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1, 0.2, 0.3]
    mock_result = MagicMock()
    mock_result.embeddings = [mock_embedding]
    mock_client.models.embed_content.return_value = mock_result
    mock_client_cls.return_value = mock_client

    from tools.gemini_tools import get_client

    get_client.cache_clear()

    vector = embed_text("A stack is LIFO.", task_type="RETRIEVAL_DOCUMENT")
    assert vector == [0.1, 0.2, 0.3]
