from unittest.mock import AsyncMock, patch

import pytest

from tools.tts_tools import TTSError, synthesize_speech


def test_synthesize_speech_rejects_empty_text():
    with pytest.raises(TTSError, match="empty"):
        synthesize_speech("   ", "/tmp/out.mp3")


@patch("tools.tts_tools.edge_tts.Communicate")
def test_synthesize_speech_calls_save(mock_communicate_cls):
    mock_instance = mock_communicate_cls.return_value
    mock_instance.save = AsyncMock()

    synthesize_speech("Hello world", "/tmp/out.mp3")

    mock_communicate_cls.assert_called_once()
    mock_instance.save.assert_awaited_once_with("/tmp/out.mp3")


@patch("tools.tts_tools.edge_tts.Communicate")
def test_synthesize_speech_wraps_failures(mock_communicate_cls):
    mock_instance = mock_communicate_cls.return_value
    mock_instance.save = AsyncMock(side_effect=RuntimeError("network down"))

    with pytest.raises(TTSError, match="Edge TTS synthesis failed"):
        synthesize_speech("Hello world", "/tmp/out.mp3")
