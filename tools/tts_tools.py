"""
Text-to-speech helper for Stage 5 (Study-Pack Agent) chapter audio.
edge-tts's Communicate.save() is async, so it's run via asyncio.run() from
the otherwise-synchronous pipeline.
"""

import asyncio

import edge_tts

from config import TTS_VOICE


class TTSError(Exception):
    """Raised when speech synthesis fails."""


async def _synthesize(text: str, output_path: str) -> None:
    communicate = edge_tts.Communicate(text, voice=TTS_VOICE)
    await communicate.save(output_path)


def synthesize_speech(text: str, output_path: str) -> None:
    """Synthesize `text` to an mp3 file at `output_path`."""
    if not text.strip():
        raise TTSError("Cannot synthesize empty text.")
    try:
        asyncio.run(_synthesize(text, output_path))
    except Exception as exc:
        raise TTSError(f"Edge TTS synthesis failed: {exc}") from exc
