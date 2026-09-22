"""
Gemini helpers used by two different Portion 2 stages:
- Stage 4 (Explanation Agent): explain_image() — vision, structured output.
- Stage 6 (Grounded Learning Agent): embed_text() — text embeddings for Qdrant.

Called directly via the google-genai SDK rather than through a CrewAI Agent:
this CrewAI version has no first-class multimodal (image) input path, while
google-genai's response_schema gives the same structured-output guarantee
CrewAI's output_pydantic gives for Groq. Using the same guarantee via the
provider's own SDK is more reliable here than forcing an image through an
interface that isn't built for it.

Gemini's free tier caps requests per minute, not per account balance — a
lecture PDF with many images can burn through that cap in seconds since
each image is its own vision call. explain_image() retries specifically on
that 429 response with a real wait (Gemini's own per-minute window), not on
other errors, which fail immediately instead of wasting three slow retries
on something a wait can't fix.
"""

import functools

import pymupdf
from google import genai
from google.genai import errors, types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config import EMBEDDING_DIMENSIONS, GEMINI_EMBEDDING_MODEL, GEMINI_MODEL, get_gemini_api_key
from models.explanation import VisualExplanation

CROP_DPI = 200


class GeminiError(Exception):
    """Raised when a Gemini call fails or returns output that fails validation."""


def _is_transient_error(exc: BaseException) -> bool:
    """
    429 = rate limit (free-tier per-minute cap); 500/503 = Gemini's own servers
    temporarily overloaded ("high demand... usually temporary"). Both resolve
    on their own with a short wait — anything else (a bad request, an invalid
    key) won't be fixed by retrying, so it fails immediately instead.
    """
    return isinstance(exc, errors.APIError) and getattr(exc, "code", None) in (429, 500, 503)


_retry_on_rate_limit = retry(
    retry=retry_if_exception(_is_transient_error),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=20, max=60),
    reraise=True,
)


@functools.lru_cache(maxsize=1)
def get_client() -> genai.Client:
    return genai.Client(
        api_key=get_gemini_api_key(),
        http_options=types.HttpOptions(timeout=60000),  # milliseconds — 60s ceiling per request
    )



def crop_page_to_png(pdf_path: str, page_number: int, bbox: list[float]) -> bytes:
    """Render just the region of one page given by bbox ([x0, y0, x1, y1]) as PNG bytes."""
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[page_number - 1]
        rect = pymupdf.Rect(*bbox)
        pixmap = page.get_pixmap(clip=rect, dpi=CROP_DPI)
        return pixmap.tobytes("png")
    finally:
        doc.close()


@_retry_on_rate_limit
def _generate_visual_explanation(image_bytes: bytes, prompt: str):
    return get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VisualExplanation,
        ),
    )


def explain_image(image_bytes: bytes, subject_hint: str) -> VisualExplanation:
    """Ask Gemini to describe one cropped image region, as structured JSON."""
    prompt = (
        f"This image is a diagram/graph/visual from a {subject_hint} lecture. "
        "Describe it for a student who cannot see it. Provide short_alt_text (a few words, "
        "for quick screen-reader navigation), detailed_explanation (full context), "
        "academic_meaning (why it matters in the lecture), a confidence score (0.0-1.0), "
        "and set is_unclear to true only if the image is genuinely too unclear/low-quality "
        "to describe safely."
    )
    try:
        response = _generate_visual_explanation(image_bytes, prompt)
    except Exception as exc:
        raise GeminiError(f"Gemini vision call failed: {exc}") from exc

    if response.parsed is None or not isinstance(response.parsed, VisualExplanation):
        raise GeminiError(f"Gemini vision response did not match VisualExplanation schema. Raw: {response.text!r}")
    return response.parsed


@_retry_on_rate_limit
def _generate_embedding(text: str, task_type: str):
    return get_client().models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )


def embed_text(text: str, task_type: str) -> list[float]:
    """
    Embed one piece of text. task_type must be "RETRIEVAL_DOCUMENT" for lecture
    chunks being stored, or "RETRIEVAL_QUERY" for a student's question — Gemini's
    embedding model produces better-aligned vectors when told which side of the
    search it's embedding.
    """
    try:
        result = _generate_embedding(text, task_type)
    except Exception as exc:
        raise GeminiError(f"Gemini embedding call failed: {exc}") from exc

    if not result.embeddings:
        raise GeminiError("Gemini embedding call returned no embeddings.")
    return result.embeddings[0].values