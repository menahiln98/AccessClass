"""
Stage 4: Explanation Agent.

Two independent paths, matching the proposal's model assignment:
- Images -> Gemini vision (tools/gemini_tools.py), one call per image, since
  each needs to see its own specific crop of the page.
- Code / table / equation -> Groq via CrewAI (batched, same pattern as
  Stage 3's Subject Interpreter), since these are already text and don't
  need vision.

A single bad image must not fail the whole document: image failures are
caught per-element and recorded as a needs_review explanation instead of
aborting Stage 4 entirely.
"""

import time

from crewai import LLM, Agent, Crew, Process, Task

from config import GROQ_BASE_URL, GROQ_MODEL, get_groq_api_key
from models.document import Element, ParsedDocument
from models.explanation import ElementExplanation, ExplanationBatch, ExplanationDocument
from models.subject import SubjectTaggedDocument
from tools.gemini_tools import GeminiError, crop_page_to_png, explain_image

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

def _is_rate_limit_error(exc: BaseException) -> bool:
    return "rate_limit_exceeded" in str(exc) or "429" in str(exc)

_retry_on_rate_limit = retry(
    retry=retry_if_exception(_is_rate_limit_error),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=10, max=30),
    reraise=True,
)

@_retry_on_rate_limit
def _kickoff_with_retry(crew, elements_block):
    return crew.kickoff(inputs={"elements_block": elements_block})

TEXT_BATCH_SIZE = 15
LOW_CONFIDENCE_THRESHOLD = 0.6
IMAGE_CALL_DELAY_SECONDS = 2  # spaces out Gemini vision calls to stay under free-tier per-minute limits


def _subject_lookup(subject_tagged: SubjectTaggedDocument | None) -> dict:
    if subject_tagged is None:
        return {}
    return {tag.element_id: tag.subject for tag in subject_tagged.tags}


def _explain_images(images: list[Element], subjects: dict, pdf_path: str) -> list[ElementExplanation]:
    results = []
    for index, element in enumerate(images):
        if index > 0:
            time.sleep(IMAGE_CALL_DELAY_SECONDS)  # avoid bursting Gemini's per-minute rate limit
        if not element.bbox:
            results.append(
                ElementExplanation(
                    element_id=element.id,
                    element_type="image",
                    explanation="This image's location on the page could not be determined, so it "
                    "could not be automatically described.",
                    confidence=0.0,
                    needs_review=True,
                    source_model="none",
                )
            )
            continue
        subject_hint = subjects.get(element.id, "technical").replace("_", " ")
        try:
            crop_bytes = crop_page_to_png(pdf_path, element.page_number, element.bbox)
            visual = explain_image(crop_bytes, subject_hint)
            explanation_text = f"{visual.detailed_explanation} {visual.academic_meaning}".strip()
            results.append(
                ElementExplanation(
                    element_id=element.id,
                    element_type="image",
                    explanation=explanation_text,
                    confidence=visual.confidence,
                    needs_review=visual.is_unclear or visual.confidence < LOW_CONFIDENCE_THRESHOLD,
                    source_model="gemini-3.5-flash",
                )
            )
        except GeminiError as exc:
            results.append(
                ElementExplanation(
                    element_id=element.id,
                    element_type="image",
                    explanation=f"Automatic description failed for this image: {exc}",
                    confidence=0.0,
                    needs_review=True,
                    source_model="gemini-3.5-flash",
                )
            )
    return results


def _build_text_crew() -> Crew:
    llm = LLM(
        model=GROQ_MODEL, provider="openai", api_key=get_groq_api_key(), base_url=GROQ_BASE_URL, temperature=0.2
    )
    agent = Agent(
        role="Explanation Agent",
        goal="Explain code, tables, and equations from a lecture in clear plain language for a student "
        "who needs an accessible explanation.",
        backstory="You are a patient teaching assistant who explains technical content step by step, "
        "in logical blocks, without adding information that isn't in the given content.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    task = Task(
        description=(
            "Explain each of the following lecture elements in plain language.\n\n"
            "Elements (format: id | type | subject | content):\n{elements_block}\n\n"
            "For code: explain what it does in logical blocks. "
            "For equations: convert notation into spoken language (e.g. 'V equals I multiplied by R') "
            "and explain its meaning. "
            "For tables: summarize what the table shows, AND set looks_like_real_table to false if the "
            "cell content looks like design/layout text rather than genuine tabular data "
            "(e.g. short unrelated labels rather than structured rows of comparable data) — "
            "leave looks_like_real_table null for non-table elements. "
            "Give a confidence score (0.0-1.0) for every element. Return every element_id exactly once."
        ),
        expected_output="An ExplanationBatch with one ExplanationItem per input element.",
        agent=agent,
        output_pydantic=ExplanationBatch,
    )
    return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)


def _explain_text_elements(elements: list[Element], subjects: dict) -> list[ElementExplanation]:
    if not elements:
        return []
    crew = _build_text_crew()
    results: list[ElementExplanation] = []
    for start in range(0, len(elements), TEXT_BATCH_SIZE):
        batch = elements[start : start + TEXT_BATCH_SIZE]
        block = "\n".join(
            f"{el.id} | {el.element_type} | {subjects.get(el.id, 'unclassified')} | {el.text[:500]}"
            for el in batch
        )
        result = _kickoff_with_retry(crew, block)
        if result.pydantic is None or not isinstance(result.pydantic, ExplanationBatch):
            for el in batch:
                results.append(
                    ElementExplanation(
                        element_id=el.id,
                        element_type=el.element_type,
                        explanation="Automatic explanation failed for this element.",
                        confidence=0.0,
                        needs_review=True,
                        source_model=GROQ_MODEL,
                    )
                )
            continue
        for item in result.pydantic.items:
            source_element = next((el for el in batch if el.id == item.element_id), None)
            element_type = source_element.element_type if source_element else "unknown"
            needs_review = item.confidence < LOW_CONFIDENCE_THRESHOLD
            if element_type == "table" and item.looks_like_real_table is False:
                needs_review = True
            results.append(
                ElementExplanation(
                    element_id=item.element_id,
                    element_type=element_type,
                    explanation=item.explanation,
                    confidence=item.confidence,
                    needs_review=needs_review,
                    source_model=GROQ_MODEL,
                )
            )
    return results


def run(
    parsed: ParsedDocument, pdf_path: str, subject_tagged: SubjectTaggedDocument | None = None
) -> ExplanationDocument:
    subjects = _subject_lookup(subject_tagged)

    images = [el for el in parsed.elements if el.element_type == "image"]
    text_elements = [el for el in parsed.elements if el.element_type in ("code", "table", "equation")]

    explanations = _explain_images(images, subjects, pdf_path) + _explain_text_elements(text_elements, subjects)

    return ExplanationDocument(document_filename=parsed.filename, explanations=explanations)