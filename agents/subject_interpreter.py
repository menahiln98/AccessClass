"""
Stage 3: Subject Interpreter.

This is the only Portion 1 stage implemented as a real CrewAI Agent/Task/Crew,
because it is the only Portion 1 stage that needs semantic reasoning (deciding
which supported subject area a chunk belongs to — see config.SUPPORTED_SUBJECTS
for the full list — and what kind of content it is) rather than deterministic
extraction/rule-checking.

Uses Groq's `openai/gpt-oss-120b`, called through CrewAI's native "openai"
provider pointed at Groq's OpenAI-compatible base URL (this CrewAI version
has no separate native "groq" provider — see config.py for details).
"""

from crewai import Agent, Crew, Process, Task

from config import GROQ_BASE_URL, GROQ_MODEL, SUPPORTED_CONTENT_TYPES, SUPPORTED_SUBJECTS, get_groq_api_key
from models.document import Element, ParsedDocument
from models.subject import SubjectTag, SubjectTagBatch, SubjectTaggedDocument

BATCH_SIZE = 20


class SubjectInterpretationError(Exception):
    """Raised when the LLM call fails or returns output that fails validation."""


def _classifiable_text(element: Element) -> str | None:
    if element.text.strip():
        return element.text.strip()
    if element.table_rows:
        return "; ".join(" | ".join(row) for row in element.table_rows)
    return None  # e.g. images: nothing textual to classify here


def _build_crew() -> Crew:
    from crewai import LLM

    llm = LLM(
        model=GROQ_MODEL,
        provider="openai",
        api_key=get_groq_api_key(),
        base_url=GROQ_BASE_URL,
        temperature=0.0,
    )

    agent = Agent(
        role="Subject Interpreter",
        goal=(
            "Classify each lecture content element by academic subject and content type, "
            "to prepare it for accessible explanation."
        ),
        backstory=(
            "You are an expert teaching assistant across multiple Computer Science courses "
            "(Programming Fundamentals/C++, Data Structures & Algorithms, ICT, OOP, Database "
            "Systems, Operating Systems, Computer Networks, and AI/ML) as well as Calculus. "
            "You read short lecture excerpts and identify which subject area they belong to "
            "and what kind of content they are, without adding any explanation of your own."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    real_subjects = [s for s in SUPPORTED_SUBJECTS if s != "unclassified"]

    task = Task(
        description=(
            "Classify each of the following lecture content elements.\n\n"
            "Elements (format: id | heuristic_type | text):\n{elements_block}\n\n"
            f"For every element, choose exactly one `subject` from {SUPPORTED_SUBJECTS} "
            f"and exactly one `content_type` from {SUPPORTED_CONTENT_TYPES}. "
            f"Use 'unclassified' as the subject only if the text truly does not relate to "
            f"any of: {real_subjects}. "
            "Give a confidence between 0.0 and 1.0 for each classification. "
            "Return every element_id you were given exactly once — do not skip any."
        ),
        expected_output="A SubjectTagBatch listing one SubjectTag per input element.",
        agent=agent,
        output_pydantic=SubjectTagBatch,
    )

    return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)


def _classify_batch(crew: Crew, elements: list[Element]) -> list[SubjectTag]:
    elements_block = "\n".join(
        f"{el.id} | {el.element_type} | {(_classifiable_text(el) or '')[:400]}" for el in elements
    )
    try:
        result = crew.kickoff(inputs={"elements_block": elements_block})
    except Exception as exc:
        raise SubjectInterpretationError(
            "Subject Interpreter failed to reach Groq. Check that GROQ_API_KEY is set "
            f"and valid. Original error: {exc}"
        ) from exc

    if result.pydantic is None or not isinstance(result.pydantic, SubjectTagBatch):
        raise SubjectInterpretationError(
            "Subject Interpreter's LLM response did not match the expected SubjectTagBatch "
            f"schema. Raw response: {result.raw!r}"
        )
    return result.pydantic.tags


def run(parsed: ParsedDocument) -> SubjectTaggedDocument:
    classifiable = [el for el in parsed.elements if _classifiable_text(el) is not None]

    if not classifiable:
        return SubjectTaggedDocument(document_filename=parsed.filename, tags=[])

    crew = _build_crew()
    all_tags: list[SubjectTag] = []
    for start in range(0, len(classifiable), BATCH_SIZE):
        batch = classifiable[start : start + BATCH_SIZE]
        all_tags.extend(_classify_batch(crew, batch))

    return SubjectTaggedDocument(document_filename=parsed.filename, tags=all_tags)