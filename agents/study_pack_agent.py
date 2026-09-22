"""
Stage 5: Study-Pack Agent.

Builds three things from Stage 1's ParsedDocument and Stage 4's
ExplanationDocument:
1. An accessible HTML page (headings kept as real headings, images described
   in text instead of shown, code in <pre><code>, tables as real <table>
   markup only when Stage 4 judged them genuinely tabular).
2. Chapter-based mp3 audio, split at heading boundaries (tools/chunking.py).
3. A short glossary of technical terms, via Groq.

HTML and audio are uploaded to Supabase Storage; only their storage paths
are kept in the StudyPack record (the files themselves can be large).
"""

import html
import os
import tempfile

from crewai import LLM, Agent, Crew, Process, Task

from config import GROQ_BASE_URL, GROQ_MODEL, get_groq_api_key
from db.supabase_client import upload_bytes, upload_file
from models.document import ParsedDocument
from models.explanation import ElementExplanation, ExplanationDocument
from models.study_pack import AudioChapter, GlossaryBatch, StudyPack
from tools.chunking import build_chapters
from tools.tts_tools import TTSError, synthesize_speech

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
def _kickoff_with_retry(crew, lecture_text):
    return crew.kickoff(inputs={"lecture_text": lecture_text})

GLOSSARY_TEXT_LIMIT = 6000


def _explanation_lookup(explanation_doc: ExplanationDocument | None) -> dict[str, ElementExplanation]:
    if explanation_doc is None:
        return {}
    return {exp.element_id: exp for exp in explanation_doc.explanations}


def _render_element_html(element, explanations: dict[str, ElementExplanation]) -> str:
    text = html.escape(element.text) if element.text else ""
    explanation = explanations.get(element.id)

    if element.element_type == "heading":
        return f"<h2>{text}</h2>"
    if element.element_type == "code":
        note = f"<p class='explanation'><em>{html.escape(explanation.explanation)}</em></p>" if explanation else ""
        return f"<pre><code>{text}</code></pre>{note}"
    if element.element_type == "equation":
        note = f"<p class='explanation'>{html.escape(explanation.explanation)}</p>" if explanation else ""
        return f"<p class='equation'>{text}</p>{note}"
    if element.element_type == "table":
        if element.table_rows:
            rows_html = "".join(
                "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
                for row in element.table_rows
            )
            caption = f"<caption>{html.escape(explanation.explanation)}</caption>" if explanation else ""
            return f"<table>{caption}{rows_html}</table>"
        return ""
    if element.element_type == "image":
        description = explanation.explanation if explanation else "No description available for this image."
        return f"<figure role='img' aria-label='{html.escape(description[:120])}'><figcaption>{html.escape(description)}</figcaption></figure>"
    return f"<p>{text}</p>" if text else ""


def build_html_study_pack(parsed: ParsedDocument, explanation_doc: ExplanationDocument | None) -> str:
    explanations = _explanation_lookup(explanation_doc)
    body_parts = [_render_element_html(el, explanations) for el in parsed.elements]
    body = "\n".join(part for part in body_parts if part)
    title = html.escape(parsed.filename)
    return (
        "<!DOCTYPE html>\n"
        f'<html lang="en">\n<head>\n<meta charset="UTF-8">\n<title>{title} — Accessible Study Pack</title>\n'
        "<style>body{font-family:sans-serif;max-width:800px;margin:2rem auto;line-height:1.6;}"
        "pre{background:#f2f2f2;padding:1rem;overflow-x:auto;} .explanation{color:#444;}"
        "table{border-collapse:collapse;} td{border:1px solid #ccc;padding:0.4rem;}</style>\n"
        f"</head>\n<body>\n<main>\n<h1>{title}</h1>\n{body}\n</main>\n</body>\n</html>\n"
    )


def _synthesize_and_upload_chapter(document_id: str, index: int, chapter: dict) -> AudioChapter | None:
    if not chapter["text"].strip():
        return None
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        synthesize_speech(chapter["text"], tmp_path)
        storage_path = upload_file(tmp_path, f"{document_id}/audio/chapter_{index}.mp3", "audio/mpeg")
    except TTSError:
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return AudioChapter(
        title=chapter["title"],
        page_start=chapter["page_start"],
        page_end=chapter["page_end"],
        storage_path=storage_path,
    )


def _build_glossary_crew() -> Crew:
    llm = LLM(model=GROQ_MODEL, provider="openai", api_key=get_groq_api_key(), base_url=GROQ_BASE_URL, temperature=0.3)
    agent = Agent(
        role="Glossary Builder",
        goal="Identify important technical terms in a lecture and define each in one clear sentence.",
        backstory="You build concise glossaries for students, using only terms that actually appear "
        "in the given lecture text.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    task = Task(
        description=(
            "Read this lecture text and list the important technical terms a student should know, "
            "each with a one-sentence definition grounded in how the term is used in this text:\n\n{lecture_text}"
        ),
        expected_output="A GlossaryBatch with 5-15 terms.",
        agent=agent,
        output_pydantic=GlossaryBatch,
    )
    return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)


def build_glossary(parsed: ParsedDocument) -> list:
    full_text = "\n".join(el.text for el in parsed.elements if el.text.strip())[:GLOSSARY_TEXT_LIMIT]
    if not full_text.strip():
        return []
    crew = _build_glossary_crew()
    result = _kickoff_with_retry(crew, full_text)
    if result.pydantic is None or not isinstance(result.pydantic, GlossaryBatch):
        return []
    return result.pydantic.terms


def run(parsed: ParsedDocument, document_id: str, explanation_doc: ExplanationDocument | None = None) -> StudyPack:
    html_content = build_html_study_pack(parsed, explanation_doc)
    html_storage_path = upload_bytes(f"{document_id}/study_pack.html", html_content.encode("utf-8"), "text/html")

    chapters = build_chapters(parsed)
    audio_chapters = []
    for index, chapter in enumerate(chapters, start=1):
        result = _synthesize_and_upload_chapter(document_id, index, chapter)
        if result:
            audio_chapters.append(result)

    glossary = build_glossary(parsed)

    return StudyPack(
        document_filename=parsed.filename,
        html_storage_path=html_storage_path,
        audio_chapters=audio_chapters,
        glossary=glossary,
    )
