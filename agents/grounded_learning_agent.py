"""
Stage 6: Grounded Learning Agent — "Ask This Lecture".

Two entry points:
- index_document(): embeds and stores every page-chunk of a processed
  lecture into Qdrant. Called once, right after Stage 1 finishes.
- ask(): embeds a student's question, retrieves the most relevant chunks
  for that specific document, and asks Groq to answer using ONLY those
  chunks — with an explicit, deterministic refusal path when nothing
  relevant enough was retrieved, so an unsupported question never reaches
  the LLM to be answered from general knowledge.

ask() calls Groq directly over HTTP rather than through a CrewAI
Agent/Task/Crew. It was a single agent with a single task and no
delegation, tool use, or multi-agent coordination — i.e. none of what
CrewAI actually provides — so the wrapper added nothing but the cost of
importing crewai (and its chromadb/lancedb dependencies) into whatever
process calls ask(). The batched, multi-agent CrewAI Crews that do the
actual document-processing work (Subject Interpreter, Explanation Agent,
Study-Pack Agent) are untouched and live elsewhere in this codebase.
"""

import time

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config import GROQ_BASE_URL, GROQ_MODEL, get_groq_api_key
from db.qdrant_client import ensure_collection, search_chunks, upsert_chunks
from models.document import ParsedDocument
from models.qa import AskAnswer
from tools.chunking import build_page_chunks
from tools.gemini_tools import embed_text


def _is_rate_limit_error(exc: BaseException) -> bool:
    return "rate_limit_exceeded" in str(exc) or "429" in str(exc)


_retry_on_rate_limit = retry(
    retry=retry_if_exception(_is_rate_limit_error),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=10, max=30),
    reraise=True,
)

ANSWER_SYSTEM_PROMPT = (
    "You are a Grounded Learning Agent. Answer a student's question using ONLY the "
    "provided lecture excerpts. You never use outside knowledge, never guess, and "
    "never help complete assignments or exams. If the excerpts don't answer the "
    "question, you say so plainly."
)

ANSWER_TIMEOUT_SECONDS = 30

MIN_RELEVANCE_SCORE = 0.55
TOP_K = 5
EMBEDDING_CALL_DELAY_SECONDS = 2  # spaces out Gemini embedding calls to stay under free-tier per-minute limits


class GroundedLearningError(Exception):
    """Raised when indexing or answering fails outright (embedding/Qdrant/LLM errors)."""


def index_document(document_id: str, parsed: ParsedDocument) -> int:
    """Embed and store every page-chunk for this document. Returns how many chunks were indexed."""
    chunks = build_page_chunks(document_id, parsed)
    if not chunks:
        return 0
    try:
        ensure_collection()
        embeddings = []
        for index, chunk in enumerate(chunks):
            if index > 0:
                time.sleep(EMBEDDING_CALL_DELAY_SECONDS)  # avoid bursting Gemini's per-minute rate limit
            embeddings.append(embed_text(chunk.text, task_type="RETRIEVAL_DOCUMENT"))
        upsert_chunks(chunks, embeddings)
    except Exception as exc:
        raise GroundedLearningError(f"Failed to index document '{document_id}': {exc}") from exc
    return len(chunks)


def _build_answer_prompt(context_block: str, question: str) -> str:
    return (
        f"Lecture excerpts (each labeled with its page number):\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using ONLY the excerpts above. If they don't contain enough information to answer, "
        "set is_supported to false and say so in the answer rather than guessing. "
        "Cite the page number(s) you actually used, each with a short supporting snippet. "
        "Do not complete assignments, write exam answers, or produce solutions to problems — "
        "explain concepts only. Respond with a single JSON object with exactly these keys: "
        '"question" (string), "answer" (string), "is_supported" (boolean), and "citations" '
        '(a list of objects, each with "page_number" (integer) and "snippet" (string)).'
    )


@_retry_on_rate_limit
def _call_groq_for_answer(context_block: str, question: str) -> AskAnswer:
    response = requests.post(
        f"{GROQ_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {get_groq_api_key()}"},
        json={
            "model": GROQ_MODEL,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": _build_answer_prompt(context_block, question)},
            ],
        },
        timeout=ANSWER_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    raw_content = response.json()["choices"][0]["message"]["content"]
    return AskAnswer.model_validate_json(raw_content)


def ask(document_id: str, question: str) -> AskAnswer:
    if not question.strip():
        raise GroundedLearningError("Question cannot be empty.")

    try:
        query_vector = embed_text(question, task_type="RETRIEVAL_QUERY")
        matches = search_chunks(document_id, query_vector, top_k=TOP_K)
    except Exception as exc:
        raise GroundedLearningError(f"Retrieval failed for document '{document_id}': {exc}") from exc

    relevant = [m for m in matches if m["score"] >= MIN_RELEVANCE_SCORE]
    if not relevant:
        return AskAnswer(
            question=question,
            answer="This question does not appear to be covered in the uploaded lecture material.",
            is_supported=False,
            citations=[],
        )

    context_block = "\n\n".join(f"[Page {m['page_start']}]\n{m['text']}" for m in relevant)
    try:
        return _call_groq_for_answer(context_block, question)
    except (requests.RequestException, KeyError, ValueError) as exc:
        # Covers malformed/unparseable model output (bad JSON, missing fields) the
        # same way the original CrewAI path handled a None result.pydantic: a
        # soft, honest fallback rather than surfacing a raw parsing error to the student.
        return AskAnswer(
            question=question,
            answer="An answer could not be generated for this question. Please try rephrasing it.",
            is_supported=False,
            citations=[],
        )
    except Exception as exc:
        raise GroundedLearningError(f"Answer generation failed: {exc}") from exc