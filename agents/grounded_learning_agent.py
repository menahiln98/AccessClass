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
"""

import time

from crewai import LLM, Agent, Crew, Process, Task

from config import GROQ_BASE_URL, GROQ_MODEL, get_groq_api_key
from db.qdrant_client import ensure_collection, search_chunks, upsert_chunks
from models.document import ParsedDocument
from models.qa import AskAnswer, Citation
from tools.chunking import build_page_chunks
from tools.gemini_tools import embed_text

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
def _kickoff_with_retry(crew, context_block, question):
    return crew.kickoff(inputs={"context_block": context_block, "question": question})

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


def _build_answer_crew() -> Crew:
    llm = LLM(model=GROQ_MODEL, provider="openai", api_key=get_groq_api_key(), base_url=GROQ_BASE_URL, temperature=0.0)
    agent = Agent(
        role="Grounded Learning Agent",
        goal="Answer a student's question using ONLY the provided lecture excerpts.",
        backstory="You never use outside knowledge, never guess, and never help complete assignments "
        "or exams. If the excerpts don't answer the question, you say so plainly.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    task = Task(
        description=(
            "Lecture excerpts (each labeled with its page number):\n{context_block}\n\n"
            "Question: {question}\n\n"
            "Answer using ONLY the excerpts above. If they don't contain enough information to answer, "
            "set is_supported to false and say so in the answer rather than guessing. "
            "Cite the page number(s) you actually used, each with a short supporting snippet. "
            "Do not complete assignments, write exam answers, or produce solutions to problems — "
            "explain concepts only."
        ),
        expected_output="An AskAnswer with the question, the answer, is_supported, and citations.",
        agent=agent,
        output_pydantic=AskAnswer,
    )
    return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)


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
    crew = _build_answer_crew()
    try:
        result = _kickoff_with_retry(crew, context_block, question)
    except Exception as exc:
        raise GroundedLearningError(f"Answer generation failed: {exc}") from exc

    if result.pydantic is None or not isinstance(result.pydantic, AskAnswer):
        return AskAnswer(
            question=question,
            answer="An answer could not be generated for this question. Please try rephrasing it.",
            is_supported=False,
            citations=[],
        )
    return result.pydantic