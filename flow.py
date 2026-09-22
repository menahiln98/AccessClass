"""
Full AccessClass pipeline: Document Reader -> Accessibility Auditor ->
Subject Interpreter -> Explanation Agent -> Study-Pack Agent -> Grounded
Learning Agent (indexing only; answering questions is a separate on-demand
call, see main.py's /ask route and agents/grounded_learning_agent.ask()).

This is one Flow, not two — Portion 2 extended the same AccessClassFlow
built in Portion 1 with three more @listen steps rather than creating a
second, separate Flow, since each new stage depends on the previous
stages' output already existing in the shared state.
"""

from typing import Optional

from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel, ConfigDict

from agents import (
    accessibility_auditor,
    document_reader,
    explanation_agent,
    grounded_learning_agent,
    study_pack_agent,
    subject_interpreter,
)
from db.supabase_client import SupabaseOperationError, create_document_record, update_document, upload_pdf
from models.audit import AccessibilityReport
from models.document import ParsedDocument
from models.explanation import ExplanationDocument
from models.study_pack import StudyPack
from models.subject import SubjectTaggedDocument


class PipelineError(Exception):
    """A single error type the FastAPI layer can catch, wrapping the real cause."""


class DocumentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    local_pdf_path: str = ""
    filename: str = ""
    document_id: str = ""
    storage_path: str = ""
    parsed_document: Optional[ParsedDocument] = None
    accessibility_report: Optional[AccessibilityReport] = None
    subject_tagged_document: Optional[SubjectTaggedDocument] = None
    explanation_document: Optional[ExplanationDocument] = None
    study_pack: Optional[StudyPack] = None
    indexed_chunk_count: int = 0
    retrieval_warning: str = ""


def _fail(document_id: Optional[str], stage: str, exc: Exception) -> None:
    """Record the failure on the document row (if it exists yet) and raise a clear error."""
    message = f"Stage '{stage}' failed: {exc}"
    if document_id:
        try:
            update_document(document_id, status="error", error_message=message)
        except Exception:
            pass  # Persisting the failure is best-effort; the raise below is what matters.
    raise PipelineError(message) from exc


# These fields were introduced after the original Portion 1 schema.  Keeping
# this compatibility path means an existing database can still process and
# display a document while the one-time SQL migration is being applied.
_PORTION_2_COLUMNS = {"explanation_document", "study_pack", "indexed_chunk_count"}


def _persist_stage(document_id: str, status: str, **fields) -> None:
    """Persist a stage without blocking a live result on an unmigrated legacy schema."""
    try:
        update_document(document_id, status=status, **fields)
    except SupabaseOperationError as exc:
        missing_column = "Could not find" in str(exc) and "schema cache" in str(exc)
        if missing_column and _PORTION_2_COLUMNS.intersection(fields):
            # ``status`` exists in the original schema.  The full result remains
            # available in the current upload response; reloading it later needs
            # the migration in db/schema.sql.
            update_document(document_id, status=status)
            return
        raise


class AccessClassFlow(Flow[DocumentState]):
    @start()
    def upload_and_register(self) -> str:
        try:
            storage_path = upload_pdf(self.state.local_pdf_path, self.state.filename)
            document_id = create_document_record(self.state.filename, storage_path)
        except Exception as exc:
            _fail(None, "upload_and_register", exc)
        self.state.storage_path = storage_path
        self.state.document_id = document_id
        return document_id

    @listen(upload_and_register)
    def read_document(self, _document_id: str) -> ParsedDocument:
        try:
            parsed = document_reader.run(self.state.local_pdf_path, self.state.filename)
            _persist_stage(
                self.state.document_id,
                status="parsed",
                parsed_document=parsed.model_dump(),
            )
        except Exception as exc:
            _fail(self.state.document_id, "read_document", exc)
        self.state.parsed_document = parsed
        return parsed

    @listen(read_document)
    def audit_document(self, parsed: ParsedDocument) -> AccessibilityReport:
        try:
            report = accessibility_auditor.run(parsed)
            _persist_stage(
                self.state.document_id,
                status="audited",
                accessibility_report=report.model_dump(),
            )
        except Exception as exc:
            _fail(self.state.document_id, "audit_document", exc)
        self.state.accessibility_report = report
        return report

    @listen(audit_document)
    def interpret_subjects(self, _report: AccessibilityReport) -> SubjectTaggedDocument:
        try:
            tagged = subject_interpreter.run(self.state.parsed_document)
            _persist_stage(
                self.state.document_id,
                status="tagged",
                subject_tagged_document=tagged.model_dump(),
            )
        except Exception as exc:
            _fail(self.state.document_id, "interpret_subjects", exc)
        self.state.subject_tagged_document = tagged
        return tagged

    @listen(interpret_subjects)
    def explain_content(self, tagged: SubjectTaggedDocument) -> ExplanationDocument:
        try:
            explanation_doc = explanation_agent.run(self.state.parsed_document, self.state.local_pdf_path, tagged)
            _persist_stage(
                self.state.document_id,
                status="explained",
                explanation_document=explanation_doc.model_dump(),
            )
        except Exception as exc:
            _fail(self.state.document_id, "explain_content", exc)
        self.state.explanation_document = explanation_doc
        return explanation_doc

    @listen(explain_content)
    def build_study_pack(self, explanation_doc: ExplanationDocument) -> StudyPack:
        try:
            pack = study_pack_agent.run(self.state.parsed_document, self.state.document_id, explanation_doc)
            _persist_stage(
                self.state.document_id,
                status="packed",
                study_pack=pack.model_dump(),
            )
        except Exception as exc:
            _fail(self.state.document_id, "build_study_pack", exc)
        self.state.study_pack = pack
        return pack

    @listen(build_study_pack)
    def index_for_retrieval(self, _pack: StudyPack) -> int:
        try:
            count = grounded_learning_agent.index_document(self.state.document_id, self.state.parsed_document)
            _persist_stage(self.state.document_id, status="done", indexed_chunk_count=count)
        except Exception as exc:
            # Retrieval is an enhancement, not a reason to discard a completed
            # accessibility report and study pack.  Qdrant outages/credentials
            # are surfaced in the results page and the document remains usable.
            warning = f"Ask This Lecture is temporarily unavailable: {exc}"
            self.state.retrieval_warning = warning
            try:
                update_document(self.state.document_id, status="done", error_message=warning)
            except Exception:
                pass
            self.state.indexed_chunk_count = 0
            return 0
        self.state.indexed_chunk_count = count
        return count


def run_pipeline(local_pdf_path: str, filename: str) -> AccessClassFlow:
    """Run the full pipeline (Stages 1-6's indexing step) and return the completed flow."""
    flow = AccessClassFlow()
    flow.kickoff(inputs={"local_pdf_path": local_pdf_path, "filename": filename})
    return flow
