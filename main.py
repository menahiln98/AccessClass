"""
AccessClass — FastAPI entry point (Portion 1 + Portion 2, one merged app).

Routes:
    GET  /                              upload form
    POST /upload                        registers the document, starts the
                                         pipeline in the background, redirects
                                         to /documents/{id}
    GET  /documents/{id}                shows a "processing" polling page
                                         while the pipeline runs, or the
                                         finished results once status="done"
    GET  /documents/{id}/ask            "Ask This Lecture" question form
    POST /documents/{id}/ask            answers a question, grounded in that
                                         document only
    POST /documents/{id}/mark-confusing adds a page to the revision queue

DEPLOYMENT NOTE: the pipeline can take 1-18 minutes depending on document
length and image count. Every hosting platform's proxy/load balancer
(Railway, Vercel, Fly, etc.) kills HTTP requests held open that long, so
/upload no longer waits for the pipeline to finish. It does the fast part
(upload to Supabase Storage + create the DB record, ~1-2s) synchronously,
then hands the rest of AccessClassFlow's stages to a background thread and
redirects the browser to /documents/{id} right away. That page polls itself
via a plain <meta http-equiv="refresh"> tag (no JavaScript, consistent with
the rest of the UI) until the document's Supabase status reaches "done" or
"error". See flow.py's AccessClassFlow.upload_and_register for the matching
change on the pipeline side.
"""

import os
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
from agents.grounded_learning_agent import GroundedLearningError, ask
from db.supabase_client import (
    SupabaseOperationError,
    add_revision_entry,
    create_document_record,
    get_document,
    get_signed_url,
    list_revision_queue,
    update_document,
    upload_pdf,
)
from dispatch import DispatchError, trigger_pipeline_workflow

# NOTE: `flow` (and therefore crewai/pymupdf/pytesseract via agents.*) is
# imported lazily inside _run_pipeline_in_background below, not here at
# module level. When PIPELINE_EXECUTOR=github_actions this process only
# ever handles routes — including the live CrewAI call in /ask, which
# agents.grounded_learning_agent above still needs — and never runs the
# pipeline itself, so it never needs flow.py's heavier import graph
# (Document Reader's OCR path, the Study-Pack Agent's Edge TTS, etc.).

templates = Jinja2Templates(directory="templates")

# Statuses flow.py's _persist_stage()/_fail() can leave a document row in.
# Anything not in this set means the pipeline is still running.
_TERMINAL_STATUSES = {"done", "error"}

# PDFs saved here must outlive the /upload request, since a background
# thread keeps processing after the response is sent. The thread deletes its
# own file when it finishes (success or failure) — see _run_pipeline_in_background.
_UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "accessclass_uploads")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Fail fast and clearly if required env vars are missing, instead of
    # erroring confusingly on the first upload. Which vars are required
    # depends on PIPELINE_EXECUTOR — see config.validate_env_for_executor().
    config.validate_env_for_executor()
    os.makedirs(_UPLOAD_DIR, exist_ok=True)
    yield


app = FastAPI(title="AccessClass", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


def _study_pack_links(study_pack) -> dict | None:
    """Normalize a StudyPack (live object or Supabase JSONB dict) into signed, playable URLs."""
    if not study_pack:
        return None
    data = study_pack.model_dump() if hasattr(study_pack, "model_dump") else study_pack
    try:
        html_url = get_signed_url(data["html_storage_path"])
        chapters = [
            {
                "title": ch["title"],
                "page_start": ch["page_start"],
                "page_end": ch["page_end"],
                "url": get_signed_url(ch["storage_path"]),
            }
            for ch in data.get("audio_chapters", [])
        ]
        return {"html_url": html_url, "audio_chapters": chapters, "glossary": data.get("glossary", [])}
    except SupabaseOperationError:
        # Non-fatal: the rest of the results page still renders without playable links.
        return None


def _run_pipeline_in_background(local_pdf_path: str, filename: str, document_id: str, storage_path: str) -> None:
    """Runs the remaining pipeline stages (2-6) after upload+registration already happened.

    Only used when PIPELINE_EXECUTOR=thread (the default, for local dev).
    See dispatch.trigger_pipeline_workflow() for the github_actions path.

    Any exception here is already caught and persisted to the document's
    Supabase row (status="error") by flow.py's _fail(), so there's nothing
    further to do with it here except make sure the temp file is cleaned up.
    """
    from flow import AccessClassFlow  # local import — see note near the top of this file

    try:
        flow = AccessClassFlow()
        flow.state.local_pdf_path = local_pdf_path
        flow.state.filename = filename
        flow.state.document_id = document_id
        flow.state.storage_path = storage_path
        flow.kickoff()
    finally:
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/upload", response_class=HTMLResponse)
def upload(request: Request, file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return templates.TemplateResponse(
            request,
            "index.html",
            {"error": "Please upload a .pdf file."},
        )

    contents = file.file.read()
    if not contents:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"error": f"'{file.filename}' is empty. Please upload a non-empty PDF."},
        )

    tmp_path = os.path.join(_UPLOAD_DIR, f"{uuid.uuid4()}.pdf")
    with open(tmp_path, "wb") as tmp_file:
        tmp_file.write(contents)

    # Fast path, kept synchronous: upload to Supabase Storage + create the DB
    # record. This is a couple of seconds, not minutes, so it's safe to keep
    # in the request/response cycle — it's what lets us hand back a real
    # document_id immediately.
    try:
        storage_path = upload_pdf(tmp_path, file.filename)
        document_id = create_document_record(file.filename, storage_path)
    except (SupabaseOperationError, Exception) as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return templates.TemplateResponse(
            request,
            "index.html",
            {"error": f"Could not start processing '{file.filename}': {exc}"},
        )

    # Everything else (OCR/parsing, accessibility audit, subject tagging,
    # explanations, study pack, retrieval indexing) happens off the request.
    # Which happens depends on PIPELINE_EXECUTOR (see config.py):
    if config.get_pipeline_executor() == config.PIPELINE_EXECUTOR_GITHUB_ACTIONS:
        # The web process itself never runs the pipeline — it hands the job
        # to GitHub Actions, which downloads this same PDF from Supabase
        # Storage itself (see worker.py), so the local temp copy isn't needed.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        try:
            trigger_pipeline_workflow(document_id, storage_path, file.filename)
        except DispatchError as exc:
            update_document(document_id, status="error", error_message=f"Could not start processing: {exc}")
    else:
        thread = threading.Thread(
            target=_run_pipeline_in_background,
            args=(tmp_path, file.filename, document_id, storage_path),
            daemon=True,
        )
        thread.start()

    return RedirectResponse(url=f"/documents/{document_id}", status_code=303)


@app.get("/documents/{document_id}", response_class=HTMLResponse)
def view_document(request: Request, document_id: str):
    try:
        record = get_document(document_id)
    except SupabaseOperationError as exc:
        return templates.TemplateResponse(request, "index.html", {"error": str(exc)})

    if record is None:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"error": f"No document found with id '{document_id}'."},
        )

    status = record.get("status", "")

    if status not in _TERMINAL_STATUSES:
        return templates.TemplateResponse(
            request,
            "processing.html",
            {"document_id": document_id, "status": status or "starting"},
        )

    if status == "error":
        return templates.TemplateResponse(
            request,
            "index.html",
            {"error": record.get("error_message") or "Processing failed. Please try uploading again."},
        )

    return templates.TemplateResponse(
        request,
        "results.html",
        {
            "document_id": record["id"],
            "parsed": record.get("parsed_document"),
            "report": record.get("accessibility_report"),
            "tagged": record.get("subject_tagged_document"),
            "explanations": record.get("explanation_document"),
            "study_pack": _study_pack_links(record.get("study_pack")),
            "indexed_chunk_count": record.get("indexed_chunk_count", 0),
            "retrieval_warning": record.get("error_message", "") if record.get("status") == "done" else "",
            "raw_mode": True,
        },
    )


@app.get("/documents/{document_id}/ask", response_class=HTMLResponse)
def ask_form(request: Request, document_id: str):
    return templates.TemplateResponse(request, "ask.html", {"document_id": document_id})


@app.post("/documents/{document_id}/ask", response_class=HTMLResponse)
def ask_question(request: Request, document_id: str, question: str = Form(...)):
    try:
        answer = ask(document_id, question)
    except GroundedLearningError as exc:
        return templates.TemplateResponse(
            request, "ask.html", {"document_id": document_id, "error": str(exc)}
        )
    return templates.TemplateResponse(
        request, "ask.html", {"document_id": document_id, "answer": answer}
    )


@app.post("/documents/{document_id}/mark-confusing", response_class=HTMLResponse)
def mark_confusing(request: Request, document_id: str, page_number: int = Form(...), note: str = Form("")):
    try:
        add_revision_entry(document_id, page_number, note)
        queue = list_revision_queue(document_id)
    except SupabaseOperationError as exc:
        return templates.TemplateResponse(request, "index.html", {"error": str(exc)})
    return templates.TemplateResponse(
        request,
        "revision_queue.html",
        {"document_id": document_id, "queue": queue},
    )