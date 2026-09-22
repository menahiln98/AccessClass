"""
AccessClass — FastAPI entry point (Portion 1 + Portion 2, one merged app).

Routes:
    GET  /                               upload form
    POST /upload                         runs the full pipeline on the uploaded PDF, shows results
    GET  /documents/{id}                 re-view a previously processed document from Supabase
    GET  /documents/{id}/ask             "Ask This Lecture" question form
    POST /documents/{id}/ask             answers a question, grounded in that document only
    POST /documents/{id}/mark-confusing  adds a page to the revision queue
"""

import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
from agents.document_reader import DocumentReadError
from agents.grounded_learning_agent import GroundedLearningError, ask
from db.supabase_client import (
    SupabaseOperationError,
    add_revision_entry,
    get_document,
    get_signed_url,
    list_revision_queue,
)
from flow import PipelineError, run_pipeline

templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Fail fast and clearly if required env vars are missing, instead of
    # erroring confusingly on the first upload. Now validates Portion 2's
    # Gemini/Qdrant credentials too, since this is one merged app.
    config.validate_portion2_env()
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

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name

        flow = run_pipeline(tmp_path, file.filename)

        return templates.TemplateResponse(
            request,
            "results.html",
            {
                "document_id": flow.state.document_id,
                "parsed": flow.state.parsed_document,
                "report": flow.state.accessibility_report,
                "tagged": flow.state.subject_tagged_document,
                "explanations": flow.state.explanation_document,
                "study_pack": _study_pack_links(flow.state.study_pack),
                "indexed_chunk_count": flow.state.indexed_chunk_count,
                "retrieval_warning": flow.state.retrieval_warning,
            },
        )

    except (DocumentReadError, PipelineError, SupabaseOperationError) as exc:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"error": str(exc)},
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


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
