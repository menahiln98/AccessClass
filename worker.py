"""
Standalone entry point for running AccessClass's full six-stage pipeline
(unchanged — same AccessClassFlow, same CrewAI agents) inside a GitHub
Actions job, for deployments whose web process can't keep a background
thread alive for the full 1-18 minute run.

Triggered by .github/workflows/process_document.yml via workflow_dispatch,
with document_id/storage_path/filename passed in as workflow inputs (see
dispatch.trigger_pipeline_workflow(), called from main.py's /upload route
when PIPELINE_EXECUTOR=github_actions).

This is the direct equivalent of main.py's local threading.Thread path
(_run_pipeline_in_background): it downloads the PDF main.py already
uploaded to Supabase Storage, then kicks off the exact same
AccessClassFlow. Nothing about Stages 1-6 changes — only which process runs
them.
"""

import argparse
import os
import sys
import tempfile

from db.supabase_client import download_pdf, update_document
from flow import AccessClassFlow, PipelineError


def run(document_id: str, storage_path: str, filename: str) -> int:
    local_pdf_path = os.path.join(tempfile.gettempdir(), f"{document_id}.pdf")

    try:
        download_pdf(storage_path, local_pdf_path)
    except Exception as exc:
        message = f"Worker could not download the uploaded PDF: {exc}"
        print(message, file=sys.stderr)
        try:
            update_document(document_id, status="error", error_message=message)
        except Exception:
            pass  # Best-effort — the job log is the fallback record either way.
        return 1

    flow = AccessClassFlow()
    flow.state.local_pdf_path = local_pdf_path
    flow.state.filename = filename
    flow.state.document_id = document_id
    flow.state.storage_path = storage_path

    try:
        flow.kickoff()
    except PipelineError as exc:
        # flow.py's own _fail() already persisted status="error" with a
        # clear message on the document row — nothing further to do here
        # except make the job log (and its exit code) reflect the failure.
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)

    print(f"Pipeline completed for document {document_id}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the AccessClass pipeline for one already-uploaded document."
    )
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--storage-path", required=True)
    parser.add_argument("--filename", required=True)
    args = parser.parse_args()
    return run(args.document_id, args.storage_path, args.filename)


if __name__ == "__main__":
    sys.exit(main())
