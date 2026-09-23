"""
Triggers Stages 1-6 (the full CrewAI pipeline, unchanged — see flow.py) as a
GitHub Actions job instead of running it in a local background thread.

Used only when config.get_pipeline_executor() == "github_actions" — for a
deployment whose web process can't be trusted to keep a background thread
alive for an 18-minute job (e.g. a free container that scales to zero after
idle time). Local dev leaves PIPELINE_EXECUTOR unset and keeps using
main.py's original threading.Thread path untouched.

The job itself is .github/workflows/process_document.yml, which runs
worker.py — a thin CLI wrapper around the exact same AccessClassFlow this
repo has always used. Nothing about the pipeline changes; only where it runs.
"""

import requests

from config import GITHUB_WORKFLOW_FILE, get_github_repo, get_github_token

GITHUB_API_VERSION = "2022-11-28"
DISPATCH_TIMEOUT_SECONDS = 15


class DispatchError(Exception):
    """Raised when triggering the GitHub Actions worker fails."""


def trigger_pipeline_workflow(
    document_id: str, storage_path: str, filename: str, ref: str = "main"
) -> None:
    """Fire a workflow_dispatch event that runs worker.py for one document.

    This call only enqueues the run — it returns as soon as GitHub accepts
    the dispatch (HTTP 204), well before the job itself starts or finishes,
    which is what lets /upload redirect immediately just like the local
    threading.Thread path always has.
    """
    url = (
        f"https://api.github.com/repos/{get_github_repo()}"
        f"/actions/workflows/{GITHUB_WORKFLOW_FILE}/dispatches"
    )
    headers = {
        "Authorization": f"Bearer {get_github_token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    payload = {
        "ref": ref,
        "inputs": {
            "document_id": document_id,
            "storage_path": storage_path,
            "filename": filename,
        },
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=DISPATCH_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise DispatchError(f"Could not reach the GitHub Actions API: {exc}") from exc

    if response.status_code != 204:
        raise DispatchError(
            f"GitHub rejected the workflow dispatch (HTTP {response.status_code}): {response.text}"
        )
