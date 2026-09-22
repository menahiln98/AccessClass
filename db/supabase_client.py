"""
Supabase Storage + Postgres access for AccessClass Portion 1.

One table (`documents`, see db/schema.sql) holds the uploaded file's
storage path plus each pipeline stage's JSON output, so progress can be
inspected at any point.
"""

import functools
import uuid
from typing import Any, Optional

from supabase import Client, create_client

from config import get_supabase_bucket, get_supabase_key, get_supabase_url

DOCUMENTS_TABLE = "documents"


class SupabaseOperationError(Exception):
    """Raised when a Supabase Storage or Postgres call fails."""


@functools.lru_cache(maxsize=1)
def get_client() -> Client:
    return create_client(get_supabase_url(), get_supabase_key())


def upload_pdf(local_path: str, filename: str) -> str:
    """Upload a local PDF file to the Supabase Storage bucket. Returns its storage path."""
    storage_path = f"{uuid.uuid4().hex}_{filename}"
    try:
        with open(local_path, "rb") as file_obj:
            get_client().storage.from_(get_supabase_bucket()).upload(
                storage_path,
                file_obj,
                {"content-type": "application/pdf"},
            )
    except Exception as exc:
        raise SupabaseOperationError(f"Failed to upload '{filename}' to Supabase Storage: {exc}") from exc
    return storage_path


def upload_bytes(storage_path: str, data: bytes, content_type: str) -> str:
    """Upload raw bytes (e.g. generated HTML) to the bucket at an exact path. Returns that path."""
    try:
        get_client().storage.from_(get_supabase_bucket()).upload(
            storage_path, data, {"content-type": content_type, "upsert": "true"}
        )
    except Exception as exc:
        raise SupabaseOperationError(f"Failed to upload bytes to '{storage_path}': {exc}") from exc
    return storage_path


def upload_file(local_path: str, storage_path: str, content_type: str) -> str:
    """Upload a local file (e.g. a synthesized mp3) to the bucket at an exact path. Returns that path."""
    try:
        with open(local_path, "rb") as file_obj:
            get_client().storage.from_(get_supabase_bucket()).upload(
                storage_path, file_obj.read(), {"content-type": content_type, "upsert": "true"}
            )
    except Exception as exc:
        raise SupabaseOperationError(f"Failed to upload '{local_path}' to '{storage_path}': {exc}") from exc
    return storage_path


def create_document_record(filename: str, storage_path: str) -> str:
    """Insert a new row for this document. Returns the new row's id."""
    try:
        response = (
            get_client()
            .table(DOCUMENTS_TABLE)
            .insert({"filename": filename, "storage_path": storage_path, "status": "uploaded"})
            .execute()
        )
    except Exception as exc:
        raise SupabaseOperationError(f"Failed to create a document record for '{filename}': {exc}") from exc

    if not response.data:
        raise SupabaseOperationError(f"Insert for '{filename}' returned no data from Supabase.")
    return response.data[0]["id"]


def update_document(document_id: str, **fields: Any) -> None:
    """Update arbitrary columns on a document row (status, stage outputs, error_message)."""
    try:
        get_client().table(DOCUMENTS_TABLE).update(fields).eq("id", document_id).execute()
    except Exception as exc:
        raise SupabaseOperationError(f"Failed to update document '{document_id}': {exc}") from exc


def get_signed_url(storage_path: str, expires_in: int = 3600) -> str:
    """Generate a temporary public URL for a private-bucket file (default: 1 hour)."""
    try:
        response = get_client().storage.from_(get_supabase_bucket()).create_signed_url(storage_path, expires_in)
    except Exception as exc:
        raise SupabaseOperationError(f"Failed to create a signed URL for '{storage_path}': {exc}") from exc
    url = response.get("signedUrl") or response.get("signedURL")
    if not url:
        raise SupabaseOperationError(f"Supabase returned no signed URL for '{storage_path}'.")
    return url


def get_document(document_id: str) -> Optional[dict]:
    try:
        response = get_client().table(DOCUMENTS_TABLE).select("*").eq("id", document_id).execute()
    except Exception as exc:
        raise SupabaseOperationError(f"Failed to fetch document '{document_id}': {exc}") from exc
    return response.data[0] if response.data else None


def add_revision_entry(document_id: str, page_number: int, note: str) -> str:
    """Mark a page as confusing. Returns the new revision_queue row's id."""
    try:
        response = (
            get_client()
            .table("revision_queue")
            .insert({"document_id": document_id, "page_number": page_number, "note": note})
            .execute()
        )
    except Exception as exc:
        raise SupabaseOperationError(f"Failed to add a revision queue entry for '{document_id}': {exc}") from exc
    if not response.data:
        raise SupabaseOperationError(f"Revision queue insert for '{document_id}' returned no data.")
    return response.data[0]["id"]


def list_revision_queue(document_id: str) -> list[dict]:
    try:
        response = (
            get_client()
            .table("revision_queue")
            .select("*")
            .eq("document_id", document_id)
            .order("created_at", desc=True)
            .execute()
        )
    except Exception as exc:
        raise SupabaseOperationError(f"Failed to list revision queue for '{document_id}': {exc}") from exc
    return response.data or []
