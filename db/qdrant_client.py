"""
Qdrant Storage for Stage 6 (Grounded Learning Agent).

Uses a single shared collection (name from config.get_qdrant_collection())
with a "document_id" payload field on every point, so a search can be
filtered down to one uploaded lecture — this lets the same Qdrant
cluster/collection safely hold chunks from many different documents (and,
per the person's own setup, coexist with unrelated projects using the same
Qdrant account, as long as the collection name doesn't clash).

Every network call retries automatically (3 attempts, exponential backoff)
before giving up, since transient connection resets against a cloud
endpoint are common and not actual bugs — see QDRANT_TIMEOUT_SECONDS and
the @retry decorator below.
"""

import functools
import logging
import uuid

from qdrant_client import QdrantClient, models
from tenacity import retry, stop_after_attempt, wait_exponential

from config import EMBEDDING_DIMENSIONS, get_qdrant_api_key, get_qdrant_collection, get_qdrant_url
from models.qa import LectureChunk

QDRANT_TIMEOUT_SECONDS = 30
logger = logging.getLogger(__name__)

# A Cloud cluster can be temporarily unreachable (for example, a TLS handshake
# may be closed by the remote host).  For a local demonstration, keep retrieval
# usable by falling back to an in-process Qdrant instance.  Its data lasts only
# until the FastAPI server stops; a healthy cloud connection remains the normal,
# persistent path.
_local_fallback_client: QdrantClient | None = None

_retry_network_call = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)


class QdrantOperationError(Exception):
    """Raised when a Qdrant call fails after all retry attempts."""


@functools.lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(url=get_qdrant_url(), api_key=get_qdrant_api_key(), timeout=QDRANT_TIMEOUT_SECONDS)


def _active_client() -> QdrantClient:
    return _local_fallback_client or get_client()


def _start_local_fallback(collection: str) -> QdrantClient:
    global _local_fallback_client
    if _local_fallback_client is None:
        local_client = QdrantClient(":memory:")
        _create_collection(local_client, collection)
        _local_fallback_client = local_client
        logger.warning(
            "Qdrant Cloud is unavailable; using temporary in-memory retrieval. "
            "Keep the server running to use Ask This Lecture for this upload."
        )
    return _local_fallback_client


@_retry_network_call
def _collection_exists(client: QdrantClient, collection: str) -> bool:
    return client.collection_exists(collection)


@_retry_network_call
def _create_collection(client: QdrantClient, collection: str) -> None:
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=EMBEDDING_DIMENSIONS, distance=models.Distance.COSINE),
    )


def ensure_collection() -> None:
    """Create the collection if it doesn't already exist. Safe to call every run."""
    collection = get_qdrant_collection()
    try:
        client = _active_client()
        if not _collection_exists(client, collection):
            _create_collection(client, collection)
    except Exception as exc:
        if _local_fallback_client is None:
            try:
                _start_local_fallback(collection)
                return
            except Exception:
                pass
        raise QdrantOperationError(
            f"Failed to ensure Qdrant collection '{collection}' after 3 attempts: {exc}"
        ) from exc


@_retry_network_call
def _upsert(client: QdrantClient, collection: str, points: list) -> None:
    client.upsert(collection_name=collection, points=points)


def upsert_chunks(chunks: list[LectureChunk], embeddings: list[list[float]]) -> None:
    if len(chunks) != len(embeddings):
        raise QdrantOperationError("Number of chunks and embeddings must match.")
    points = [
        models.PointStruct(
            # Qdrant accepts an unsigned integer or a UUID point ID, not an
            # arbitrary string such as "<document-id>-p1".  UUID5 keeps the
            # ID deterministic, so re-indexing the same page updates it.
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
            vector=embedding,
            payload={
                "document_id": chunk.document_id,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "text": chunk.text,
            },
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    try:
        _upsert(_active_client(), get_qdrant_collection(), points)
    except Exception as exc:
        raise QdrantOperationError(
            f"Failed to upsert {len(points)} chunk(s) to Qdrant after 3 attempts: {exc}"
        ) from exc


@_retry_network_call
def _query(client: QdrantClient, collection: str, query_vector: list[float], document_id: str, top_k: int):
    return client.query_points(
        collection_name=collection,
        query=query_vector,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
        ),
        limit=top_k,
    )


def search_chunks(document_id: str, query_vector: list[float], top_k: int = 5) -> list[dict]:
    """Returns up to top_k payload dicts ({page_start, page_end, text}) for this document, most relevant first."""
    try:
        response = _query(_active_client(), get_qdrant_collection(), query_vector, document_id, top_k)
    except Exception as exc:
        raise QdrantOperationError(f"Qdrant search failed for document '{document_id}' after 3 attempts: {exc}") from exc

    return [
        {
            "page_start": point.payload["page_start"],
            "page_end": point.payload["page_end"],
            "text": point.payload["text"],
            "score": point.score,
        }
        for point in response.points
    ]
