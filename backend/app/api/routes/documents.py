"""Document ingestion endpoints: upload, list, delete."""

import re
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.schemas import (
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    UploadResponse,
)
from app.services.chunking import chunk_documents
from app.services.document_loader import (
    UnsupportedFileTypeError,
    load_document,
    supported_extensions,
)
from app.services.vectorstore import (
    DocumentNotFoundError,
    get_vectorstore_manager,
)

router = APIRouter(prefix="/documents", tags=["documents"])

_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_filename(filename: str) -> str:
    name = Path(filename).name  # strip any path components
    return _FILENAME_SAFE.sub("_", name) or "upload"


def _to_document_info(entry: dict) -> DocumentInfo:
    return DocumentInfo(
        doc_id=entry["doc_id"],
        filename=entry["filename"],
        chunks=int(entry.get("chunks", 0)),
        size_bytes=int(entry.get("size_bytes", 0)),
        uploaded_at=datetime.fromisoformat(entry["uploaded_at"]),
    )


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    ext = Path(file.filename).suffix.lower()
    if ext not in supported_extensions():
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext or '(none)'}'. Supported: {', '.join(supported_extensions())}",
        )

    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    doc_id = uuid.uuid4().hex
    safe_name = _safe_filename(file.filename)
    stored_path = settings.upload_dir / f"{doc_id}_{safe_name}"

    # Stream to disk in chunks, enforcing the size limit.
    size = 0
    try:
        with stored_path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):  # 1 MiB
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {settings.max_upload_size_mb} MB limit.",
                    )
                out.write(chunk)
    except BaseException:
        stored_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if size == 0:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Parse, chunk and embed (blocking work -> threadpool).
    try:
        chunks = await run_in_threadpool(
            lambda: chunk_documents(
                load_document(stored_path, original_filename=file.filename),
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        )
        manager = get_vectorstore_manager()
        added = await run_in_threadpool(
            lambda: manager.add_documents(
                chunks, doc_id=doc_id, filename=file.filename, size_bytes=size
            )
        )
    except UnsupportedFileTypeError as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except HTTPException:
        stored_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {exc}") from exc

    info = DocumentInfo(
        doc_id=doc_id,
        filename=file.filename,
        chunks=added,
        size_bytes=size,
        uploaded_at=datetime.now().astimezone(),
    )
    return UploadResponse(message="Document ingested successfully.", document=info)


@router.get("", response_model=DocumentListResponse)
def list_documents(settings: Settings = Depends(get_settings)) -> DocumentListResponse:
    manager = get_vectorstore_manager()
    docs = [_to_document_info(e) for e in manager.list_documents()]
    return DocumentListResponse(
        documents=docs,
        total=len(docs),
        embedding_model=settings.embedding_model,
    )


@router.delete("/{doc_id}", response_model=DeleteResponse)
def delete_document(
    doc_id: str,
    settings: Settings = Depends(get_settings),
) -> DeleteResponse:
    manager = get_vectorstore_manager()
    try:
        deleted = manager.delete_document(doc_id)
    except DocumentNotFoundError:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.") from None

    # Best-effort removal of the stored upload as well.
    for candidate in settings.upload_dir.glob(f"{doc_id}_*"):
        try:
            candidate.unlink()
        except OSError:
            pass

    return DeleteResponse(
        message="Document removed from the index.",
        doc_id=doc_id,
        chunks_deleted=deleted,
    )
