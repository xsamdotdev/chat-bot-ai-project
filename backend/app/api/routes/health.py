"""Liveness / readiness probe + diagnostics."""

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.schemas import DiagnosticsResponse, HealthResponse
from app.services.vectorstore import get_vectorstore_manager

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    try:
        indexed = get_vectorstore_manager().count_documents()
    except Exception:
        indexed = -1  # store unreachable; report degraded rather than crash
    return HealthResponse(
        status="ok",
        provider=settings.provider,
        chatbot_name=settings.chatbot_display_name,
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        embedding_provider="pinecone",
        vector_store="pinecone",
        documents_indexed=indexed,
    )


def _namespace_vector_count(stats: dict, namespace: str) -> int:
    namespaces = stats.get("namespaces") or {}
    if namespace in namespaces:
        return int(namespaces[namespace] or 0)
    # Some SDK versions report Pinecone's default namespace as "__default__".
    if namespace == "" and "__default__" in namespaces:
        return int(namespaces["__default__"] or 0)
    return 0


@router.get("/diagnostics", response_model=DiagnosticsResponse)
def diagnostics(settings: Settings = Depends(get_settings)) -> DiagnosticsResponse:
    """Compare the local manifest with the live Pinecone index.

    Helps diagnose the classic "upload succeeded but /api/chat returns no sources"
    case: documents ingested into another index/namespace, or vectors missing.
    """
    manager = get_vectorstore_manager()
    documents = manager.list_documents()
    warnings: list[str] = []

    for doc in documents:
        name = doc.get("filename") or doc.get("doc_id")
        if (
            doc.get("pinecone_index") != settings.pinecone_index
            or doc.get("pinecone_host") != settings.pinecone_host
        ):
            warnings.append(
                f"'{name}' was ingested into index={doc.get('pinecone_index')!r} "
                f"host={doc.get('pinecone_host')!r}, but the app now uses "
                f"index={settings.pinecone_index!r} host={settings.pinecone_host!r}. "
                "Re-upload the document (or restore the original settings)."
            )
        if doc.get("pinecone_namespace") != settings.pinecone_namespace:
            warnings.append(
                f"'{name}' was ingested into namespace {doc.get('pinecone_namespace')!r}, "
                f"but the app now queries namespace {settings.pinecone_namespace!r}."
            )
        if doc.get("embedding_model") != settings.embedding_model:
            warnings.append(
                f"'{name}' was embedded with {doc.get('embedding_model')!r}, but the app now "
                f"uses {settings.embedding_model!r}. Re-upload the document."
            )

    index_stats = None
    try:
        index_stats = manager.describe_index_stats()
    except Exception as exc:
        warnings.append(f"Could not read Pinecone index stats: {exc}")

    if documents and index_stats is not None:
        total = int(index_stats.get("total_vector_count") or 0)
        if total == 0:
            warnings.append(
                "Pinecone reports 0 vectors in this index, even though uploads returned 201. "
                "The write may have gone to a different index/host — check PINECONE_INDEX / "
                "PINECONE_HOST in .env."
            )
        elif _namespace_vector_count(index_stats, settings.pinecone_namespace) == 0:
            namespaces = sorted((index_stats.get("namespaces") or {}).keys())
            warnings.append(
                f"Pinecone has {total} vector(s), but none in namespace "
                f"{settings.pinecone_namespace!r}. Namespaces that do have vectors: "
                f"{namespaces}."
            )

    return DiagnosticsResponse(
        provider=settings.provider,
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        pinecone_index=settings.pinecone_index,
        pinecone_host=settings.pinecone_host,
        pinecone_namespace=settings.pinecone_namespace,
        documents_indexed=len(documents),
        manifest_documents=documents,
        index_stats=index_stats,
        warnings=warnings,
    )
