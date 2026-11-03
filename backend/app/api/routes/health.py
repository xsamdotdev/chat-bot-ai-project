"""Liveness / readiness probe."""

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.schemas import HealthResponse
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
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        embedding_provider="pinecone",
        vector_store="pinecone",
        documents_indexed=indexed,
    )
