"""Pydantic request/response models for the API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- health
class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    provider: str            # gemini | fake
    chatbot_name: str        # persona name (default: Lisa)
    chat_model: str
    embedding_model: str     # Pinecone hosted embedding model
    embedding_provider: str  # always "pinecone"
    vector_store: str        # always "pinecone"
    documents_indexed: int


class DiagnosticsResponse(BaseModel):
    """Everything needed to explain 'documents uploaded but no sources returned'."""

    provider: str
    chat_model: str
    embedding_model: str
    pinecone_index: str | None = None
    pinecone_host: str | None = None
    pinecone_namespace: str
    documents_indexed: int
    manifest_documents: list[dict]
    index_stats: dict | None = None  # total_vector_count + per-namespace counts
    warnings: list[str] = Field(default_factory=list)


# ------------------------------------------------------------- documents
class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    size_bytes: int
    uploaded_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int
    embedding_model: str


class UploadResponse(BaseModel):
    message: str
    document: DocumentInfo


class DeleteResponse(BaseModel):
    message: str
    doc_id: str
    chunks_deleted: int


# ------------------------------------------------------------------ chat
class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, gt=0, le=20)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=40)


class Source(BaseModel):
    doc_id: str | None = None
    filename: str
    chunk_index: int | None = None
    page: int | None = None  # 0-based page number for PDFs, when available
    snippet: str
    score: float | None = None  # Pinecone cosine similarity (higher = more similar)


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    provider: str
    model: str
    chatbot_name: str = "Lisa"


# ---------------------------------------------------------------- errors
class ErrorDetail(BaseModel):
    detail: str
