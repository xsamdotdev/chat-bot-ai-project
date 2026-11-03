"""Pinecone vector store manager with a local JSON manifest of ingested documents.

Vectors (and their embeddings) live in Pinecone. A small manifest is kept on
disk so the API can list documents and delete their vectors by id:

    storage/vectorstore/<embedding-model>/
        manifest.json   # {doc_id: {doc_id, filename, chunks, size_bytes, ...}}

Chunk ids are `{doc_id}:{chunk_index}`, so deleting a document removes exactly
its vectors from the configured index/namespace.
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document

from app.config import Settings
from app.services.embeddings import get_embeddings
from app.services.pinecone_client import get_pinecone_client, get_pinecone_index


class DocumentNotFoundError(KeyError):
    """Raised when a doc_id is not present in the manifest."""


class VectorStoreNotReadyError(RuntimeError):
    """Raised when a query arrives before any document has been ingested."""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _translate_store_error(exc: Exception) -> Exception:
    """Turn Pinecone's terse index-type errors into actionable configuration errors."""
    text = str(exc).lower()
    if "document schema" in text:
        return RuntimeError(
            "The configured Pinecone index uses a document schema (documents API), but this "
            "backend stores raw vectors. Recreate the index as a standard vector index "
            "(dimension 1024, metric cosine, no integrated embedding / schema), then point "
            "PINECONE_INDEX or PINECONE_HOST at it. See README → Troubleshooting."
        )
    if "dimension" in text and ("does not match" in text or "mismatch" in text):
        return RuntimeError(
            "The Pinecone index dimension does not match the embedding model. "
            "llama-text-embed-v2 and multilingual-e5-large output 1024-dim vectors — "
            "create the index with dimension 1024 (metric cosine) or change "
            "PINECONE_EMBEDDING_MODEL. See README → Troubleshooting."
        )
    return exc


class VectorStoreManager:
    def __init__(self, settings: Settings, embeddings=None):
        self.settings = settings
        self._embeddings = embeddings
        self.index_dir: Path = settings.index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path: Path = self.index_dir / "manifest.json"
        self._client = None
        self._index = None
        self._store = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- plumbing
    @property
    def embeddings(self):
        if self._embeddings is None:
            if self.settings.provider == "fake":
                self._embeddings = get_embeddings(self.settings)
            else:
                self._embeddings = get_embeddings(self.settings, client=self._pinecone_client())
        return self._embeddings

    def _pinecone_client(self):
        if self._client is None:
            self._client = get_pinecone_client(self.settings)
        return self._client

    def _pinecone_index(self):
        if self._index is None:
            self._index = get_pinecone_index(self.settings, client=self._pinecone_client())
        return self._index

    def _vector_store(self):
        if self._store is None:
            from app.services.pinecone_store import PineconeVectorStore

            self._store = PineconeVectorStore(
                index=self._pinecone_index(),
                embedding=self.embeddings,
                namespace=self.settings.pinecone_namespace,
            )
        return self._store

    # --------------------------------------------------------------- manifest
    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt manifest should not take the service down; keep a backup
            # for manual recovery and start empty.
            backup = self.manifest_path.with_suffix(".json.bak")
            try:
                os.replace(self.manifest_path, backup)
            except OSError:
                pass
            return {}

    def _save_manifest(self, manifest: dict) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp, self.manifest_path)

    # ------------------------------------------------------------------- API
    def add_documents(
        self,
        docs: list[Document],
        *,
        doc_id: str,
        filename: str,
        size_bytes: int,
    ) -> int:
        if not docs:
            raise ValueError("No content could be extracted from the document.")

        for i, doc in enumerate(docs):
            doc.metadata["doc_id"] = doc_id
            doc.metadata.setdefault("chunk_index", i)

        ids = [f"{doc_id}:{doc.metadata['chunk_index']}" for doc in docs]

        with self._lock:
            try:
                self._vector_store().add_documents(docs, ids=ids)
            except Exception as exc:
                raise _translate_store_error(exc) from exc

            manifest = self._load_manifest()
            manifest[doc_id] = {
                "doc_id": doc_id,
                "filename": filename,
                "chunks": len(docs),
                "size_bytes": size_bytes,
                "uploaded_at": _utcnow_iso(),
                "embedding_model": self.settings.embedding_model,
                "pinecone_index": self.settings.pinecone_index,
                "pinecone_host": self.settings.pinecone_host,
                "pinecone_namespace": self.settings.pinecone_namespace,
            }
            self._save_manifest(manifest)
        return len(docs)

    def similarity_search_with_score(self, query: str, k: int):
        manifest = self._load_manifest()
        if not manifest:
            raise VectorStoreNotReadyError(
                "No documents have been ingested yet. Upload a document first."
            )
        try:
            return self._vector_store().similarity_search_with_score(query, k=k)
        except Exception as exc:
            raise _translate_store_error(exc) from exc

    def list_documents(self) -> list[dict]:
        manifest = self._load_manifest()
        return sorted(manifest.values(), key=lambda d: d.get("uploaded_at", ""), reverse=True)

    def count_documents(self) -> int:
        return len(self._load_manifest())

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            manifest = self._load_manifest()
            entry = manifest.get(doc_id)
            if entry is None:
                raise DocumentNotFoundError(doc_id)

            chunk_count = int(entry.get("chunks", 0))
            ids = [f"{doc_id}:{i}" for i in range(chunk_count)]
            if ids:
                try:
                    self._vector_store().delete(ids)
                except Exception as exc:
                    raise _translate_store_error(exc) from exc

            del manifest[doc_id]
            self._save_manifest(manifest)
            return len(ids)


# Lazily created singleton used by the API layer. Constructed on first use so
# importing the app (e.g. in tests) does not hit Pinecone.
_manager: VectorStoreManager | None = None
_manager_lock = threading.Lock()


def get_vectorstore_manager() -> VectorStoreManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                from app.config import get_settings

                _manager = VectorStoreManager(get_settings())
    return _manager


def reset_vectorstore_manager() -> None:
    """Testing hook: drop the cached singleton."""
    global _manager
    with _manager_lock:
        _manager = None
