"""Persistent FAISS vector store with a JSON manifest of ingested documents.

Layout (per embedding model, so switching providers never mixes vectors):

    storage/vectorstore/<embedding_slug>/
        index.faiss
        index.pkl
        manifest.json      # {doc_id: {doc_id, filename, chunks, size_bytes, uploaded_at}}
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document

from app.config import Settings
from app.services.embeddings import get_embeddings


class DocumentNotFoundError(KeyError):
    """Raised when a doc_id is not present in the manifest."""


class VectorStoreNotReadyError(RuntimeError):
    """Raised when a query arrives before any document has been ingested."""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class VectorStoreManager:
    def __init__(self, settings: Settings, embeddings=None):
        self.settings = settings
        self.embeddings = embeddings if embeddings is not None else get_embeddings(settings)
        self.index_dir: Path = settings.index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path: Path = self.index_dir / "manifest.json"
        self._store = None  # lazy-loaded FAISS instance
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ FAISS
    def _load_store(self):
        if self._store is not None:
            return self._store
        if not (self.index_dir / "index.faiss").exists():
            return None
        from langchain_community.vectorstores import FAISS  # langchain-faiss is a placeholder; community still ships FAISS

        self._store = FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,  # our own index, created locally
        )
        return self._store

    def _save_store(self) -> None:
        if self._store is not None:
            self._store.save_local(str(self.index_dir))

    # --------------------------------------------------------------- manifest
    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Corrupt manifest should not take the service down; start empty
            # but keep a backup for manual recovery.
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
            store = self._load_store()
            if store is None:
                from langchain_community.vectorstores import FAISS  # see _load_store

                store = FAISS.from_documents(docs, self.embeddings, ids=ids)
            else:
                store.add_documents(docs, ids=ids)
            self._store = store
            self._save_store()

            manifest = self._load_manifest()
            manifest[doc_id] = {
                "doc_id": doc_id,
                "filename": filename,
                "chunks": len(docs),
                "size_bytes": size_bytes,
                "uploaded_at": _utcnow_iso(),
            }
            self._save_manifest(manifest)
        return len(docs)

    def similarity_search_with_score(self, query: str, k: int):
        store = self._load_store()
        ntotal = getattr(getattr(store, "index", None), "ntotal", None) if store else None
        if store is None or (ntotal is not None and ntotal == 0):
            raise VectorStoreNotReadyError(
                "No documents have been ingested yet. Upload a document first."
            )
        return store.similarity_search_with_score(query, k=k)

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

            store = self._load_store()
            deleted = 0
            if store is not None and ids:
                try:
                    store.delete(ids=ids)
                    deleted = chunk_count
                except Exception as exc:  # pragma: no cover - version-dependent
                    raise RuntimeError(f"Failed to remove vectors for {doc_id}: {exc}") from exc
                self._save_store()

            del manifest[doc_id]
            self._save_manifest(manifest)
            return deleted


# Lazily created singleton used by the API layer. Constructed on first use so
# importing the app (e.g. in tests) does not hit provider clients.
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
