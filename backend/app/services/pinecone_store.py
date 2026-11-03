"""Pinecone vector store backed by the official `pinecone` SDK.

Why not the LangChain Pinecone package? On PyPI, `langchain-pinecone` is an
abandoned 0.0.1 stub (pins numpy<2, which cannot build on modern Python), and
the integration was removed from langchain-community. This module implements
the minimal LangChain `VectorStore` surface directly — no LangChain Pinecone
package required.
"""

import uuid
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

_UPSERT_BATCH = 100  # Pinecone upsert limit per call
_DELETE_BATCH = 1000  # Pinecone delete-by-id limit per call
_EMBED_BATCH = 64


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Pinecone metadata must be flat str/number/bool (or list of str). Drop the rest."""
    clean: dict[str, Any] = {}
    for key, value in (meta or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            clean[key] = value
    return clean


class PineconeVectorStore(VectorStore):
    """Minimal LangChain vector store for a single Pinecone index (+ namespace)."""

    def __init__(self, *, index, embedding: Embeddings, namespace: str = "", text_key: str = "text"):
        self.index = index  # pinecone.Index instance
        self._embedding = embedding
        self.namespace = namespace
        self.text_key = text_key

    # ------------------------------------------------------------- plumbing
    @property
    def embedding_function(self) -> Embeddings:
        return self._embedding

    def _select_relevance_score_fn(self):
        return lambda score: score

    def _embed_batches(self, texts: list[str]) -> Iterable[list[list[float]]]:
        for start in range(0, len(texts), _EMBED_BATCH):
            yield self._embedding.embed_documents(texts[start : start + _EMBED_BATCH])

    # ------------------------------------------------------------- writing
    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs,
    ) -> list[str]:
        if not texts:
            return []
        ids = ids or [uuid.uuid4().hex for _ in texts]
        metadatas = metadatas or [{} for _ in texts]

        embedded_texts: list[list[float]] = []
        for batch in self._embed_batches(texts):
            embedded_texts.extend(batch)

        vectors = []
        for doc_id, text, meta, values in zip(ids, texts, metadatas, embedded_texts):
            metadata = _sanitize_metadata(meta)
            metadata[self.text_key] = text[:8000]  # keep metadata well under limits
            vectors.append((doc_id, values, metadata))

        for start in range(0, len(vectors), _UPSERT_BATCH):
            self.index.upsert(vectors=vectors[start : start + _UPSERT_BATCH], namespace=self.namespace)
        return ids

    def delete(self, ids: list[str]) -> None:
        for start in range(0, len(ids), _DELETE_BATCH):
            self.index.delete(ids=ids[start : start + _DELETE_BATCH], namespace=self.namespace)

    # ----------------------------------------------------------------- ctors
    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, ids=None, **kwargs):
        index = kwargs.get("index")
        if index is None:
            raise ValueError("PineconeVectorStore.from_texts requires an `index=` kwarg (a pinecone Index).")
        store = cls(index=index, embedding=embedding, namespace=kwargs.get("namespace", ""))
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    # ------------------------------------------------------------- reading
    def similarity_search(self, query: str, k: int = 4, filter: dict | None = None, **kwargs):  # noqa: A002
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k, filter=filter, **kwargs)]

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict | None = None,  # noqa: A002 - langchain-core signature
        **kwargs,
    ) -> list[Document]:
        return [doc for doc, _ in self._query(embedding, k=k, filter=filter)]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,  # noqa: A002
        **kwargs,
    ) -> list[tuple[Document, float]]:
        return self._query(self._embedding.embed_query(query), k=k, filter=filter)

    def _query(self, vector: list[float], k: int, filter: dict | None) -> list[tuple[Document, float]]:  # noqa: A002
        response = self.index.query(
            vector=vector,
            top_k=k,
            include_metadata=True,
            namespace=self.namespace,
            filter=filter,
        )
        results: list[tuple[Document, float]] = []
        for match in getattr(response, "matches", []) or []:
            metadata = dict(getattr(match, "metadata", None) or {})
            text = metadata.pop(self.text_key, "")
            results.append((Document(page_content=text, metadata=metadata), float(getattr(match, "score", 0.0) or 0.0)))
        return results
