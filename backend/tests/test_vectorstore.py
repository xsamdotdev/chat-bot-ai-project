"""VectorStoreManager unit tests (Pinecone backend, fully offline)."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.config import Settings
from app.services import vectorstore as vs


def _settings(tmp_path) -> Settings:
    return Settings(
        provider="fake",
        _env_file=None,
        pinecone_api_key="test-key",
        pinecone_index="test-index",
        upload_dir=tmp_path / "uploads",
        vectorstore_dir=tmp_path / "vectorstore",
        chunk_size=100,
        chunk_overlap=10,
    )


def test_search_add_delete_flow(tmp_path, fake_pinecone):
    settings = _settings(tmp_path)
    manager = vs.VectorStoreManager(settings, embeddings=DeterministicFakeEmbedding(size=64))

    with pytest.raises(vs.VectorStoreNotReadyError):
        manager.similarity_search_with_score("anything", k=1)

    docs = [
        Document(page_content=f"chunk number {i} about topic {i % 3}", metadata={"chunk_index": i, "source": "t.txt"})
        for i in range(6)
    ]
    added = manager.add_documents(docs, doc_id="abc123", filename="t.txt", size_bytes=123)
    assert added == 6

    # Vectors landed in Pinecone with chunk ids.
    index = fake_pinecone.instances[-1]._index
    assert set(index.vectors) == {f"abc123:{i}" for i in range(6)}

    results = manager.similarity_search_with_score("topic 1", k=2)
    assert len(results) == 2
    assert all(doc.metadata["doc_id"] == "abc123" for doc, _ in results)

    entry = manager.list_documents()[0]
    assert entry["doc_id"] == "abc123"
    assert entry["embedding_model"] == "fake-embedding"
    assert entry["pinecone_index"] == "test-index"

    deleted = manager.delete_document("abc123")
    assert deleted == 6
    assert index.vectors == {}
    assert manager.list_documents() == []
    assert manager.count_documents() == 0
    with pytest.raises(vs.VectorStoreNotReadyError):
        manager.similarity_search_with_score("topic", k=1)


def test_delete_unknown_raises(tmp_path, fake_pinecone):
    settings = _settings(tmp_path)
    manager = vs.VectorStoreManager(settings, embeddings=DeterministicFakeEmbedding(size=64))
    with pytest.raises(vs.DocumentNotFoundError):
        manager.delete_document("nope")


def test_empty_documents_rejected(tmp_path, fake_pinecone):
    settings = _settings(tmp_path)
    manager = vs.VectorStoreManager(settings, embeddings=DeterministicFakeEmbedding(size=64))
    with pytest.raises(ValueError):
        manager.add_documents([], doc_id="x", filename="x.txt", size_bytes=0)
