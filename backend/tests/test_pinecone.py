"""Pinecone wiring: store adapter, hosted-inference embeddings, config errors."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.config import Settings
from app.services import vectorstore as vs
from app.services.embeddings import PineconeInferenceEmbeddings, get_embeddings
from app.services.pinecone_store import PineconeVectorStore, _sanitize_metadata
from fakes import FakeIndex, FakePinecone


def _settings(tmp_path, **overrides) -> Settings:
    base = {
        "_env_file": None,
        "provider": "fake",
        "pinecone_api_key": "test-key",
        "pinecone_index": "test-index",
        "pinecone_namespace": "",
        "upload_dir": tmp_path / "uploads",
        "vectorstore_dir": tmp_path / "vectorstore",
    }
    return Settings(**{**base, **overrides})


def _doc(text: str = "hello"):
    from langchain_core.documents import Document

    return Document(page_content=text, metadata={})


# ------------------------------------------------------------------ store
def test_sanitize_metadata_drops_none_and_nested():
    assert _sanitize_metadata({"a": 1, "b": None, "c": {"nested": 1}, "d": "x", "e": ["ok"], "f": [1, 2]}) == {
        "a": 1,
        "d": "x",
        "e": ["ok"],
    }


def test_store_add_search_delete_and_namespace_isolation():
    index = FakeIndex()
    embedding = DeterministicFakeEmbedding(size=16)
    store_a = PineconeVectorStore(index=index, embedding=embedding, namespace="a")
    store_b = PineconeVectorStore(index=index, embedding=embedding, namespace="b")

    ids = store_a.add_texts(
        ["alpha chunk", "beta chunk"],
        metadatas=[{"source": "a.txt", "page": None}, {"source": "a.txt"}],
        ids=["a:0", "a:1"],
    )
    assert ids == ["a:0", "a:1"]
    # None-valued metadata is dropped (Pinecone rejects nulls).
    assert "page" not in index.vectors["a:0"][1]
    # The chunk text is stored so it can be returned as page_content.
    assert index.vectors["a:0"][1]["text"] == "alpha chunk"

    assert store_b.similarity_search("alpha", k=5) == []
    hits = store_a.similarity_search_with_score("alpha", k=2)
    assert len(hits) == 2
    assert hits[0][0].page_content in {"alpha chunk", "beta chunk"}
    assert hits[0][0].metadata["source"] == "a.txt"

    store_a.delete(["a:0", "a:1"])
    assert index.vectors == {}
    assert index.deleted[-1] == (["a:0", "a:1"], "a")


# ------------------------------------------------------------- embeddings
def test_inference_embeddings_use_passage_and_query_input_types():
    class Client:
        def __init__(self):
            self.inference = FakePinecone().inference

    client = Client()
    embeddings = PineconeInferenceEmbeddings(client=client, model="llama-text-embed-v2")

    docs = embeddings.embed_documents(["first passage", "second passage"])
    query = embeddings.embed_query("a question")

    assert len(docs) == 2 and len(docs[0]) == 1024
    assert len(query) == 1024
    calls = client.inference.calls
    assert [call["parameters"]["input_type"] for call in calls] == ["passage", "query"]
    assert all(call["model"] == "llama-text-embed-v2" for call in calls)


def test_get_embeddings_uses_pinecone_inference_for_gemini():
    settings = Settings(
        _env_file=None,
        provider="gemini",
        pinecone_api_key="test-key",
        pinecone_index="test-index",
        pinecone_embedding_model="multilingual-e5-large",
    )
    client = FakePinecone()
    embeddings = get_embeddings(settings, client=client)
    assert isinstance(embeddings, PineconeInferenceEmbeddings)
    assert embeddings.model == "multilingual-e5-large"
    assert len(embeddings.embed_query("hello")) == 1024


# ----------------------------------------------------------------- errors
def test_missing_pinecone_key_raises_clear_error(tmp_path, fake_pinecone):
    settings = _settings(tmp_path, pinecone_api_key=None)
    manager = vs.VectorStoreManager(settings, embeddings=DeterministicFakeEmbedding(size=8))
    with pytest.raises(RuntimeError, match="PINECONE_API_KEY"):
        manager.add_documents([_doc()], doc_id="d1", filename="x.txt", size_bytes=1)


def test_missing_pinecone_index_raises_clear_error(tmp_path, fake_pinecone):
    settings = _settings(tmp_path, pinecone_index=None)
    manager = vs.VectorStoreManager(settings, embeddings=DeterministicFakeEmbedding(size=8))
    with pytest.raises(RuntimeError, match="PINECONE_INDEX"):
        manager.add_documents([_doc()], doc_id="d1", filename="x.txt", size_bytes=1)


def test_pinecone_host_connects_by_url(tmp_path, fake_pinecone):
    host = "https://chat-bot-idx-b8h006u.svc.aped-4627-b74a.pinecone.io"
    settings = _settings(tmp_path, pinecone_index=None, pinecone_host=host)
    manager = vs.VectorStoreManager(settings, embeddings=DeterministicFakeEmbedding(size=8))

    manager.add_documents([_doc("hello")], doc_id="d1", filename="x.txt", size_bytes=1)

    client = fake_pinecone.instances[-1]
    assert client.index_host == host
    assert client.index_name is None
    assert manager.list_documents()[0]["pinecone_host"] == host
