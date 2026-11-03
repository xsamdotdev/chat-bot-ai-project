from langchain_core.documents import Document

from app.services.chunking import chunk_documents


def _make_docs() -> list[Document]:
    text = "Sentence one. " * 120  # ~1800 chars
    return [Document(page_content=text, metadata={"source": "doc.txt"})]


def test_chunking_splits_and_indexes():
    chunks = chunk_documents(_make_docs(), chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["source"] == "doc.txt" for c in chunks)
    assert all(len(c.page_content) <= 100 for c in chunks)


def test_chunking_preserves_short_documents():
    docs = [Document(page_content="short text", metadata={"source": "s.txt"})]
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) == 1
    assert chunks[0].page_content == "short text"
