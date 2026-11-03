"""Load uploaded files into LangChain Documents."""

from pathlib import Path

from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}


class UnsupportedFileTypeError(ValueError):
    """Raised when a file extension has no registered loader."""


def supported_extensions() -> list[str]:
    return sorted(SUPPORTED_EXTENSIONS)


def load_document(path: Path, *, original_filename: str | None = None) -> list[Document]:
    """Parse a file on disk into LangChain Documents.

    Metadata always includes ``source`` (original filename) so downstream
    chunks can be traced back to the uploaded document.
    """
    ext = path.suffix.lower()
    filename = original_filename or path.name

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        docs = PyPDFLoader(str(path)).load()
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        docs = Docx2txtLoader(str(path)).load()
    elif ext in {".txt", ".md", ".markdown"}:
        from langchain_community.document_loaders import TextLoader

        docs = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()
    else:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(supported_extensions())}"
        )

    for doc in docs:
        doc.metadata["source"] = filename
    return docs
