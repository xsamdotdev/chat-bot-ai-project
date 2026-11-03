"""Pinecone client/index helpers shared by embeddings and the vector store."""

from app.config import Settings


def get_pinecone_client(settings: Settings):
    """Create a Pinecone client, failing with actionable messages when unconfigured."""
    if not settings.pinecone_api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Set it in the repo-root .env (Docker) or "
            "backend/.env (local uvicorn) — get a key at https://app.pinecone.io."
        )
    from pinecone import Pinecone  # imported lazily so tests can monkeypatch it

    return Pinecone(api_key=settings.pinecone_api_key)


def get_pinecone_index(settings: Settings, client=None):
    """Return the configured Pinecone index handle (by name or by host URL)."""
    if not settings.pinecone_host and not settings.pinecone_index:
        raise RuntimeError(
            "Set either PINECONE_INDEX (index name) or PINECONE_HOST (index host URL, "
            "e.g. https://my-index-xxxx.svc.aped-1234.pinecone.io). Create a serverless "
            f"index whose dimension matches the embedding model "
            f"({settings.pinecone_embedding_model}) — see README."
        )
    client = client or get_pinecone_client(settings)
    if settings.pinecone_host:
        return client.Index(host=settings.pinecone_host)
    return client.Index(settings.pinecone_index)
