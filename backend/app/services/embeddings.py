"""Embeddings via Pinecone's hosted inference API (`pc.inference.embed`).

Pinecone hosts dense embedding models, so no separate embedding provider is
needed: the same client that stores the vectors also generates them.

The `input_type` parameter matters for these models:
  - "passage" for documents being ingested
  - "query"   for the search query
"""

from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.services.pinecone_client import get_pinecone_client

_EMBED_BATCH = 96  # Pinecone inference accepts batches of documents per call


class PineconeInferenceEmbeddings(Embeddings):
    """LangChain `Embeddings` backed by Pinecone hosted inference."""

    def __init__(self, client, model: str):
        self._client = client
        self.model = model

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = texts[start : start + _EMBED_BATCH]
            response = self._client.inference.embed(
                model=self.model,
                inputs=batch,
                parameters={"input_type": input_type},
            )
            vectors.extend([item.values for item in response])
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(list(texts), "passage")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]


def get_embeddings(settings: Settings, client=None) -> Embeddings:
    """Return the embeddings implementation for the configured mode."""
    if settings.provider == "fake":
        from langchain_core.embeddings import DeterministicFakeEmbedding

        # 1024 dims matches the dense Pinecone embedding models, so fake vectors
        # can be used against a mocked index in tests.
        return DeterministicFakeEmbedding(size=1024)

    client = client or get_pinecone_client(settings)
    return PineconeInferenceEmbeddings(client=client, model=settings.pinecone_embedding_model)
