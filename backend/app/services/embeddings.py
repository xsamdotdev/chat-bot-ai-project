"""Provider-aware embedding factory."""

from langchain_core.embeddings import Embeddings

from app.config import Settings


def get_embeddings(settings: Settings) -> Embeddings:
    """Return the embeddings implementation for the configured provider."""
    if settings.provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to backend/.env "
                "(or switch PROVIDER to 'ollama' or 'fake')."
            )
        kwargs: dict = {"model": settings.openai_embedding_model, "api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return OpenAIEmbeddings(**kwargs)

    if settings.provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=settings.ollama_embedding_model, base_url=settings.ollama_base_url)

    if settings.provider == "fake":
        # Deterministic, offline embeddings for dev/smoke tests. NOT semantic.
        from langchain_core.embeddings import DeterministicFakeEmbedding

        return DeterministicFakeEmbedding(size=384)

    raise RuntimeError(f"Unknown provider: {settings.provider!r}")
