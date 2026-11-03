"""Provider factories: Mistral wiring (offline, no network calls)."""

import pytest

from app.config import Settings
from app.services.embeddings import get_embeddings
from app.services.llm import get_chat_model


def _settings(**overrides) -> Settings:
    base = {"_env_file": None, "provider": "mistral"}
    return Settings(**{**base, **overrides})


def test_mistral_missing_key_raises_clear_error():
    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        get_embeddings(_settings(mistral_api_key=None))
    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        get_chat_model(_settings(mistral_api_key=None))


def test_mistral_factories_construct_with_key():
    settings = _settings(mistral_api_key="dummy-key-for-construction-only")
    embeddings = get_embeddings(settings)
    model = get_chat_model(settings)
    assert settings.embedding_model == "mistral-embed"
    assert settings.chat_model == "mistral-small"
    assert model.model == "mistral-small"
    assert embeddings.model == "mistral-embed"


def test_mistral_endpoint_override_is_plumbed():
    settings = _settings(
        mistral_api_key="dummy-key-for-construction-only",
        mistral_base_url="https://example.test/v1",
        mistral_chat_model="mistral-large",
        mistral_embedding_model="mistral-embed",
    )
    model = get_chat_model(settings)
    assert model.model == "mistral-large"
    assert "example.test" in str(model.endpoint)
    assert settings.chat_model == "mistral-large"
