"""Gemini chat wiring + config defaults (offline, no network calls)."""

import pytest

from app.config import Settings
from app.services.llm import EchoChatModel, get_chat_model


def _settings(**overrides) -> Settings:
    return Settings(**{"_env_file": None, **overrides})


def test_defaults_are_gemini_and_pinecone_embedding_model():
    settings = _settings()
    assert settings.provider == "gemini"
    assert settings.chat_model == "gemini-2.5-flash"
    assert settings.embedding_model == "llama-text-embed-v2"


def test_fake_mode_uses_offline_stubs():
    settings = _settings(provider="fake")
    assert isinstance(get_chat_model(settings), EchoChatModel)
    assert settings.embedding_model == "fake-embedding"


def test_google_api_key_alias_accepted(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "alias-key")
    settings = _settings()
    assert settings.google_api_key == "alias-key"


def test_blank_provider_env_means_default(monkeypatch):
    monkeypatch.setenv("PROVIDER", "")
    assert _settings().provider == "gemini"


def test_gemini_missing_key_raises_clear_error():
    settings = _settings(google_api_key=None)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        get_chat_model(settings)


def test_gemini_chat_model_constructs_offline():
    settings = _settings(google_api_key="dummy-key-for-construction-only")
    model = get_chat_model(settings)
    assert model.model == "gemini-2.5-flash"


def test_embedding_slug_tracks_embedding_model():
    settings = _settings(pinecone_embedding_model="multilingual-e5-large")
    assert settings.embedding_slug == "multilingual-e5-large"
