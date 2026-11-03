"""Shared pytest fixtures. All tests run fully offline (PROVIDER=fake)."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient wired to isolated temp storage and the offline fake provider."""
    monkeypatch.setenv("PROVIDER", "fake")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("VECTORSTORE_DIR", str(tmp_path / "vectorstore"))
    monkeypatch.setenv("CHUNK_SIZE", "200")
    monkeypatch.setenv("CHUNK_OVERLAP", "40")

    from app.config import get_settings

    get_settings.cache_clear()

    from app.services.vectorstore import reset_vectorstore_manager

    reset_vectorstore_manager()

    from app.main import create_app

    try:
        with TestClient(create_app()) as test_client:
            yield test_client
    finally:
        get_settings.cache_clear()
        reset_vectorstore_manager()


@pytest.fixture()
def sample_text_file(tmp_path):
    content = (
        "The mitochondria is the powerhouse of the cell. "
        "Photosynthesis occurs in the chloroplasts of plant cells. "
        "Newton published the Principia Mathematica in 1687. "
        "The human genome contains about three billion base pairs. "
    ) * 12
    path = tmp_path / "sample.txt"
    path.write_text(content, encoding="utf-8")
    return path
