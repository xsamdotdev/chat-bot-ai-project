def test_health_reports_fake_provider(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["provider"] == "fake"
    assert body["chatbot_name"] == "Lisa"
    assert body["documents_indexed"] == 0


def test_chatbot_name_is_configurable(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("CHATBOT_NAME", "Ava")
    settings = Settings(_env_file=None, provider="fake")
    assert settings.chatbot_display_name == "Ava"

    monkeypatch.setenv("CHATBOT_NAME", "")
    settings = Settings(_env_file=None, provider="fake")
    assert settings.chatbot_display_name == "Lisa"
