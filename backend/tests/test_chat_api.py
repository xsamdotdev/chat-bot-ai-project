def _ingest_sample(client, sample_text_file):
    with open(sample_text_file, "rb") as handle:
        response = client.post(
            "/api/documents/upload",
            files={"file": ("notes.txt", handle, "text/plain")},
        )
    assert response.status_code == 201, response.text


def test_chat_without_documents_still_replies_as_persona(client):
    """No documents yet: answer conversationally instead of returning an error."""
    response = client.post("/api/chat", json={"query": "Hi Lisa, who are you?"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sources"] == []
    assert body["chatbot_name"] == "Lisa"


def test_system_prompt_defines_lisa_persona():
    from app.config import Settings
    from app.services.rag import build_system_prompt

    settings = Settings(_env_file=None, provider="fake")
    prompt = build_system_prompt(settings)

    assert settings.chatbot_display_name == "Lisa"
    assert "Lisa" in prompt
    assert "female" in prompt.lower()
    assert "chatbot" in prompt.lower()
    assert "not a human" in prompt.lower()
    # The answer must not reveal where the knowledge comes from.
    assert "never mention documents" in prompt.lower()


def test_retrieval_context_has_no_source_labels(client, sample_text_file, monkeypatch):
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    captured: dict = {}

    class CaptureChatModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "capture-fake"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            captured["messages"] = messages
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    _ingest_sample(client, sample_text_file)
    monkeypatch.setattr("app.services.rag.get_chat_model", lambda settings: CaptureChatModel())

    response = client.post("/api/chat", json={"query": "Where does photosynthesis happen?"})
    assert response.status_code == 200, response.text

    human = next(m for m in captured["messages"] if m.type == "human").content
    assert "Reference information:" in human
    assert "uploaded documents" not in human.lower()
    assert "[Source 1]" not in human


def test_content_to_text_handles_gemini_content_blocks():
    from app.services.rag import _content_to_text

    assert _content_to_text("plain answer") == "plain answer"
    assert _content_to_text(None) == ""
    assert (
        _content_to_text(
            [
                {"type": "text", "text": "Hello", "extras": {"signature": "sig"}},
                {"type": "thinking", "text": "internal reasoning"},
                {"type": "text", "text": "world"},
            ]
        )
        == "Hello\nworld"
    )


def test_chat_normalizes_content_blocks_from_gemini(client, sample_text_file, monkeypatch):
    """Gemini 3.x returns a list of content blocks; the API must return plain text."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class BlockChatModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "block-fake"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=[
                                {
                                    "type": "text",
                                    "text": "Answer from content blocks.",
                                    "extras": {"signature": "sig"},
                                }
                            ]
                        )
                    )
                ]
            )

    _ingest_sample(client, sample_text_file)
    monkeypatch.setattr("app.services.rag.get_chat_model", lambda settings: BlockChatModel())

    response = client.post("/api/chat", json={"query": "Where does photosynthesis happen?"})
    assert response.status_code == 200, response.text
    assert response.json()["answer"] == "Answer from content blocks."
    assert response.json()["chatbot_name"] == "Lisa"


def test_chat_reports_missing_vectors_instead_of_guessing(client, sample_text_file):
    """Manifest non-empty but Pinecone has no matches -> actionable error, not 'I don't know'."""
    from fakes import FakePinecone

    _ingest_sample(client, sample_text_file)
    FakePinecone.instances[-1]._index.vectors.clear()

    response = client.post("/api/chat", json={"query": "anything"})
    assert response.status_code == 409
    assert "no matches" in response.json()["detail"]
    assert "/api/diagnostics" in response.json()["detail"]


def test_chat_returns_answer_and_sources(client, sample_text_file):
    _ingest_sample(client, sample_text_file)

    response = client.post(
        "/api/chat",
        json={"query": "Where does photosynthesis happen?", "top_k": 3},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["provider"] == "fake"
    assert "[fake-echo]" in body["answer"]
    assert len(body["sources"]) == 3
    for source in body["sources"]:
        assert source["filename"] == "notes.txt"
        assert source["snippet"]
        assert source["doc_id"]


def test_chat_after_deleting_all_documents_still_replies(client, sample_text_file):
    _ingest_sample(client, sample_text_file)
    doc_id = client.get("/api/documents").json()["documents"][0]["doc_id"]
    assert client.delete(f"/api/documents/{doc_id}").status_code == 200
    response = client.post("/api/chat", json={"query": "still there?"})
    assert response.status_code == 200
    assert response.json()["sources"] == []


def test_chat_respects_history_and_validation(client, sample_text_file):
    _ingest_sample(client, sample_text_file)

    ok = client.post(
        "/api/chat",
        json={
            "query": "And after that?",
            "history": [{"role": "user", "content": "Tell me about cells."}],
        },
    )
    assert ok.status_code == 200

    bad = client.post("/api/chat", json={"query": ""})
    assert bad.status_code == 422
