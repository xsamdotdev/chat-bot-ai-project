def _ingest_sample(client, sample_text_file):
    with open(sample_text_file, "rb") as handle:
        response = client.post(
            "/api/documents/upload",
            files={"file": ("notes.txt", handle, "text/plain")},
        )
    assert response.status_code == 201, response.text


def test_chat_before_any_upload_returns_409(client):
    response = client.post("/api/chat", json={"query": "What is this about?"})
    assert response.status_code == 409


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


def test_chat_after_deleting_all_documents_returns_409(client, sample_text_file):
    _ingest_sample(client, sample_text_file)
    doc_id = client.get("/api/documents").json()["documents"][0]["doc_id"]
    assert client.delete(f"/api/documents/{doc_id}").status_code == 200
    response = client.post("/api/chat", json={"query": "still there?"})
    assert response.status_code == 409


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
