def _upload(client, path, filename="notes.txt"):
    with open(path, "rb") as handle:
        return client.post(
            "/api/documents/upload",
            files={"file": (filename, handle, "text/plain")},
        )


def test_upload_list_delete_roundtrip(client, sample_text_file):
    upload = _upload(client, sample_text_file)
    assert upload.status_code == 201, upload.text
    body = upload.json()
    doc_id = body["document"]["doc_id"]
    assert body["document"]["chunks"] >= 1
    assert body["document"]["size_bytes"] > 0

    listing = client.get("/api/documents")
    assert listing.status_code == 200
    assert listing.json()["total"] == 1
    assert listing.json()["documents"][0]["doc_id"] == doc_id

    delete = client.delete(f"/api/documents/{doc_id}")
    assert delete.status_code == 200
    assert delete.json()["chunks_deleted"] == body["document"]["chunks"]

    assert client.get("/api/documents").json()["total"] == 0


def test_upload_rejects_unsupported_extension(client, tmp_path):
    exe = tmp_path / "evil.exe"
    exe.write_bytes(b"MZ\x90\x00")
    response = _upload(client, exe, filename="evil.exe")
    assert response.status_code == 415


def test_upload_rejects_empty_file(client, tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    response = _upload(client, empty, filename="empty.txt")
    assert response.status_code == 400


def test_delete_unknown_document_returns_404(client):
    response = client.delete("/api/documents/doesnotexist")
    assert response.status_code == 404
