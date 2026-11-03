"""`/api/diagnostics`: compare the local manifest with live Pinecone stats."""

import json


def _ingest_sample(client, sample_text_file):
    with open(sample_text_file, "rb") as handle:
        response = client.post(
            "/api/documents/upload",
            files={"file": ("notes.txt", handle, "text/plain")},
        )
    assert response.status_code == 201, response.text


def test_diagnostics_reports_manifest_and_index_stats(client, sample_text_file):
    _ingest_sample(client, sample_text_file)

    response = client.get("/api/diagnostics")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["documents_indexed"] == 1
    assert body["pinecone_index"] == "test-index"
    assert body["pinecone_namespace"] == ""
    assert body["manifest_documents"][0]["filename"] == "notes.txt"
    assert body["index_stats"]["total_vector_count"] > 0
    assert body["warnings"] == []


def test_diagnostics_warns_when_manifest_points_at_another_index(
    client, sample_text_file, tmp_path
):
    _ingest_sample(client, sample_text_file)

    manifest_path = tmp_path / "vectorstore" / "fake-embedding" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest.values():
        entry["pinecone_index"] = "old-index"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    body = client.get("/api/diagnostics").json()
    assert any("old-index" in warning for warning in body["warnings"])


def test_diagnostics_warns_when_namespace_has_no_vectors(client, sample_text_file, monkeypatch):
    _ingest_sample(client, sample_text_file)

    # Vectors still live in the default namespace, but the app now queries another one.
    from app.config import get_settings

    monkeypatch.setenv("PINECONE_NAMESPACE", "other-namespace")
    get_settings.cache_clear()

    body = client.get("/api/diagnostics").json()
    assert body["pinecone_namespace"] == "other-namespace"
    assert any("none in namespace 'other-namespace'" in warning for warning in body["warnings"])
