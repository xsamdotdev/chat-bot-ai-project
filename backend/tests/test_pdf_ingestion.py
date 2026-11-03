"""End-to-end PDF ingestion: build a real text-layer PDF, upload it, query it."""

from fpdf import FPDF


def _build_pdf(tmp_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    pdf.multi_cell(
        0,
        8,
        "The Eiffel Tower was completed in 1889 and stands 330 metres tall. "
        "It was designed by the engineering company of Gustave Eiffel. "
        "Originally criticized by some artists, it is now one of the most visited "
        "monuments in the world, attracting millions of tourists each year. "
        "The tower is located on the Champ de Mars in Paris, France.",
    )
    path = tmp_path / "eiffel.pdf"
    pdf.output(str(path))
    return path


def test_pdf_upload_and_chat(client, tmp_path):
    pdf_path = _build_pdf(tmp_path)
    with open(pdf_path, "rb") as handle:
        upload = client.post(
            "/api/documents/upload",
            files={"file": ("eiffel.pdf", handle, "application/pdf")},
        )
    assert upload.status_code == 201, upload.text
    assert upload.json()["document"]["chunks"] >= 1

    chat = client.post("/api/chat", json={"query": "How tall is the Eiffel Tower?"})
    assert chat.status_code == 200
    sources = chat.json()["sources"]
    assert sources and sources[0]["filename"] == "eiffel.pdf"
    assert sources[0]["page"] == 0  # PDF page number, 0-based
