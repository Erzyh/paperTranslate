"""Bulk download (POST/GET /api/archives) — no pipeline involved."""
from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from app import config, db


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERTRANSLATE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PAPERTRANSLATE_TRANSLATOR", "stub")
    from app.main import app

    with TestClient(app) as c:
        yield c


def finished(doc_id: str, filename: str, body: bytes = b"%PDF-1.4 translated") -> str:
    db.insert_document(doc_id, filename, 1, status="done")
    (config.get_outputs_dir() / f"{doc_id}.pdf").write_bytes(body)
    return doc_id


def download(client, payload) -> zipfile.ZipFile:
    created = client.post("/api/archives", json=payload)
    assert created.status_code == 200, created.text
    response = client.get(created.json()["url"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "paperTranslate" in response.headers["content-disposition"]
    return zipfile.ZipFile(io.BytesIO(response.content))


def test_zip_keeps_folder_structure(client):
    a = finished("a" * 32, "attention.pdf", b"A")
    b = finished("b" * 32, "resnet.pdf", b"B")
    zf = download(client, {"items": [
        {"id": a, "path": "papers/2025/attention.pdf"},
        {"id": b},
    ]})
    assert sorted(zf.namelist()) == ["papers/2025/attention.pdf", "resnet.pdf"]
    assert zf.read("papers/2025/attention.pdf") == b"A"


def test_duplicate_names_do_not_overwrite(client):
    a = finished("a" * 32, "paper.pdf", b"A")
    b = finished("b" * 32, "paper.pdf", b"B")
    zf = download(client, {"items": [{"id": a}, {"id": b}]})
    assert sorted(zf.namelist()) == ["paper (2).pdf", "paper.pdf"]


def test_unsafe_paths_are_cleaned(client):
    a = finished("a" * 32, "x.pdf")
    zf = download(client, {"items": [{"id": a, "path": "../../C:/evil<>?.pdf"}]})
    (name,) = zf.namelist()
    assert ".." not in name and ":" not in name and "<" not in name
    assert name.endswith(".pdf")


def test_unfinished_and_unknown_documents_are_skipped(client):
    a = finished("a" * 32, "done.pdf")
    db.insert_document("c" * 32, "running.pdf", 1, status="translating")
    created = client.post("/api/archives", json={"items": [
        {"id": a}, {"id": "c" * 32}, {"id": "d" * 32}]}).json()
    assert created["count"] == 1
    assert created["skipped"] == ["c" * 32, "d" * 32]


def test_nothing_finished_is_404(client):
    db.insert_document("c" * 32, "running.pdf", 1, status="translating")
    response = client.post("/api/archives", json={"items": [{"id": "c" * 32}]})
    assert response.status_code == 404


def test_archive_name_is_validated(client):
    assert client.get("/api/archives/..%2Fapp.db").status_code == 404
    assert client.get("/api/archives/" + "0" * 32 + ".zip").status_code == 404
