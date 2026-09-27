"""Assets API tests: figure/table detection, mentions, references, PNG crops
and cache invalidation (GET /assets, GET /figures/{key}.png).

These tests run the real pipeline modules (no fakes) against the generated
sample PDF with the stub translator; nothing touches Ollama or the network.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    monkeypatch.setenv("PAPERTRANSLATE_DATA_DIR", str(d))
    monkeypatch.setenv("PAPERTRANSLATE_TRANSLATOR", "stub")
    return d


@pytest.fixture()
def client(data_dir):
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def doc_id(client, sample_pdf) -> str:
    content = Path(sample_pdf).read_bytes()
    resp = client.post(
        "/api/documents",
        files={"file": ("sample.pdf", content, "application/pdf")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


def _get_assets(client, doc_id) -> dict:
    resp = client.get(f"/api/documents/{doc_id}/assets")
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_assets_unknown_document_404(client):
    assert client.get("/api/documents/nope/assets").status_code == 404
    assert client.get(
        "/api/documents/nope/figures/figure-1.png").status_code == 404


def test_assets_figures_detected(client, doc_id):
    data = _get_assets(client, doc_id)
    assert set(data) == {"figures", "mentions", "references"}

    figures = {f["key"]: f for f in data["figures"]}
    assert set(figures) == {"figure-1", "table-1"}

    fig = figures["figure-1"]
    assert fig["page"] == 0
    assert fig["caption"].startswith("Figure 1:")
    x0, y0, x1, y1 = fig["bbox"]
    assert x0 < x1 and y0 < y1
    # The raster image sits in the right column of page 0.
    assert x0 >= 300.0 and x1 <= 612.0

    tab = figures["table-1"]
    assert tab["page"] == 1
    assert tab["caption"].startswith("TABLE 1.")
    tx0, ty0, tx1, ty1 = tab["bbox"]
    assert tx0 < tx1 and ty0 < ty1
    # The drawn grid lives in the left column of page 1.
    assert tx1 <= 306.0
    # IEEE table style: the grid sits BELOW its caption. The caption bbox is
    # not part of the payload, but the region must start below the caption
    # text, which ends above y=380 in the sample.
    assert ty0 > 360.0


def test_assets_mentions_found(client, doc_id):
    data = _get_assets(client, doc_id)
    mentions = data["mentions"]
    # Document not translated yet: only original-side mentions.
    assert mentions and all(m["side"] == "original" for m in mentions)
    for m in mentions:
        assert len(m["bbox"]) == 4
        assert m["bbox"][0] < m["bbox"][2] and m["bbox"][1] < m["bbox"][3]

    # Exactly one body mention per figure key; the caption itself is not a
    # mention even though search_for hits it.
    fig_mentions = [m for m in mentions if m["key"] == "figure-1"]
    assert len(fig_mentions) == 1  # "as shown in Fig. 1"
    assert fig_mentions[0]["kind"] == "figure"
    assert fig_mentions[0]["page"] == 0
    assert fig_mentions[0]["ref"] is None

    tab_mentions = [m for m in mentions if m["key"] == "table-1"]
    assert len(tab_mentions) == 1  # "see Table 1"
    assert tab_mentions[0]["kind"] == "table"
    assert tab_mentions[0]["page"] == 1

    # Citations: only "[2]" appears in the body; the "[1]"/"[2]" bibliography
    # entry markers must not count as citation mentions.
    cites = [m for m in mentions if m["kind"] == "cite"]
    assert len(cites) == 1
    assert cites[0]["key"] == "cite-2"
    assert cites[0]["ref"] == 2
    assert cites[0]["page"] == 1


def test_assets_references_parsed(client, doc_id):
    refs = _get_assets(client, doc_id)["references"]
    assert set(refs) == {"1", "2"}
    assert refs["1"]["entry"].startswith("[1] A. Author")
    assert refs["1"]["title"] == "Layout aware translation of scholarly documents"
    assert refs["2"]["entry"].startswith("[2] C. Kim")
    assert refs["2"]["title"] == "Column detection for academic typesetting"


def test_figure_png_rendered_and_cached(client, doc_id, data_dir):
    for key in ("figure-1", "table-1"):
        resp = client.get(f"/api/documents/{doc_id}/figures/{key}.png")
        assert resp.status_code == 200, key
        assert resp.headers["content-type"] == "image/png"
        assert resp.content.startswith(_PNG_SIGNATURE)
        cache_file = data_dir / "figures" / doc_id / f"{key}.png"
        assert cache_file.exists()

    # Second request is served from the cache file with identical bytes.
    first = client.get(f"/api/documents/{doc_id}/figures/figure-1.png")
    second = client.get(f"/api/documents/{doc_id}/figures/figure-1.png")
    assert first.content == second.content


def test_figure_png_unknown_key_404(client, doc_id):
    assert client.get(
        f"/api/documents/{doc_id}/figures/figure-9.png").status_code == 404
    assert client.get(
        f"/api/documents/{doc_id}/figures/bogus.png").status_code == 404
    assert client.get(
        f"/api/documents/{doc_id}/figures/..%2F..%2Fapp.png").status_code == 404


def _wait_for_terminal_status(client, doc_id, timeout=120.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/api/documents/{doc_id}").json()["status"]
        if status in ("done", "error"):
            return status
        time.sleep(0.1)
    pytest.fail("translation never reached a terminal status")


def test_assets_cached_and_invalidated_on_retranslate(client, doc_id):
    first = _get_assets(client, doc_id)
    # The first call computed and cached the payload.
    assert db.get_doc_assets(doc_id) == first

    # A cache hit returns the stored payload verbatim (no recomputation):
    # plant a sentinel and observe it coming back.
    sentinel = {"figures": [], "mentions": [],
                "references": {"9": {"entry": "[9] sentinel", "title": None}}}
    db.set_doc_assets(doc_id, sentinel)
    assert _get_assets(client, doc_id) == sentinel

    # Starting a (re-)translation invalidates the cache; after completion a
    # fresh payload is computed against the done document.
    assert client.post(f"/api/documents/{doc_id}/translate").status_code == 200
    assert _wait_for_terminal_status(client, doc_id) == "done"
    assert db.get_doc_assets(doc_id) is None

    fresh = _get_assets(client, doc_id)
    assert fresh != sentinel
    assert {f["key"] for f in fresh["figures"]} == {"figure-1", "table-1"}
    assert fresh["references"] == first["references"]
    # Original-side mentions are stable across translation.
    original = [m for m in fresh["mentions"] if m["side"] == "original"]
    assert original == first["mentions"]
    # Translated-side mentions (if any) may only come from the output PDF.
    for m in fresh["mentions"]:
        if m["side"] == "translated":
            assert m["kind"] in ("figure", "table", "cite")
