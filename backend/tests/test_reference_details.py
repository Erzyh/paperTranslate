"""Reference details API tests (GET /references/{n}/details).

Safety contract: every Semantic Scholar lookup is answered in-process by an
httpx.MockTransport handler injected through the semantic_scholar module
hook, and the base URL is patched to a non-resolvable .invalid host so an
accidental real transport could not reach semanticscholar.org anyway.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi.testclient import TestClient

from app import db
from app.api import semantic_scholar

_S2_PAPER = {
    "paperId": "abc123",
    "title": "Layout aware translation of scholarly documents",
    "abstract": "We study layout preserving translation.",
    "year": 2021,
    "url": "https://www.semanticscholar.org/paper/abc123",
    "authors": [{"authorId": "1", "name": "A. Author"},
                {"authorId": "2", "name": "B. Builder"}],
}

_NOT_FOUND_PAYLOAD = {"found": False, "title": None, "authors": [],
                      "year": None, "abstract": None, "url": None}


@pytest.fixture(autouse=True)
def s2_isolation(monkeypatch):
    """Point the client at a dead host and clear any leftover transport."""
    monkeypatch.setattr(semantic_scholar, "API_BASE_URL", "https://s2.invalid")
    monkeypatch.setattr(semantic_scholar, "TRANSPORT", None)


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


def _install_transport(monkeypatch, handler) -> list[httpx.Request]:
    """Install a counting MockTransport; returns the recorded request list."""
    requests: list[httpx.Request] = []

    def counting_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    monkeypatch.setattr(semantic_scholar, "TRANSPORT",
                        httpx.MockTransport(counting_handler))
    return requests


def _details(client, doc_id, num) -> httpx.Response:
    return client.get(f"/api/documents/{doc_id}/references/{num}/details")


# ------------------------------------------------------------ success path


def test_details_success_parsed_and_cached(client, doc_id, monkeypatch):
    requests = _install_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200, json={"total": 1, "offset": 0, "data": [_S2_PAPER]}),
    )

    resp = _details(client, doc_id, 1)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "found": True,
        "title": "Layout aware translation of scholarly documents",
        "authors": ["A. Author", "B. Builder"],
        "year": 2021,
        "abstract": "We study layout preserving translation.",
        "url": "https://www.semanticscholar.org/paper/abc123",
    }

    # Exactly one outbound request, with the contract's query parameters:
    # the parsed title, the fixed field list and limit=1.
    assert len(requests) == 1
    parsed = urlparse(str(requests[0].url))
    assert parsed.hostname == "s2.invalid"
    assert parsed.path == "/graph/v1/paper/search"
    params = parse_qs(parsed.query)
    assert params["query"] == ["Layout aware translation of scholarly documents"]
    assert params["fields"] == ["title,authors,year,abstract,url"]
    assert params["limit"] == ["1"]

    # The successful payload is cached in the DB.
    assert db.get_ref_details(doc_id, 1) == resp.json()


def test_details_cache_hit_makes_no_external_call(client, doc_id, monkeypatch):
    requests = _install_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200, json={"total": 1, "offset": 0, "data": [_S2_PAPER]}),
    )
    first = _details(client, doc_id, 1)
    assert first.status_code == 200
    assert len(requests) == 1

    second = _details(client, doc_id, 1)
    assert second.status_code == 200
    assert second.json() == first.json()
    assert len(requests) == 1  # served from the DB cache, zero new calls


def test_details_entry_prefix_query_when_title_missing(client, doc_id,
                                                       monkeypatch):
    # A reference without a parsed title is queried by the first 120 chars
    # of the raw entry. Plant a doc_assets payload to control the entry.
    entry = "[7] " + "x" * 200
    db.set_doc_assets(doc_id, {
        "figures": [], "mentions": [],
        "references": {"7": {"entry": entry, "title": None}},
    })
    requests = _install_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200, json={"total": 0, "offset": 0, "data": []}),
    )

    resp = _details(client, doc_id, 7)
    assert resp.status_code == 200
    assert len(requests) == 1
    params = parse_qs(urlparse(str(requests[0].url)).query)
    assert params["query"] == [entry[:120]]
    assert len(params["query"][0]) == 120


# ------------------------------------------------------------ failure paths


def _timeout_handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectTimeout("connect timed out")


@pytest.mark.parametrize("handler", [
    pytest.param(_timeout_handler, id="timeout"),
    pytest.param(lambda request: httpx.Response(500, text="server error"),
                 id="http-error"),
    pytest.param(lambda request: httpx.Response(200, text="not json{"),
                 id="bad-json"),
    pytest.param(
        lambda request: httpx.Response(
            200, json={"total": 0, "offset": 0, "data": []}),
        id="no-match"),
])
def test_details_failure_is_found_false_and_not_cached(client, doc_id,
                                                       monkeypatch, handler):
    requests = _install_transport(monkeypatch, handler)

    resp = _details(client, doc_id, 1)
    assert resp.status_code == 200
    assert resp.json() == _NOT_FOUND_PAYLOAD
    assert len(requests) == 1
    # Failures are never cached: the next request tries the API again.
    assert db.get_ref_details(doc_id, 1) is None
    assert _details(client, doc_id, 1).json() == _NOT_FOUND_PAYLOAD
    assert len(requests) == 2


# -------------------------------------------------------------------- 404s


def test_details_unknown_reference_404(client, doc_id, monkeypatch):
    requests = _install_transport(
        monkeypatch, lambda request: httpx.Response(500))

    # The sample bibliography has entries [1] and [2] only.
    resp = _details(client, doc_id, 99)
    assert resp.status_code == 404
    assert requests == []  # rejected before any external call


def test_details_unknown_document_404(client, monkeypatch):
    requests = _install_transport(
        monkeypatch, lambda request: httpx.Response(500))
    assert _details(client, "nope", 1).status_code == 404
    assert requests == []
