"""POST /explain tests — stub notice, Ollama stream proxy, 503 preflight.

Safety contract: nothing here may open a network connection. The Ollama
preflight and the explain stream both go through injected
httpx.MockTransport handlers, and every base URL uses the non-resolvable
.invalid TLD so an accidental real transport could not reach a server.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.pipeline.translate import (
    OllamaTranslator,
    TranslatorUnavailableError,
    build_explain_prompt,
)

BASE_URL = "http://ollama.invalid"

# The sample PDF renders real formula glyphs only when this font exists
# (scripts/make_sample.py falls back to plain helv text without it).
MATH_FONT_FILE = Path(r"C:\Windows\Fonts\seguisym.ttf")
needs_math_font = pytest.mark.skipif(
    not MATH_FONT_FILE.exists(), reason="needs seguisym math font"
)


# ------------------------------------------------------------------ fixtures


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    monkeypatch.setenv("PAPERTRANSLATE_DATA_DIR", str(d))
    monkeypatch.setenv("PAPERTRANSLATE_TRANSLATOR", "stub")
    monkeypatch.setenv("PAPERTRANSLATE_OLLAMA_URL", BASE_URL)
    monkeypatch.setenv("PAPERTRANSLATE_OLLAMA_MODEL", "qwen3:8b")
    return d


@pytest.fixture()
def client(data_dir):
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def pdf_file(tmp_path):
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Hello world")
    doc.save(str(path))
    doc.close()
    return path


def _upload(client, pdf_file) -> str:
    resp = client.post(
        "/api/documents",
        files={"file": ("sample.pdf", pdf_file.read_bytes(),
                        "application/pdf")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


# ------------------------------------------------------------------- helpers


def _ndjson_stream(chunks: list[str]) -> httpx.Response:
    """Fake Ollama /api/chat stream=true body (NDJSON, terminal done line)."""
    lines = [json.dumps({"message": {"role": "assistant", "content": c}},
                        ensure_ascii=False)
             for c in chunks]
    lines.append(json.dumps(
        {"done": True, "message": {"role": "assistant", "content": ""}}))
    body = "\n".join(lines) + "\n"
    return httpx.Response(200, content=body.encode("utf-8"))


def _tags_ok(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/api/tags"
    return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})


def _must_not_be_called(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected request: {request.url}")


def _install_transports(monkeypatch, preflight_handler, explain_handler):
    from app.api import routes

    monkeypatch.setattr(routes, "OLLAMA_PREFLIGHT_TRANSPORT",
                        httpx.MockTransport(preflight_handler))
    monkeypatch.setattr(routes, "OLLAMA_EXPLAIN_TRANSPORT",
                        httpx.MockTransport(explain_handler))


# --------------------------------------------------- explain_stream (direct)


def make_translator(handler) -> OllamaTranslator:
    return OllamaTranslator(model="qwen3:8b", base_url=BASE_URL,
                            transport=httpx.MockTransport(handler))


def test_explain_stream_yields_content_chunks_and_skips_empty():
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        captured.append(json.loads(request.content.decode("utf-8")))
        return _ndjson_stream(["첫 번째", "", " 두 번째"])

    chunks = list(make_translator(handler).explain_stream("prompt text"))

    assert chunks == ["첫 번째", " 두 번째"], "empty chunks must be dropped"
    payload = captured[0]
    assert payload["model"] == "qwen3:8b"
    assert payload["stream"] is True
    assert payload["think"] is False
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1] == {"role": "user", "content": "prompt text"}


def test_explain_stream_drops_think_block_within_chunk():
    def handler(request: httpx.Request) -> httpx.Response:
        return _ndjson_stream(["<think>hidden reasoning</think>답은 이렇다."])

    chunks = list(make_translator(handler).explain_stream("p"))
    assert "".join(chunks) == "답은 이렇다."


def test_explain_stream_drops_think_block_split_across_chunks():
    # Both tags and the block body are split across chunk boundaries.
    def handler(request: httpx.Request) -> httpx.Response:
        return _ndjson_stream(
            ["<thi", "nk>hidden", " more</th", "ink>보이", "는 답이다."])

    chunks = list(make_translator(handler).explain_stream("p"))
    assert "".join(chunks) == "보이는 답이다."


def test_explain_stream_connect_error_raises_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(TranslatorUnavailableError) as excinfo:
        list(make_translator(handler).explain_stream("p"))
    assert "Ollama 서버에 연결할 수 없습니다" in str(excinfo.value)


def test_explain_stream_http_error_raises_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(TranslatorUnavailableError) as excinfo:
        list(make_translator(handler).explain_stream("p"))
    assert "HTTP 500" in str(excinfo.value)


# ------------------------------------------------------ build_explain_prompt


def test_build_explain_prompt_selection_with_title():
    prompt = build_explain_prompt("An excerpt.", "selection",
                                  "Attention Is All You Need")
    assert "논문 'Attention Is All You Need'의 발췌다" in prompt
    assert "학부생에게 설명하듯" in prompt
    assert "5문장 이내" in prompt
    assert prompt.endswith("An excerpt.")


def test_build_explain_prompt_selection_without_title():
    prompt = build_explain_prompt("An excerpt.", "selection", None)
    assert "다음은 논문의 발췌다" in prompt
    assert "''" not in prompt


def test_build_explain_prompt_formula():
    prompt = build_explain_prompt("E = mc^2", "formula", "Some Title")
    assert prompt.startswith("다음 수식을 항별로 한국어로 해설하라")
    assert prompt.endswith("E = mc^2")
    assert "Some Title" not in prompt, "formula prompt carries no title"


def test_build_explain_prompt_caps_text_at_4000_chars():
    prompt = build_explain_prompt("x" * 5000, "selection", None)
    assert "x" * 4000 in prompt
    assert "x" * 4001 not in prompt


# ------------------------------------------------------------ route: stub


def test_explain_stub_streams_fixed_notice(client, pdf_file, monkeypatch):
    from app.api import routes

    # The stub path must never talk to Ollama at all.
    _install_transports(monkeypatch, _must_not_be_called, _must_not_be_called)

    doc_id = _upload(client, pdf_file)
    resp = client.post(f"/api/documents/{doc_id}/explain",
                       json={"text": "What is attention?",
                             "kind": "selection"})

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/plain; charset=utf-8"
    assert resp.text == routes.STUB_EXPLAIN_MESSAGE
    assert "스텁 모드에서는 설명을 제공하지 않습니다" in resp.text


# ---------------------------------------------------------- route: ollama


def test_explain_proxies_ollama_stream(client, pdf_file, monkeypatch):
    def explain_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["stream"] is True
        return _ndjson_stream(["어텐션은 ", "", "가중 평균이다."])

    _install_transports(monkeypatch, _tags_ok, explain_handler)
    monkeypatch.setenv("PAPERTRANSLATE_TRANSLATOR", "ollama")

    doc_id = _upload(client, pdf_file)
    with client.stream("POST", f"/api/documents/{doc_id}/explain",
                       json={"text": "Attention weights",
                             "kind": "selection"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/plain; charset=utf-8"
        body = "".join(resp.iter_text())
    assert body == "어텐션은 가중 평균이다."


def test_explain_stream_excludes_think_blocks_via_route(client, pdf_file,
                                                        monkeypatch):
    def explain_handler(request: httpx.Request) -> httpx.Response:
        return _ndjson_stream(
            ["<think>추론 중", "이라 숨긴다</think>", "실제 설명이다."])

    _install_transports(monkeypatch, _tags_ok, explain_handler)
    monkeypatch.setenv("PAPERTRANSLATE_TRANSLATOR", "ollama")

    doc_id = _upload(client, pdf_file)
    resp = client.post(f"/api/documents/{doc_id}/explain",
                       json={"text": "Some text", "kind": "selection"})
    assert resp.status_code == 200
    assert resp.text == "실제 설명이다."


def test_explain_503_when_ollama_down(client, pdf_file, monkeypatch):
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    # The chat stream must never start when the preflight fails.
    _install_transports(monkeypatch, down, _must_not_be_called)
    monkeypatch.setenv("PAPERTRANSLATE_TRANSLATOR", "ollama")

    doc_id = _upload(client, pdf_file)
    resp = client.post(f"/api/documents/{doc_id}/explain",
                       json={"text": "Some text", "kind": "selection"})
    assert resp.status_code == 503
    assert "Ollama 서버에 연결할 수 없습니다" in resp.json()["detail"]


def test_explain_503_when_model_missing(client, pdf_file, monkeypatch):
    def tags_without_model(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "llama3:8b"}]})

    _install_transports(monkeypatch, tags_without_model, _must_not_be_called)
    monkeypatch.setenv("PAPERTRANSLATE_TRANSLATOR", "ollama")

    doc_id = _upload(client, pdf_file)
    resp = client.post(f"/api/documents/{doc_id}/explain",
                       json={"text": "Some text", "kind": "selection"})
    assert resp.status_code == 503
    assert "qwen3:8b" in resp.json()["detail"]


def test_explain_text_capped_at_4000_chars_in_prompt(client, pdf_file,
                                                     monkeypatch):
    captured: list[dict] = []

    def explain_handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return _ndjson_stream(["요약이다."])

    _install_transports(monkeypatch, _tags_ok, explain_handler)
    monkeypatch.setenv("PAPERTRANSLATE_TRANSLATOR", "ollama")

    doc_id = _upload(client, pdf_file)
    resp = client.post(f"/api/documents/{doc_id}/explain",
                       json={"text": "x" * 5000, "kind": "selection"})
    assert resp.status_code == 200

    user_msg = captured[0]["messages"][1]["content"]
    assert "x" * 4000 in user_msg
    assert "x" * 4001 not in user_msg


def test_explain_selection_prompt_uses_first_heading_title(client, pdf_file,
                                                           monkeypatch):
    from app import db

    captured: list[dict] = []

    def explain_handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return _ndjson_stream(["설명이다."])

    _install_transports(monkeypatch, _tags_ok, explain_handler)
    monkeypatch.setenv("PAPERTRANSLATE_TRANSLATOR", "ollama")

    doc_id = _upload(client, pdf_file)
    # Body first, heading second: the first *heading* wins, not the first row.
    db.upsert_segment(doc_id, "p0_s0", 0, (0, 0, 10, 10), "body",
                      "Some body text.", None, status="pending")
    db.upsert_segment(doc_id, "p0_s1", 0, (0, 20, 10, 30), "heading",
                      "Attention Is All You Need", None, status="pending")

    resp = client.post(f"/api/documents/{doc_id}/explain",
                       json={"text": "The dot-product attention.",
                             "kind": "selection"})
    assert resp.status_code == 200

    user_msg = captured[0]["messages"][1]["content"]
    assert "논문 'Attention Is All You Need'의 발췌다" in user_msg
    assert "5문장 이내" in user_msg
    assert user_msg.endswith("The dot-product attention.")


def test_explain_formula_prompt_via_route(client, pdf_file, monkeypatch):
    captured: list[dict] = []

    def explain_handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return _ndjson_stream(["항별 해설이다."])

    _install_transports(monkeypatch, _tags_ok, explain_handler)
    monkeypatch.setenv("PAPERTRANSLATE_TRANSLATOR", "ollama")

    doc_id = _upload(client, pdf_file)
    resp = client.post(f"/api/documents/{doc_id}/explain",
                       json={"text": "E = mc^2", "kind": "formula"})
    assert resp.status_code == 200
    assert resp.text == "항별 해설이다."

    user_msg = captured[0]["messages"][1]["content"]
    assert user_msg.startswith("다음 수식을 항별로 한국어로 해설하라")


# ------------------------------------------------ route: segment_id clip path


@needs_math_font
def test_explain_segment_id_uses_pdf_clip_text(client, sample_pdf,
                                               sample_segments, monkeypatch):
    """segment_id: the model input is re-extracted from the PDF region, so
    it carries the real formula characters and no ⟦EQn⟧ placeholder, and
    the request body text is ignored."""
    from app import db

    captured: list[dict] = []

    def explain_handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return _ndjson_stream(["항별 해설이다."])

    _install_transports(monkeypatch, _tags_ok, explain_handler)
    monkeypatch.setenv("PAPERTRANSLATE_TRANSLATOR", "ollama")

    doc_id = _upload(client, Path(sample_pdf))
    seg = next(s for s in sample_segments
               if s.page == 1 and s.kind == "formula" and "0.42" in s.text)
    # Precondition: the stored source really is placeholder-tokenized.
    assert "⟦EQ" in seg.text
    db.upsert_segment(doc_id, seg.id, seg.page, seg.bbox, seg.kind, seg.text,
                      None, status="pending")

    resp = client.post(f"/api/documents/{doc_id}/explain",
                       json={"text": "⟦EQ9⟧ ignored body text",
                             "kind": "formula", "segment_id": seg.id})
    assert resp.status_code == 200
    assert resp.text == "항별 해설이다."

    user_msg = captured[0]["messages"][1]["content"]
    assert user_msg.startswith("다음 수식을 항별로 한국어로 해설하라")
    # Real characters from the PDF clip, math and regular font alike.
    assert "∑" in user_msg
    assert "0.42" in user_msg
    # No placeholder token anywhere and the body text was ignored.
    assert "⟦" not in user_msg
    assert "ignored body text" not in user_msg


def test_explain_unknown_segment_id_404(client, pdf_file, monkeypatch):
    # Applies in stub mode too, before any Ollama traffic could happen.
    _install_transports(monkeypatch, _must_not_be_called, _must_not_be_called)

    doc_id = _upload(client, pdf_file)
    resp = client.post(f"/api/documents/{doc_id}/explain",
                       json={"text": "t", "kind": "formula",
                             "segment_id": "nope"})
    assert resp.status_code == 404


def test_explain_without_segment_id_keeps_body_text(client, pdf_file,
                                                    monkeypatch):
    """Regression: the plain text path is untouched when segment_id is absent."""
    captured: list[dict] = []

    def explain_handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return _ndjson_stream(["설명이다."])

    _install_transports(monkeypatch, _tags_ok, explain_handler)
    monkeypatch.setenv("PAPERTRANSLATE_TRANSLATOR", "ollama")

    doc_id = _upload(client, pdf_file)
    resp = client.post(f"/api/documents/{doc_id}/explain",
                       json={"text": "The dot-product attention.",
                             "kind": "selection"})
    assert resp.status_code == 200
    assert captured[0]["messages"][1]["content"].endswith(
        "The dot-product attention.")


# ------------------------------------------------- route: segment crop PNG


def test_segment_png_returns_png(client, pdf_file):
    from app import db

    doc_id = _upload(client, pdf_file)
    db.upsert_segment(doc_id, "p0_s0", 0, (70.0, 88.0, 160.0, 106.0),
                      "formula", "⟦EQ0⟧", None, status="pending")

    resp = client.get(f"/api/documents/{doc_id}/segments/p0_s0.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_segment_png_unknown_segment_404(client, pdf_file):
    doc_id = _upload(client, pdf_file)
    resp = client.get(f"/api/documents/{doc_id}/segments/nope.png")
    assert resp.status_code == 404


def test_segment_png_unknown_document_404(client):
    resp = client.get("/api/documents/nope/segments/p0_s0.png")
    assert resp.status_code == 404


# ------------------------------------------------------------- route: misc


def test_explain_unknown_document_404(client):
    resp = client.post("/api/documents/nope/explain",
                       json={"text": "t", "kind": "selection"})
    assert resp.status_code == 404


def test_explain_invalid_kind_422(client, pdf_file):
    doc_id = _upload(client, pdf_file)
    resp = client.post(f"/api/documents/{doc_id}/explain",
                       json={"text": "t", "kind": "summary"})
    assert resp.status_code == 422
