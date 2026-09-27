"""OpenAITranslator tests — httpx.MockTransport only, no real API calls.

Safety contract: nothing here may open a network connection. Every request
is answered in-process by a MockTransport handler, and the base URL uses the
non-resolvable .invalid TLD so an accidental real transport could not reach
api.openai.com anyway.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.pipeline.translate import (
    OpenAITranslator,
    TranslatorUnavailableError,
)

BASE_URL = "https://openai.invalid"
TEST_KEY = "sk-test-not-a-real-key-123"


def make_translator(handler, **kwargs) -> OpenAITranslator:
    return OpenAITranslator(
        api_key=TEST_KEY,
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant",
                                       "content": content}}]},
    )


# ------------------------------------------------------- request contract


def test_request_bearer_header_model_and_url():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return chat_response("번역된 문장이다.")

    translator = make_translator(handler)
    result = translator.translate("Hello world.")
    assert result == "번역된 문장이다."
    assert len(requests) == 1
    req = requests[0]
    assert req.url.path == "/v1/chat/completions"
    assert req.headers["Authorization"] == f"Bearer {TEST_KEY}"
    payload = json.loads(req.content)
    assert payload["model"] == "gpt-5.6-luna"
    # GPT-5-family reasoning models reject sampling overrides (HTTP 400), so
    # no temperature or other sampling params may be sent.
    assert "temperature" not in payload
    assert payload["messages"][-1]["role"] == "user"


def test_api_key_not_stored_as_attribute():
    translator = make_translator(lambda r: chat_response("x"))
    leaked = [name for name, value in vars(translator).items()
              if isinstance(value, str) and TEST_KEY in value]
    assert leaked == []


# ------------------------------------------------- inherited guard chain


def test_placeholder_guard_inherited_corrective_retry():
    # First answer drops the mask token; the corrective retry must run and
    # the second (correct) answer must be accepted — proving the Ollama
    # guard chain is reused by inheritance.
    answers = ["토큰을 잃어버린 번역.", "옳은 번역 ⟦M0⟧ 이다."]
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return chat_response(answers[min(len(calls) - 1, len(answers) - 1)])

    translator = make_translator(handler)
    result = translator.translate("Value ⟦M0⟧ here.")
    assert result == "옳은 번역 ⟦M0⟧ 이다."
    assert len(calls) == 2
    assert translator.last_fallback is False


# --------------------------------------------------------- error mapping


@pytest.mark.parametrize("status,fragment", [
    (401, "API 키가 유효하지 않습니다"),
    (429, "요청 한도 초과"),
])
def test_http_error_mapping(status, fragment):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "nope"}})

    translator = make_translator(handler)
    with pytest.raises(TranslatorUnavailableError) as excinfo:
        translator.translate("Hello.")
    assert fragment in str(excinfo.value)
    # The key must never leak into exception messages.
    assert TEST_KEY not in str(excinfo.value)


def test_connect_error_maps_to_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    translator = make_translator(handler)
    with pytest.raises(TranslatorUnavailableError):
        translator.translate("Hello.")


# --------------------------------------------------------- route layer


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    monkeypatch.setenv("PAPERTRANSLATE_DATA_DIR", str(d))
    monkeypatch.setenv("PAPERTRANSLATE_TRANSLATOR", "stub")
    return d


@pytest.fixture()
def fake_pipeline(monkeypatch):
    """Fake app.pipeline.{runner,translate} with an OpenAITranslator spy."""
    state = {"openai_keys": [], "openai_models": [], "openai_kwargs": [],
             "claude": [], "get_translator_calls": 0}

    def fake_run_pipeline(src_path, out_path, translator,
                          on_progress=None, segment_cb=None):
        Path(out_path).write_bytes(b"%PDF-1.4\n% fake\n")
        return types.SimpleNamespace(overflow_segments=[], scaled_segments={})

    class FakeOpenAITranslator:
        def __init__(self, api_key, transport=None, model=None, **kwargs):
            state["openai_keys"].append(api_key)
            state["openai_models"].append(model)
            state["openai_kwargs"].append(kwargs)

        def translate(self, text, context=None, continuation=False):
            return text

    class FakeClaudeTranslator:
        def __init__(self, api_key, model=None, effort=None, client=None):
            state["claude"].append({"key": api_key, "model": model,
                                    "effort": effort})

        def translate(self, text, context=None, continuation=False):
            return text

    class FakeStubTranslator:
        def translate(self, text, context=None, continuation=False):
            return text

    def fake_get_translator(name):
        state["get_translator_calls"] += 1
        return FakeStubTranslator()

    pipeline_pkg = types.ModuleType("app.pipeline")
    pipeline_pkg.__path__ = []
    runner_mod = types.ModuleType("app.pipeline.runner")
    runner_mod.run_pipeline = fake_run_pipeline
    translate_mod = types.ModuleType("app.pipeline.translate")
    translate_mod.get_translator = fake_get_translator
    translate_mod.OpenAITranslator = FakeOpenAITranslator
    claude_mod = types.ModuleType("app.pipeline.claude")
    claude_mod.ClaudeTranslator = FakeClaudeTranslator
    pipeline_pkg.runner = runner_mod
    pipeline_pkg.translate = translate_mod
    pipeline_pkg.claude = claude_mod
    monkeypatch.setitem(sys.modules, "app.pipeline", pipeline_pkg)
    monkeypatch.setitem(sys.modules, "app.pipeline.runner", runner_mod)
    monkeypatch.setitem(sys.modules, "app.pipeline.translate", translate_mod)
    monkeypatch.setitem(sys.modules, "app.pipeline.claude", claude_mod)
    return state


@pytest.fixture()
def client(data_dir, fake_pipeline):
    from app.main import app

    with TestClient(app) as c:
        yield c


def _upload(client, tmp_path) -> str:
    import pymupdf

    path = tmp_path / "one.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(str(path))
    with path.open("rb") as fh:
        res = client.post("/api/documents",
                          files={"file": ("one.pdf", fh, "application/pdf")})
    assert res.status_code == 200
    return res.json()["id"]


def _wait_done(client, doc_id):
    with client.stream("GET", f"/api/documents/{doc_id}/events") as res:
        for line in res.iter_lines():
            if line.startswith("event: done") or line.startswith("event: error"):
                break


def test_translate_with_key_selects_openai(client, fake_pipeline, tmp_path,
                                           data_dir, caplog):
    doc_id = _upload(client, tmp_path)
    res = client.post(f"/api/documents/{doc_id}/translate",
                      json={"openai_api_key": TEST_KEY})
    assert res.status_code == 200
    _wait_done(client, doc_id)
    assert fake_pipeline["openai_keys"] == [TEST_KEY]
    assert fake_pipeline["get_translator_calls"] == 0
    # The key must not appear in logs or in any file under the data dir.
    assert TEST_KEY not in caplog.text
    for file in data_dir.rglob("*"):
        if file.is_file():
            assert TEST_KEY.encode() not in file.read_bytes(), str(file)


def test_translate_with_bad_key_format_400(client, tmp_path):
    doc_id = _upload(client, tmp_path)
    res = client.post(f"/api/documents/{doc_id}/translate",
                      json={"openai_api_key": "not-a-key"})
    assert res.status_code == 400


def test_translate_without_key_uses_configured_path(client, fake_pipeline,
                                                    tmp_path):
    doc_id = _upload(client, tmp_path)
    res = client.post(f"/api/documents/{doc_id}/translate")
    assert res.status_code == 200
    _wait_done(client, doc_id)
    assert fake_pipeline["openai_keys"] == []
    assert fake_pipeline["get_translator_calls"] == 1


def test_translate_uses_chosen_model(client, fake_pipeline, tmp_path):
    doc_id = _upload(client, tmp_path)
    res = client.post(f"/api/documents/{doc_id}/translate",
                      json={"openai_api_key": TEST_KEY,
                            "openai_model": "gpt-6-sol"})
    assert res.status_code == 200
    _wait_done(client, doc_id)
    assert fake_pipeline["openai_models"] == ["gpt-6-sol"]


def test_translate_without_model_uses_default(client, fake_pipeline, tmp_path):
    from app import config

    doc_id = _upload(client, tmp_path)
    client.post(f"/api/documents/{doc_id}/translate",
                json={"openai_api_key": TEST_KEY})
    _wait_done(client, doc_id)
    assert fake_pipeline["openai_models"] == [config.DEFAULT_OPENAI_MODEL]


def test_translate_with_unknown_model_400(client, tmp_path):
    doc_id = _upload(client, tmp_path)
    res = client.post(f"/api/documents/{doc_id}/translate",
                      json={"openai_api_key": TEST_KEY,
                            "openai_model": "gpt-99"})
    assert res.status_code == 400


# ------------------------------------------------------- other API providers


def test_gemini_model_uses_openai_compatible_endpoint(client, fake_pipeline, tmp_path):
    doc_id = _upload(client, tmp_path)
    res = client.post(f"/api/documents/{doc_id}/translate",
                      json={"api_key": "AIza-test-key", "api_model": "gemini-3.8-flash"})
    assert res.status_code == 200, res.text
    _wait_done(client, doc_id)
    assert fake_pipeline["openai_keys"] == ["AIza-test-key"]
    assert fake_pipeline["openai_models"] == ["gemini-3.8-flash"]
    kwargs = fake_pipeline["openai_kwargs"][0]
    assert kwargs["base_url"] == "https://generativelanguage.googleapis.com"
    assert kwargs["chat_path"] == "/v1beta/openai/chat/completions"
    assert kwargs["provider"] == "Gemini"


def test_claude_model_uses_claude_translator(client, fake_pipeline, tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERTRANSLATE_CLAUDE_EFFORT", "high")
    doc_id = _upload(client, tmp_path)
    res = client.post(f"/api/documents/{doc_id}/translate",
                      json={"api_key": "sk-ant-test", "api_model": "claude-opus-5-5"})
    assert res.status_code == 200, res.text
    _wait_done(client, doc_id)
    assert fake_pipeline["claude"] == [
        {"key": "sk-ant-test", "model": "claude-opus-5-5", "effort": "high"}]
    assert fake_pipeline["openai_keys"] == []


def test_claude_model_rejects_non_anthropic_key(client, tmp_path):
    doc_id = _upload(client, tmp_path)
    res = client.post(f"/api/documents/{doc_id}/translate",
                      json={"api_key": "sk-proj-openai", "api_model": "claude-opus-5-5"})
    assert res.status_code == 400
    assert "sk-ant-" in res.json()["detail"]


def test_unknown_api_model_rejected(client, tmp_path):
    doc_id = _upload(client, tmp_path)
    res = client.post(f"/api/documents/{doc_id}/translate",
                      json={"api_key": "sk-x", "api_model": "gpt-99"})
    assert res.status_code == 400
