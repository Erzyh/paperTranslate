"""ClaudeTranslator — a fake Anthropic client only; nothing reaches the network."""
from __future__ import annotations

from types import SimpleNamespace

from app.pipeline.claude import ClaudeTranslator


class FakeClaude:
    """Stands in for anthropic.Anthropic: records create() calls."""

    def __init__(self, *responses):
        self.calls: list[dict] = []
        self._responses = list(responses)
        self.closed = False
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses[min(len(self.calls), len(self._responses)) - 1]

    def close(self):
        self.closed = True


def answer(text: str, stop_reason: str = "end_turn"):
    # Thinking blocks (empty under the default display) precede the text.
    return SimpleNamespace(stop_reason=stop_reason, content=[
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text=text),
    ])


def make(*responses) -> tuple[ClaudeTranslator, FakeClaude]:
    fake = FakeClaude(*responses)
    return ClaudeTranslator(api_key="sk-ant-test", client=fake), fake


def test_request_shape_for_opus_5_5():
    translator, fake = make(answer("모델은 빠르게 수렴한다."))
    assert translator.translate("The model converges fast.") == "모델은 빠르게 수렴한다."

    call = fake.calls[0]
    assert call["model"] == "claude-opus-5-5"
    # Thinking can't be disabled and sampling parameters are rejected on
    # Opus 5.5: neither is sent; effort is set explicitly.
    assert "thinking" not in call and "temperature" not in call
    assert call["output_config"] == {"effort": "medium"}
    # Refusal fallback opted in from day one.
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    # The per-paper system prompt is cached; no system role in messages.
    system = call["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert "[번역 예시" in system[0]["text"]
    assert [m["role"] for m in call["messages"]] == ["user"]


def test_guard_retry_keeps_conversation_shape():
    # A lost identifier triggers the corrective retry: the failed answer
    # goes back as an assistant turn, followed by the correction (no prefill).
    translator, fake = make(answer("버트는 강력하다."), answer("BERT는 강력하다."))
    assert translator.translate("BERT is strong.") == "BERT는 강력하다."
    roles = [m["role"] for m in fake.calls[1]["messages"]]
    assert roles == ["user", "assistant", "user"]


def test_refusal_keeps_source_text():
    translator, _ = make(answer("", stop_reason="refusal"))
    assert translator.translate("A paragraph.") == "A paragraph."
    assert translator.last_fallback is True


def test_effort_is_configurable_and_close_releases_client():
    fake = FakeClaude(answer("번역이다."))
    translator = ClaudeTranslator(api_key="sk-ant-test", effort="low", client=fake)
    translator.translate("Hello.")
    assert fake.calls[0]["output_config"] == {"effort": "low"}
    translator.close()
    assert fake.closed
