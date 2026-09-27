"""OllamaTranslator tests — httpx.MockTransport only, no real server.

Safety contract: nothing here may open a network connection. Every request
is answered in-process by a MockTransport handler, and the base URL uses the
non-resolvable .invalid TLD so an accidental real transport could not reach
localhost:11434 anyway.
"""
from __future__ import annotations

import json

import httpx
import pytest

from app.pipeline.translate import (
    MASK_TOKEN_RE,
    OllamaTranslator,
    TranslatorUnavailableError,
    find_echo_markers,
    find_foreign_script_chars,
    find_placeholder_errors,
    get_translator,
    mask,
    strip_echo_lines,
    strip_think_blocks,
)

BASE_URL = "http://ollama.invalid"


def make_translator(handler, **kwargs) -> OllamaTranslator:
    return OllamaTranslator(
        model=kwargs.pop("model", "qwen3:8b"),
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200, json={"message": {"role": "assistant", "content": content}}
    )


# ---------------------------------------------------------------- helpers


def test_strip_think_blocks_multiline_and_multiple():
    text = "<think>step 1\nstep 2</think>결과다.<think>again</think> 끝."
    assert strip_think_blocks(text) == "결과다. 끝."


def test_find_placeholder_errors_missing_and_duplicated():
    source = "A ⟦M0⟧ B ⟦M1⟧ C"
    assert find_placeholder_errors(source, "A ⟦M0⟧ B ⟦M1⟧") == []
    assert find_placeholder_errors(source, "B ⟦M1⟧") == ["⟦M0⟧"]
    assert find_placeholder_errors(source, "⟦M0⟧ ⟦M0⟧ ⟦M1⟧") == ["⟦M0⟧"]
    assert find_placeholder_errors("plain", "아무거나") == []


def test_find_placeholder_errors_hallucinated_token():
    # Tokens absent from the source must not appear in the translation —
    # they would be restored as literal garbage in the output PDF.
    source = "A ⟦M0⟧ B"
    assert find_placeholder_errors(source, "가 ⟦M0⟧ 나 ⟦M7⟧") == ["⟦M7⟧"]
    assert find_placeholder_errors("plain", "아무거나 ⟦M3⟧") == ["⟦M3⟧"]


# ------------------------------------------------------- request contract


def test_request_payload_shape():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return chat_response("번역된 문장이다.")

    translator = make_translator(handler)
    result = translator.translate("Hello world.")

    assert result == "번역된 문장이다."
    assert translator.last_fallback is False
    assert len(requests) == 1
    request = requests[0]
    assert request.url.path == "/api/chat"
    payload = json.loads(request.content.decode("utf-8"))
    assert payload["model"] == "qwen3:8b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"] == {"temperature": 0.2, "num_ctx": 8192}
    assert payload["messages"][0]["role"] == "system"
    assert "플레이스홀더" in payload["messages"][0]["content"]
    assert payload["messages"][1]["role"] == "user"
    assert "Hello world." in payload["messages"][1]["content"]


def test_context_included_and_truncated_to_400_chars():
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return chat_response("번역이다.")

    translator = make_translator(handler)
    translator.translate("Hello.", context="x" * 500)

    user_msg = captured[0]["messages"][1]["content"]
    assert "이전 문단(문맥 참고용, 번역하지 말 것):" in user_msg
    assert "x" * 400 in user_msg
    assert "x" * 401 not in user_msg


def test_no_context_means_no_context_header():
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return chat_response("번역이다.")

    make_translator(handler).translate("Hello.")
    assert "이전 문단" not in captured[0]["messages"][1]["content"]


# ------------------------------------------------------------ think blocks


def test_think_block_removed_from_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return chat_response(
            "<think>let me reason\nacross lines</think>\n번역 결과다."
        )

    assert make_translator(handler).translate("Hello.") == "번역 결과다."


# ------------------------------------------------------ placeholder guard


def test_placeholder_missing_then_corrective_retry_succeeds():
    source_masked, _ = mask("The loss ⟦EQ1⟧ holds [12].")
    tokens = MASK_TOKEN_RE.findall(source_masked)
    assert len(tokens) == 2
    good = f"손실 {tokens[0]} 은 {tokens[1]} 에서 성립한다."
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        if len(captured) == 1:
            return chat_response(f"손실은 성립한다 {tokens[0]}")  # drops tokens[1]
        return chat_response(good)

    translator = make_translator(handler)
    result = translator.translate(source_masked)

    assert result == good
    assert translator.last_fallback is False
    assert len(captured) == 2
    # The retry keeps the failed answer and appends a corrective instruction.
    retry_messages = captured[1]["messages"]
    assert retry_messages[-2]["role"] == "assistant"
    assert retry_messages[-1]["role"] == "user"
    assert "플레이스홀더 오류가 있다" in retry_messages[-1]["content"]
    assert tokens[1] in retry_messages[-1]["content"]


def test_placeholder_failure_after_two_retries_falls_back_to_source():
    source_masked, _ = mask("See ⟦EQ1⟧ and [3].")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return chat_response("플레이스홀더가 몽땅 사라진 번역이다.")

    translator = make_translator(handler)
    result = translator.translate(source_masked)

    assert calls["n"] == 3, "initial attempt + 2 corrective retries"
    assert result == source_masked, "fallback must return the source unchanged"
    assert translator.last_fallback is True

    # A following successful call resets the flag.
    ok_source, _ = mask("Plain sentence.")

    def ok_handler(request: httpx.Request) -> httpx.Response:
        return chat_response("평범한 문장이다.")

    translator2 = make_translator(ok_handler)
    translator2.last_fallback = True
    assert translator2.translate(ok_source) == "평범한 문장이다."
    assert translator2.last_fallback is False


def test_sentence_split_retry_recovers_dense_paragraph_partially():
    # Two sentences; the paragraph-level pass and the token sentence keep
    # failing, but the plain sentence succeeds -> mixed output, no fallback.
    source_masked, _ = mask("The loss ⟦EQ1⟧ holds. Plain closing remark.")
    tokens = MASK_TOKEN_RE.findall(source_masked)
    assert len(tokens) == 1

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        user_texts = [m["content"] for m in body["messages"] if m["role"] == "user"]
        if any("Plain closing remark." in t and tokens[0] not in t for t in user_texts):
            return chat_response("평이한 마무리 문장이다.")
        return chat_response("토큰이 사라진 번역이다.")

    translator = make_translator(handler)
    result = translator.translate(source_masked)

    assert "평이한 마무리 문장이다." in result
    assert f"The loss {tokens[0]} holds." in result, "failed sentence keeps its source"
    assert translator.last_fallback is False


def test_sentence_split_all_fail_still_falls_back_to_source():
    source_masked, _ = mask("First ⟦EQ1⟧ fails. Second ⟦EQ2⟧ fails too.")

    def handler(request: httpx.Request) -> httpx.Response:
        return chat_response("토큰 없는 번역이다.")

    translator = make_translator(handler)
    result = translator.translate(source_masked)

    assert result == source_masked
    assert translator.last_fallback is True


# -------------------------------------------------------------- echo guard


def test_find_echo_markers_and_strip_lines():
    echoed = (
        "다음 텍스트를 한국어로 번역하라:\n"
        "제안한 방법은 각 페이지의 레이아웃을 보존한다."
    )
    assert find_echo_markers(echoed) == ["다음 텍스트를 한국어로 번역하라"]
    assert strip_echo_lines(echoed) == "제안한 방법은 각 페이지의 레이아웃을 보존한다."
    # Plain translations that merely contain the word "번역" never match.
    assert find_echo_markers("이 논문은 기계 번역 품질을 다룬다.") == []


def test_echo_response_triggers_corrective_retry_then_succeeds():
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        if len(captured) == 1:
            return chat_response(
                "다음 텍스트를 한국어로 번역하라:\n손실 함수는 수렴한다."
            )
        return chat_response("손실 함수는 수렴한다.")

    translator = make_translator(handler)
    result = translator.translate("The loss function converges.")

    assert result == "손실 함수는 수렴한다."
    assert translator.last_fallback is False
    assert len(captured) == 2
    # The retry keeps the echoed answer and appends the echo correction.
    retry_messages = captured[1]["messages"]
    assert retry_messages[-2]["role"] == "assistant"
    assert retry_messages[-1]["role"] == "user"
    assert "지시문이나 문맥 라벨을 출력하지 말고" in retry_messages[-1]["content"]


def test_persistent_echo_recovered_by_line_strip():
    # Every response echoes both the context label and the instruction line;
    # after the retry budget the marker lines are stripped away.
    echoed = (
        "이전 문단(문맥 참고용, 번역하지 말 것):\n"
        "다음 텍스트를 한국어로 번역하라:\n"
        "제안한 방법은 각 페이지의 레이아웃을 보존한다."
    )
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return chat_response(echoed)

    translator = make_translator(handler)
    result = translator.translate(
        "The proposed method preserves the layout of each page.",
        context="Earlier paragraph.",
    )

    assert calls["n"] == 3, "initial attempt + 2 corrective retries"
    assert result == "제안한 방법은 각 페이지의 레이아웃을 보존한다."
    assert translator.last_fallback is False


def test_echo_strip_too_short_falls_back_to_source():
    # Stripping leaves almost nothing (<30% of the source), so the guard
    # refuses the remainder and falls back to the source text.
    source = "This is a fairly long sentence that the model refuses to translate."

    def handler(request: httpx.Request) -> httpx.Response:
        return chat_response("다음 텍스트를 한국어로 번역하라:\n짧다.")

    translator = make_translator(handler)
    result = translator.translate(source)

    assert result == source
    assert translator.last_fallback is True


def test_echo_strip_result_must_still_pass_placeholder_guard():
    # The only copy of the mask token sits on the echoed instruction line, so
    # stripping loses it; the placeholder guard must reject the remainder.
    source_masked, _ = mask("The loss ⟦EQ1⟧ converges quickly in practice.")
    tokens = MASK_TOKEN_RE.findall(source_masked)
    assert len(tokens) == 1

    def handler(request: httpx.Request) -> httpx.Response:
        return chat_response(
            f"다음 텍스트를 한국어로 번역하라: 손실 {tokens[0]}\n"
            "실제로는 빠르게 수렴한다는 것이 알려져 있다."
        )

    translator = make_translator(handler)
    result = translator.translate(source_masked)

    assert result == source_masked
    assert translator.last_fallback is True


def test_word_translation_in_output_is_not_false_positive():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return chat_response("이 논문은 기계 번역과 요약 기법을 다룬다.")

    translator = make_translator(handler)
    result = translator.translate(
        "This paper covers machine translation and summarization."
    )

    assert calls["n"] == 1, "no corrective retry may be triggered"
    assert result == "이 논문은 기계 번역과 요약 기법을 다룬다."
    assert translator.last_fallback is False


def test_sentence_split_path_applies_echo_guard():
    # Paragraph pass keeps losing the token; in the per-sentence pass the
    # plain sentence first echoes, then recovers via the echo retry.
    source_masked, _ = mask("First ⟦EQ1⟧ stays. Plain closing remark.")
    tokens = MASK_TOKEN_RE.findall(source_masked)
    assert len(tokens) == 1
    plain_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        first_user = next(
            m["content"] for m in body["messages"] if m["role"] == "user"
        )
        if "Plain closing remark." in first_user and tokens[0] not in first_user:
            plain_calls["n"] += 1
            if plain_calls["n"] == 1:
                return chat_response(
                    "다음 텍스트를 한국어로 번역하라:\n평이한 마무리 문장이다."
                )
            return chat_response("평이한 마무리 문장이다.")
        return chat_response("토큰이 사라진 번역이다.")

    translator = make_translator(handler)
    result = translator.translate(source_masked)

    assert "평이한 마무리 문장이다." in result
    assert "다음 텍스트를" not in result
    assert f"First {tokens[0]} stays." in result, "failed sentence keeps its source"
    assert translator.last_fallback is False
    assert plain_calls["n"] == 2, "echo detected once, corrected once"


# ------------------------------------------------------ foreign-script guard


def test_find_foreign_script_chars_detects_han_kana_cyrillic():
    assert find_foreign_script_chars("src", "활성화된但我们 상태다.") == \
        sorted(set("但我们"))
    assert find_foreign_script_chars("src", "결과는 テスト 값이다.") == \
        sorted(set("テスト"))
    assert find_foreign_script_chars("src", "결과는 Привет 값이다.") == \
        sorted(set("Привет"))
    # Korean, Latin, digits and punctuation are all fine.
    assert find_foreign_script_chars("src", "정상 한국어 English 123 (fig).") == []


def test_find_foreign_script_chars_exempts_source_characters():
    # A CJK title cited in the source may legitimately appear in the output.
    source = "See the paper '模型分析' for details."
    assert find_foreign_script_chars(source, "논문 '模型分析'을 참조한다.") == []
    # Characters beyond the source's own set are still flagged.
    assert find_foreign_script_chars(source, "논문 '模型分析'을但 참조한다.") == ["但"]


def test_han_mix_triggers_corrective_retry_then_succeeds():
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        if len(captured) == 1:
            return chat_response("보상이 활성화된但我们 상태를 유지한다.")
        return chat_response("보상이 활성화된 상태를 유지한다.")

    translator = make_translator(handler)
    result = translator.translate("The reward remains active.")

    assert result == "보상이 활성화된 상태를 유지한다."
    assert translator.last_fallback is False
    assert len(captured) == 2
    # The retry keeps the failed answer and appends the script correction.
    retry_messages = captured[1]["messages"]
    assert retry_messages[-2]["role"] == "assistant"
    assert retry_messages[-1]["role"] == "user"
    assert "한자·가나·키릴 문자를 쓰지 말고" in retry_messages[-1]["content"]
    assert "但" in retry_messages[-1]["content"]


@pytest.mark.parametrize("bad", [
    "결과는 テスト 값이다.",       # kana
    "결과는 Привет 값이다.",      # Cyrillic
])
def test_kana_and_cyrillic_trigger_corrective_retry(bad):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return chat_response(bad)
        return chat_response("결과는 검증된 값이다.")

    translator = make_translator(handler)
    result = translator.translate("The result is a validated value.")

    assert result == "결과는 검증된 값이다."
    assert calls["n"] == 2
    assert translator.last_fallback is False


def test_clean_korean_translation_never_triggers_script_retry():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return chat_response("이 방법은 콜드 스타트 문제를 다룬다.")

    translator = make_translator(handler)
    result = translator.translate("This method addresses the cold-start problem.")

    assert calls["n"] == 1, "no corrective retry may be triggered"
    assert result == "이 방법은 콜드 스타트 문제를 다룬다."
    assert translator.last_fallback is False


def test_persistent_han_falls_back_to_sentence_path():
    # The paragraph pass and the first sentence keep mixing Han even in the
    # plain last-chance call, so that sentence keeps its source; the second
    # sentence comes out clean. No Han character may survive into the result.
    source = "First finding holds. Second finding holds too."

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        user_texts = [m["content"] for m in body["messages"] if m["role"] == "user"]
        if any("Second finding holds too." in t and "First finding" not in t
               for t in user_texts):
            return chat_response("두 번째 발견도 성립한다.")
        return chat_response("첫 번째 발견은 성립한다但我们")

    translator = make_translator(handler)
    result = translator.translate(source)

    assert "두 번째 발견도 성립한다." in result
    assert "First finding holds." in result, "failed sentence keeps its source"
    assert "但" not in result
    assert translator.last_fallback is False


def test_mask_free_sentence_gets_plain_last_chance_retry():
    # Guarded attempts (1 initial + 2 corrective) keep emitting Han for the
    # first sentence; the 4th call for it is the plain last-chance retry and
    # its clean answer is accepted. This also pins the 1->2 sentence budget.
    source = "First finding holds. Second finding holds too."
    first_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        user_texts = [m["content"] for m in body["messages"] if m["role"] == "user"]
        if any("First finding holds." in t and "Second finding" not in t
               for t in user_texts):
            first_calls["n"] += 1
            if first_calls["n"] >= 4:
                return chat_response("첫 번째 발견은 성립한다.")
            return chat_response("첫 번째 발견은但 성립한다.")
        if any("Second finding holds too." in t and "First finding" not in t
               for t in user_texts):
            return chat_response("두 번째 발견도 성립한다.")
        return chat_response("문단 번역이但 실패한다.")  # paragraph pass fails

    translator = make_translator(handler)
    result = translator.translate(source)

    assert result == "첫 번째 발견은 성립한다. 두 번째 발견도 성립한다."
    assert first_calls["n"] == 4, \
        "1 attempt + 2 corrective retries + 1 plain last chance"
    assert translator.last_fallback is False


def test_plain_last_chance_retry_rejects_hallucinated_mask_token():
    # The sentence has no mask token, so a token in the plain-retry answer
    # would collide with the paragraph mapping at unmask time -> keep source.
    source = "First finding holds. Second finding holds too."

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        user_texts = [m["content"] for m in body["messages"] if m["role"] == "user"]
        if any("First finding holds." in t and "Second finding" not in t
               for t in user_texts):
            # Guarded attempts hallucinate a token; so does the plain retry.
            return chat_response("첫 번째 발견은 ⟦M9⟧ 성립한다.")
        if any("Second finding holds too." in t and "First finding" not in t
               for t in user_texts):
            return chat_response("두 번째 발견도 성립한다.")
        return chat_response("문단 번역이 ⟦M9⟧ 실패한다.")

    translator = make_translator(handler)
    result = translator.translate(source)

    assert "First finding holds." in result, "hallucinated token must be refused"
    assert "⟦M9⟧" not in result
    assert "두 번째 발견도 성립한다." in result
    assert translator.last_fallback is False


# --------------------------------------------------------- error mapping


def test_connect_error_raises_translator_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(TranslatorUnavailableError) as excinfo:
        make_translator(handler).translate("Hello.")
    assert "Ollama 서버에 연결할 수 없습니다" in str(excinfo.value)


def test_timeout_raises_translator_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(TranslatorUnavailableError) as excinfo:
        make_translator(handler).translate("Hello.")
    # Timeouts get their own message so a busy GPU is not misreported as a
    # connection failure.
    assert "응답 시간이 초과" in str(excinfo.value)


# ----------------------------------------------------------- get_translator


def test_get_translator_ollama_reads_env_config(monkeypatch):
    monkeypatch.setenv("PAPERTRANSLATE_OLLAMA_MODEL", "qwen3:4b")
    monkeypatch.setenv("PAPERTRANSLATE_OLLAMA_URL", "http://ollama.invalid:11435")
    monkeypatch.setenv("PAPERTRANSLATE_OLLAMA_TIMEOUT", "42")

    translator = get_translator("ollama")
    # Constructing the translator opens no connection; no request is sent.
    assert isinstance(translator, OllamaTranslator)
    assert translator.model == "qwen3:4b"
    assert translator.base_url == "http://ollama.invalid:11435"
    assert translator.timeout == 42.0


def test_get_translator_ollama_defaults(monkeypatch):
    for var in ("PAPERTRANSLATE_OLLAMA_MODEL", "PAPERTRANSLATE_OLLAMA_URL",
                "PAPERTRANSLATE_OLLAMA_TIMEOUT"):
        monkeypatch.delenv(var, raising=False)

    translator = get_translator("ollama")
    assert isinstance(translator, OllamaTranslator)
    assert translator.model == "qwen3:8b"
    assert translator.base_url == "http://localhost:11434"
    assert translator.timeout == 180.0


# ------------------------------------------------------ runner integration


class _AlwaysFallbackTranslator:
    """Fake translator mimicking the guard's terminal fallback behavior."""

    def __init__(self) -> None:
        self.last_fallback = False

    def translate(self, text: str, context: str | None = None) -> str:
        self.last_fallback = True
        return text


def _run_pipeline_collecting_statuses(sample_pdf, tmp_path, monkeypatch,
                                      translator):
    from app.pipeline import runner
    from app.pipeline.models import RenderReport

    # Rendering is covered by test_retypeset; skip it to keep this test fast.
    monkeypatch.setattr(
        runner, "render_translated_pdf",
        lambda src, segments, translations, out: RenderReport(
            overflow_segments=[], scaled_segments={}),
    )
    calls: list[tuple[object, str | None, str | None]] = []

    def segment_cb(segment, translated=None, **kwargs):
        calls.append((segment, translated, kwargs.get("status")))

    runner.run_pipeline(sample_pdf, str(tmp_path / "out.pdf"), translator,
                        segment_cb=segment_cb)
    return calls


def test_runner_marks_fallback_segments(sample_pdf, tmp_path, monkeypatch):
    calls = _run_pipeline_collecting_statuses(
        sample_pdf, tmp_path, monkeypatch, _AlwaysFallbackTranslator())

    translated_calls = [c for c in calls if c[1] is not None]
    assert translated_calls, "expected translatable segments in the sample"
    for segment, translated, status in translated_calls:
        assert status == "fallback"
        # The guard returns the source; unmask restores the original text.
        assert translated == segment.text
    # header_footer segments are reported without a status keyword.
    for _, translated, status in calls:
        if translated is None:
            assert status is None


def test_runner_marks_done_segments_for_stub(sample_pdf, tmp_path, monkeypatch):
    from app.pipeline.translate import StubTranslator

    calls = _run_pipeline_collecting_statuses(
        sample_pdf, tmp_path, monkeypatch, StubTranslator())

    translated_calls = [c for c in calls if c[1] is not None]
    assert translated_calls
    assert all(status == "done" for _, _, status in translated_calls)
