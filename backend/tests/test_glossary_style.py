"""Glossary and academic-style tests — httpx.MockTransport only, no real server.

Safety contract: nothing here may open a network connection. Every request
is answered in-process by a MockTransport handler, and the base URL uses the
non-resolvable .invalid TLD so an accidental real transport could not reach
localhost:11434 anyway.
"""
from __future__ import annotations

import json

import httpx

from app.pipeline.translate import (
    OllamaTranslator,
    StubTranslator,
    mask,
    parse_glossary,
    strip_leading_label,
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


# ------------------------------------------------------ style system prompt


def test_system_prompt_contains_style_rules():
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return chat_response("번역이다.")

    make_translator(handler).translate("Hello.")

    system = captured[0]["messages"][0]["content"]
    # Loanword rule with its canonical examples.
    assert "Persona→페르소나" in system
    assert "pipeline→파이프라인" in system
    # Anti word-for-word rule and no-label rule.
    assert "직역을 피하고" in system
    assert "라벨·머리말·요약도 붙이지 않는다" in system


# ---------------------------------------------------------- build_glossary


def test_build_glossary_parses_pairs_and_sends_sample():
    sample = "Persona-driven dialogue with a pipeline and baseline models."
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return chat_response(
            "Persona = 페르소나\npipeline = 파이프라인\nbaseline = 베이스라인"
        )

    translator = make_translator(handler)
    glossary = translator.build_glossary(sample)

    # Seed merge canonicalizes the keys of seed terms (Persona/pipeline/baseline
    # are all in the seed and present in the sample).
    assert glossary == {
        "Persona": "페르소나",
        "Pipeline": "파이프라인",
        "Baseline": "베이스라인",
    }
    assert len(captured) == 1
    user_msg = captured[0]["messages"][1]["content"]
    assert sample in user_msg
    assert "영어 = 한국어 관례 표기" in user_msg


def test_build_glossary_ignores_malformed_lines():
    sample = "The transformer attention mechanism is standard."
    answer = (
        "다음은 용어집이다.\n"  # no '=' -> dropped
        "- transformer = 트랜스포머\n"  # bullet prefix tolerated
        " = 좌변 없음\n"  # empty English side -> dropped
        "attention =\n"  # empty Korean side -> dropped
        "attention = 어텐션\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return chat_response(answer)

    glossary = make_translator(handler).build_glossary(sample)
    assert glossary == {"Transformer": "트랜스포머", "Attention": "어텐션"}


def test_build_glossary_drops_terms_absent_from_sample():
    sample = "A pipeline for document translation."

    def handler(request: httpx.Request) -> httpx.Response:
        return chat_response("pipeline = 파이프라인\nhallucinated = 환각 용어")

    glossary = make_translator(handler).build_glossary(sample)
    assert glossary == {"Pipeline": "파이프라인"}, "hallucinated term dropped, seed key kept"


def test_build_glossary_caps_at_40_entries():
    terms = [f"term{i}" for i in range(50)]
    sample = " ".join(terms)
    answer = "\n".join(f"{t} = 용어{i}" for i, t in enumerate(terms))

    def handler(request: httpx.Request) -> httpx.Response:
        return chat_response(answer)

    glossary = make_translator(handler).build_glossary(sample)
    assert len(glossary) == 40


def test_build_glossary_returns_empty_dict_on_transport_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    # Extraction fails but the seed (pipeline occurs in the sample) survives.
    assert make_translator(handler).build_glossary("Some pipeline text.") == {
        "Pipeline": "파이프라인"
    }
    assert make_translator(handler).build_glossary("No seed words here.") == {}


def test_build_glossary_returns_empty_dict_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    assert make_translator(handler).build_glossary("Some pipeline text.") == {
        "Pipeline": "파이프라인"
    }


def test_build_glossary_empty_answer_and_empty_sample():
    def handler(request: httpx.Request) -> httpx.Response:
        return chat_response("")

    translator = make_translator(handler)
    assert translator.build_glossary("Some pipeline text.") == {
        "Pipeline": "파이프라인"
    }

    calls = {"n": 0}

    def counting_handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return chat_response("x = y")

    translator2 = make_translator(counting_handler)
    assert translator2.build_glossary("   ") == {}
    assert calls["n"] == 0, "blank sample must not trigger a chat call"


def test_parse_glossary_direct():
    sample = "Persona pipeline"
    assert parse_glossary("Persona = 페르소나", sample) == {"Persona": "페르소나"}
    assert parse_glossary("no pairs here", sample) == {}
    # Case-insensitive presence check against the sample.
    assert parse_glossary("PIPELINE = 파이프라인", sample) == {
        "PIPELINE": "파이프라인"
    }


def test_glossary_seed_force_included():
    from app.pipeline.translate import apply_glossary_seed

    sample = "A Persona-Driven pipeline for rhythm games."
    # Seed terms present in the sample are included even with empty extraction.
    merged = apply_glossary_seed({}, sample)
    assert merged["Persona"] == "페르소나"
    assert merged["Pipeline"] == "파이프라인"
    assert "Baseline" not in merged, "seed terms absent from the sample are skipped"
    # A conflicting extracted rendering for a seed term is overridden.
    merged = apply_glossary_seed({"persona": "인격"}, sample)
    assert merged["Persona"] == "페르소나"
    assert "persona" not in merged
    # Non-seed extracted entries survive the merge.
    merged = apply_glossary_seed({"Beatmap": "비트맵"}, sample)
    assert merged["Beatmap"] == "비트맵"


def test_glossary_seed_includes_new_academic_terms():
    from app.pipeline.translate import apply_glossary_seed

    sample = (
        "A cold-start setting evaluated against a gold standard label set. "
        "We run an ablation with a fixed random seed, publish a repository, "
        "and study retrieval, entailment, and hallucination."
    )
    merged = apply_glossary_seed({}, sample)

    assert merged["Cold-start"] == "콜드 스타트"
    assert merged["Gold standard"] == "골드 스탠더드"
    assert merged["Ablation"] == "애블레이션"
    assert merged["Random seed"] == "랜덤 시드"
    assert merged["Repository"] == "리포지토리"
    assert merged["Retrieval"] == "검색"
    assert merged["Entailment"] == "함의"
    assert merged["Hallucination"] == "할루시네이션"
    # Terms absent from the sample stay out.
    assert "Baseline" not in merged


def test_build_glossary_failure_still_returns_seed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    translator = make_translator(handler)
    glossary = translator.build_glossary("The Persona baseline experiment.")
    assert glossary["Persona"] == "페르소나"
    assert glossary["Baseline"] == "베이스라인"


# ------------------------------------------------------- glossary injection


def test_glossary_appended_to_system_prompt():
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return chat_response("페르소나 기반 방법이다.")

    translator = make_translator(handler)
    translator.glossary = {"Persona": "페르소나", "baseline": "베이스라인"}
    translator.translate("A persona-driven baseline.")

    system = captured[0]["messages"][0]["content"]
    assert "용어집(반드시 이 표기를 따르라)" in system
    assert "Persona=페르소나" in system
    assert "baseline=베이스라인" in system


def test_glossary_block_echo_triggers_corrective_retry():
    """Echoing the injected glossary scaffold must trip the echo guard."""
    from app.pipeline.translate import find_echo_markers

    echoed = "용어집(반드시 이 표기를 따르라): Persona=페르소나\n페르소나 기반 방법이다."
    assert find_echo_markers(echoed), "glossary header must be an echo marker"

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return chat_response(echoed)
        return chat_response("페르소나 기반 방법이다.")

    translator = make_translator(handler)
    translator.glossary = {"Persona": "페르소나"}
    result = translator.translate("A persona-driven method.")

    assert result == "페르소나 기반 방법이다."
    assert calls["n"] == 2, "echoed glossary block must consume one retry"
    assert translator.last_fallback is False


def test_empty_glossary_leaves_system_prompt_clean():
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return chat_response("번역이다.")

    make_translator(handler).translate("Hello.")
    assert "용어집" not in captured[0]["messages"][0]["content"]


# ------------------------------------------------- leading-label post-process


def test_strip_leading_label_removes_model_added_label():
    source = "We propose a persona-driven method."
    out = strip_leading_label("초록:\n본 논문은 페르소나 기반 방법을 제안한다.", source)
    assert out == "본 논문은 페르소나 기반 방법을 제안한다."


def test_strip_leading_label_keeps_label_when_source_has_colon():
    source = "Abstract: We propose a method."
    text = "초록:\n본 논문은 방법을 제안한다."
    assert strip_leading_label(text, source) == text


def test_strip_leading_label_keeps_long_token_or_single_line():
    source = "We propose a method."
    long_label = "이것은 스무 글자를 넘는 아주 긴 라벨 줄이다:"
    assert len(long_label) > 20
    text_long = long_label + "\n본문이다."
    assert strip_leading_label(text_long, source) == text_long
    # A first line carrying a mask token must never be dropped.
    text_token = "⟦M0⟧ 라벨:\n본문이다."
    assert strip_leading_label(text_token, source) == text_token
    # A single-line answer has no separate label line to strip.
    assert strip_leading_label("요약:", source) == "요약:"
    # A first line not ending with a colon is not a label.
    text_plain = "첫 문장이다.\n둘째 문장이다."
    assert strip_leading_label(text_plain, source) == text_plain


def test_translate_strips_label_first_line():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return chat_response("초록:\n본 논문은 페르소나 기반 방법을 제안한다.")

    translator = make_translator(handler)
    result = translator.translate("We propose a persona-driven method.")

    assert result == "본 논문은 페르소나 기반 방법을 제안한다."
    assert calls["n"] == 1, "label strip must not consume a corrective retry"
    assert translator.last_fallback is False


def test_translate_keeps_label_for_colon_labelled_source():
    def handler(request: httpx.Request) -> httpx.Response:
        return chat_response("초록:\n본 논문은 방법을 제안한다.")

    translator = make_translator(handler)
    result = translator.translate("Abstract: We propose a method.")

    assert result == "초록:\n본 논문은 방법을 제안한다."
    assert translator.last_fallback is False


def test_label_strip_result_must_still_pass_placeholder_guard():
    source_masked, _ = mask("The loss ⟦EQ1⟧ holds.")

    def handler(request: httpx.Request) -> httpx.Response:
        return chat_response("요약:\n토큰 없는 번역이다.")

    translator = make_translator(handler)
    result = translator.translate(source_masked)

    assert result == source_masked, "stripped answer without tokens must be rejected"
    assert translator.last_fallback is True


# ------------------------------------------------------- runner integration


class _GlossaryRecordingTranslator:
    """Fake translator exposing build_glossary; records the sample it gets."""

    def __init__(self, error: Exception | None = None) -> None:
        self.last_fallback = False
        self.glossary: dict[str, str] = {}
        self.samples: list[str] = []
        self._error = error

    def build_glossary(self, sample_text: str) -> dict[str, str]:
        self.samples.append(sample_text)
        if self._error is not None:
            raise self._error
        return {"layout": "레이아웃"}

    def translate(self, text: str, context: str | None = None) -> str:
        return "번역된 문장이다."


def _run_pipeline(sample_pdf, tmp_path, monkeypatch, translator):
    from app.pipeline import runner
    from app.pipeline.models import RenderReport

    # Rendering is covered by test_retypeset; skip it to keep this test fast.
    monkeypatch.setattr(
        runner, "render_translated_pdf",
        lambda src, segments, translations, out: RenderReport(
            overflow_segments=[], scaled_segments={}),
    )
    return runner.run_pipeline(sample_pdf, str(tmp_path / "out.pdf"), translator)


def test_runner_builds_glossary_from_the_whole_paper(
        sample_pdf, tmp_path, monkeypatch):
    translator = _GlossaryRecordingTranslator()
    _run_pipeline(sample_pdf, tmp_path, monkeypatch, translator)

    assert len(translator.samples) == 1, "one glossary call per document"
    sample = translator.samples[0]
    assert len(sample) <= 6000
    assert "Introduction" in sample, "headings must be sampled"
    assert "Recent advances" in sample, "first-page body opening must be sampled"
    assert "Our corpus contains" in sample, "later sections' openings must be sampled"
    assert translator.glossary == {"layout": "레이아웃"}


def test_runner_swallows_glossary_errors(sample_pdf, tmp_path, monkeypatch):
    translator = _GlossaryRecordingTranslator(error=RuntimeError("boom"))
    _run_pipeline(sample_pdf, tmp_path, monkeypatch, translator)

    assert len(translator.samples) == 1
    assert translator.glossary == {}, "failed glossary must stay empty"


def test_runner_skips_glossary_for_stub(sample_pdf, tmp_path, monkeypatch):
    stub = StubTranslator()
    _run_pipeline(sample_pdf, tmp_path, monkeypatch, stub)

    assert not hasattr(stub, "build_glossary")
    assert not hasattr(stub, "glossary"), "runner must not attach a glossary"


def test_glossary_sample_selection_and_cap():
    from app.pipeline.models import Segment
    from app.pipeline.runner import _glossary_sample

    def seg(idx, kind, page, text):
        return Segment(id=f"p{page}_s{idx}", page=page, column=0,
                       bbox=(0.0, 0.0, 1.0, 1.0), text=text, kind=kind,
                       font_size=9.0)

    segments = [
        seg(0, "heading", 0, "Persona-Driven Dialogue"),
        seg(1, "body", 0, "b" * 1500),
        seg(2, "body", 0, "c" * 1500),
        seg(3, "heading", 1, "2 Method"),
        seg(4, "body", 1, "method opening " + "m" * 600),
        seg(5, "body", 1, "second method paragraph"),
        seg(6, "caption", 1, "Figure 1: caption text"),
    ]
    sample = _glossary_sample(segments)

    assert len(sample) <= 6000
    assert sample.startswith("Persona-Driven Dialogue\n2 Method")
    # First-page opening capped at 2000 characters.
    assert "b" * 1500 + "\n" + "c" * 500 in sample
    assert "c" * 501 not in sample
    # First paragraph after each later heading, capped at 400 characters.
    assert "method opening" in sample and "m" * 400 not in sample
    assert "second method paragraph" not in sample
    assert "Figure 1: caption text" in sample


def test_glossary_sample_capped_at_6000_chars():
    from app.pipeline.models import Segment
    from app.pipeline.runner import _glossary_sample

    segments = [
        Segment(id=f"p{i}_s0", page=i, column=0, bbox=(0.0, 0.0, 1.0, 1.0),
                text=f"Heading {i} " + "h" * 200, kind="heading", font_size=9.0)
        for i in range(60)
    ]
    assert len(_glossary_sample(segments)) == 6000
