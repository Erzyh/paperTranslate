"""Translation-quality round: context, examples, brief, and the guards that
keep untranslatable text (names, acronyms, placeholders, identifiers) intact.

httpx.MockTransport only — nothing here opens a network connection.
"""
from __future__ import annotations

import json

import httpx

from app.pipeline.translate import (
    OllamaTranslator,
    find_missing_identifiers,
    find_placeholder_errors,
    mask,
    unmask,
)

BASE_URL = "http://ollama.invalid"


def make_translator(handler) -> OllamaTranslator:
    return OllamaTranslator(model="qwen3:8b", base_url=BASE_URL,
                            transport=httpx.MockTransport(handler))


def chat_response(content: str) -> httpx.Response:
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})


def scripted(*answers):
    """Handler answering in order and recording each request body."""
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return chat_response(answers[min(len(captured), len(answers)) - 1])

    return handler, captured


# ------------------------------------------------------------ identifiers


def test_identifiers_kept_verbatim_pass():
    src = "We fine-tune GPT-4 and two LLMs on ImageNet with PyTorch."
    out = "우리는 PyTorch로 ImageNet에서 GPT-4와 두 LLM을 파인튜닝한다."
    assert find_missing_identifiers(src, out) == []


def test_transliterated_identifiers_reported():
    src = "BERT outperforms ResNet-50 on COCO."
    out = "버트는 레즈넷-50보다 코코에서 뛰어나다."
    assert find_missing_identifiers(src, out) == ["BERT", "ResNet-50", "COCO"]


def test_ordinary_words_are_not_identifiers():
    # Title-Case words, roman numerals, dotted abbreviations and a single
    # capital are ordinary text and may be translated.
    src = "Multi-Task Learning in the U.S. is covered in Section IV, e.g. here."
    assert find_missing_identifiers(src, "미국의 멀티태스크 학습은 IV절에서 다룬다.") == []


def test_all_caps_heading_is_not_checked():
    assert find_missing_identifiers("VII. CONCLUSION AND FUTURE WORK", "VII. 결론 및 향후 과제") == []


def test_missing_identifier_triggers_corrective_retry():
    handler, captured = scripted("버트는 강력하다.", "BERT는 강력하다.")
    result = make_translator(handler).translate("BERT is strong.")
    assert result == "BERT는 강력하다."
    assert len(captured) == 2
    assert "BERT" in captured[1]["messages"][-1]["content"]
    assert "원문 표기 그대로" in captured[1]["messages"][-1]["content"]


def test_missing_identifier_is_soft_after_retries():
    # A name the model keeps transliterating must not throw the paragraph back
    # to English: the last answer is accepted.
    handler, captured = scripted("버트는 강력하다.")
    translator = make_translator(handler)
    assert translator.translate("BERT is strong.") == "버트는 강력하다."
    assert translator.last_fallback is False
    assert len(captured) == 3  # first answer + 2 corrective retries


# ------------------------------------------------------------ placeholders


def test_hallucinated_equation_token_is_an_error():
    # ⟦EQn⟧ tokens come from context the model saw; copying one into the
    # answer would be read as an equation slot downstream.
    assert find_placeholder_errors("손실 ⟦M0⟧", "손실 ⟦M0⟧ ⟦EQ3⟧") == ["⟦EQ3⟧"]


def test_mask_protects_email_doi_and_arxiv():
    text = ("Contact a.kim@lab.ac.kr, see doi 10.1145/3544548.3581388 "
            "and arXiv:2401.01234v2 [3].")
    masked, mapping = mask(text)
    for original in ("a.kim@lab.ac.kr", "10.1145/3544548.3581388",
                     "arXiv:2401.01234v2", "[3]"):
        assert original not in masked
        assert original in mapping.values()
    assert unmask(masked, mapping) == text


# ------------------------------------------------ context, examples, brief


def test_previous_translation_in_user_message_and_sanitized():
    handler, captured = scripted("두 번째 문단이다.")
    make_translator(handler).translate(
        "Second paragraph.", context="First ⟦EQ1⟧ paragraph.",
        previous_translation="첫 문단 ⟦EQ1⟧이다.")
    user = captured[0]["messages"][1]["content"]
    assert "이전 문단의 번역(" in user
    assert "첫 문단 (수식)이다." in user
    assert "⟦EQ1⟧" not in user, "context must not carry copyable tokens"


def test_repeating_previous_translation_is_an_echo():
    previous = "이 연구는 레이아웃을 보존하는 문서 번역 기법을 제안하고 그 효과를 검증한다."
    handler, captured = scripted(previous, "새 문단의 번역이다.")
    result = make_translator(handler).translate(
        "A new paragraph.", previous_translation=previous)
    assert result == "새 문단의 번역이다."
    assert len(captured) == 2


def test_system_prompt_has_examples_and_brief():
    handler, captured = scripted("번역이다.")
    translator = make_translator(handler)
    translator.document_brief = "제목: Layout-Preserving Translation\n초록 일부: We study ..."
    translator.translate("Hello.")
    system = captured[0]["messages"][0]["content"]
    assert "[번역 예시" in system
    assert "[논문 정보" in system and "Layout-Preserving Translation" in system
    assert "모델·데이터셋·도구 이름" in system


def test_echoed_few_shot_answer_is_rejected():
    handler, captured = scripted(
        "우리는 ImageNet으로 GPT-4를 파인튜닝하고, 표 2에 BLEU 점수를 보고한다.",
        "모델은 빠르게 수렴한다.")
    assert make_translator(handler).translate("The model converges fast.") == "모델은 빠르게 수렴한다."


# ------------------------------------------------------------------ runner


class _RecordingTranslator:
    """Records translate() kwargs; the 2nd call reports a fallback."""

    def __init__(self):
        self.calls: list[dict] = []
        self.document_brief = ""
        self.last_fallback = False

    def translate(self, text, context=None, continuation=False,
                  previous_translation=None):
        self.calls.append({"text": text, "previous": previous_translation})
        self.last_fallback = len(self.calls) == 2
        return text if self.last_fallback else f"번역{len(self.calls)}"


def test_runner_passes_previous_translation_and_brief(sample_pdf, tmp_path, monkeypatch):
    from app.pipeline import runner
    from app.pipeline.models import RenderReport

    monkeypatch.setattr(runner, "render_translated_pdf",
                        lambda *a: RenderReport(overflow_segments=[], scaled_segments={}))
    translator = _RecordingTranslator()
    runner.run_pipeline(sample_pdf, str(tmp_path / "out.pdf"), translator)

    calls = translator.calls
    assert calls[0]["previous"] is None
    assert calls[1]["previous"].endswith("번역1")
    # The 2nd paragraph fell back to English: it is not offered as context.
    assert calls[2]["previous"] is None
    assert calls[3]["previous"].endswith("번역3")
    assert translator.document_brief.startswith("제목: ")
