# Tests for mask/unmask and StubTranslator (Ollama is never instantiated).
from __future__ import annotations

import re

import pytest

from app.pipeline.translate import (
    MASK_TOKEN_RE,
    StubTranslator,
    get_translator,
    mask,
    unmask,
)

SOURCE = (
    "The loss ⟦EQ3⟧ follows prior work [12] and [3, 7]; see "
    "https://example.org/dataset for the corpus, then ⟦EQ4⟧ applies."
)


def test_mask_hides_equations_citations_urls():
    masked, mapping = mask(SOURCE)
    assert "⟦EQ3⟧" not in masked
    assert "⟦EQ4⟧" not in masked
    assert "[12]" not in masked
    assert "[3, 7]" not in masked
    assert "https://example.org/dataset" not in masked
    assert len(mapping) == 5
    assert set(mapping.keys()) == set(MASK_TOKEN_RE.findall(masked))
    assert sorted(mapping.keys()) == [f"⟦M{i}⟧" for i in range(5)]


def test_mask_unmask_roundtrip():
    masked, mapping = mask(SOURCE)
    assert unmask(masked, mapping) == SOURCE


def test_mask_on_plain_text_is_identity():
    text = "No special tokens in this sentence."
    masked, mapping = mask(text)
    assert masked == text
    assert mapping == {}
    assert unmask(masked, mapping) == text


def test_stub_translator_outputs_korean_and_keeps_tokens():
    masked, _ = mask(SOURCE)
    tokens = MASK_TOKEN_RE.findall(masked)
    result = StubTranslator().translate(masked)
    assert re.search(r"[가-힣]", result), "stub output must contain Hangul"
    assert sorted(MASK_TOKEN_RE.findall(result)) == sorted(tokens)
    for token in tokens:
        assert result.count(token) == 1


def test_stub_translator_length_ratio():
    text = "word " * 80  # 400 chars
    result = StubTranslator().translate(text)
    ratio = len(result) / len(text)
    assert 0.7 <= ratio <= 1.3, f"unexpected length ratio {ratio:.2f}"


def test_get_translator_stub_and_invalid():
    assert isinstance(get_translator("stub"), StubTranslator)
    assert isinstance(get_translator("STUB"), StubTranslator)
    with pytest.raises(ValueError):
        get_translator("papago")
