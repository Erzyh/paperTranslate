# Tests for extract.analyze_pdf on the generated sample (criterion group 1).
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.pipeline.extract import _is_math_font, math_span_literal

MATH_FONT_AVAILABLE = Path(r"C:\Windows\Fonts\seguisym.ttf").exists()


def test_analyze_detects_block_types(sample_layout):
    # Pages 3-4 carry the boundary/micro-token constructs (make_sample N-Q).
    assert len(sample_layout.pages) == 4
    page0 = sample_layout.pages[0]
    assert (page0.width, page0.height) == (612.0, 792.0)
    types0 = [block.type for block in page0.blocks]
    assert types0.count("image") == 1
    assert types0.count("drawing") >= 1
    assert types0.count("text") >= 10
    types1 = [block.type for block in sample_layout.pages[1].blocks]
    assert types1.count("text") >= 8
    assert types1.count("image") == 0


def test_text_blocks_have_lines_and_spans(sample_layout):
    for page in sample_layout.pages:
        for block in page.blocks:
            if block.type == "text":
                assert block.lines
                for line in block.lines:
                    assert line.spans
                    for span in line.spans:
                        assert span.size > 0
                        assert len(span.bbox) == 4
            else:
                assert block.lines == []


@pytest.mark.skipif(not MATH_FONT_AVAILABLE, reason="math font sample needs seguisym")
def test_math_font_spans_become_placeholders(sample_layout):
    """Math spans are masked as ⟦EQn⟧ — except literal micro tokens.

    Formula runs must become placeholders, while 1-2 character
    punctuation/relation glyphs (the decimal points of "0.949"/"0.227",
    the "=" of the split row) keep their literal text (micro-token
    literalization, see extract.math_span_literal).
    """
    eq_texts = []
    for page in sample_layout.pages:
        for block in page.blocks:
            for line in block.lines:
                for span in line.spans:
                    if _is_math_font(span.font) and span.text.strip():
                        eq_texts.append(span.text)
    assert eq_texts, "expected math-font spans in the sample"
    masked = [t for t in eq_texts if re.fullmatch(r"⟦EQ\d+⟧", t)]
    literal = [t for t in eq_texts if not re.fullmatch(r"⟦EQ\d+⟧", t)]
    assert masked, "expected masked formula spans"
    assert literal, "expected literalized micro-token spans"
    assert all(math_span_literal(t) is not None for t in literal)


def test_is_math_font_heuristic():
    assert _is_math_font("CMMI10")
    assert _is_math_font("CMSY7")
    assert _is_math_font("CMEX10")
    assert _is_math_font("CambriaMath")
    # MathTime/AMS families observed in real IEEE productions (subset
    # prefixes included, as reported by PyMuPDF).
    assert _is_math_font("KYGEMB+RMTMI")
    assert _is_math_font("ODNYBN+RMTMIB")
    assert _is_math_font("MPBGMH+MTSYN")
    assert _is_math_font("VNGSFB+MTSYB")
    assert _is_math_font("QIPYIM+MSAM10")
    assert _is_math_font("QDTWCG+MSBM10")
    assert _is_math_font("SWLNXX+BLEX")
    assert not _is_math_font("Helvetica")
    assert not _is_math_font("NimbusRomNo9L-Regu")
    assert not _is_math_font("SWRRFW+TimesLTStd-Roman")
    assert not _is_math_font("ENXOBD+FormataOTF-Bold")
    # STIXGeneral is a body-text font in some journals: never a math marker.
    assert not _is_math_font("STIXGeneral-Regular")
