# Tests for extract.analyze_pdf on the generated sample (criterion group 1).
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.pipeline.extract import (
    _attach_drop_caps,
    _is_math_font,
    is_drop_cap_span,
    is_masked_math_span,
    math_span_literal,
)
from app.pipeline.models import Block, Line, Span

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


def test_text_family_named_math_is_not_math():
    # Nature sets standfirsts in GlosaMath-Roman: a text face, not math.
    assert not _is_math_font("GlosaMath-Roman")
    assert not _is_math_font("ABCDEF+GlosaMath-RomanItalic")
    assert _is_math_font("STIXMath-Italic")
    assert _is_math_font("CambriaMath")
    assert _is_math_font("LatinModernMath-Regular")


def test_drop_cap_is_never_masked():
    assert is_drop_cap_span("M", 54.0)
    assert not is_drop_cap_span("M", 10.0)
    assert not is_drop_cap_span("Mo", 54.0)
    assert not is_masked_math_span("M", "CambriaMath", 54.0)
    assert is_masked_math_span("x", "CambriaMath", 10.0)


def test_drop_cap_joins_paragraph_start():
    cap = Span("M", (34, 505, 60, 545), "Display", 54.0, 0)
    first = Span("ore than 70%", (62, 507, 290, 517), "Body", 9.5, 0)
    second = Span("of researchers", (62, 519, 290, 529), "Body", 9.5, 0)
    blocks = [
        Block("text", (34, 505, 60, 545), [Line([cap], (34, 505, 60, 545))]),
        Block("text", (62, 507, 290, 529), [
            Line([first], (62, 507, 290, 517)),
            Line([second], (62, 519, 290, 529)),
        ]),
    ]
    _attach_drop_caps(blocks)
    assert len(blocks) == 1
    line = blocks[0].lines[0]
    assert "".join(sp.text for sp in line.spans) == "More than 70%"
    assert line.spans[0].size == 9.5
    assert line.bbox[0] == 34 and line.bbox[1] == 507
