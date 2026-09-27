# Regression tests for the boundary/micro-token symptom fixes (S1-S5):
# S1 far-apart "(n)" paragraph enumerations split via the document-wide
#    consecutive-sequence rule (across columns and pages),
# S2 mid-sentence segment starts get the continuation prompt instruction,
# S3 visual rows split by inline math are re-assembled in x order (with
#    hyphen restoration) before paragraph merging,
# S4 math-font micro tokens (decimal points, "=", digits) stay literal
#    instead of masking into ⟦EQn⟧ placeholders,
# S5 a flush one-line formula tail closing a body sentence joins the body
#    paragraph instead of stranding as an untranslated fragment.
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.pipeline.extract import (
    _is_math_font,
    is_masked_math_span,
    math_span_literal,
)
from app.pipeline.models import Block, DocumentLayout, Line, PageLayout, Span
from app.pipeline.runner import is_continuation_start, run_pipeline
from app.pipeline.segment import build_segments
from app.pipeline.translate import OllamaTranslator

MATH_FONT_AVAILABLE = Path(r"C:\Windows\Fonts\seguisym.ttf").exists()
BASE_URL = "http://ollama.invalid"


def _line(x0, y0, x1, y1, text, size=10.0, font="Helvetica", flags=0):
    span = Span(text=text, bbox=(x0, y0, x1, y1), font=font, size=size,
                flags=flags)
    return Line(spans=[span], bbox=(x0, y0, x1, y1))


def _block(lines):
    bbox = (
        min(ln.bbox[0] for ln in lines),
        min(ln.bbox[1] for ln in lines),
        max(ln.bbox[2] for ln in lines),
        max(ln.bbox[3] for ln in lines),
    )
    return Block(type="text", bbox=bbox, lines=lines)


def _layout(pages_blocks):
    return DocumentLayout(
        pages=[
            PageLayout(number=i, width=612.0, height=792.0, blocks=blocks)
            for i, blocks in enumerate(pages_blocks)
        ]
    )


# ------------------------------------------------- S4: micro-token literals


def test_math_span_literal_unit():
    # Literal micro tokens: 1-2 whitelisted characters, whitespace kept.
    assert math_span_literal(".") == "."
    assert math_span_literal("=") == "="
    assert math_span_literal(" =") == " ="
    assert math_span_literal(" = −") == " = −"
    assert math_span_literal("12") == "12"
    assert math_span_literal("×") == "×"
    assert math_span_literal("•") == "•"
    assert math_span_literal("≈−") == "≈−"
    # Masked: letters, long runs, control characters, empty.
    assert math_span_literal("ρ") is None
    assert math_span_literal("P") is None
    assert math_span_literal(", . . . ,") is None
    assert math_span_literal("\x11") is None
    assert math_span_literal("∑ α x") is None
    assert math_span_literal("") is None
    assert math_span_literal("   ") is None


def test_is_masked_math_span_unit():
    assert is_masked_math_span("ρ", "KYGEMB+RMTMI")
    assert not is_masked_math_span("=", "MPBGMH+MTSYN")  # literalized
    assert not is_masked_math_span("=", "Helvetica")  # not a math font
    assert not is_masked_math_span("   ", "KYGEMB+RMTMI")  # whitespace


@pytest.mark.skipif(not MATH_FONT_AVAILABLE, reason="needs seguisym math font")
def test_s4_decimal_points_stay_literal_in_layout(sample_layout):
    """The math-font "." spans of the decimal paragraph keep their text."""
    dots = []
    for block in sample_layout.pages[2].blocks:
        for line in block.lines:
            for span in line.spans:
                if _is_math_font(span.font) and span.text == ".":
                    dots.append(span)
    assert len(dots) >= 2, "literal decimal-point math spans missing"


def test_s4_decimals_survive_in_segment_text(sample_segments):
    seg = next(s for s in sample_segments
               if "The two proxy metrics" in s.text)
    assert seg.kind == "body"
    assert "0.949 and 0.227" in seg.text
    assert "⟦EQ" not in seg.text


# ---------------------------------------------- S1: enumeration sequences


def test_s1_sequence_items_split_across_columns_and_pages(sample_segments):
    one = next(s for s in sample_segments if s.text.startswith("(1) Delta"))
    two = next(s for s in sample_segments if s.text.startswith("(2) Spatial"))
    three = next(s for s in sample_segments
                 if s.text.startswith("(3) Vertical"))
    assert (one.page, two.page, three.page) == (2, 2, 3)
    assert one.column != two.column  # items sit in different columns
    for seg, tail in ((one, "saturates the span."),
                      (two, "relocation between beats."),
                      (three, "same lane column.")):
        assert seg.kind == "body"
        assert seg.text.endswith(tail), seg.text[-60:]
    # The flush follow-up paragraph never glues onto the last item.
    follow = next(s for s in sample_segments
                  if "observed rank drifts" in s.text)
    assert follow.id != three.id
    assert "(3)" not in follow.text


def test_s1_sequence_unit_far_apart_items_split():
    """Items far beyond the ±8-line window split via the sequence rule."""
    def item(n, y, label):
        lines = [_line(64, y, 290, y + 10,
                       f"({n}) {label} opens this item paragraph.")]
        for i in range(1, 10):  # 10-line items defeat the neighbor window
            lines.append(_line(54, y + 12 * i, 290, y + 10 + 12 * i,
                               f"item {n} continuation line {i} keeps the"
                               " flush left edge of the body flow."))
        return lines

    intro = [
        _line(54, 76, 290, 86, "The corpus statistics are described next."),
    ]
    page0 = _block(intro + item(1, 100, "Proxy dependence") +
                   item(2, 232, "Sample size"))
    page1 = _block(item(3, 100, "Modality limits"))
    segments = build_segments(_layout([[page0], [page1]]))

    for n, page in ((1, 0), (2, 0), (3, 1)):
        seg = next(s for s in segments if s.text.startswith(f"({n})"))
        assert seg.page == page
        assert seg.kind == "body"
        assert f"item {n} continuation line 9" in seg.text
        for other in (1, 2, 3):
            if other != n:
                assert f"({other})" not in seg.text


def test_s1_isolated_marker_does_not_split():
    """A lone paragraph-leading "(n)" without a sequence stays merged."""
    lines = [
        _line(54, 100, 290, 110, "The first sentence ends cleanly here."),
        _line(64, 112, 290, 122, "(8) Isolated aside continues the flow"),
        _line(54, 124, 290, 134, "of the same paragraph without a list."),
    ]
    segments = build_segments(_layout([[_block(lines)]]))
    starts = [s for s in segments if s.text.startswith("(8)")]
    assert not starts, "isolated (8) must not open its own segment"


# ------------------------------------------------- S3: visual-row reassembly


def test_s3_split_row_reassembled_in_sample(sample_segments):
    seg = next(s for s in sample_segments
               if "External raters" in s.text)
    assert seg.kind == "body"
    assert ("validation pass on a shared pilot (N = 5 raters)"
            " before release." in seg.text)
    # No stranded fragment segment survives.
    assert not any(s.text.startswith("5 raters)") for s in sample_segments)


def test_s3_row_reassembly_unit():
    """Fragments of one visual row re-join in x order (hyphen restored)."""
    lines = [
        _line(54, 100, 290, 110, "The pilot study sections report the"),
        _line(54, 112, 290, 122, "aggregate results for every rater and"),
        _line(54, 124, 240, 134, "close with the note on validation."),
        _line(54, 136, 290, 146, "Limited human vali-"),
        # The right fragment sorts BEFORE the left one on (y0, x0) — the
        # scrambled real-paper shape the reassembly must fix.
        _line(238, 147.0, 290, 157, "= 5 raters,"),
        _line(54, 148.2, 234, 158.6, "dation is limited to a pilot (N"),
        _line(54, 160, 290, 170, "that did not yield a usable ground"),
        _line(54, 172, 200, 182, "truth for the mid-tier charts."),
    ]
    segments = build_segments(_layout([[_block(lines)]]))
    seg = next(s for s in segments if "Limited human" in s.text)
    assert ("Limited human validation is limited to a pilot"
            " (N = 5 raters, that did not yield" in seg.text)


# --------------------------------------------------- S5: flush formula tail


def test_s5_flush_formula_tail_joins_body_unit():
    """A flush one-line formula closing a sentence joins the body flow."""
    filler = _block([
        _line(54, 100 + 12 * i, 290, 110 + 12 * i,
              f"plain filler sentence number {i} sets the body size")
        for i in range(5)
    ])
    # The tail line shares the paragraph's extraction block, as real
    # extraction groups a column's contiguous lines into one block.
    tail_line = Line(
        spans=[
            Span(text="p", bbox=(54, 224, 60, 236),
                 font="TimesLTStd-Italic", size=10.0, flags=2),
            Span(text="<", bbox=(60, 224, 66, 236),
                 font="MPBGMH+MTSYN", size=10.0, flags=0),
            Span(text=" 10−5.", bbox=(66, 224, 120, 236),
                 font="TimesLTStd-Roman", size=10.0, flags=0),
        ],
        bbox=(54, 224, 120, 236),
    )
    para = Block(type="text", bbox=(54, 200, 290, 236), lines=[
        _line(54, 200, 290, 210, "the binomial test rejects the null"),
        _line(54, 212, 290, 222, "hypothesis of random choice at"),
        tail_line,
    ])
    segments = build_segments(_layout([[filler, para]]))
    seg = next(s for s in segments if "binomial test" in s.text)
    assert seg.kind == "body"
    assert seg.text.endswith("random choice at p< 10−5.")
    assert not any(s.kind == "formula" for s in segments)


# --------------------------------------------- S2: continuation-flag prompts


def test_is_continuation_start_unit():
    # Lowercase openings and sentence tails continue the previous segment.
    assert is_continuation_start("truth, and a pairwise extreme-case study")
    assert is_continuation_start("remaining charts with their ranks kept")
    assert is_continuation_start("0.949) that did not yield a ground truth")
    assert is_continuation_start("≈ −0.05. The mean rater correlation was")
    assert is_continuation_start("⟦EQ0⟧ ≈−0.05. The mean rater correlation")
    # Fresh paragraph starts are never continuations.
    assert not is_continuation_start("The proposed method analyzes blocks")
    assert not is_continuation_start("Vertical release timing counts holds")
    assert not is_continuation_start("2026 marks the corpus cutoff year")
    assert not is_continuation_start("본 연구는 레이아웃을 보존한다")
    assert not is_continuation_start("")
    assert not is_continuation_start("⟦EQ0⟧ ⟦EQ1⟧")


def _make_translator(handler) -> OllamaTranslator:
    return OllamaTranslator(
        model="qwen3:8b", base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
    )


def test_s2_continuation_instruction_injected():
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200, json={"message": {"role": "assistant",
                                   "content": "이어지는 번역이다."}}
        )

    translator = _make_translator(handler)
    translator.translate("tail of a broken sentence.", context="이전 문단",
                         continuation=True)
    user_msg = captured[0]["messages"][1]["content"]
    assert "직전 문단에서 이어지는 문장의 뒷부분" in user_msg
    assert "새 문장처럼 시작하지 말고" in user_msg
    assert "tail of a broken sentence." in user_msg

    translator.translate("A fresh paragraph.", context="이전 문단")
    assert "직전 문단에서 이어지는" not in captured[1]["messages"][1]["content"]
    translator.close()


def test_s2_runner_passes_continuation_flag(sample_pdf, tmp_path):
    calls: list[tuple[str, bool]] = []

    class RecordingTranslator:
        def translate(self, text, context=None, continuation=False):
            calls.append((text, continuation))
            return "번역 " + text  # mask tokens preserved verbatim

    out = tmp_path / "cont_out.pdf"
    run_pipeline(sample_pdf, str(out), RecordingTranslator())

    tail = next(c for c in calls if c[0].startswith("remaining charts"))
    assert tail[1] is True, "page-boundary tail must carry continuation=True"
    fresh = next(c for c in calls if c[0].startswith("Delta lane density"))
    assert fresh[1] is False
    proxy = next(c for c in calls if c[0].startswith("The two proxy"))
    assert proxy[1] is False


def test_s2_runner_tolerates_translators_without_the_flag(sample_pdf,
                                                          tmp_path):
    """Legacy translate(text, context) signatures keep working (S2)."""
    seen: list[str] = []

    class LegacyTranslator:
        def translate(self, text, context=None):
            seen.append(text)
            return "번역 " + text

    out = tmp_path / "legacy_out.pdf"
    run_pipeline(sample_pdf, str(out), LegacyTranslator())
    assert any(text.startswith("remaining charts") for text in seen)


# ---------------------------------------- S6: multi-line title heading merge
def test_s6_two_line_title_is_one_heading_segment(sample_segments):
    """Both title lines merge into ONE heading segment (S6): each full-width
    line forms its own band, so only the adjacent-heading merge can join
    them; split halves used to translate separately with different shrink
    factors."""
    titles = [seg for seg in sample_segments
              if seg.page == 0 and seg.kind == "heading"
              and "Layout-Preserving" in seg.text]
    assert len(titles) == 1
    assert titles[0].text == ("Layout-Preserving Neural Translation "
                              "of Two-Column Academic Documents")


def test_s6_numbered_heading_never_joins_previous_heading_unit():
    """Adjacent same-size heading lines merge, but a numbered section title
    keeps its own segment and never absorbs the paragraph below it."""
    title = _block([
        _line(150, 80, 460, 100, "A Two Line Centered Paper", size=20.0),
        _line(170, 104, 440, 124, "Title About Segmentation", size=20.0),
    ])
    body = _block([
        _line(54, 200, 300, 210, "IV. GUARANTEES", size=10.0,
              font="Helvetica-Bold"),
        _line(54, 212, 300, 222, "The guarantees rest on the invariants"),
        _line(54, 224, 300, 234, "that the following sections state."),
    ])
    segments = build_segments(_layout([[title, body]]))
    merged = [seg for seg in segments if "Centered Paper" in seg.text]
    assert len(merged) == 1
    assert merged[0].kind == "heading"
    assert merged[0].text == ("A Two Line Centered Paper "
                              "Title About Segmentation")
    heading = next(seg for seg in segments if "GUARANTEES" in seg.text)
    assert heading.text == "IV. GUARANTEES", \
        "numbered heading must not absorb the following body text"


# ------------------------------------- S7: theorem-environment forced breaks
def test_s7_theorem_paragraphs_split_in_sample(sample_segments):
    """"Definition 2 ..." and "Proof." lines at plain line steps open their
    own segments instead of gluing onto the preceding prose (S7)."""
    def starts(prefix):
        return [seg for seg in sample_segments if seg.text.startswith(prefix)]

    defs = starts("Definition 2 (Transition).")
    proofs = starts("Proof.")
    assert len(defs) == 1 and len(proofs) == 1
    assert defs[0].kind == "body" and proofs[0].kind == "body"
    # The header line keeps its statement body in the SAME segment.
    assert "maps responses to committed states." in defs[0].text
    assert "invariant by construction of the rules." in proofs[0].text
    prev = next(seg for seg in sample_segments
                if "generated by the engine." in seg.text)
    assert "Definition" not in prev.text and "Proof." not in prev.text


def test_s7_mid_sentence_theorem_mention_stays_merged_unit():
    """A wrap line starting with "Theorem 1." mid-sentence (previous line
    does not end a sentence) must stay inside its paragraph."""
    para = _block([
        _line(54, 200, 300, 210, "The claimed bound follows directly from"),
        _line(54, 212, 300, 222, "Theorem 1. For the second regime, the"),
        _line(54, 224, 300, 234, "gate accepted the committed instance."),
    ])
    segments = build_segments(_layout([[para]]))
    bodies = [seg for seg in segments if seg.kind == "body"]
    assert len(bodies) == 1
    assert "follows directly from Theorem 1." in bodies[0].text


def test_s7_theorem_after_sentence_end_splits_unit():
    """A "Lemma 1 (Name)." line after a sentence-ending line opens a new
    segment; "Proofread" style words never trigger the rule."""
    para = _block([
        _line(54, 200, 300, 210, "The previous paragraph closes here."),
        _line(54, 212, 300, 222, "Lemma 1 (Realizability). Every emitted"),
        _line(54, 224, 300, 234, "transcript is realizable by some device."),
        _line(54, 236, 300, 246, "Proofread copies of the text were kept"),
        _line(54, 248, 300, 258, "with the archived experiment bundles."),
    ])
    segments = build_segments(_layout([[para]]))
    texts = [seg.text for seg in segments if seg.kind == "body"]
    assert any(t.startswith("Lemma 1 (Realizability).") for t in texts)
    lemma = next(t for t in texts if t.startswith("Lemma 1"))
    assert "Proofread copies" in lemma, \
        "'Proofread' must not fire the theorem-head rule"
