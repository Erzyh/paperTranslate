# Tests for formula rendering: whole-bbox protection of untranslated
# segments overlapped by redactions (symptom A) and inline-flow insertion
# of ⟦EQn⟧ runs into the translated text (symptom B), plus the safety
# fallback when token/run matching fails.
from __future__ import annotations

import re
from pathlib import Path

import pymupdf
import pytest

MATH_FONT_FILE = Path(r"C:\Windows\Fonts\seguisym.ttf")
needs_math_font = pytest.mark.skipif(
    not MATH_FONT_FILE.exists(), reason="needs seguisym math font"
)


@pytest.fixture(scope="module")
def flow_out(sample_pdf, tmp_path_factory):
    """Sample pipeline output rendered once for this module."""
    from app.pipeline.runner import run_pipeline
    from app.pipeline.translate import StubTranslator

    out = tmp_path_factory.mktemp("flow") / "flow_out.pdf"
    report = run_pipeline(sample_pdf, str(out), StubTranslator())
    return str(out), report


def _nonwhite_pixels(page: pymupdf.Page, clip: pymupdf.Rect) -> int:
    pix = page.get_pixmap(clip=clip, matrix=pymupdf.Matrix(2, 2))
    n = pix.n
    samples = pix.samples
    count = 0
    for offset in range(0, len(samples), n):
        if any(samples[offset + c] < 250 for c in range(min(n, 3))):
            count += 1
    return count


def _hangul_line_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    rects = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s.get("text", "") for s in line.get("spans", []))
            if any("가" <= ch <= "힣" for ch in text):
                rects.append(pymupdf.Rect(line["bbox"]))
    return rects


def _math_span_rects(path: str, pno: int) -> list[pymupdf.Rect]:
    from app.pipeline.extract import _is_math_font

    doc = pymupdf.open(path)
    try:
        rects = []
        for block in doc.load_page(pno).get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["text"].strip() and _is_math_font(span["font"]):
                        rects.append(pymupdf.Rect(span["bbox"]))
        return rects
    finally:
        doc.close()


def test_htmlbox_image_probe_positive():
    """This platform must support at least one htmlbox image mechanism."""
    from app.pipeline.retypeset import _htmlbox_supports_images

    assert _htmlbox_supports_images() in ("data", "archive")


@needs_math_font
def test_overlapped_formula_bbox_fully_protected(
        sample_pdf, sample_segments, flow_out):
    """Symptom A: a formula segment overlapped by a translated paragraph's
    redact rect keeps ALL its glyphs (math font AND regular font), verified
    by comparing rendered ink inside the formula bbox with the source."""
    formula = next(
        s for s in sample_segments
        if s.page == 1 and s.kind == "formula" and "0.42" in s.text
    )
    body = next(
        s for s in sample_segments
        if s.page == 1 and "bullet criteria" in s.text
    )
    frect = pymupdf.Rect(*formula.bbox)
    # Reproduction precondition: the two bboxes really do overlap, so the
    # redaction of the body paragraph would erase formula glyphs.
    assert frect.intersects(pymupdf.Rect(*body.bbox))

    out_path, _report = flow_out
    src = pymupdf.open(sample_pdf)
    out = pymupdf.open(out_path)
    try:
        src_ink = _nonwhite_pixels(src.load_page(1), frect)
        out_ink = _nonwhite_pixels(out.load_page(1), frect)
        # The whole-bbox snapshot restores every glyph: the regular-font
        # tail "= 0.42 (3)" must not vanish. Translated text may add ink
        # inside the overlap strip, hence >=.
        assert src_ink > 0
        assert out_ink >= 0.9 * src_ink
        # The protection is image-based: one restored image covers the
        # formula segment bbox.
        infos = out.load_page(1).get_image_info()
        assert any(
            all(abs(a - b) <= 2.0 for a, b in zip(info["bbox"], formula.bbox))
            for info in infos
        ), "whole-bbox snapshot image missing"
    finally:
        src.close()
        out.close()


@needs_math_font
def test_inline_formulas_flow_with_translated_text(
        sample_pdf, sample_segments, flow_out):
    """Symptom B: inline ⟦EQn⟧ runs are inserted into the text flow as
    images inside the translated lines, not restored at their original
    page coordinates."""
    seg = next(
        s for s in sample_segments
        if s.page == 1 and "sensitivity analysis" in s.text
    )
    assert len(re.findall(r"⟦EQ\d+⟧", seg.text)) == 2
    seg_rect = pymupdf.Rect(*seg.bbox)
    span_rects = [
        r for r in _math_span_rects(sample_pdf, 1)
        if r.intersects(seg_rect)
    ]
    assert len(span_rects) == 2

    out_path, _report = flow_out
    src = pymupdf.open(sample_pdf)
    out = pymupdf.open(out_path)
    try:
        # The page gains exactly the restored formula images: two inline
        # runs plus one whole-bbox formula snapshot.
        src_count = len(src.load_page(1).get_image_info())
        out_page = out.load_page(1)
        infos = out_page.get_image_info()
        assert len(infos) == src_count + 3

        inline_imgs = [
            pymupdf.Rect(info["bbox"]) for info in infos
            if seg_rect.contains(
                pymupdf.Point(
                    (info["bbox"][0] + info["bbox"][2]) / 2.0,
                    (info["bbox"][1] + info["bbox"][3]) / 2.0,
                )
            )
        ]
        assert len(inline_imgs) == 2, "expected two inline formula images"

        hangul_lines = [
            r for r in _hangul_line_rects(out_page) if r.intersects(seg_rect)
        ]
        for img in inline_imgs:
            # Inside the segment bbox and vertically inside a translated
            # text line: the run flows with the Korean sentence.
            assert img.y0 >= seg_rect.y0 - 2.0 and img.y1 <= seg_rect.y1 + 2.0
            assert any(
                line.y0 - 1.5 <= img.y0 and img.y1 <= line.y1 + 1.5
                for line in hangul_lines
            ), f"inline image {img} not inside any translated text line"
        # At least one run moved away from its original coordinates (the
        # stub translator reorders the tokens into the dummy sentences). A
        # run counts as moved when no original run sits at the same spot;
        # comparing the line alone misjudged a run that flowed onto the line
        # where the OTHER run originally was.
        assert any(
            all(abs(img.y0 - r.y0) > 4.0 or abs(img.x0 - r.x0) > 4.0
                for r in span_rects)
            for img in inline_imgs
        ), "no inline image left its original position"

        text = out_page.get_text()
        assert "⟦EQ" not in text, "literal placeholder leaked into output"
        assert re.search(r"[가-힣]", text)
    finally:
        src.close()
        out.close()


@needs_math_font
def test_token_mismatch_appends_runs_to_text_flow(tmp_path):
    """Mismatch policy: when the translation loses an ⟦EQn⟧ token, the
    orphaned run is appended after the segment's last line instead of being
    restored at its original coordinates (which would float over the blank
    space a shorter translation leaves behind). No formula ink is lost."""
    from scripts.make_sample import MATH_FONT_ALIAS, _rename_math_font

    from app.pipeline.runner import run_pipeline
    from app.pipeline.translate import MASK_TOKEN_RE, StubTranslator

    class TokenDroppingTranslator(StubTranslator):
        """Stub that loses the last mask token of every segment."""

        def translate(self, text: str, context: str | None = None) -> str:
            result = StubTranslator.translate(self, text, context=context)
            tokens = MASK_TOKEN_RE.findall(result)
            if tokens:
                result = result.replace(tokens[-1], " ", 1)
            return result

    src = tmp_path / "inline2.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    font = pymupdf.Font(fontfile=str(MATH_FONT_FILE))
    prefix1 = "The convergence bound "
    w1 = pymupdf.get_text_length(prefix1, fontname="helv", fontsize=10)
    page.insert_text((72, 100), prefix1, fontname="helv", fontsize=10)
    page.insert_text((72 + w1, 100), "∑ α ≈ β", fontname=MATH_FONT_ALIAS,
                     fontfile=str(MATH_FONT_FILE), fontsize=10)
    prefix2 = "holds and the rate "
    w2 = pymupdf.get_text_length(prefix2, fontname="helv", fontsize=10)
    page.insert_text((72, 112), prefix2, fontname="helv", fontsize=10)
    page.insert_text((72 + w2, 112), "ρ < 1", fontname=MATH_FONT_ALIAS,
                     fontfile=str(MATH_FONT_FILE), fontsize=10)
    wm = font.text_length("ρ < 1", fontsize=10)
    page.insert_text((72 + w2 + wm + 4, 112), "is stable in practice.",
                     fontname="helv", fontsize=10)
    _rename_math_font(doc)
    doc.save(str(src))
    doc.close()

    span_rects = _math_span_rects(str(src), 0)
    assert len(span_rects) == 2

    out = tmp_path / "inline2_out.pdf"
    run_pipeline(str(src), str(out), TokenDroppingTranslator())

    result = pymupdf.open(str(out))
    try:
        out_page = result.load_page(0)
        text = out_page.get_text()
        infos = out_page.get_image_info()
        images = [pymupdf.Rect(info["bbox"]) for info in infos]
    finally:
        result.close()
    assert "⟦EQ" not in text, "literal placeholder leaked in fallback path"
    assert re.search(r"[가-힣]", text)
    # Ink preservation: both runs survive as images even though the
    # translation lost one token per line pairing.
    assert len(images) == 2
    # No floating over blank space: every image sits vertically inside a
    # translated (Hangul) text line, i.e. it flows with the text.
    result = pymupdf.open(str(out))
    try:
        hangul_lines = _hangul_line_rects(result.load_page(0))
    finally:
        result.close()
    assert hangul_lines
    for img in images:
        assert any(
            line.y0 - 1.5 <= img.y0 and img.y1 <= line.y1 + 1.5
            for line in hangul_lines
        ), f"image {img} floats outside every translated text line"
