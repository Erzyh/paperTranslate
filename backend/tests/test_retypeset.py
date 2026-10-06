# Tests for retypeset: expansion bounds (issue: neighbour collision),
# inline-formula restoration (issue: equation destruction) and the loss-free
# overflow rescue ladder (issue: silent text drop after redaction).
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pymupdf
import pytest

from app.pipeline.models import Segment
from app.pipeline.retypeset import (
    MIN_SCALE,
    _expand_rect,
    _force_expand_rect,
    render_translated_pdf,
)

MATH_FONT_FILE = Path(r"C:\Windows\Fonts\seguisym.ttf")


@pytest.fixture()
def blank_page():
    doc = pymupdf.open()
    try:
        yield doc.new_page(width=612, height=792)
    finally:
        doc.close()


def test_expand_rect_stops_above_next_block(blank_page):
    rect = pymupdf.Rect(54, 600, 294, 650)
    obstacle = pymupdf.Rect(54, 680, 294, 700)
    expanded = _expand_rect(blank_page, rect, 200.0, [obstacle])
    assert expanded is not None
    # Safety gap above the next block (3pt since the A4 overflow tuning).
    assert expanded.y1 == pytest.approx(obstacle.y0 - 3.0)


def test_expand_rect_none_when_next_block_adjacent(blank_page):
    rect = pymupdf.Rect(54, 600, 294, 650)
    obstacle = pymupdf.Rect(100, 650.5, 200, 700)  # right below, overlapping x
    assert _expand_rect(blank_page, rect, 50.0, [obstacle]) is None


def test_expand_rect_ignores_non_overlapping_neighbour(blank_page):
    rect = pymupdf.Rect(54, 600, 294, 650)
    obstacle = pymupdf.Rect(318, 655, 558, 700)  # other column: no x overlap
    expanded = _expand_rect(blank_page, rect, 50.0, [obstacle])
    assert expanded is not None
    assert expanded.y1 == pytest.approx(700.0)


def test_expand_rect_bounded_by_page_bottom(blank_page):
    rect = pymupdf.Rect(54, 700, 294, 780)
    expanded = _expand_rect(blank_page, rect, 100.0, [])
    assert expanded is not None
    assert expanded.y1 <= blank_page.rect.y1 - 2.0
    # No room at all -> None
    rect2 = pymupdf.Rect(54, 700, 294, 790.0)
    assert _expand_rect(blank_page, rect2, 100.0, []) is None


def test_force_expand_rect_ignores_obstacles(blank_page):
    """The terminal rescue grows past a blocking neighbour by one line
    (cap tightened from 1.5 lines in the A4 overflow tuning)."""
    rect = pymupdf.Rect(54, 600, 294, 650)
    expanded = _force_expand_rect(blank_page, rect, 10.0)
    assert expanded.y1 == pytest.approx(650 + 10.0 * 1.04 * 1.0)
    assert (expanded.x0, expanded.y0, expanded.x1) == (54, 600, 294)


def test_force_expand_rect_clamps_to_page_bottom(blank_page):
    rect = pymupdf.Rect(54, 700, 294, 785)
    expanded = _force_expand_rect(blank_page, rect, 12.0)
    assert expanded.y1 == pytest.approx(blank_page.rect.y1)


def test_force_expand_rect_no_room_returns_rect(blank_page):
    rect = pymupdf.Rect(54, 700, 294, blank_page.rect.y1)
    expanded = _force_expand_rect(blank_page, rect, 10.0)
    assert expanded == rect


# Consecutive lines inside one rendered paragraph legitimately overlap: the
# CJK glyph box (ascender+descender, ~1.39em) exceeds the 1.04em line step
# of the htmlbox CSS, giving up to ~31% overlap between SAME-HEIGHT lines of
# one insertion. Cross-segment collisions are therefore caught by two
# orthogonal signals instead of one global ratio:
# (a) overlapping lines with DIFFERENT heights - lines of one insertion
#     share font size and scale exactly, while the pre-fix intrusion pair
#     (p0_s2 last line h=11.2pt vs p0_s3 first line h=9.5pt) overlapped by
#     25-27%; measured legitimate value today: 0%.
# (b) a line that starts well above another translated segment's bbox top
#     (more than the own-first-line glyph bleed) yet dips into that bbox -
#     the expansion-crowding signature; measured legitimate maximum today:
#     1.65pt (source bboxes of adjacent segments overlap slightly).
_MAX_SAME_HEIGHT_OVERLAP_RATIO = 0.35   # intra-paragraph packing bound
_MAX_CROSS_HEIGHT_OVERLAP_RATIO = 0.20  # different-height line pairs
_LINE_HEIGHT_MATCH_PT = 0.6             # same-insertion height tolerance
_MAX_BBOX_TOP_DIP_PT = 2.5              # dip into the next block's bbox
_FIRST_LINE_BLEED_FACTOR = 0.45         # own first-line glyph bleed, in em


def _hangul_line_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Collect bboxes of extracted text lines that carry translated Hangul."""
    rects: list[pymupdf.Rect] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            if any("가" <= ch <= "힣" for ch in text):
                rects.append(pymupdf.Rect(line["bbox"]))
    return rects


def test_translated_lines_do_not_collide(sample_pdf, sample_segments, tmp_path):
    """Regression (overflow expansion): no line of one segment may intrude
    into another segment's lines or bbox on the same page/column."""
    from app.pipeline.runner import run_pipeline
    from app.pipeline.segment import UNTRANSLATED_KINDS
    from app.pipeline.translate import StubTranslator

    out = tmp_path / "collide_out.pdf"
    report = run_pipeline(sample_pdf, str(out), StubTranslator())

    # Paragraphs flow within their column, so compare against where each
    # translation was actually placed rather than the source bbox.
    segs_by_page: dict[int, list] = {}
    for seg in sample_segments:
        if seg.kind not in UNTRANSLATED_KINDS:
            placed = report.placed_rects.get(seg.id)
            if placed is not None:
                seg = replace(seg, bbox=placed)
            segs_by_page.setdefault(seg.page, []).append(seg)

    doc = pymupdf.open(str(out))
    try:
        for pno in range(doc.page_count):
            lines = sorted(
                _hangul_line_rects(doc.load_page(pno)),
                key=lambda r: (r.y0, r.x0),
            )
            for i, ra in enumerate(lines):
                for rb in lines[i + 1:]:
                    x_overlap = min(ra.x1, rb.x1) - max(ra.x0, rb.x0)
                    y_overlap = min(ra.y1, rb.y1) - max(ra.y0, rb.y0)
                    if x_overlap <= 2.0 or y_overlap <= 0.0:
                        continue  # different columns or no contact
                    ratio = y_overlap / min(ra.height, rb.height)
                    if abs(ra.height - rb.height) > _LINE_HEIGHT_MATCH_PT:
                        # Different heights -> different insertions: any
                        # notable overlap is a cross-segment collision.
                        assert ratio <= _MAX_CROSS_HEIGHT_OVERLAP_RATIO, (
                            f"page {pno}: line {ra} (h={ra.height:.2f}) "
                            f"collides with {rb} (h={rb.height:.2f}) by "
                            f"{y_overlap:.2f}pt ({ratio:.1%})"
                        )
                    else:
                        # Same height: legitimate intra-paragraph packing,
                        # bounded by the 1.04em line step vs the glyph box.
                        assert ratio <= _MAX_SAME_HEIGHT_OVERLAP_RATIO, (
                            f"page {pno}: line {ra} overlaps {rb} by "
                            f"{y_overlap:.2f}pt ({ratio:.1%} of line height)"
                        )
            # Expansion-crowding signature: a line that starts clearly above
            # another translated segment's bbox (beyond the first-line glyph
            # bleed of that segment) must not dip into that bbox.
            for line in lines:
                for seg in segs_by_page.get(pno, []):
                    bbox = pymupdf.Rect(*seg.bbox)
                    if line.y0 >= bbox.y0 - _FIRST_LINE_BLEED_FACTOR * line.height:
                        continue  # the segment's own first line
                    x_overlap = min(line.x1, bbox.x1) - max(line.x0, bbox.x0)
                    dip = line.y1 - bbox.y0
                    if x_overlap <= 2.0 or dip <= 0.0:
                        continue
                    assert dip <= _MAX_BBOX_TOP_DIP_PT, (
                        f"page {pno}: line {line} dips {dip:.2f}pt into "
                        f"{seg.id} ({seg.kind}) bbox {bbox}"
                    )
    finally:
        doc.close()


@pytest.mark.skipif(not MATH_FONT_FILE.exists(), reason="needs seguisym math font")
def test_inline_equation_restored_not_typeset(tmp_path):
    """An inline math run inside a translated paragraph must survive.

    The paragraph bbox is redacted (erasing the math glyphs), so retypeset
    must restore the original run as an image and must not typeset the
    literal placeholder token.
    """
    from scripts.make_sample import MATH_FONT_ALIAS, _rename_math_font

    from app.pipeline.runner import run_pipeline
    from app.pipeline.translate import StubTranslator

    src = tmp_path / "inline.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    prefix = "The convergence bound "
    formula = "∑ α ≈ β"
    w1 = pymupdf.get_text_length(prefix, fontname="helv", fontsize=10)
    w2 = pymupdf.Font(fontfile=str(MATH_FONT_FILE)).text_length(formula, fontsize=10)
    page.insert_text((72, 100), prefix, fontname="helv", fontsize=10)
    page.insert_text((72 + w1, 100), formula, fontname=MATH_FONT_ALIAS,
                     fontfile=str(MATH_FONT_FILE), fontsize=10)
    page.insert_text((72 + w1 + w2 + 4, 100), "holds for every input.",
                     fontname="helv", fontsize=10)
    page.insert_text((72, 112), "This second sentence pads the paragraph.",
                     fontname="helv", fontsize=10)
    _rename_math_font(doc)
    doc.save(str(src))
    doc.close()

    out = tmp_path / "inline_out.pdf"
    run_pipeline(str(src), str(out), StubTranslator())

    result = pymupdf.open(str(out))
    try:
        text = result.load_page(0).get_text()
        images = result.load_page(0).get_image_info()
    finally:
        result.close()
    assert "⟦EQ" not in text, "literal placeholder typeset into the output"
    assert re.search(r"[가-힣]", text), "no Korean text rendered"
    assert "convergence" not in text, "English body survived redaction"
    assert len(images) == 1, "restored formula snippet image missing"


# --- Loss-free overflow rescue ladder (issue: silent text drop) ----------
# A segment whose text was redacted must ALWAYS get its translation back
# into the page: when scaling to MIN_SCALE fails, a 0.55 pre-expansion
# shrink runs first (smaller text beats crowding the neighbour), then
# collision-free expansion; when both fail, the scale floor drops stepwise
# (0.5 -> 0.35) and finally the bbox is force-expanded past obstacles
# (overlap beats loss, DESIGN 2.5 update).

_LONG_KO = "겹침이 유실보다 낫다는 손실 제로 정책 검증 문장. " * 45


def _has_hangul(text: str) -> bool:
    return any("가" <= ch <= "힣" for ch in text)


def _render_single(
    tmp_path: Path, src_doc: pymupdf.Document, segment: Segment,
    translated: str,
):
    """Save the source doc, render one translated segment, return
    (extracted page text, report)."""
    src = tmp_path / "rescue_src.pdf"
    src_doc.save(str(src))
    src_doc.close()
    out = tmp_path / "rescue_out.pdf"
    report = render_translated_pdf(
        str(src), [segment], {segment.id: translated}, str(out))
    doc = pymupdf.open(str(out))
    try:
        text = doc.load_page(0).get_text()
    finally:
        doc.close()
    return text, report


def test_pre_expand_shrink_preferred_over_expansion(tmp_path):
    """A4 overflow tuning: text that fits at scale 0.55-0.65 must be placed
    by the extra shrink step INSIDE the original bbox, not by growing the
    bbox toward the block below (even though room for expansion exists)."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 110), "Paragraph to be redacted.", fontsize=10)
    # A drawing 10pt below the bbox: collision-free expansion WOULD be
    # possible, but the shrink step must win first.
    page.draw_rect(pymupdf.Rect(72, 170, 300, 200), color=None,
                   fill=(0.8, 0.8, 0.8))
    segment = Segment(id="p0_s0", page=0, column=0,
                      bbox=(72.0, 100.0, 300.0, 160.0),
                      text="Paragraph to be redacted.", kind="body",
                      font_size=10.0)
    translated = "겹침 대신 축소를 선호하는 사용자 정책 검증 문장이다. " * 16
    src = tmp_path / "preexpand_src.pdf"
    doc.save(str(src))
    doc.close()
    out = tmp_path / "preexpand_out.pdf"
    report = render_translated_pdf(
        str(src), [segment], {segment.id: translated}, str(out))
    result = pymupdf.open(str(out))
    try:
        page0 = result.load_page(0)
        inside = page0.get_text(clip=pymupdf.Rect(72, 100, 300, 160))
        below = page0.get_text(clip=pymupdf.Rect(72, 162, 300, 792))
    finally:
        result.close()
    assert _has_hangul(inside), "translation missing from the original bbox"
    assert not _has_hangul(below), (
        "bbox was expanded although the 0.55 shrink step should have fit")
    assert "p0_s0" in report.overflow_segments
    scale = report.scaled_segments["p0_s0"]
    assert 0.55 - 0.01 <= scale < MIN_SCALE


def test_rescue_ladder_inserts_despite_blocked_expansion(tmp_path):
    """Expansion blocked by a drawing right below: the ladder must still
    place the (far too long) translation instead of dropping it."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 712), "This line will be redacted.", fontsize=10)
    page.draw_rect(pymupdf.Rect(72, 717, 300, 770), color=None,
                   fill=(0.8, 0.8, 0.8))
    segment = Segment(id="p0_s0", page=0, column=0,
                      bbox=(72.0, 700.0, 300.0, 716.0),
                      text="This line will be redacted.", kind="body",
                      font_size=10.0)
    text, report = _render_single(tmp_path, doc, segment, _LONG_KO)
    assert _has_hangul(text), "translation lost despite rescue ladder"
    assert "redacted" not in text, "English source survived redaction"
    assert "p0_s0" in report.overflow_segments
    assert report.scaled_segments["p0_s0"] < MIN_SCALE


def test_rescue_ladder_inserts_at_page_bottom(tmp_path):
    """A very low bbox with almost no room below: forced expansion clamps to
    the page bottom and the unbounded-shrink step still places the text."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 782), "Bottom line to be redacted.", fontsize=10)
    segment = Segment(id="p0_s0", page=0, column=0,
                      bbox=(72.0, 770.0, 300.0, 788.0),
                      text="Bottom line to be redacted.", kind="body",
                      font_size=10.0)
    text, report = _render_single(tmp_path, doc, segment, _LONG_KO)
    assert _has_hangul(text), "translation lost at the page bottom"
    assert "p0_s0" in report.overflow_segments
    assert report.scaled_segments["p0_s0"] < MIN_SCALE


def test_thin_redacted_bbox_still_renders_text(tmp_path):
    """A sub-2pt bbox is redacted, so skipping it would lose the text: the
    rect must be grown to a usable height and the translation inserted."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 700), "Thin segment source.", fontsize=10)
    segment = Segment(id="p0_s0", page=0, column=0,
                      bbox=(72.0, 698.0, 300.0, 699.0),
                      text="Thin segment source.", kind="body",
                      font_size=10.0)
    text, report = _render_single(
        tmp_path, doc, segment, "얇은 상자에서도 유실되지 않는다.")
    assert _has_hangul(text), "translation lost for a thin bbox"
    assert "p0_s0" in report.overflow_segments


_REAL_PDF = Path(__file__).resolve().parents[1] / "data" / "uploads" / (
    "d1032a73020346eb9a4cb016eb2b0a04.pdf")
# Fragile short/list segments that vanished entirely (redacted, 0
# insertions) before the rescue ladder existed. Anchored by text because
# segment ids drift with segmentation improvements; the old one-line
# equation fragments ("N", "⟦EQn⟧offset ⟦EQm⟧", "3") are gone from this
# list on purpose - the formula-zone banding now absorbs them into
# untranslated formula segments, so they are never redacted at all.
_PREVIOUSLY_LOST_TEXTS = (
    "E. COMPOSITE LOSS FORMULATION",
    "Training the MT-BCN necessitates",
    "NPC baseline term:",
    "Cognitive overload term",
    "Execution error term",
    "Biomechanical ceiling term",
    "REFERENCES",
)


@pytest.mark.skipif(not _REAL_PDF.exists(),
                    reason="needs the locally uploaded real paper")
def test_real_pdf_stub_render_loses_no_segment(tmp_path):
    """Regression on the real paper (slow, ~40s): every translated segment
    leaves extractable Hangul near its bbox - in particular the seven
    one-line list segments that used to disappear."""
    from app.pipeline.extract import analyze_pdf
    from app.pipeline.retypeset import _EQ_TOKEN_RE
    from app.pipeline.segment import (
        UNTRANSLATED_KINDS,
        build_segments,
        split_list_marker,
    )
    from app.pipeline.translate import StubTranslator, mask, unmask

    layout = analyze_pdf(str(_REAL_PDF))
    segments = build_segments(layout)
    stub = StubTranslator()
    translations: dict[str, str] = {}
    for seg in segments:
        if seg.kind in UNTRANSLATED_KINDS:
            continue
        marker, item = split_list_marker(seg.text)
        masked, mapping = mask(item)
        translated = unmask(stub.translate(masked), mapping)
        if marker:
            translated = marker + translated.lstrip()
        translations[seg.id] = translated
    for anchor in _PREVIOUSLY_LOST_TEXTS:
        anchored = next(
            (seg for seg in segments
             if anchor in seg.text and seg.kind not in UNTRANSLATED_KINDS),
            None,
        )
        assert anchored is not None, f"regression fixture drifted: {anchor!r}"
        assert anchored.id in translations

    out = tmp_path / "real_out.pdf"
    render_translated_pdf(str(_REAL_PDF), segments, translations, str(out))

    by_id = {seg.id: seg for seg in segments}
    lost: list[str] = []
    doc = pymupdf.open(str(out))
    try:
        for seg_id, translated in translations.items():
            if not _EQ_TOKEN_RE.sub(" ", translated).strip():
                continue  # formula-only translation: legitimately skipped
            seg = by_id[seg_id]
            page = doc.load_page(seg.page)
            clip = pymupdf.Rect(*seg.bbox)
            clip.y1 += 30.0  # allow for downward expansion
            clip.intersect(page.rect)
            if clip.is_empty:
                continue
            if not _has_hangul(page.get_text(clip=clip)):
                lost.append(seg_id)
    finally:
        doc.close()
    assert lost == [], f"segments lost their translation: {lost}"


def _flow_source(tmp_path, with_formula: bool) -> tuple[str, list[Segment]]:
    """Two long English paragraphs in one column, optionally with a small
    formula fragment sitting inside the second paragraph's box."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(pymupdf.Rect(72, 100, 300, 300), "Lorem ipsum " * 120,
                        fontsize=10)
    page.insert_textbox(pymupdf.Rect(72, 310, 300, 500), "Dolor sit " * 120,
                        fontsize=10)
    segments = [
        Segment(id="p0_s0", page=0, column=0, bbox=(72.0, 100.0, 300.0, 300.0),
                text="Lorem ipsum " * 120, kind="body", font_size=10.0),
        Segment(id="p0_s1", page=0, column=0, bbox=(72.0, 310.0, 300.0, 500.0),
                text="Dolor sit " * 120, kind="body", font_size=10.0),
    ]
    if with_formula:
        page.insert_text((260, 330), "xy", fontsize=10)
        segments.append(Segment(id="p0_s2", page=0, column=0,
                                bbox=(258.0, 320.0, 280.0, 334.0), text="xy",
                                kind="formula", font_size=10.0))
    src = tmp_path / "flow_src.pdf"
    doc.save(str(src))
    doc.close()
    return str(src), segments


def test_short_translation_flows_up_at_one_size(tmp_path):
    """Column flow: a short translation leaves no gap under its paragraph;
    the next paragraph follows it, and both share one size."""
    src, segments = _flow_source(tmp_path, with_formula=False)
    translations = {"p0_s0": "짧은 번역 문장이다. " * 4,
                    "p0_s1": "두 번째 문단의 번역이다. " * 4}
    report = render_translated_pdf(src, segments, translations,
                                   str(tmp_path / "flow_out.pdf"))
    first, second = report.placed_rects["p0_s0"], report.placed_rects["p0_s1"]
    assert first[1] == pytest.approx(100.0)
    assert first[3] < 160.0, "short translation should not fill the old box"
    # Follows the first paragraph, keeping the source gap (10pt).
    assert second[1] == pytest.approx(first[3] + 10.0, abs=0.5)
    assert (report.scaled_segments.get("p0_s0", 1.0)
            == report.scaled_segments.get("p0_s1", 1.0))
    assert not report.overflow_segments


def test_paragraph_holding_fixed_fragment_stays_put(tmp_path):
    """A paragraph overlapping something kept in place (an inline formula
    fragment) keeps its source position instead of flowing away from it."""
    src, segments = _flow_source(tmp_path, with_formula=True)
    translations = {"p0_s0": "짧은 번역 문장이다. " * 4,
                    "p0_s1": "두 번째 문단의 번역이다. " * 4}
    report = render_translated_pdf(src, segments, translations,
                                   str(tmp_path / "flow_out.pdf"))
    assert report.placed_rects["p0_s1"][1] == pytest.approx(310.0)
