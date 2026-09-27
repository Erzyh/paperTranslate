# Real-paper (OSCAR) render regressions for two user-reported symptoms:
# (a) adjacent Korean lines overlapping vertically (line-height 0.96 packed
#     the ~1.39em CJK glyph boxes too tightly, worst under sub-MIN_SCALE
#     rescue rendering) - fixed by restoring _LINE_HEIGHT_EM to 1.04;
# (b) inline formula images floating over the blank space left by shorter
#     translations (restore-at-origin fallback on token/run mismatch) -
#     fixed by the min(n,m) in-order matching + append-after-last-line
#     policy in retypeset.
# Uses only the StubTranslator; no LLM calls. Skipped when the locally
# uploaded paper is absent.
from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

_OSCAR_PDF = Path(__file__).resolve().parents[1] / "data" / "uploads" / (
    "e891481d768e43a4b6ba4c1c1307f4ea.pdf")

pytestmark = pytest.mark.skipif(
    not _OSCAR_PDF.exists(), reason="needs the locally uploaded OSCAR paper")

# Measured maxima on the OSCAR stub render after the fix (2026-08):
# same-height 0.200, cross-height 0.031. Bounds keep headroom below the
# pre-fix regime (0.96em step: legitimate intra-paragraph packing alone
# exceeded 0.31).
_MAX_SAME_HEIGHT_OVERLAP_RATIO = 0.27
_MAX_CROSS_HEIGHT_OVERLAP_RATIO = 0.20
_LINE_HEIGHT_MATCH_PT = 0.6
# Sub-MIN_SCALE rescue lines can be <5pt tall; a ~1.3pt bbox kiss between
# adjacent segments then inflates the ratio although it is invisible in
# print. Overlaps below this absolute depth are ignored.
_MIN_OVERLAP_PT = 1.5
# Lines below this height come from the sub-floor rescue ladder, whose
# design explicitly allows overlapping a neighbour rather than dropping
# text ("overlap beats loss"); cross-height pairs involving such a line
# are exempt from the collision bound.
_RESCUE_LINE_HEIGHT_PT = 6.0
# An appended run may land in a force-expanded last line slightly below the
# segment bbox; allow one expanded line of downward tolerance.
_BELOW_BBOX_TOLERANCE_PT = 20.0


@pytest.fixture(scope="module")
def oscar_render(tmp_path_factory):
    """OSCAR paper rendered once with the stub translator."""
    from app.pipeline.extract import analyze_pdf
    from app.pipeline.retypeset import render_translated_pdf
    from app.pipeline.segment import (
        UNTRANSLATED_KINDS,
        build_segments,
        split_list_marker,
    )
    from app.pipeline.translate import StubTranslator, mask, unmask

    segments = build_segments(analyze_pdf(str(_OSCAR_PDF)))
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
    out = tmp_path_factory.mktemp("oscar") / "oscar_out.pdf"
    render_translated_pdf(str(_OSCAR_PDF), segments, translations, str(out))
    return str(out), segments


def _hangul_line_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    rects = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s.get("text", "") for s in line.get("spans", []))
            if any("가" <= ch <= "힣" for ch in text):
                rects.append(pymupdf.Rect(line["bbox"]))
    return rects


def test_oscar_adjacent_lines_do_not_overlap_excessively(oscar_render):
    """(i) Same-column adjacent translated lines stay readable: the 1.04em
    line step bounds the vertical overlap of extracted line boxes."""
    out_path, _segments = oscar_render
    doc = pymupdf.open(out_path)
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
                    if x_overlap <= 2.0 or y_overlap <= _MIN_OVERLAP_PT:
                        continue  # different columns or invisible contact
                    same_height = (
                        abs(ra.height - rb.height) <= _LINE_HEIGHT_MATCH_PT)
                    if (not same_height
                            and min(ra.height, rb.height)
                            < _RESCUE_LINE_HEIGHT_PT):
                        continue  # permitted rescue-ladder overlap
                    ratio = y_overlap / min(ra.height, rb.height)
                    bound = (
                        _MAX_SAME_HEIGHT_OVERLAP_RATIO
                        if abs(ra.height - rb.height) <= _LINE_HEIGHT_MATCH_PT
                        else _MAX_CROSS_HEIGHT_OVERLAP_RATIO
                    )
                    assert ratio <= bound, (
                        f"page {pno}: line {ra} (h={ra.height:.2f}) overlaps "
                        f"{rb} (h={rb.height:.2f}) by {y_overlap:.2f}pt "
                        f"({ratio:.1%})"
                    )
    finally:
        doc.close()


def test_oscar_no_floating_formula_images(oscar_render):
    """(ii) Zero restore-at-origin strays: every image the render added must
    sit inside some segment bbox (small tolerance; appended runs may trail
    into a force-expanded last line just below it)."""
    out_path, segments = oscar_render
    src = pymupdf.open(str(_OSCAR_PDF))
    out = pymupdf.open(out_path)
    try:
        floating: list[str] = []
        for pno in range(out.page_count):
            src_boxes = [
                pymupdf.Rect(info["bbox"])
                for info in src.load_page(pno).get_image_info()
            ]
            page_segs = [s for s in segments if s.page == pno]
            for info in out.load_page(pno).get_image_info():
                rect = pymupdf.Rect(info["bbox"])
                if any(
                    all(abs(a - b) <= 2.0
                        for a, b in zip(info["bbox"], tuple(sb)))
                    for sb in src_boxes
                ):
                    continue  # pre-existing figure, unchanged
                cx = (rect.x0 + rect.x1) / 2.0
                cy = (rect.y0 + rect.y1) / 2.0
                if not any(
                    s.bbox[0] - 2.0 <= cx <= s.bbox[2] + 2.0
                    and s.bbox[1] - 2.0 <= cy
                    <= s.bbox[3] + _BELOW_BBOX_TOLERANCE_PT
                    for s in page_segs
                ):
                    floating.append(f"page {pno}: {rect}")
        assert floating == [], f"floating formula images: {floating}"
    finally:
        src.close()
        out.close()


def test_oscar_formula_ink_preserved(oscar_render):
    """(iii) Every masked math run eligible for inline flow survives as an
    image (273 on the current paper), and no placeholder token leaks."""
    from app.pipeline.retypeset import _rect_mostly_inside, _scan_math_spans
    from app.pipeline.segment import UNTRANSLATED_KINDS

    out_path, segments = oscar_render
    src = pymupdf.open(str(_OSCAR_PDF))
    out = pymupdf.open(out_path)
    try:
        for pno in range(out.page_count):
            spans = _scan_math_spans(src.load_page(pno))
            translated_boxes = [
                pymupdf.Rect(*s.bbox) for s in segments
                if s.page == pno and s.kind not in UNTRANSLATED_KINDS
            ]
            untranslated_boxes = [
                pymupdf.Rect(*s.bbox) for s in segments
                if s.page == pno and s.kind in UNTRANSLATED_KINDS
            ]
            eligible = sum(
                1 for _key, rect, _oy in spans
                if not any(_rect_mostly_inside(rect, u)
                           for u in untranslated_boxes)
                and any(_rect_mostly_inside(rect, t)
                        for t in translated_boxes)
            )
            src_count = len(src.load_page(pno).get_image_info())
            out_count = len(out.load_page(pno).get_image_info())
            assert out_count - src_count >= eligible, (
                f"page {pno}: {eligible} inline-eligible math runs but only "
                f"{out_count - src_count} images were added"
            )
            assert "⟦EQ" not in out.load_page(pno).get_text()
    finally:
        src.close()
        out.close()
