# Tests for the layout-model / OCR integration (layout_model.py, ocr.py and
# the region rules in segment.py). The pure-logic tests always run; the ones
# that need the ONNX models skip when scripts/fetch_models.py has not run.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf
import pytest

from app.config import get_models_dir
from app.pipeline import layout_model, ocr
from app.pipeline.models import Line, Region, Segment, Span
from app.pipeline.segment import _layout_kind, _region_key

MODELS = Path(get_models_dir())
HAS_LAYOUT = (MODELS / layout_model.MODEL_FILE).exists()
HAS_OCR = (MODELS / ocr.DET_FILE).exists() and (MODELS / ocr.REC_FILE).exists()


def _seg(text, kind="body", bbox=(50, 100, 300, 140)):
    return Segment(id="p0_s0", page=0, column=0, bbox=bbox, text=text,
                   kind=kind, font_size=10.0)


def _line(bbox, text="text"):
    return Line(spans=[Span(text, bbox, "Helvetica", 10.0, 0)], bbox=bbox)


# --- region rules -----------------------------------------------------------

def test_chart_text_is_figure_text():
    region = Region("chart", 0.9, (40, 90, 320, 150))
    assert _layout_kind(_seg("52% Yes, a significant crisis"), region, 1.0) \
        == "figure_text"


def test_long_prose_inside_image_box_stays_translated():
    prose = ("The survey asked scientists what led to problems in "
             "reproducibility and more than half of them said that the "
             "pressure to publish and selective reporting were to blame "
             "for most of the failures they had seen in the lab.")
    region = Region("image", 0.9, (40, 90, 320, 150))
    assert _layout_kind(_seg(prose), region, 1.0) == "body"
    data = "Solubility 0.20 ** 0.34 *** 0.30 *** " * 4
    assert _layout_kind(_seg(data), region, 1.0) == "figure_text"


def test_footer_region_becomes_header_footer():
    region = Region("footer", 0.9, (40, 90, 320, 150))
    assert _layout_kind(_seg("452 | NATURE | VOL 533"), region, 1.0) \
        == "header_footer"


def test_long_heading_in_text_region_becomes_body():
    standfirst = ("A Nature survey lifts the lid on how researchers view the "
                  "crisis rocking science and what they think will help "
                  "their fields to recover.")
    region = Region("text", 0.95, (40, 90, 320, 150))
    assert _layout_kind(_seg(standfirst, kind="heading"), region, 1.0) == "body"


def test_stray_short_text_outside_regions_is_label():
    assert _layout_kind(_seg("38% Yes, a slight crisis"), None, 0.0) \
        == "figure_text"
    long_text = " ".join(["word"] * 30)
    assert _layout_kind(_seg(long_text), None, 0.0) == "body"


def test_panel_letter_is_not_a_caption():
    region = Region("figure_title", 0.6, (300, 50, 312, 61))
    assert _layout_kind(_seg("b", bbox=(305, 50, 310, 60)), region, 1.0) \
        == "figure_text"


def test_region_key_attaches_clipped_last_line():
    regions = [Region("text", 0.9, (40, 100, 300, 150))]
    inside = _line((42, 110, 298, 120))
    clipped = _line((42, 151, 200, 161))  # just below the box
    far = _line((42, 400, 200, 410))
    assert _region_key(inside, regions) == 0
    assert _region_key(clipped, regions) == 0
    assert _region_key(far, regions) == -1


# --- model post-processing / OCR components (no model needed) ------------

def test_postprocess_nms_and_reading_order(monkeypatch):
    monkeypatch.setattr(layout_model, "_labels", ["text", "image", "footer"])
    raw = np.array([
        # label, score, x0, y0, x1, y1, order, tie
        [0, 0.95, 10, 200, 100, 260, 2, 0],
        [0, 0.90, 10, 100, 100, 160, 1, 0],
        [0, 0.60, 12, 102, 98, 158, 1, 0],   # duplicate of the one above
        [2, 0.30, 10, 700, 100, 710, 3, 0],  # below the score threshold
    ], dtype=np.float32)
    regions = layout_model._postprocess(raw, 612, 792)
    assert [r.bbox[1] for r in regions] == [100, 200]
    assert [r.order for r in regions] == [1, 2]


def test_components_finds_separate_boxes():
    mask = np.zeros((20, 40), dtype=bool)
    mask[2:5, 3:15] = True
    mask[10:14, 20:35] = True
    mask[4:6, 14:16] = True  # touches the first box diagonally
    boxes = sorted(ocr._components(mask))
    assert boxes == [[3, 2, 16, 6], [20, 10, 35, 14]]


def test_is_scanned_page():
    doc = pymupdf.open()
    text_page = doc.new_page()
    text_page.insert_text((72, 72), "A born-digital page with a text layer. " * 3)
    scan_page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 80), False)
    pix.clear_with(255)
    scan_page.insert_image(scan_page.rect, pixmap=pix)
    assert not ocr.is_scanned_page(doc[0])
    assert ocr.is_scanned_page(doc[1])


# --- with models ----------------------------------------------------------

@pytest.mark.skipif(not HAS_LAYOUT, reason="layout model not fetched")
def test_layout_model_finds_text_and_title(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 90), "A Study of Layout Detection", fontsize=22)
    y = 140
    for _ in range(12):
        page.insert_text(
            (72, y), "This paragraph describes the method in plain words "
                     "and continues for a while.", fontsize=10)
        y += 13
    regions = layout_model.detect_regions(page)
    labels = {r.label for r in regions}
    assert "text" in labels
    assert labels & {"doc_title", "paragraph_title"}


@pytest.mark.skipif(not (HAS_OCR and HAS_LAYOUT), reason="OCR models not fetched")
def test_scanned_page_is_read_by_ocr(tmp_path):
    from app.pipeline.extract import analyze_pdf

    # Build a born-digital page, rasterize it, and wrap the picture as a
    # "scanned" PDF without a text layer.
    src = pymupdf.open()
    page = src.new_page(width=612, height=792)
    page.insert_text((72, 120), "Reproducibility is a defining feature",
                     fontsize=14)
    page.insert_text((72, 140), "of science and deserves attention.",
                     fontsize=14)
    pix = page.get_pixmap(dpi=200)
    scan = pymupdf.open()
    scan_page = scan.new_page(width=612, height=792)
    scan_page.insert_image(scan_page.rect, pixmap=pix)
    path = tmp_path / "scan.pdf"
    scan.save(str(path))

    layout = analyze_pdf(str(path))
    assert layout.pages[0].ocr
    text = " ".join(sp.text for b in layout.pages[0].blocks if b.type == "text"
                    for ln in b.lines for sp in ln.spans)
    assert "Reproducibility" in text and "science" in text
