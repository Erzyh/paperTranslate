# Page layout detection with a pretrained model (PP-DocLayoutV2, ONNX).
#
# Rule-based segmentation (segment.py) reads fonts, gaps and positions; it is
# precise on the layouts it was tuned for and blind to the rest (magazine
# covers, infographics, side columns, drop caps). The layout model looks at
# the rendered page instead and returns labelled regions - text, titles,
# charts, tables, formulas, headers/footers, footnotes - in reading order.
# segment.py uses those regions as paragraph boundaries and as the final say
# on what gets translated; the rules stay in charge of everything the model
# does not cover (author blocks, list markers, formula zones).
#
# Only onnxruntime + numpy are needed: the page is rendered by PyMuPDF
# straight at the model's input size, and the post-processing below is a
# trimmed port of RapidLayout's PP-DocLayout handler (Apache-2.0).
# When the model file or onnxruntime is missing, detect_regions() returns []
# and the pipeline runs on rules alone.
from __future__ import annotations

import logging
import threading
from pathlib import Path

import pymupdf

from ..config import get_models_dir, layout_model_enabled
from .models import Region

log = logging.getLogger(__name__)

MODEL_FILE = "layout_int8.onnx"
_INPUT_SIZE = 800  # the model's fixed square input (pixels)
_SCORE_MIN = 0.5  # boxes below this confidence are dropped
_NMS_IOU_SAME = 0.6  # same-label boxes overlapping more than this collapse
_NMS_IOU_DIFF = 0.98  # different-label boxes collapse only when ~identical
_OVERLAP_DROP = 0.7  # a box mostly inside another same-kind box is dropped
_MIN_SIDE_PT = 3.0  # degenerate boxes (pt)

# Regions that float outside the reading flow (RapidLayout SKIP_ORDER_LABELS).
_FLOATING = frozenset({
    "figure_title", "vision_footnote", "image", "chart", "table", "header",
    "header_image", "footer", "footer_image", "footnote", "aside_text",
})

_lock = threading.Lock()
_session = None  # onnxruntime.InferenceSession | False (unavailable)
_labels: list[str] = []


def _load():
    """Load the ONNX session once; False when unavailable."""
    global _session, _labels
    with _lock:
        if _session is not None:
            return _session
        path = Path(get_models_dir()) / MODEL_FILE
        if not path.exists():
            log.info("layout model not found: %s (rules only)", path)
            _session = False
            return _session
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            sess = ort.InferenceSession(
                str(path), opts, providers=["CPUExecutionProvider"])
            meta = sess.get_modelmeta().custom_metadata_map
            _labels = meta["character"].splitlines()
            _session = sess
        except Exception:  # noqa: BLE001 - any failure falls back to rules
            log.exception("layout model failed to load (rules only)")
            _session = False
        return _session


def available() -> bool:
    """True when the layout model can run."""
    return layout_model_enabled() and bool(_load())


def _render_input(page: pymupdf.Page):
    """Page rendered straight to the model's 800x800 BGR float input."""
    import numpy as np

    rect = page.rect
    matrix = pymupdf.Matrix(_INPUT_SIZE / rect.width, _INPUT_SIZE / rect.height)
    pix = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    if (pix.h, pix.w) != (_INPUT_SIZE, _INPUT_SIZE):
        # rounding can leave the pixmap a pixel short/long: pad or crop
        canvas = np.full((_INPUT_SIZE, _INPUT_SIZE, 3), 255, dtype=np.uint8)
        h, w = min(pix.h, _INPUT_SIZE), min(pix.w, _INPUT_SIZE)
        canvas[:h, :w] = img[:h, :w]
        img = canvas
    bgr = img[:, :, ::-1].astype(np.float32) / 255.0
    return bgr.transpose(2, 0, 1)[None]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def _overlap_small(a, b) -> float:
    """Intersection over the smaller box's area."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / small if small > 0 else 0.0


def _postprocess(raw, page_w: float, page_h: float) -> list[Region]:
    """Threshold, NMS, reading-order sort and overlap filtering."""
    import numpy as np

    rows = [r for r in np.asarray(raw) if r[1] > _SCORE_MIN and r[0] > -1]
    if not rows:
        return []
    # NMS: same-label boxes collapse at IoU 0.6, different labels only
    # when practically identical.
    rows.sort(key=lambda r: -float(r[1]))
    kept: list = []
    for row in rows:
        box = row[2:6]
        if all(_iou(box, k[2:6]) < (_NMS_IOU_SAME if row[0] == k[0]
                                     else _NMS_IOU_DIFF) for k in kept):
            kept.append(row)
    # Columns 6/7 carry the model's reading order (primary asc, tie desc).
    if len(kept[0]) >= 8:
        kept.sort(key=lambda r: (float(r[6]), -float(r[7])))

    regions: list[Region] = []
    for row in kept:
        idx = int(row[0])
        label = _labels[idx] if 0 <= idx < len(_labels) else str(idx)
        x0, y0, x1, y1 = (float(v) for v in row[2:6])
        bbox = (max(0.0, x0), max(0.0, y0), min(page_w, x1), min(page_h, y1))
        if label == "reference":  # container box; items are reference_content
            continue
        if bbox[2] - bbox[0] < _MIN_SIDE_PT or bbox[3] - bbox[1] < _MIN_SIDE_PT:
            continue
        regions.append(Region(label=label, score=float(row[1]), bbox=bbox))

    # A box mostly inside another is a duplicate detection: keep the larger
    # one - except image-vs-other pairs (text over a picture is legitimate).
    dropped: set[int] = set()
    for i, a in enumerate(regions):
        for j in range(i + 1, len(regions)):
            if i in dropped or j in dropped:
                continue
            b = regions[j]
            if _overlap_small(a.bbox, b.bbox) <= _OVERLAP_DROP:
                continue
            if "image" in (a.label, b.label) and a.label != b.label:
                continue
            area_a = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1])
            area_b = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1])
            dropped.add(j if area_a >= area_b else i)
    regions = [r for k, r in enumerate(regions) if k not in dropped]

    order = 1
    for region in regions:
        if region.label not in _FLOATING:
            region.order = order
            order += 1
    return regions


def detect_regions(page: pymupdf.Page) -> list[Region]:
    """Labelled layout regions of a page in PDF points ([] when unavailable)."""
    if not layout_model_enabled():
        return []
    sess = _load()
    if not sess:
        return []
    import numpy as np

    rect = page.rect
    if rect.width <= 0 or rect.height <= 0:
        return []
    image = _render_input(page)
    # scale_factor maps model pixels back to PDF points directly.
    feeds = {
        "im_shape": np.array([[_INPUT_SIZE, _INPUT_SIZE]], dtype=np.float32),
        "image": image,
        "scale_factor": np.array(
            [[_INPUT_SIZE / rect.height, _INPUT_SIZE / rect.width]],
            dtype=np.float32),
    }
    try:
        boxes, counts = sess.run(None, feeds)
    except Exception:  # noqa: BLE001 - a bad page must not stop the run
        log.exception("layout model failed on page %s", page.number)
        return []
    raw = boxes[: int(counts[0])]
    return _postprocess(raw, rect.width, rect.height)
