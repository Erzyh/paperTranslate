# OCR for scanned pages (PP-OCRv6 small detector + recognizer, ONNX).
#
# A scanned PDF is one picture per page with no text layer, so extraction
# finds nothing to translate. For such pages the text lines are detected and
# read with PaddleOCR models; the result is handed to segmentation as
# ordinary lines (font "OCR"), and retypeset blanks the scanned pixels under
# each translated paragraph before writing the translation.
#
# Only onnxruntime + numpy are needed. The detector's DB probability map is
# turned into axis-aligned line boxes with a run-length connected-component
# pass (scanned papers are horizontal text; rotated boxes are not needed),
# and every line is re-rendered by PyMuPDF straight at the recognizer's
# 48 px input height, so no image library is involved.
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from ..config import get_models_dir, layout_model_enabled

log = logging.getLogger(__name__)

DET_FILE = "ocr_det.onnx"
REC_FILE = "ocr_rec.onnx"

# A page is "scanned" when its text layer is (almost) empty and one image
# covers most of it. Rotated margin stamps ("Downloaded from ...") are not
# counted: publishers add them to scans too.
_SCAN_MAX_TEXT_CHARS = 40
_SCAN_MIN_IMAGE_COVER = 0.5

_DET_SCALE = 2.0  # render zoom for detection (~144 dpi)
_DET_MAX_SIDE = 2400  # px cap for the detector input
_DB_THRESH = 0.3  # probability threshold of the text map
_DB_BOX_THRESH = 0.5  # mean probability a box must reach
_DB_UNCLIP = 1.6  # DB box expansion ratio
_MIN_BOX_PX = 3
_REC_HEIGHT = 48  # recognizer input height (px)
_REC_MAX_WIDTH = 2400  # px cap for one line image
_REC_BATCH = 8
_REC_MIN_SCORE = 0.5  # lines read with lower confidence are dropped


@dataclass
class OcrLine:
    bbox: tuple[float, float, float, float]  # PDF points
    text: str
    score: float


_lock = threading.Lock()
_sessions = None  # (det, rec) | False
_chars: list[str] = []


def _load():
    global _sessions, _chars
    with _lock:
        if _sessions is not None:
            return _sessions
        det_path = Path(get_models_dir()) / DET_FILE
        rec_path = Path(get_models_dir()) / REC_FILE
        if not (det_path.exists() and rec_path.exists()):
            log.info("OCR models not found in %s", det_path.parent)
            _sessions = False
            return _sessions
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            det = ort.InferenceSession(
                str(det_path), opts, providers=["CPUExecutionProvider"])
            rec = ort.InferenceSession(
                str(rec_path), opts, providers=["CPUExecutionProvider"])
            meta = rec.get_modelmeta().custom_metadata_map
            # CTC classes: blank, the dictionary, then a space.
            _chars = ["", *meta["character"].splitlines(), " "]
            _sessions = (det, rec)
        except Exception:  # noqa: BLE001
            log.exception("OCR models failed to load")
            _sessions = False
        return _sessions


def is_scanned_page(page: pymupdf.Page) -> bool:
    """True for a page that is a picture without a usable text layer."""
    chars = 0
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            if abs(line.get("dir", (1.0, 0.0))[1]) > 0.2:
                continue  # rotated stamp, not page text
            chars += sum(len(s["text"].strip()) for s in line.get("spans", []))
    if chars > _SCAN_MAX_TEXT_CHARS:
        return False
    area = abs(page.rect)
    if area <= 0:
        return False
    for info in page.get_image_info():
        box = pymupdf.Rect(info["bbox"]) & page.rect
        if abs(box) >= area * _SCAN_MIN_IMAGE_COVER:
            return True
    return False


def available() -> bool:
    return layout_model_enabled() and bool(_load())


def _pixmap_bgr(pix: pymupdf.Pixmap):
    import numpy as np

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    return img[:, :, 2::-1]  # RGB -> BGR (PaddleOCR models are BGR-trained)


def _components(mask):
    """Bounding boxes of 8-connected components (run-length union-find)."""
    import numpy as np

    parent: list[int] = []

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    runs: list[tuple[int, int, int, int]] = []  # (row, x0, x1, id)
    prev: list[tuple[int, int, int]] = []  # (x0, x1, id) of the previous row
    padded = np.zeros((mask.shape[0], mask.shape[1] + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    edges = np.diff(padded, axis=1)
    for y in range(mask.shape[0]):
        starts = np.flatnonzero(edges[y] == 1)
        if starts.size == 0:
            prev = []
            continue
        ends = np.flatnonzero(edges[y] == -1)  # exclusive
        cur: list[tuple[int, int, int]] = []
        j = 0
        for x0, x1 in zip(starts.tolist(), ends.tolist()):
            rid = len(parent)
            parent.append(rid)
            # previous-row runs touching [x0-1, x1] (8-connectivity)
            while j < len(prev) and prev[j][1] < x0:
                j += 1
            k = j
            while k < len(prev) and prev[k][0] <= x1:
                ra, rb = find(prev[k][2]), find(rid)
                if ra != rb:
                    parent[rb] = ra
                k += 1
            cur.append((x0, x1, rid))
            runs.append((y, x0, x1, rid))
        prev = cur

    boxes: dict[int, list[int]] = {}
    for y, x0, x1, rid in runs:
        root = find(rid)
        box = boxes.get(root)
        if box is None:
            boxes[root] = [x0, y, x1, y + 1]
        else:
            box[0] = min(box[0], x0)
            box[1] = min(box[1], y)
            box[2] = max(box[2], x1)
            box[3] = y + 1
    return list(boxes.values())


def _detect(page: pymupdf.Page, det) -> list[tuple[float, float, float, float]]:
    """Text-line boxes of the page in PDF points."""
    import numpy as np

    rect = page.rect
    zoom = min(_DET_SCALE, _DET_MAX_SIDE / max(rect.width, rect.height))
    # the detector needs sides that are multiples of 32
    w = max(32, int(round(rect.width * zoom / 32)) * 32)
    h = max(32, int(round(rect.height * zoom / 32)) * 32)
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(w / rect.width, h / rect.height),
        colorspace=pymupdf.csRGB, alpha=False)
    img = _pixmap_bgr(pix)
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    ch, cw = min(h, img.shape[0]), min(w, img.shape[1])
    canvas[:ch, :cw] = img[:ch, :cw]
    x = (canvas.astype(np.float32) / 255.0 - 0.5) / 0.5
    prob = det.run(None, {"x": x.transpose(2, 0, 1)[None]})[0][0, 0]
    mask = prob > _DB_THRESH
    # 2x2 dilation, as in PaddleOCR's DB post-processing
    mask[:-1, :] |= mask[1:, :]
    mask[:, :-1] |= mask[:, 1:]

    sx, sy = rect.width / w, rect.height / h
    out = []
    for x0, y0, x1, y1 in _components(mask):
        bw, bh = x1 - x0, y1 - y0
        if min(bw, bh) < _MIN_BOX_PX:
            continue
        if float(prob[y0:y1, x0:x1].mean()) < _DB_BOX_THRESH:
            continue
        # DB predicts shrunk kernels: grow by area*ratio/perimeter
        d = bw * bh * _DB_UNCLIP / (2.0 * (bw + bh))
        out.append((
            max(0.0, (x0 - d) * sx), max(0.0, (y0 - d) * sy),
            min(rect.width, (x1 + d) * sx), min(rect.height, (y1 + d) * sy),
        ))
    return out


def _recognize(page: pymupdf.Page, rec, boxes) -> list[OcrLine]:
    import numpy as np

    crops = []
    for box in boxes:
        clip = pymupdf.Rect(box)
        zoom = _REC_HEIGHT / clip.height
        if clip.width * zoom > _REC_MAX_WIDTH:
            continue
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip,
                              colorspace=pymupdf.csRGB, alpha=False)
        img = _pixmap_bgr(pix)
        if img.shape[0] != _REC_HEIGHT:  # rounding: pad/crop to 48 rows
            fixed = np.full((_REC_HEIGHT, img.shape[1], 3), 255, dtype=np.uint8)
            rows = min(_REC_HEIGHT, img.shape[0])
            fixed[:rows] = img[:rows]
            img = fixed
        crops.append((box, img))

    lines: list[OcrLine] = []
    crops.sort(key=lambda c: c[1].shape[1])  # similar widths batch together
    for i in range(0, len(crops), _REC_BATCH):
        batch = crops[i:i + _REC_BATCH]
        width = max(img.shape[1] for _b, img in batch)
        x = np.zeros((len(batch), 3, _REC_HEIGHT, width), dtype=np.float32)
        for k, (_b, img) in enumerate(batch):
            norm = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
            x[k, :, :, : img.shape[1]] = norm.transpose(2, 0, 1)
        probs = rec.run(None, {"x": x})[0]
        for (box, _img), seq in zip(batch, probs):
            idx = seq.argmax(axis=1)
            conf = seq.max(axis=1)
            keep = (idx != 0) & np.concatenate(([True], idx[1:] != idx[:-1]))
            if not keep.any():
                continue
            text = "".join(_chars[j] for j in idx[keep] if j < len(_chars))
            score = float(conf[keep].mean())
            if text.strip() and score >= _REC_MIN_SCORE:
                lines.append(OcrLine(bbox=box, text=text.strip(), score=score))
    lines.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
    return lines


def read_drop_cap(page: pymupdf.Page, rect: pymupdf.Rect) -> str | None:
    """Read a single oversized initial letter inside rect (or None).

    The line detector does not report drop caps (one huge glyph is not a
    text line to it), so extract.py asks for the gap left of a paragraph
    whose first line starts in lower case.
    """
    import numpy as np

    sessions = _load()
    if not sessions or rect.is_empty or rect.height < 4:
        return None
    _det, rec = sessions
    # Crop to the glyph's ink (a tight or clipped crop makes "R" read as
    # "B"), then pad it the way line crops are padded by the detector.
    zoom = 4.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=rect,
                          colorspace=pymupdf.csGRAY, alpha=False)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
    rows = np.flatnonzero((gray < 128).any(axis=1))
    cols = np.flatnonzero((gray < 128).any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    ink = pymupdf.Rect(rect.x0 + cols[0] / zoom, rect.y0 + rows[0] / zoom,
                       rect.x0 + (cols[-1] + 1) / zoom,
                       rect.y0 + (rows[-1] + 1) / zoom)
    pad = ink.height * 0.15
    crop = pymupdf.Rect(ink.x0 - pad, ink.y0 - pad, ink.x1 + pad, ink.y1 + pad)
    lines = _recognize(page, rec, [tuple(crop)])
    if not lines:
        return None
    text = lines[0].text.strip()
    if (len(text) == 1 and text.isalpha() and text.isupper()
            and lines[0].score >= 0.6):
        return text
    return None


def ocr_page(page: pymupdf.Page) -> list[OcrLine]:
    """Read the text lines of a scanned page ([] when OCR is unavailable)."""
    if not layout_model_enabled():
        return []
    sessions = _load()
    if not sessions:
        return []
    det, rec = sessions
    try:
        return _recognize(page, rec, _detect(page, det))
    except Exception:  # noqa: BLE001 - a bad page must not stop the run
        log.exception("OCR failed on page %s", page.number)
        return []
