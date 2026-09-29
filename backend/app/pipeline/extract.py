# PDF layout extraction based on PyMuPDF.
from __future__ import annotations

import os
from collections import OrderedDict

import pymupdf

from ..config import layout_model_enabled
from . import layout_model, ocr
from .models import Block, DocumentLayout, Line, PageLayout, Region, Span

# Font-name markers that indicate math glyph runs. The list is grounded in
# fonts actually observed in real papers:
# - TeX Computer Modern math families (CMMI*, CMSY*, CMEX*),
# - MathTime families used by IEEE productions (RMTMI/RMTMIB -> "MTMI",
#   MTSYN/MTSYB -> "MTSY", plus the MTEX/BLEX extension fonts),
# - AMS symbol fonts (MSAM10, MSBM10),
# - anything advertising itself as a math font (CambriaMath, STIXMath,
#   *-Math, ...). Bare "STIX" is intentionally NOT a marker: STIXGeneral is
#   a body-text font in some journals and must never be masked.
#
# Measured against an ACL 2025 production (NimbusRomNo9L body text): its
# display equations use exactly CMMI10/8/9, CMSY10/6/8/9, CMEX10 (all
# covered above) PLUS CMR10/8/9 for roman math material ("(", ") = min",
# fraction digits). "CMR" is deliberately NOT a marker: Computer Modern
# Roman is the BODY font of plain-TeX papers, so masking it would destroy
# running prose there (same reasoning as STIX). Equations therefore mix
# masked math-font spans with literal text-font spans ("-score" words,
# CMR operators); grouping those into one untouched segment is the job of
# the font-agnostic formula-zone pass in segment.py, not of this list.
_MATH_FONT_MARKERS = (
    "CMMI", "CMSY", "CMEX",
    "MTMI", "MTSY", "MTEX", "BLEX",
    "MSAM", "MSBM",
    "MATH",
)


# The generic "MATH" marker also hits text families that merely carry the
# word in their name: Nature sets its standfirsts and drop caps in
# GlosaMath-Roman / GlosaMath-RomanItalic, and masking those erased whole
# sentences from the translation. Real math fonts ship a single style
# (CambriaMath, LatinModernMath-Regular, XITSMath-Bold) or a Regular/Italic
# pair (STIXMath-Italic in Elsevier papers). A "Roman" style name, the
# typographer's word for a text face's upright cut, marks a text family.
_TEXT_STYLE_WORDS = ("ROMAN",)


def _is_math_font(font_name: str) -> bool:
    """Return True when the span font looks like a math font."""
    upper = font_name.upper()
    if not any(marker in upper for marker in _MATH_FONT_MARKERS):
        return False
    if any(marker in upper for marker in _MATH_FONT_MARKERS[:-1]):
        return True  # TeX / MathTime families are math whatever the style
    style = upper.rsplit("+", 1)[-1].partition("-")[2]
    return not any(word in style for word in _TEXT_STYLE_WORDS)


# --- micro-token literalization -----------------------------------------
# Real papers render single punctuation/relation glyphs of running prose
# with math fonts: the decimal point of "0.949" (RMTMI "."), the "=" of
# "N = 5" (MTSYN), exponents like the "6" of "10^-6". Masking those as
# ⟦EQn⟧ makes the visible characters vanish from the translated text
# ("0 949") and floods the placeholder guard with micro tokens. A math-font
# span whose text is such a micro token (1-2 visible characters, all from
# the whitelist below) therefore keeps its literal text and flows with the
# surrounding prose instead of being masked. Letters (Greek variables, big
# operators with broken ToUnicode like BLEX "\x11") are never literalized.
# Hyphen/dash characters are excluded on purpose: a literal line-final "-"
# would trigger the hyphenation-restore join in segmentation.
_LITERAL_MAX_CHARS = 2
_LITERAL_CHARS = frozenset(
    "0123456789"
    ".,;:!?()[]{}|/\\%"
    "=<>+±−×÷·∗*≤≥≈≃≅∼~≠≲≳"
    "•◦▪"
)


def math_span_literal(text: str) -> str | None:
    """Literal replacement for a micro math-font span, or None to mask.

    Returns ``text`` unchanged (whitespace preserved) when its visible core
    is 1-2 characters long and consists only of whitelisted punctuation,
    digits and relation/operator symbols; None means "mask as ⟦EQn⟧".
    """
    core = "".join(text.split())
    if 0 < len(core) <= _LITERAL_MAX_CHARS \
            and all(ch in _LITERAL_CHARS for ch in core):
        return text
    return None


# --- drop caps -----------------------------------------------------------
# Magazine-style layouts (Nature features, Science news) open a paragraph
# with one oversized initial letter. Some productions set that letter in a
# font whose name contains "Math" (e.g. GlosaMath-Roman), which used to get
# it masked as a formula and silently dropped from the translation
# ("ore than 70%" instead of "More than 70%"). A lone letter this large is
# never math in practice: display formulas use body-sized glyphs, and big
# operators (sum, integral) are not letters.
_DROP_CAP_MIN_SIZE = 20.0


def is_drop_cap_span(text: str, size: float) -> bool:
    """True when the span is a single oversized letter (a drop cap)."""
    core = "".join(text.split())
    return len(core) == 1 and core.isalpha() and size >= _DROP_CAP_MIN_SIZE


def is_masked_math_span(
    text: str, font_name: str, size: float | None = None
) -> bool:
    """True when analyze_pdf masks this span as an ⟦EQn⟧ placeholder.

    Shared with retypeset: only masked runs take part in snapshot
    restoration and inline-flow matching; literalized micro tokens are
    ordinary text (redacted and re-typeset with the translation). Drop caps
    are never masked (see is_drop_cap_span); callers that know the span
    size must pass it so both stages agree.
    """
    if size is not None and is_drop_cap_span(text, size):
        return False
    return bool(text.strip()) and _is_math_font(font_name) \
        and math_span_literal(text) is None


# A line whose writing direction deviates from horizontal by more than this
# (sine of the angle) is rotated text: vertical download stamps in the page
# margin ("Downloaded from science.org ..."), rotated axis labels. It is never
# running prose, so it is left out of the layout entirely; its glyphs stay in
# the output PDF untouched instead of being translated as body text.
_ROTATED_LINE_SIN = 0.2

# Drop-cap attachment window, relative to the drop cap box.
_DROP_CAP_MAX_GAP = 30.0  # pt between the cap's right edge and the line start
_DROP_CAP_TOP_RATIO = 0.6  # line top within this fraction of the cap height


_DUP_TOL = 1.0  # pt: same text drawn this close again is a duplicate


def _drop_duplicate_lines(blocks: list[Block]) -> None:
    """Remove text lines drawn twice at the same spot (shadow/outline effect).

    Infographics often print a label twice for a halo or drop shadow; both
    copies were extracted and merged into one line ("38% 38% Yes, a slight
    Yes, a slight crisis crisis").
    """
    seen: list[tuple[str, tuple]] = []
    for block in blocks:
        if block.type != "text":
            continue
        kept = []
        for line in block.lines:
            text = "".join(sp.text for sp in line.spans).strip()
            dup = any(
                text == other and all(abs(a - b) <= _DUP_TOL
                                      for a, b in zip(line.bbox, obox))
                for other, obox in seen
            )
            if dup:
                continue
            seen.append((text, line.bbox))
            kept.append(line)
        block.lines = kept
    blocks[:] = [b for b in blocks if b.type != "text" or b.lines]


def _attach_drop_caps(blocks: list[Block]) -> None:
    """Glue each drop cap onto the first line of the paragraph it opens.

    The cap is extracted as its own line, far larger than the text beside
    it, so segmentation would either lose it or make it a one-letter
    heading. It is moved to the front of the line that starts just right of
    it at the same height, taking that line's font size so size-based
    heading/paragraph logic keeps treating the line as body text. Only the
    line's left edge is widened to cover the cap: widening it vertically
    would make the following lines overlap it and merge into one row. The
    widened line is what later gets redacted, so the original cap glyph is
    erased together with the paragraph instead of floating over the
    translation.
    """
    lines = [(b, ln) for b in blocks if b.type == "text" for ln in b.lines]
    for _block, line in lines:
        visible = [sp for sp in line.spans if sp.text.strip()]
        if len(visible) != 1 or not is_drop_cap_span(visible[0].text, visible[0].size):
            continue
        cap = visible[0]
        cx0, cy0, cx1, cy1 = cap.bbox
        height = cy1 - cy0
        target: Line | None = None
        for _other_block, other in lines:
            if other is line or not other.spans:
                continue
            lx0, ly0, _lx1, _ly1 = other.bbox
            if not (cx1 - 4.0 <= lx0 <= cx1 + _DROP_CAP_MAX_GAP):
                continue
            if not (cy0 - 2.0 <= ly0 <= cy0 + height * _DROP_CAP_TOP_RATIO):
                continue
            # another oversized line (a headline) is not the paragraph start
            if max(sp.size for sp in other.spans) >= cap.size * 0.6:
                continue
            if target is None or ly0 < target.bbox[1]:
                target = other
        if target is None:
            continue
        first = target.spans[0]
        target.spans.insert(0, Span(
            text=cap.text.strip(),
            bbox=cap.bbox,
            font=first.font,
            size=first.size,
            flags=first.flags,
        ))
        target.bbox = (min(target.bbox[0], cx0), target.bbox[1],
                       target.bbox[2], target.bbox[3])
        line.spans = [sp for sp in line.spans if sp is not cap]

    for block in blocks:
        if block.type == "text":
            block.lines = [ln for ln in block.lines
                           if any(sp.text.strip() for sp in ln.spans)]
    blocks[:] = [b for b in blocks if b.type != "text" or b.lines]


# --- model results cache -------------------------------------------------
# analyze_pdf runs more than once per document (translation run, figure
# assets API); layout detection and OCR are the expensive part, so their
# per-page results are memoized by file identity (path, size, mtime).
_MODEL_CACHE_DOCS = 4
_model_cache: OrderedDict[tuple, dict[int, tuple[list[Region], list]]] = (
    OrderedDict())


def _doc_key(path: str) -> tuple | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (os.path.abspath(path), st.st_size, st.st_mtime_ns,
            layout_model_enabled())


def _page_models(page: pymupdf.Page, cache: dict) -> tuple[list[Region], list]:
    """(layout regions, OCR lines) of a page, memoized per document."""
    hit = cache.get(page.number)
    if hit is None:
        lines = ocr.ocr_page(page) if ocr.is_scanned_page(page) else []
        hit = (layout_model.detect_regions(page), lines)
        cache[page.number] = hit
    return hit


# OCR line boxes include ascenders/descenders: font size ~ 80% of the height.
_OCR_SIZE_RATIO = 0.8
_OCR_FONT = "OCR"


_OCR_CAP_MIN_GAP = 10.0  # pt between the region's left edge and the line


def _restore_ocr_drop_cap(page: pymupdf.Page, first: Line, region: Region
                          ) -> None:
    """Put back a scanned drop cap the OCR line detector skipped.

    A paragraph region whose first line starts in lower case well right of
    the region's left edge has its initial letter in that gap.
    """
    span = first.spans[0]
    if not span.text[:1].islower():
        return
    x0, y0, _x1, y1 = first.bbox
    if x0 - region.bbox[0] < _OCR_CAP_MIN_GAP:
        return
    height = y1 - y0
    gap = pymupdf.Rect(region.bbox[0], y0 - 1.0, x0 - 0.5,
                       min(region.bbox[3], y0 + height * 4.5))
    letter = ocr.read_drop_cap(page, gap)
    if letter:
        span.text = letter + span.text
        first.bbox = (region.bbox[0], y0, first.bbox[2], y1)
        span.bbox = first.bbox


def _ocr_blocks(page: pymupdf.Page, lines: list, regions: list[Region]
                ) -> list[Block]:
    """Text blocks for a scanned page: OCR lines grouped by layout region.

    Lines sharing a layout region form one block (one paragraph candidate);
    lines outside every region become single-line blocks. Lines in the side
    margins are dropped like the rotated stamps of born-digital pages.
    """
    width = page.rect.width
    grouped: dict[int, list[Line]] = {}
    loose: list[Line] = []
    for ocr_line in lines:
        x0, y0, x1, y1 = ocr_line.bbox
        if x0 >= width * 0.94 or x1 <= width * 0.06:
            continue  # margin stamp read as text
        size = max(4.0, (y1 - y0) * _OCR_SIZE_RATIO)
        span = Span(text=ocr_line.text, bbox=ocr_line.bbox, font=_OCR_FONT,
                    size=size, flags=0)
        line = Line(spans=[span], bbox=ocr_line.bbox)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        home = next((i for i, r in enumerate(regions)
                     if r.bbox[0] <= cx <= r.bbox[2]
                     and r.bbox[1] <= cy <= r.bbox[3]), None)
        if home is None:
            loose.append(line)
        else:
            grouped.setdefault(home, []).append(line)
    for home, group in grouped.items():
        group.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
        _restore_ocr_drop_cap(page, group[0], regions[home])
    blocks: list[Block] = []
    for group in [*grouped.values(), *[[ln] for ln in loose]]:
        group.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
        bbox = (min(ln.bbox[0] for ln in group), min(ln.bbox[1] for ln in group),
                max(ln.bbox[2] for ln in group), max(ln.bbox[3] for ln in group))
        blocks.append(Block(type="text", bbox=bbox, lines=group))
    return blocks


def analyze_pdf(path: str) -> DocumentLayout:
    """Read a PDF and return its block/line/span layout.

    Spans rendered with a math font keep their bbox/font/size/flags but their
    text is replaced by an ``(EQn)`` placeholder (U+27E6/U+27E7 brackets) so
    that downstream translation never touches formula content. Micro tokens
    (single punctuation/digit/relation glyphs such as the decimal point of
    "0.949" or the "=" of "N = 5", see math_span_literal) are exempt: they
    keep their literal text and stay part of the prose.
    """
    doc = pymupdf.open(path)
    eq_counter = 0
    pages: list[PageLayout] = []
    key = _doc_key(path)
    cache = _model_cache.pop(key, {}) if key else {}
    if key:
        _model_cache[key] = cache
        while len(_model_cache) > _MODEL_CACHE_DOCS:
            _model_cache.popitem(last=False)
    try:
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            regions, ocr_lines = _page_models(page, cache)
            if ocr_lines:
                # Scanned page: the OCR lines are the text; the page-sized
                # scan image is not a figure and is left out of the layout.
                area = abs(page.rect)
                blocks = [
                    Block(type="image", bbox=tuple(info["bbox"]), lines=[])
                    for info in page.get_image_info()
                    if abs(pymupdf.Rect(info["bbox"]) & page.rect) < area * 0.5
                ]
                blocks.extend(_ocr_blocks(page, ocr_lines, regions))
                pages.append(PageLayout(
                    number=pno, width=page.rect.width,
                    height=page.rect.height, blocks=blocks,
                    regions=regions, ocr=True))
                continue
            raw = page.get_text("dict")
            blocks: list[Block] = []
            for raw_block in raw.get("blocks", []):
                if raw_block.get("type") == 1:  # raster image block
                    blocks.append(
                        Block(type="image", bbox=tuple(raw_block["bbox"]), lines=[])
                    )
                    continue
                lines: list[Line] = []
                for raw_line in raw_block.get("lines", []):
                    direction = raw_line.get("dir", (1.0, 0.0))
                    if abs(direction[1]) > _ROTATED_LINE_SIN:
                        continue  # rotated margin stamp / axis label
                    spans: list[Span] = []
                    for raw_span in raw_line.get("spans", []):
                        text = raw_span["text"]
                        if is_masked_math_span(
                                text, raw_span["font"], raw_span["size"]):
                            text = f"⟦EQ{eq_counter}⟧"
                            eq_counter += 1
                        spans.append(
                            Span(
                                text=text,
                                bbox=tuple(raw_span["bbox"]),
                                font=raw_span["font"],
                                size=raw_span["size"],
                                flags=raw_span["flags"],
                            )
                        )
                    if spans:
                        lines.append(Line(spans=spans, bbox=tuple(raw_line["bbox"])))
                if lines:
                    blocks.append(
                        Block(type="text", bbox=tuple(raw_block["bbox"]), lines=lines)
                    )
            _drop_duplicate_lines(blocks)
            _attach_drop_caps(blocks)
            for drawing in page.get_drawings():
                rect = drawing["rect"]
                blocks.append(
                    Block(
                        type="drawing",
                        bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                        lines=[],
                    )
                )
            pages.append(
                PageLayout(
                    number=pno,
                    width=page.rect.width,
                    height=page.rect.height,
                    blocks=blocks,
                    regions=regions,
                )
            )
    finally:
        doc.close()
    return DocumentLayout(pages=pages)
