# Retypesetting: redact source text and re-render Korean into the same bboxes.
from __future__ import annotations

import base64
import html
import re
from dataclasses import replace

import pymupdf

from app import config

from .extract import is_masked_math_span
from .models import RenderReport, Segment
from .segment import UNTRANSLATED_KINDS

MIN_SCALE = 0.65  # preferred lowest font scale
# --- Loss-free overflow policy (DESIGN 2.5 principle update) -------------
# The original policy (shrink to MIN_SCALE, then expand downward but never
# into the next block, else give up) could silently DROP a segment whose
# source text had already been redacted. Updated principle: overlap beats
# loss - when every collision-free option fails, the scale floor is lowered
# stepwise (0.5 -> 0.35) and, as the terminal step, the bbox is force-
# expanded downward past obstacles by up to one line height and inserted
# through a path that cannot fail. "Insertion succeeded 0 times after the
# redact" must be structurally impossible.
# User-feedback refinement: slightly smaller text is preferred over any
# visible crowding of the next block, so a _PRE_EXPAND_SCALE shrink step
# runs BEFORE the collision-free expansion, and the forced expansion cap
# was tightened from 1.5 to 1.0 line heights.
_PRE_EXPAND_SCALE = 0.55  # extra shrink tried before any bbox growth
_RESCUE_SCALES = (0.5, 0.35)  # stepwise scale floors after expansion fails
_FLOOR_SCALE = _RESCUE_SCALES[-1]  # scale used with the forced expansion
# Line-height factor shared with the htmlbox CSS. User-feedback update:
# 0.96 packed CJK glyph boxes (~1.39em tall) so tightly that adjacent lines
# visibly overlapped once sub-MIN_SCALE rescue rendering kicked in. 1.04
# removes the overlap; the capacity loss is absorbed by the loss-free
# rescue ladder (a slightly smaller scale beats overlapping lines).
_LINE_HEIGHT_EM = 1.04
_FORCED_EXPAND_LINES = 1.0  # forced downward growth cap, in line heights
_NEIGHBOR_GAP = 3.0  # safety gap kept above the next block when expanding
# Minimum usable rect height: a redacted bbox thinner than this is grown to
# one line height before insertion instead of being skipped (skipping would
# lose the text).
_MIN_RECT_HEIGHT = 2.0
_FALLBACK_FONTNAME = "korea"  # PyMuPDF built-in CJK font for insert_textbox
# --- Body font (Korean serif) -------------------------------------------
# Translated text is rendered with a serif (Batang-style) font when one is
# available. Resolution chain: explicit env path (PAPERTRANSLATE_FONT_PATH)
# -> HANBatang (Windows HWP font) -> NotoSerifKR -> None (keep the current
# sans-serif behaviour). An explicit env path that cannot be loaded disables
# the custom font entirely (predictable sans fallback) instead of silently
# switching to a system font.
_BODY_FONT_FAMILY = "pt-bodyserif"  # CSS family name used in @font-face
_BODY_FONT_REGULAR_MEMBER = "pt-body-regular.ttf"  # Archive member names
_BODY_FONT_BOLD_MEMBER = "pt-body-bold.ttf"
_TEXTBOX_FONTNAME = "ptserif"  # page fontname for the insert_textbox path
_DEFAULT_FONT_CANDIDATES: tuple[tuple[str, str | None], ...] = (
    (r"C:\Windows\Fonts\HANBatang.ttf", r"C:\Windows\Fonts\HANBatangB.ttf"),
    (r"C:\Windows\Fonts\NotoSerifKR-VF.ttf", None),
)
# (regular, bold) buffers cached per env-path pair: font files are read once
# per process, never per document. A changed env value (tests) re-resolves.
_body_font_cache: dict[tuple[str, str], tuple[bytes, bytes] | None] = {}


def _read_font(path: str) -> bytes | None:
    """Read and validate a TTF; None when missing or not a usable font."""
    try:
        with open(path, "rb") as fh:
            buf = fh.read()
        pymupdf.Font(fontbuffer=buf)  # raises on corrupt/unsupported data
        return buf
    except Exception:
        return None


def _load_body_font() -> tuple[bytes, bytes] | None:
    """Resolve the body font as (regular, bold) TTF buffers, or None.

    With no bold file the regular buffer doubles as the bold face so that
    headings keep identical glyph coverage (just without the heavier weight).
    """
    key = (config.get_font_path(), config.get_font_bold_path())
    if key in _body_font_cache:
        return _body_font_cache[key]
    env_regular, env_bold = key
    if env_regular:
        candidates: list[tuple[str, str | None]] = [
            (env_regular, env_bold or None)
        ]
    else:
        candidates = list(_DEFAULT_FONT_CANDIDATES)
    result: tuple[bytes, bytes] | None = None
    for regular_path, bold_path in candidates:
        regular = _read_font(regular_path)
        if regular is None:
            continue
        bold = _read_font(bold_path) if bold_path else None
        result = (regular, bold if bold is not None else regular)
        break
    _body_font_cache[key] = result
    return result
# Literal equation placeholders must never be typeset: the original formula
# glyphs stay in place, or are restored as images (DESIGN 1.3-4 / 2.2).
_EQ_TOKEN_RE = re.compile(r"⟦EQ\d+⟧")
_MATH_CLIP_ZOOM = 3.0  # rasterization zoom for restored formula snippets
# Height cap for inline formula images, in em of the segment font size.
# Empirical Story metric: an inline image taller than the text ascent
# (~0.8em for the CJK fallback) shifts the neighbouring baselines by half
# the excess, producing irregular line spacing. Snapshots are therefore
# cropped to their glyph ink (dropping the font's asc/desc padding) and
# capped at 0.72em (tightened from 0.8: with the descender part of the
# vertical-align shift the 0.8em cap could still grow the line box), which
# keeps every line step uniform.
_INLINE_IMG_MAX_EM = 0.72
# White-ish threshold for the ink-cropping scan (antialiased glyph pixels
# are darker than this in every channel).
_INK_THRESHOLD = 250
# Minimum share of a math-run rect that must lie inside a container rect for
# the run to count as belonging to it (protection / inline-flow matching).
_INSIDE_RATIO = 0.7

_htmlbox_korean_ok: bool | None = None
_htmlbox_image_mode: str | None = None


def _htmlbox_renders_korean() -> bool:
    """Empirical probe: does insert_htmlbox produce extractable Hangul?

    The result decides whether insert_htmlbox is used at all; when the probe
    fails, the insert_textbox(fontname="korea") fallback is promoted to the
    default path. Cached per process.
    """
    global _htmlbox_korean_ok
    if _htmlbox_korean_ok is not None:
        return _htmlbox_korean_ok
    try:
        doc = pymupdf.open()
        try:
            page = doc.new_page(width=300, height=200)
            page.insert_htmlbox(
                pymupdf.Rect(10, 10, 290, 190), "<div>한글 검증</div>",
                scale_low=MIN_SCALE,
            )
            data = doc.tobytes()
        finally:
            doc.close()
        probe = pymupdf.open(stream=data, filetype="pdf")
        try:
            text = probe.load_page(0).get_text()
        finally:
            probe.close()
        _htmlbox_korean_ok = any("가" <= ch <= "힣" for ch in text)
    except Exception:
        _htmlbox_korean_ok = False
    return _htmlbox_korean_ok


def _htmlbox_supports_images() -> str:
    """Empirical probe: how can insert_htmlbox display an inline <img>?

    Tries a data: URI first, then an in-memory pymupdf.Archive member.
    Returns "data", "archive" or "" (no image support: inline-flow formula
    insertion is disabled and the restore-at-origin path is used instead).
    Cached per process like the Korean probe above.
    """
    global _htmlbox_image_mode
    if _htmlbox_image_mode is not None:
        return _htmlbox_image_mode
    try:
        tmp = pymupdf.open()
        try:
            probe_page = tmp.new_page(width=40, height=20)
            probe_page.draw_rect(
                pymupdf.Rect(0, 0, 40, 20), color=None, fill=(1, 0, 0)
            )
            png = probe_page.get_pixmap().tobytes("png")
        finally:
            tmp.close()
    except Exception:
        _htmlbox_image_mode = ""
        return ""
    for mode in ("data", "archive"):
        try:
            doc = pymupdf.open()
            try:
                page = doc.new_page(width=300, height=200)
                if mode == "data":
                    src = ("data:image/png;base64,"
                           + base64.b64encode(png).decode("ascii"))
                    archive = None
                else:
                    src = "probe.png"
                    archive = pymupdf.Archive()
                    archive.add(png, src)
                page.insert_htmlbox(
                    pymupdf.Rect(10, 10, 290, 190),
                    f'<div>x <img src="{src}" '
                    'style="width:20pt;height:10pt"> y</div>',
                    archive=archive,
                )
                data = doc.tobytes()
            finally:
                doc.close()
            check = pymupdf.open(stream=data, filetype="pdf")
            try:
                ok = bool(check.load_page(0).get_image_info())
            finally:
                check.close()
            if ok:
                _htmlbox_image_mode = mode
                return mode
        except Exception:
            continue
    _htmlbox_image_mode = ""
    return ""


def _record_overflow(report: RenderReport, segment_id: str) -> None:
    if segment_id not in report.overflow_segments:
        report.overflow_segments.append(segment_id)


def _expand_rect(
    page: pymupdf.Page,
    rect: pymupdf.Rect,
    step: float,
    obstacles: list[pymupdf.Rect],
) -> pymupdf.Rect | None:
    """Grow the rect downward, bounded by the page and by the next block.

    Expansion stops _NEIGHBOR_GAP (3pt) above the closest obstacle (another
    segment, an image, a drawing or a preserved formula) that lies below the
    rect and overlaps it horizontally (DESIGN 2.5-3: never collide with the
    next block); with nothing below, the page bottom margin is the limit.
    When no room is left, None is returned so callers keep the overflow
    recorded instead of scaling below MIN_SCALE.
    """
    limit = page.rect.y1 - 2.0
    for obs in obstacles:
        if obs.y0 >= rect.y1 - 0.1 and obs.x0 < rect.x1 and obs.x1 > rect.x0:
            limit = min(limit, obs.y0 - _NEIGHBOR_GAP)
    new_y1 = min(rect.y1 + step, limit)
    if new_y1 <= rect.y1 + 0.1:
        return None
    return pymupdf.Rect(rect.x0, rect.y0, rect.x1, new_y1)


def _force_expand_rect(
    page: pymupdf.Page, rect: pymupdf.Rect, font_size: float
) -> pymupdf.Rect:
    """Terminal rescue: grow the rect downward IGNORING obstacles.

    Growth is capped at _FORCED_EXPAND_LINES line heights and at the page
    bottom. Deliberately blind to neighbouring blocks - this runs only after
    every collision-free option has failed, and overlapping the next block
    is preferred over silently dropping translated text (DESIGN 2.5
    principle update: overlap beats loss).
    """
    new_y1 = min(
        rect.y1 + font_size * _LINE_HEIGHT_EM * _FORCED_EXPAND_LINES,
        page.rect.y1,
    )
    if new_y1 <= rect.y1:
        return rect
    return pymupdf.Rect(rect.x0, rect.y0, rect.x1, new_y1)


def _insert_html(
    page: pymupdf.Page,
    segment: Segment,
    payload: str,
    report: RenderReport,
    obstacles: list[pymupdf.Rect],
    archive: pymupdf.Archive | None = None,
) -> bool:
    """Primary path: insert_htmlbox with automatic down-scaling.

    ``payload`` is the ready-made inner HTML (escaped text, optionally with
    inline <img> formula snippets); ``archive`` resolves relative image
    sources when the platform requires Archive-based images.
    """
    rect = pymupdf.Rect(*segment.bbox)
    if segment.kind == "heading":
        payload = f"<b>{payload}</b>"
    # Body font injection: register the serif faces via @font-face backed by
    # Archive members. The font archive merges with the inline-formula image
    # archive (member names never collide with the "<segid>_eqN.png" images).
    font = _load_body_font()
    if font is not None:
        if archive is None:
            archive = pymupdf.Archive()
        archive.add(font[0], _BODY_FONT_REGULAR_MEMBER)
        archive.add(font[1], _BODY_FONT_BOLD_MEMBER)
        family = _BODY_FONT_FAMILY
        face_css = (
            f"@font-face {{font-family:{family};"
            f"src:url({_BODY_FONT_REGULAR_MEMBER});}}"
            f"@font-face {{font-family:{family};"
            f"src:url({_BODY_FONT_BOLD_MEMBER});font-weight:bold;}}"
        )
    else:
        family = "sans-serif"
        face_css = ""
    css = (
        f"{face_css}* {{margin:0;padding:0;font-family:{family};"
        f"font-size:{segment.font_size:.1f}pt;"
        f"line-height:{_LINE_HEIGHT_EM};}}"
    )

    def attempt(target: pymupdf.Rect, low: float) -> tuple[float, float]:
        # A failed insert_htmlbox call writes nothing, so retrying with a
        # different rect/floor never duplicates text.
        return page.insert_htmlbox(
            target, f"<div>{payload}</div>", css=css, scale_low=low,
            archive=archive,
        )

    spare, scale = attempt(rect, MIN_SCALE)
    if spare < 0:
        # Did not fit even at MIN_SCALE: escalate along the rescue ladder
        # (loss-free policy, see module header): (1) modest extra shrink to
        # _PRE_EXPAND_SCALE - slightly smaller text beats crowding the next
        # block, (2) collision-free downward expansion at MIN_SCALE,
        # (3) stepwise lower scale floors, (4) forced expansion past
        # obstacles at the floor scale, (5) unbounded shrink - scale_low=0
        # always fits, so the ladder cannot end in a text drop.
        _record_overflow(report, segment.id)
        spare, scale = attempt(rect, _PRE_EXPAND_SCALE)
        if spare < 0:
            expanded = _expand_rect(
                page, rect, rect.height * 0.5 + 4.0, obstacles)
            if expanded is not None:
                rect = expanded
                spare, scale = attempt(rect, MIN_SCALE)
        if spare < 0:
            for low in _RESCUE_SCALES:
                spare, scale = attempt(rect, low)
                if spare >= 0:
                    break
        if spare < 0:
            rect = _force_expand_rect(page, rect, segment.font_size)
            spare, scale = attempt(rect, _FLOOR_SCALE)
        if spare < 0:
            spare, scale = attempt(rect, 0)
    if spare < 0:
        return False  # unreachable in practice; keeps the caller's fallback
    if scale < 1.0:
        report.scaled_segments[segment.id] = round(scale, 3)
    return True


def _insert_textbox(
    page: pymupdf.Page,
    segment: Segment,
    text: str,
    report: RenderReport,
    obstacles: list[pymupdf.Rect],
) -> bool:
    """Fallback path: insert_textbox with the body serif font when it is
    available, else the built-in Korean font."""
    rect = pymupdf.Rect(*segment.bbox)
    fontname = _FALLBACK_FONTNAME
    font = _load_body_font()
    if font is not None:
        try:
            # Register once per page; repeated same-name/same-buffer calls
            # are deduplicated by PyMuPDF.
            page.insert_font(fontname=_TEXTBOX_FONTNAME, fontbuffer=font[0])
            fontname = _TEXTBOX_FONTNAME
        except Exception:
            fontname = _FALLBACK_FONTNAME
    def attempt(target: pymupdf.Rect, fontsize: float) -> float:
        # A failed insert_textbox call writes nothing, so retrying with a
        # different rect/size never duplicates text.
        return page.insert_textbox(
            target, text, fontname=fontname, fontsize=fontsize,
            align=pymupdf.TEXT_ALIGN_LEFT,
        )

    size = segment.font_size
    min_size = segment.font_size * MIN_SCALE
    pre_expand_size = max(segment.font_size * _PRE_EXPAND_SCALE, 1.0)
    leftover = -1.0
    while True:
        leftover = attempt(rect, size)
        if leftover >= 0:
            break
        if size - 0.5 >= min_size:
            size -= 0.5
            continue
        _record_overflow(report, segment.id)
        if size - 0.5 >= pre_expand_size:
            # Slightly smaller text beats crowding the next block: keep
            # shrinking down to _PRE_EXPAND_SCALE before any bbox growth.
            size -= 0.5
            continue
        expanded = _expand_rect(page, rect, 12.0, obstacles)
        if expanded is not None:
            rect = expanded
            continue
        # No collision-free room left: rescue ladder (loss-free policy, see
        # module header). Lower the size floor stepwise, then force-expand
        # past obstacles and keep shrinking until the text fits - a small
        # enough font always fits, so the loop below terminates with the
        # text present.
        for low in _RESCUE_SCALES:
            size = max(segment.font_size * low, 1.0)
            leftover = attempt(rect, size)
            if leftover >= 0:
                break
        if leftover < 0:
            rect = _force_expand_rect(page, rect, segment.font_size)
            size = max(segment.font_size * _FLOOR_SCALE, 1.0)
            leftover = attempt(rect, size)
            while leftover < 0 and size > 1.0:
                size = max(size - 0.5, 1.0)
                leftover = attempt(rect, size)
        break
    if leftover < 0:
        return False  # pathological; the caller records the overflow
    if size < segment.font_size:
        report.scaled_segments[segment.id] = round(size / segment.font_size, 3)
    return True


def _rect_mostly_inside(
    rect: pymupdf.Rect, container: pymupdf.Rect, ratio: float = _INSIDE_RATIO
) -> bool:
    """True when at least ``ratio`` of rect's area lies inside container."""
    x0 = max(rect.x0, container.x0)
    y0 = max(rect.y0, container.y0)
    x1 = min(rect.x1, container.x1)
    y1 = min(rect.y1, container.y1)
    if x1 <= x0 or y1 <= y0:
        return False
    area = rect.get_area()
    return area > 0 and (x1 - x0) * (y1 - y0) >= ratio * area


def _scan_math_spans(
    page: pymupdf.Page,
) -> list[tuple[tuple[float, float, float], pymupdf.Rect, float]]:
    """All masked math-font glyph runs on the page, with an order key.

    Returns (order_key, rect, baseline_y) triples; order_key = (line y0,
    line x0, span x0) mirrors the line ordering used by segmentation, so
    sorting a segment's runs by it reproduces the ⟦EQn⟧ token order of the
    segment text. ``baseline_y`` is the span origin used for the inline-flow
    vertical alignment. Only spans that extraction masked as ⟦EQn⟧ count:
    literalized micro tokens (extract.math_span_literal) are ordinary text
    whose glyphs are redacted and re-typeset with the translation.
    """
    spans: list[tuple[tuple[float, float, float], pymupdf.Rect, float]] = []
    raw = page.get_text("dict")
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            lbox = line.get("bbox", (0.0, 0.0, 0.0, 0.0))
            for span in line.get("spans", []):
                if not is_masked_math_span(
                        span.get("text", ""), span.get("font", "")):
                    continue
                srect = pymupdf.Rect(span["bbox"])
                if srect.is_empty:
                    continue
                origin_y = float(span.get("origin", (0.0, srect.y1))[1])
                spans.append(((lbox[1], lbox[0], srect.x0), srect, origin_y))
    return spans


def _snapshot(page: pymupdf.Page, rect: pymupdf.Rect) -> pymupdf.Pixmap:
    """Rasterize a page region before redaction (zoom keeps glyphs sharp)."""
    matrix = pymupdf.Matrix(_MATH_CLIP_ZOOM, _MATH_CLIP_ZOOM)
    return page.get_pixmap(clip=rect, matrix=matrix)


def _ink_clip(page: pymupdf.Page, rect: pymupdf.Rect) -> pymupdf.Rect:
    """Tighten a glyph-run rect to its rendered ink (white margins cut).

    A text span rect spans the font's full ascender..descender box, mostly
    empty for a short inline formula; displaying that box inline would blow
    up the line height. When no ink is found (or the background is not
    white-ish) the original rect is returned and the size cap alone applies.
    """
    pix = _snapshot(page, rect)
    n = pix.n
    samples = pix.samples
    width, height = pix.width, pix.height
    min_x, min_y, max_x, max_y = width, height, -1, -1
    for row in range(height):
        base = row * pix.stride
        for col in range(width):
            offset = base + col * n
            if any(samples[offset + c] < _INK_THRESHOLD for c in range(min(n, 3))):
                if col < min_x:
                    min_x = col
                if col > max_x:
                    max_x = col
                if row < min_y:
                    min_y = row
                max_y = row
    if max_x < 0:
        return rect
    pad = 1.0 / _MATH_CLIP_ZOOM  # one rendered pixel of padding
    return pymupdf.Rect(
        max(rect.x0, rect.x0 + min_x / _MATH_CLIP_ZOOM - pad),
        max(rect.y0, rect.y0 + min_y / _MATH_CLIP_ZOOM - pad),
        min(rect.x1, rect.x0 + (max_x + 1) / _MATH_CLIP_ZOOM + pad),
        min(rect.y1, rect.y0 + (max_y + 1) / _MATH_CLIP_ZOOM + pad),
    )


def _inline_math_payload(
    text: str,
    runs: dict[str, tuple[pymupdf.Rect, bytes, float, float, float]],
    image_mode: str,
    names: dict[str, str] | None,
) -> str:
    """HTML body for a segment whose inline formulas flow with the text.

    Every ⟦EQn⟧ token is replaced by an <img> holding the pre-redaction
    ink snapshot of its math run, displayed at its natural size (capped at
    _INLINE_IMG_MAX_EM em) and baseline-aligned via a negative
    vertical-align offset, so the snippet sits on the text line it now
    belongs to instead of its original page coordinates.

    Robustness (loss-free): a token with no matching run renders as a
    space; a run whose token never occurs in ``text`` (or occurs again
    after being used once) is appended after the last line instead - no
    formula ink is dropped and no image floats at its original page
    coordinates over now-empty space.
    """

    def img_tag(token: str) -> str:
        _rect, png, disp_w, disp_h, valign = runs[token]
        if image_mode == "data":
            src = ("data:image/png;base64,"
                   + base64.b64encode(png).decode("ascii"))
        else:
            src = (names or {})[token]
        return (
            f'<img src="{src}" style="width:{disp_w:.2f}pt;'
            f'height:{disp_h:.2f}pt;vertical-align:{valign:.2f}pt">'
        )

    parts: list[str] = []
    pos = 0
    used: set[str] = set()
    for match in _EQ_TOKEN_RE.finditer(text):
        parts.append(html.escape(text[pos:match.start()]))
        token = match.group(0)
        if token in runs and token not in used:
            used.add(token)
            parts.append(img_tag(token))
        else:
            parts.append(" ")
        pos = match.end()
    parts.append(html.escape(text[pos:]))
    # Leftover runs (dict order = source/reading order) trail the text.
    for token in runs:
        if token not in used:
            parts.append(" ")
            parts.append(img_tag(token))
    return "".join(parts).replace("\n", "<br>")


def render_translated_pdf(
    src_path: str,
    segments: list[Segment],
    translations: dict[str, str],
    out_path: str,
) -> RenderReport:
    """Produce the translated PDF.

    Per page: redact every translated segment bbox (keeping images and line
    art untouched), then insert the Korean text into the same bbox. The
    primary renderer is insert_htmlbox(scale_low=0.65) with the resolved
    body serif font (see _load_body_font); when the empirical Korean probe
    fails or a segment cannot be placed, the insert_textbox fallback is used
    with the same font (or the built-in "korea" font without one). Both
    paths end in the loss-free rescue ladder (module header): a redacted
    segment always gets its translation inserted, at worst overlapping the
    next block or scaled below MIN_SCALE, with the overflow recorded.
    """
    report = RenderReport(overflow_segments=[], scaled_segments={})
    by_page: dict[int, list[Segment]] = {}
    all_by_page: dict[int, list[Segment]] = {}
    for segment in segments:
        all_by_page.setdefault(segment.page, []).append(segment)
        if segment.kind in UNTRANSLATED_KINDS:
            # Defensive guard: untranslated kinds (header_footer, formula,
            # author, reference) are never redacted nor re-inserted, even if
            # a stray translation was supplied for them.
            continue
        translated = translations.get(segment.id)
        if translated is None or not translated.strip():
            continue
        by_page.setdefault(segment.page, []).append(segment)

    use_htmlbox = _htmlbox_renders_korean()
    image_mode = _htmlbox_supports_images() if use_htmlbox else ""
    doc = pymupdf.open(src_path)
    try:
        for pno in sorted(by_page):
            page = doc.load_page(pno)
            # Fixed page furniture acts as expansion obstacles: images,
            # drawings and (below) preserved formula runs.
            base_obstacles: list[pymupdf.Rect] = []
            for info in page.get_image_info():
                base_obstacles.append(pymupdf.Rect(info["bbox"]))
            for drawing in page.get_drawings():
                base_obstacles.append(pymupdf.Rect(drawing["rect"]))

            redact_rects: list[pymupdf.Rect] = []
            for segment in by_page[pno]:
                rect = pymupdf.Rect(*segment.bbox)
                if rect.is_empty:
                    continue
                redact_rects.append(rect)

            # Whole-bbox protection: an untranslated segment (formula/author/
            # reference/header_footer) whose bbox intersects a redact rect
            # would lose the glyphs inside the overlap - including regular
            # font glyphs (digits, parens, "=") that the math-run restore
            # below cannot bring back. Snapshot the ENTIRE segment bbox
            # before redaction and re-insert it verbatim afterwards.
            protected: list[tuple[pymupdf.Rect, pymupdf.Pixmap]] = []
            for other in all_by_page.get(pno, []):
                if other.kind not in UNTRANSLATED_KINDS:
                    continue
                orect = pymupdf.Rect(*other.bbox)
                orect.intersect(page.rect)
                if orect.is_empty:
                    continue
                if any(orect.intersects(rr) for rr in redact_rects):
                    protected.append((orect, _snapshot(page, orect)))
            protected_rects = [rect for rect, _pix in protected]

            # Math glyph runs on the page: expansion obstacles, inline-flow
            # candidates and (last resort) restore-at-origin clips.
            math_spans = _scan_math_spans(page)
            math_rects = [rect for _key, rect, _oy in math_spans]
            base_obstacles.extend(math_rects)
            consumed: set[int] = set()
            for i, (_key, rect, _oy) in enumerate(math_spans):
                # Runs covered by a whole-bbox snapshot need no second copy.
                if any(_rect_mostly_inside(rect, prot)
                       for prot in protected_rects):
                    consumed.add(i)

            # Inline-flow assignment: a translated segment's ⟦EQn⟧ tokens are
            # matched 1:1 (source-text order vs page reading order) with the
            # math runs inside its bbox. Mismatch policy (user-feedback
            # update): the old restore-at-origin fallback left formula
            # images floating over the blank space that shorter Korean
            # translations leave behind. Instead, min(tokens, runs) pairs
            # are matched in order; surplus runs get synthetic keys that
            # never occur in the text, so _inline_math_payload appends them
            # after the last line. No formula ink is ever lost and no image
            # stays at its original coordinates. Run values are
            # (ink_rect, png, display_w, display_h, vertical_align).
            inline_runs: dict[
                str, dict[str, tuple[pymupdf.Rect, bytes, float, float, float]]
            ] = {}
            if use_htmlbox and image_mode:
                for segment in by_page[pno]:
                    source_tokens = _EQ_TOKEN_RE.findall(segment.text)
                    if not source_tokens:
                        continue
                    seg_rect = pymupdf.Rect(*segment.bbox)
                    candidates = [
                        (key, i, rect, oy)
                        for i, (key, rect, oy) in enumerate(math_spans)
                        if i not in consumed
                        and _rect_mostly_inside(rect, seg_rect)
                    ]
                    if not candidates:
                        continue  # nothing to place -> tokens render blank
                    candidates.sort(key=lambda item: item[0])
                    runs: dict[
                        str, tuple[pymupdf.Rect, bytes, float, float, float]
                    ] = {}
                    for pair_no, (_k, i, rect, oy) in enumerate(candidates):
                        if (pair_no < len(source_tokens)
                                and source_tokens[pair_no] not in runs):
                            token = source_tokens[pair_no]
                        else:
                            # Surplus (or duplicate-token) run: synthetic
                            # key -> appended after the segment's last line.
                            token = f"⟦EXTRA{pair_no}⟧"
                        ink = _ink_clip(page, rect)
                        png = _snapshot(page, ink).tobytes("png")
                        cap = segment.font_size * _INLINE_IMG_MAX_EM
                        factor = (min(1.0, cap / ink.height)
                                  if ink.height > 0 else 1.0)
                        disp_w = ink.width * factor
                        disp_h = ink.height * factor
                        # Align the run's internal baseline with the text
                        # baseline: shift the image down by its (scaled)
                        # descender part.
                        valign = -max(0.0, ink.y1 - oy) * factor
                        runs[token] = (ink, png, disp_w, disp_h, valign)
                        consumed.add(i)
                    inline_runs[segment.id] = runs

            # Snapshot the remaining math runs the redaction below would
            # destroy; they are re-inserted verbatim right after (DESIGN
            # 1.3-4: formulas keep both their position and content).
            math_clips = [
                (rect, _snapshot(page, rect))
                for i, (_key, rect, _oy) in enumerate(math_spans)
                if i not in consumed
                and any(rect.intersects(rr) for rr in redact_rects)
            ]

            for rect in redact_rects:
                page.add_redact_annot(rect, fill=False)
            page.apply_redactions(
                images=pymupdf.PDF_REDACT_IMAGE_NONE,
                graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
            )
            for rect, pix in protected:
                page.insert_image(rect, pixmap=pix)
            for rect, pix in math_clips:
                page.insert_image(rect, pixmap=pix)
            for segment in by_page[pno]:
                rect = pymupdf.Rect(*segment.bbox)
                if rect.is_empty:
                    continue  # never redacted above -> no text was removed
                if rect.height < _MIN_RECT_HEIGHT:
                    # This bbox WAS redacted; a too-thin rect must not drop
                    # its text (loss-free policy). Grow it to one line
                    # height, clamped to the page, before inserting.
                    _record_overflow(report, segment.id)
                    new_y1 = min(
                        rect.y0 + max(_MIN_RECT_HEIGHT,
                                      segment.font_size * _LINE_HEIGHT_EM),
                        page.rect.y1,
                    )
                    new_y0 = max(0.0, min(rect.y0, new_y1 - _MIN_RECT_HEIGHT))
                    segment = replace(
                        segment, bbox=(rect.x0, new_y0, rect.x1, new_y1))
                raw_text = translations[segment.id]
                runs = inline_runs.get(segment.id)
                # Legacy policy: equation placeholders are never typeset as
                # literal text - without inline flow they are removed and the
                # original glyphs stay (or were restored) in place.
                text = _EQ_TOKEN_RE.sub(" ", raw_text)
                if runs is None and not text.strip():
                    continue
                obstacles = base_obstacles + [
                    pymupdf.Rect(*other.bbox)
                    for other in all_by_page.get(pno, [])
                    if other.id != segment.id
                ]
                placed = False
                if use_htmlbox:
                    archive = None
                    if runs is not None:
                        names = None
                        if image_mode == "archive":
                            archive = pymupdf.Archive()
                            names = {}
                            for k, (token, run) in enumerate(runs.items()):
                                names[token] = f"{segment.id}_eq{k}.png"
                                archive.add(run[1], names[token])
                        payload = _inline_math_payload(
                            raw_text, runs, image_mode, names,
                        )
                    else:
                        payload = html.escape(text).replace("\n", "<br>")
                    placed = _insert_html(
                        page, segment, payload, report, obstacles,
                        archive=archive,
                    )
                if not placed:
                    if text.strip():
                        placed = _insert_textbox(
                            page, segment, text, report, obstacles)
                    if runs is not None:
                        # Inline insertion failed: restore the runs at their
                        # original coordinates so no formula content is lost.
                        for orect, png, _w, _h, _va in runs.values():
                            page.insert_image(orect, stream=png)
                if not placed and text.strip():
                    _record_overflow(report, segment.id)
        try:
            # insert_htmlbox embeds the full CJK fallback font per page;
            # subsetting shrinks the output by orders of magnitude.
            doc.subset_fonts()
        except Exception:
            pass  # size optimization only; never fail the render for it
        doc.save(out_path, garbage=3, deflate=True)
    finally:
        doc.close()
    return report
