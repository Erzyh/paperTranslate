# Figure/table region geometry shared by segmentation and the assets API.
#
# The caption-adjacency logic used to live inside assets.py; it is factored
# out here so that build_segments can classify text lying inside a detected
# figure/table region (kind "figure_text") with EXACTLY the same region
# definition the GET /assets endpoint reports. Everything is derived from
# the layout and the segments alone - no PDF or DB access.
from __future__ import annotations

import re

from .models import DocumentLayout, Segment

# Caption prefix that yields an asset key: "Figure 3", "FIGURE 3.", "Fig. 3",
# "Table 2", "TABLE IV." (case-insensitive, decimal or Roman number captured).
_CAPTION_KEY_RE = re.compile(
    r"^\s*(fig(?:ure)?\.?|table)\s*(\d+|[IVXLCDM]+\b)", re.IGNORECASE
)

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500,
                 "M": 1000}


def _caption_number(token: str) -> int:
    """Decimal or Roman caption number as an int ("IV" -> 4)."""
    if token.isdigit():
        return int(token)
    total = 0
    prev = 0
    for ch in reversed(token.upper()):
        value = _ROMAN_VALUES[ch]
        total = total - value if value < prev else total + value
        prev = max(prev, value)
    return total

# Max vertical gap (pt) between a caption and its nearest graphic block, and
# between graphic blocks absorbed into one figure region.
_ADJACENT_GAP = 50.0
# Slop (pt) for "above"/"below" side tests against the caption edges.
_EDGE_SLOP = 2.0


def h_overlap(a: tuple, b: tuple) -> float:
    """Horizontal overlap width of two bboxes (<= 0 means disjoint)."""
    return min(a[2], b[2]) - max(a[0], b[0])


def intersects(a: tuple, b: tuple) -> bool:
    """True when two bboxes share a positive area."""
    return (min(a[2], b[2]) - max(a[0], b[0]) > 0
            and min(a[3], b[3]) - max(a[1], b[1]) > 0)


def _blocked_by_caption(block: tuple, caption: tuple, side: str,
                        barriers: list[tuple]) -> bool:
    """True when another caption sits vertically between block and caption.

    A graphic separated from the caption by a different caption belongs to
    that other caption's figure, never to this one.
    """
    if side == "above":
        lo, hi = block[3], caption[1]
    else:
        lo, hi = caption[3], block[1]
    if lo >= hi:
        return False
    for bar in barriers:
        if h_overlap(bar, caption) <= 0 and h_overlap(bar, block) <= 0:
            continue
        center_y = (bar[1] + bar[3]) / 2.0
        if lo < center_y < hi:
            return True
    return False


def _grow_region(caption: tuple, graphics: list[tuple], side: str,
                 barriers: list[tuple],
                 claimed: list[tuple]) -> tuple | None:
    """Union of image/drawing blocks adjacent to a caption on one side.

    Eligible blocks horizontally overlap the caption, lie on the given side,
    are not separated from the caption by another caption and do not touch a
    region already claimed by a previous caption. The nearest eligible block
    within _ADJACENT_GAP seeds the region; the region then absorbs remaining
    eligible blocks to a fixpoint while their vertical gap to the region
    stays within _ADJACENT_GAP.
    """
    candidates: list[tuple[float, tuple]] = []
    for box in graphics:
        if h_overlap(box, caption) <= 0:
            continue
        if side == "above":
            if box[3] > caption[1] + _EDGE_SLOP:
                continue
            dist = caption[1] - box[3]
        else:
            if box[1] < caption[3] - _EDGE_SLOP:
                continue
            dist = box[1] - caption[3]
        if _blocked_by_caption(box, caption, side, barriers):
            continue
        # Inflate before the claimed test: zero-area blocks (axis/rule
        # lines) have no intersection area yet still belong to the region
        # that surrounds them.
        inflated = (box[0] - 0.5, box[1] - 0.5, box[2] + 0.5, box[3] + 0.5)
        if any(intersects(inflated, region) for region in claimed):
            continue
        candidates.append((max(dist, 0.0), box))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    seed_dist, seed = candidates[0]
    if seed_dist > _ADJACENT_GAP:
        return None
    region = list(seed)
    remaining = [box for _, box in candidates[1:]]
    changed = True
    while changed:
        changed = False
        for box in list(remaining):
            gap = max(box[1] - region[3], region[1] - box[3], 0.0)
            if gap > _ADJACENT_GAP:
                continue
            region[0] = min(region[0], box[0])
            region[1] = min(region[1], box[1])
            region[2] = max(region[2], box[2])
            region[3] = max(region[3], box[3])
            remaining.remove(box)
            changed = True
    return tuple(region)


def select_captions(segments: list[Segment]) -> list[tuple[Segment, str, int]]:
    """Pick one caption segment per figure/table key.

    A body paragraph starting with "Fig. 8 highlights ..." is classified as
    kind "caption" by the segmenter and would steal the key from the real
    caption. Real captions punctuate the number ("FIGURE 8. Bar chart ...",
    "Figure 1: Synthetic ..."), so a candidate whose number is followed by
    "." or ":" is strong and always wins over a weak one; among candidates
    of equal strength the first in reading order wins.
    """
    chosen: dict[str, tuple[Segment, str, int, bool]] = {}
    order: list[str] = []
    for seg in segments:
        if seg.kind != "caption":
            continue
        match = _CAPTION_KEY_RE.match(seg.text)
        if match is None:
            continue
        kind = "table" if match.group(1).lower().startswith("tab") else "figure"
        number = _caption_number(match.group(2))
        key = f"{kind}-{number}"
        rest = seg.text[match.end(2):].lstrip()
        strong = bool(rest) and rest[0] in ".:"
        if key not in chosen:
            chosen[key] = (seg, kind, number, strong)
            order.append(key)
        elif strong and not chosen[key][3]:
            chosen[key] = (seg, kind, number, strong)
    return [chosen[key][:3] for key in order]


def detect_figures(
    layout: DocumentLayout, segments: list[Segment], header_band: float
) -> tuple[list[dict], dict[str, tuple[int, tuple]]]:
    """Detect figure/table regions from caption segments.

    Returns (figures, caption_boxes) where figures is the payload list and
    caption_boxes maps each key to its caption's (page, bbox) for mention
    filtering. A caption without an adjacent graphic yields no figure.
    ``header_band`` is the top/bottom page-height fraction whose graphics
    are page furniture (header rules, footer decorations), never figure
    content.
    """
    captions = select_captions(segments)

    graphics_by_page: dict[int, list[tuple]] = {}
    for page in layout.pages:
        top = page.height * header_band
        bottom = page.height * (1.0 - header_band)
        graphics_by_page[page.number] = [
            tuple(block.bbox)
            for block in page.blocks
            if block.type in ("image", "drawing")
            and block.bbox[3] > top and block.bbox[1] < bottom
        ]

    figures: list[dict] = []
    caption_boxes: dict[str, tuple[int, tuple]] = {}
    claimed_by_page: dict[int, list[tuple]] = {}
    for seg, kind, num in captions:
        key = f"{kind}-{num}"
        graphics = graphics_by_page.get(seg.page, [])
        barriers = [other.bbox for other, _, _ in captions
                    if other is not seg and other.page == seg.page]
        claimed = claimed_by_page.setdefault(seg.page, [])
        region = _grow_region(seg.bbox, graphics, "above", barriers, claimed)
        if kind == "table":
            below = _grow_region(seg.bbox, graphics, "below", barriers, claimed)
            if region is None:
                region = below
            elif below is not None:
                # Both sides have graphics: keep the nearer group.
                if (below[1] - seg.bbox[3]) < (seg.bbox[1] - region[3]):
                    region = below
        if region is None:
            continue
        claimed.append(region)
        caption_boxes[key] = (seg.page, tuple(seg.bbox))
        figures.append({
            "key": key,
            "page": seg.page,
            "bbox": [round(v, 2) for v in region],
            "caption": seg.text,
        })
    return figures, caption_boxes
