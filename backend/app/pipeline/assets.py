# Figure/table mention search and reference parsing for the assets API
# (GET /api/documents/{id}/assets).
#
# Everything here is derived from the PDFs alone (analyze_pdf + search_for),
# never from DB state, so assets can be computed for a document that has not
# been translated yet. The API layer caches the resulting payload. The
# caption-adjacency region logic lives in regions.py, shared with the
# segmenter's "figure_text" classification.
from __future__ import annotations

import re

import pymupdf

from .extract import analyze_pdf
from .models import Segment
from .regions import detect_figures, intersects as _intersects
from .segment import HEADER_FOOTER_BAND, build_segments

# Bibliography entry marker "[12]".
_REF_MARKER_RE = re.compile(r"\[(\d+)\]")
# Best-effort quoted-title spans inside a bibliography entry, tried in
# order: TeX-style double apostrophes (IEEE Access: ‘‘title,’’) first —
# their inner text may contain a single apostrophe — then straight or curly
# double quotes.
_TITLE_QUOTE_RES = (
    re.compile(r"‘‘\s*(.{4,}?)\s*’’"),
    re.compile(r"[\"“]\s*([^\"“”]{4,}?)\s*[\"”]"),
)

# Two mention rects sharing an origin within these tolerances are the same
# text hit (used to prune "Fig. 1" matching inside "Fig. 12").
_PRUNE_X_TOL = 1.5
_PRUNE_Y_TOL = 2.0


def _parse_references(segments: list[Segment]) -> dict[str, dict]:
    """Split concatenated reference segments into numbered entries.

    Entries are cut at "[n]" markers that continue the 1,2,3,... chain, so a
    stray bracketed number inside an entry body cannot open a bogus entry.
    Title extraction is best-effort: the first double-quoted span wins.
    """
    text = " ".join(seg.text for seg in segments
                    if seg.kind == "reference").strip()
    if not text:
        return {}
    chain: list[tuple[int, int]] = []  # (number, start offset)
    expected = 1
    for match in _REF_MARKER_RE.finditer(text):
        if int(match.group(1)) == expected:
            chain.append((expected, match.start()))
            expected += 1
    references: dict[str, dict] = {}
    for i, (num, start) in enumerate(chain):
        end = chain[i + 1][1] if i + 1 < len(chain) else len(text)
        entry = text[start:end].strip()
        title = None
        for pattern in _TITLE_QUOTE_RES:
            title_match = pattern.search(entry)
            if title_match is not None:
                title = title_match.group(1).strip().rstrip(",.;").strip() or None
                break
        references[str(num)] = {"entry": entry, "title": title}
    return references


def _mention_needles(kind: str, num: int, side: str) -> list[str]:
    """Search strings for one figure/table key on one PDF side.

    search_for is case-insensitive, so "Figure 3" also hits "FIGURE 3".
    """
    if side == "original":
        if kind == "figure":
            return [f"Figure {num}", f"Fig. {num}"]
        return [f"Table {num}"]
    if kind == "figure":
        return [f"그림 {num}"]  # "그림 N"
    return [f"표 {num}"]  # "표 N"


def _prune_prefix_hits(mentions: list[dict]) -> list[dict]:
    """Drop hits of a shorter number inside a longer one ("Fig. 1" in
    "Fig. 12"): both needles match at the same text origin, so the shorter
    rect starting at the same point as a longer rect is a false positive."""
    pruned: list[dict] = []
    for m in mentions:
        shadowed = False
        for other in mentions:
            if other is m or other["side"] != m["side"] \
                    or other["kind"] != m["kind"] or other["page"] != m["page"]:
                continue
            if (abs(other["bbox"][0] - m["bbox"][0]) <= _PRUNE_X_TOL
                    and abs(other["bbox"][1] - m["bbox"][1]) <= _PRUNE_Y_TOL
                    and other["bbox"][2] > m["bbox"][2] + 0.5):
                shadowed = True
                break
        if not shadowed:
            pruned.append(m)
    return pruned


def _collect_mentions(
    pdf_path: str,
    side: str,
    figures: list[dict],
    caption_boxes: dict[str, tuple[int, tuple]],
    reference_boxes: list[tuple[int, tuple]],
    reference_numbers: list[int],
) -> list[dict]:
    """Search one PDF for figure/table/citation mentions.

    Hits inside any caption region are dropped (the caption itself is not a
    mention of its figure); citation hits inside bibliography segments are
    dropped (the entry marker "[n]" is not a citation). Caption/reference
    bboxes come from the source layout; retypesetting keeps segments in
    place, so the same boxes apply to the translated side.
    """
    all_caption_boxes = list(caption_boxes.values())
    mentions: list[dict] = []
    doc = pymupdf.open(pdf_path)
    try:
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            for fig in figures:
                kind, num = fig["key"].rsplit("-", 1)
                for needle in _mention_needles(kind, int(num), side):
                    for rect in page.search_for(needle):
                        box = (rect.x0, rect.y0, rect.x1, rect.y1)
                        if any(cpage == pno and _intersects(box, cbox)
                               for cpage, cbox in all_caption_boxes):
                            continue
                        mentions.append({
                            "key": fig["key"],
                            "kind": kind,
                            "side": side,
                            "page": pno,
                            "bbox": [round(v, 2) for v in box],
                            "ref": None,
                        })
            for num in reference_numbers:
                for rect in page.search_for(f"[{num}]"):
                    box = (rect.x0, rect.y0, rect.x1, rect.y1)
                    if any(rpage == pno and _intersects(box, rbox)
                           for rpage, rbox in reference_boxes):
                        continue
                    mentions.append({
                        "key": f"cite-{num}",
                        "kind": "cite",
                        "side": side,
                        "page": pno,
                        "bbox": [round(v, 2) for v in box],
                        "ref": num,
                    })
    finally:
        doc.close()
    return _prune_prefix_hits(mentions)


def compute_assets(src_path: str, translated_path: str | None) -> dict:
    """Compute the full assets payload for one document.

    translated_path is the output PDF when the document is done, else None
    (translated-side mentions stay empty per the API contract).
    """
    layout = analyze_pdf(src_path)
    segments = build_segments(layout)
    figures, caption_boxes = detect_figures(layout, segments,
                                            HEADER_FOOTER_BAND)
    references = _parse_references(segments)
    reference_boxes = [(seg.page, tuple(seg.bbox)) for seg in segments
                       if seg.kind == "reference"]
    reference_numbers = sorted(int(n) for n in references)
    mentions = _collect_mentions(src_path, "original", figures, caption_boxes,
                                 reference_boxes, reference_numbers)
    if translated_path is not None:
        mentions += _collect_mentions(translated_path, "translated", figures,
                                      caption_boxes, reference_boxes,
                                      reference_numbers)
    return {"figures": figures, "mentions": mentions, "references": references}
