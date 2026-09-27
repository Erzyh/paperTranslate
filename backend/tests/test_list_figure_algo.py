# Tests for the four segment rules added for real-paper fidelity:
# (1) bullet-list items become individual segments with their marker kept,
# (2) "1)/2)"-style enumerations behave the same,
# (3) text inside figure/table regions is kind "figure_text" (untranslated),
# (4) Algorithm/pseudocode blocks are kind "algorithm" (untranslated),
# plus false-positive guards (lone numbered lines, years, prose mentions).
from __future__ import annotations

import re
from pathlib import Path

import pymupdf
import pytest

from app.pipeline.models import Block, DocumentLayout, Line, PageLayout, Span
from app.pipeline.segment import build_segments, split_list_marker

LIST_FONT_AVAILABLE = Path(r"C:\Windows\Fonts\arial.ttf").exists()
BULLET = "•" if LIST_FONT_AVAILABLE else "*"


def _line(x0, y0, x1, y1, text, size=9.0, font="Helvetica", flags=0):
    span = Span(text=text, bbox=(x0, y0, x1, y1), font=font, size=size,
                flags=flags)
    return Line(spans=[span], bbox=(x0, y0, x1, y1))


def _block(lines):
    bbox = (
        min(ln.bbox[0] for ln in lines),
        min(ln.bbox[1] for ln in lines),
        max(ln.bbox[2] for ln in lines),
        max(ln.bbox[3] for ln in lines),
    )
    return Block(type="text", bbox=bbox, lines=lines)


def _page(blocks):
    return DocumentLayout(
        pages=[PageLayout(number=0, width=612.0, height=792.0, blocks=blocks)]
    )


@pytest.fixture(scope="module")
def rules_output(sample_pdf, tmp_path_factory):
    """One stub pipeline run shared by the output-side tests; also captures
    the per-segment translations via the segment callback."""
    from app.pipeline.runner import run_pipeline
    from app.pipeline.translate import StubTranslator

    seen: list[tuple] = []

    def segment_cb(seg, translated=None, **kwargs):
        seen.append((seg, translated))

    out = tmp_path_factory.mktemp("rules") / "rules_out.pdf"
    run_pipeline(sample_pdf, str(out), StubTranslator(), segment_cb=segment_cb)
    return str(out), seen


def _page_lines(path: str, pno: int) -> list[tuple[float, float, str]]:
    """(y0, x0, text) of every extraction line on a page of the PDF."""
    doc = pymupdf.open(path)
    try:
        lines = []
        for block in doc.load_page(pno).get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(s.get("text", "") for s in line.get("spans", []))
                bbox = line["bbox"]
                lines.append((bbox[1], bbox[0], text))
        return lines
    finally:
        doc.close()


def _full_text(path: str) -> str:
    doc = pymupdf.open(path)
    try:
        return "\n".join(
            doc.load_page(i).get_text() for i in range(doc.page_count)
        )
    finally:
        doc.close()


# ------------------------------------------------------- (1)/(2) list splits


def test_bullet_items_are_individual_segments(sample_segments):
    items = [seg for seg in sample_segments
             if seg.page == 1 and seg.text.lstrip().startswith(BULLET)]
    assert len(items) == 3, "each bullet item must be its own segment"
    assert all(seg.kind == "body" for seg in items)
    texts = [seg.text for seg in items]
    assert any("Delta timing" in t for t in texts)
    assert any("Spatial lanes" in t for t in texts)
    assert any("Note topology" in t for t in texts)
    # Items never bleed into each other.
    for text in texts:
        assert sum(m in text for m in
                   ("Delta timing", "Spatial lanes", "Note topology")) == 1
    # The indented continuation line stays inside its item.
    delta = next(t for t in texts if "Delta timing" in t)
    assert "playback lane" in delta


def test_enum_items_are_individual_segments(sample_segments):
    one = [seg for seg in sample_segments
           if seg.page == 0 and seg.text.startswith("1) Extract")]
    two = [seg for seg in sample_segments
           if seg.page == 0 and seg.text.startswith("2) Group")]
    assert len(one) == 1 and len(two) == 1
    assert one[0] is not two[0]
    assert one[0].kind == "body" and two[0].kind == "body"
    assert "Group the spans" not in one[0].text
    # Hanging-indent continuation lines stay inside their items.
    assert "font and position" in one[0].text
    assert "segments for translation" in two[0].text


def test_markers_reprefixed_deterministically(rules_output):
    """The runner strips the marker prefix before translation and re-prefixes
    it verbatim, so every stored translation keeps its marker up front."""
    _out, seen = rules_output
    bullets = [(seg, tr) for seg, tr in seen
               if tr is not None and seg.text.lstrip().startswith(BULLET)]
    assert len(bullets) == 3
    for seg, translated in bullets:
        prefix, _rest = split_list_marker(seg.text)
        assert prefix and translated.startswith(prefix)
        assert re.search(r"[가-힣]", translated)
    for marker in ("1)", "2)"):
        found = [(seg, tr) for seg, tr in seen
                 if tr is not None and seg.text.startswith(marker)]
        assert found
        seg, translated = found[0]
        prefix, _rest = split_list_marker(seg.text)
        assert prefix.startswith(marker) and translated.startswith(prefix)


def test_markers_survive_at_line_start_in_output(rules_output):
    out_path, _seen = rules_output
    lines = _page_lines(out_path, 1)
    marker_lines = [(y, x, t) for y, x, t in lines
                    if t.strip().startswith(BULLET)
                    and re.search(r"[가-힣]", t)]
    assert len(marker_lines) >= 3, "translated bullet lines missing"
    # y-separation: the three items sit on distinct lines.
    ys = sorted(y for y, _x, _t in marker_lines)
    assert all(b - a > 2.0 for a, b in zip(ys, ys[1:]))
    # Enumeration markers stay at the line start on page 0.
    p0 = _page_lines(out_path, 0)
    for prefix in ("1)", "2)"):
        assert any(t.strip().startswith(prefix) and re.search(r"[가-힣]", t)
                   for _y, _x, t in p0), prefix


def test_list_item_english_translated_away(rules_output):
    out_path, _seen = rules_output
    text = _full_text(out_path)
    for sentinel in ("Delta timing preserves", "Spatial lanes anchor",
                     "Note topology survives", "Extract every span",
                     "Group the spans into"):
        assert sentinel not in text, f"list item survived: {sentinel!r}"


def test_split_list_marker_unit():
    # The prefix keeps the original separator: prefix + rest == text.
    assert split_list_marker("• item text") == ("• ", "item text")
    assert split_list_marker("•\xa0item text") == ("•\xa0", "item text")
    assert split_list_marker("* item") == ("* ", "item")
    assert split_list_marker("1) item") == ("1) ", "item")
    assert split_list_marker("(3) item") == ("(3) ", "item")
    assert split_list_marker("a. item") == ("a. ", "item")
    assert split_list_marker("⟦EQ7⟧ Delta Time: x") == ("⟦EQ7⟧ ", "Delta Time: x")
    # Non-markers pass through untouched.
    for text in ("plain sentence", "(2026) saw progress", "e.g. something",
                 "123) too many digits", "⟦EQ7⟧⟦EQ8⟧ glued", "no. 3, pp. 4"):
        assert split_list_marker(text) == ("", text)


def test_lone_numbered_line_is_not_split_unit():
    """A single "2)" wrap line without an aligned sibling stays inside its
    paragraph (adjacency requirement for enumeration markers)."""
    para = _block([
        _line(54, 100, 290, 109, "The best configuration turned out to be"),
        _line(54, 111, 290, 120, "2) with a larger beam, which we adopt"),
        _line(54, 122, 290, 131, "for the remaining experiments."),
    ])
    segments = build_segments(_page([para]))
    matching = [seg for seg in segments if "best configuration" in seg.text]
    assert len(matching) == 1
    assert "larger beam" in matching[0].text
    assert "remaining experiments" in matching[0].text


def test_year_and_wide_number_lines_are_not_markers_unit():
    """4-digit years and long numbers never match the marker shape."""
    para = _block([
        _line(54, 100, 290, 109, "(2026) marked a turning point for the"),
        _line(54, 111, 290, 120, "2020. field, as several surveys note."),
    ])
    segments = build_segments(_page([para]))
    assert len([seg for seg in segments if "turning point" in seg.text]) == 1
    assert all("surveys note" in seg.text for seg in segments
               if "turning point" in seg.text)


def test_lone_dash_wrap_line_is_not_split_unit():
    """An en-dash wrap line (continuation of a parenthetical dash) must not
    open a list item; real dash lists have adjacent dash siblings."""
    para = _block([
        _line(54, 100, 290, 109, "such sweeping claims of superiority"),
        _line(54, 111, 290, 120, "– are not supported by the present data"),
        _line(54, 122, 290, 131, "and are deliberately avoided here."),
    ])
    segments = build_segments(_page([para]))
    matching = [seg for seg in segments if "sweeping claims" in seg.text]
    assert len(matching) == 1
    assert "not supported" in matching[0].text
    assert "deliberately avoided" in matching[0].text


def test_adjacent_dash_lines_split_unit():
    para = _block([
        _line(54, 100, 290, 109, "– first dashed point of the outline"),
        _line(54, 111, 290, 120, "– second dashed point of the outline"),
    ])
    segments = build_segments(_page([para]))
    firsts = [seg for seg in segments if "first dashed" in seg.text]
    seconds = [seg for seg in segments if "second dashed" in seg.text]
    assert len(firsts) == 1 and len(seconds) == 1
    assert firsts[0] is not seconds[0]


def test_adjacent_numbered_lines_split_unit():
    para = _block([
        _line(54, 100, 290, 109, "1) First enumerated point of the method."),
        _line(54, 111, 290, 120, "2) Second enumerated point of the method."),
    ])
    segments = build_segments(_page([para]))
    firsts = [seg for seg in segments if "First enumerated" in seg.text]
    seconds = [seg for seg in segments if "Second enumerated" in seg.text]
    assert len(firsts) == 1 and len(seconds) == 1
    assert firsts[0] is not seconds[0]


def test_masked_bullet_glyph_items_split_unit():
    """Real-paper shape: the bullet is a math-font glyph masked to ⟦EQn⟧.
    A narrow leading math span with an aligned sibling starts an item; the
    indented continuation line joins it and the next flush-left paragraph
    does not."""
    def item(y, token, text):
        marker = Span(text=token, bbox=(46.1, y + 1.0, 49.9, y + 9.0),
                      font="MTSYN", size=9.0, flags=0)
        rest = Span(text=f" {text}", bbox=(49.9, y, 290, y + 10.0),
                    font="TimesLTStd-Roman", size=9.0, flags=0)
        return Line(spans=[marker, rest], bbox=(46.1, y, 290, y + 10.0))

    # A right column keeps the left blocks below the full-width threshold,
    # so intro/items/outro land in ONE column band and paragraph merging
    # really runs across the block boundaries.
    right = _block([
        _line(318, 100 + 11 * i, 558, 109 + 11 * i,
              f"right column filler sentence number {i}")
        for i in range(8)
    ])
    intro = _block([
        _line(36.2, 100, 290, 109, "The features are encoded as follows"),
    ])
    # 11pt line steps throughout: without the marker rule everything below
    # would merge into a single paragraph. The outro line is flush left
    # (x0=36.2 < marker x0) and only 5pt below the last item line, so only
    # the continuation-indent rule keeps it out of the second item.
    items = _block([
        item(130, "⟦EQ0⟧", "Delta Time: The absolute NoteTime is"),
        _line(55.9, 141, 290, 150, "converted to a relative offset value."),
        item(152, "⟦EQ1⟧", "Spatial Lane: The NoteLane index is"),
        _line(55.9, 163, 290, 172, "normalized by a fixed lane count."),
    ])
    outro = _block([
        _line(36.2, 177, 290, 186, "This tensor is the input to the model."),
    ])
    segments = build_segments(_page([right, intro, items, outro]))

    delta = [seg for seg in segments if "Delta Time" in seg.text]
    lane = [seg for seg in segments if "Spatial Lane" in seg.text]
    assert len(delta) == 1 and len(lane) == 1
    assert delta[0] is not lane[0]
    assert delta[0].text.startswith("⟦EQ0⟧")
    assert "relative offset" in delta[0].text  # continuation absorbed
    assert "Spatial Lane" not in delta[0].text
    outro_segs = [seg for seg in segments if "input to the model" in seg.text]
    assert outro_segs and "Spatial Lane" not in outro_segs[0].text


def test_wide_leading_math_span_is_not_a_marker_unit():
    """A genuinely wide leading formula span must not start a list item."""
    def formula_first(y, width):
        eq = Span(text="⟦EQ0⟧", bbox=(54, y, 54 + width, y + 10.0),
                  font="MTSYN", size=9.0, flags=0)
        rest = Span(text=" denotes the loss used in stage one",
                    bbox=(54 + width, y, 290, y + 10.0),
                    font="TimesLTStd-Roman", size=9.0, flags=0)
        return Line(spans=[eq, rest], bbox=(54, y, 290, y + 10.0))

    para = Block(type="text", bbox=(54, 100, 290, 121), lines=[
        formula_first(100, 30.0),
        formula_first(111, 30.0),
    ])
    filler = _block([
        _line(54, 200 + 12 * i, 290, 209 + 12 * i,
              f"plain body sentence number {i} for the size mode")
        for i in range(4)
    ])
    segments = build_segments(_page([para, filler]))
    matching = [seg for seg in segments if "denotes the loss" in seg.text]
    assert len(matching) == 1, "wide math spans must not split as list items"


# ------------------------------------------------------------ (3) figure_text


def test_axis_labels_classified_figure_text(sample_segments):
    labels = {seg.text: seg for seg in sample_segments
              if seg.text in ("Accuracy", "Epochs")}
    assert set(labels) == {"Accuracy", "Epochs"}
    for seg in labels.values():
        assert seg.kind == "figure_text"
        assert seg.page == 0


def test_figure_text_stays_english_in_output(rules_output):
    out_path, seen = rules_output
    text = _full_text(out_path)
    assert "Accuracy" in text
    assert "Epochs" in text
    skipped = {seg.text for seg, tr in seen if tr is None}
    assert {"Accuracy", "Epochs"} <= skipped


def test_figure_text_unit():
    """Text whose center lies inside a caption-adjacent graphic region is
    figure_text; the caption itself stays a caption."""
    image = Block(type="image", bbox=(318, 100, 558, 220), lines=[])
    label = _block([_line(340, 150, 380, 158, "F-score", size=7.5)])
    caption = _block([
        _line(318, 232, 540, 241, "Figure 2: Latency curves per domain."),
    ])
    body = _block([
        _line(54, 100 + 12 * i, 290, 110 + 12 * i,
              f"ordinary body sentence number {i} of the page")
        for i in range(5)
    ])
    segments = build_segments(_page([image, label, caption, body]))

    lab = next(seg for seg in segments if seg.text == "F-score")
    assert lab.kind == "figure_text"
    cap = next(seg for seg in segments if seg.text.startswith("Figure 2:"))
    assert cap.kind == "caption"
    assert all(seg.kind == "body" for seg in segments
               if "ordinary body" in seg.text)


# ------------------------------------------------------------- (4) algorithm


def test_algorithm_block_classified(sample_segments):
    algos = [seg for seg in sample_segments if seg.kind == "algorithm"]
    assert algos, "algorithm block not classified"
    joined = " ".join(seg.text for seg in algos)
    assert "Greedy Column Packing" in joined
    assert "end for" in joined
    # The surrounding sections stay translatable.
    concl = next(seg for seg in sample_segments
                 if seg.text.strip() == "VII. CONCLUSION")
    assert concl.kind == "heading"


def test_algorithm_stays_english_in_output(rules_output):
    out_path, _seen = rules_output
    text = _full_text(out_path)
    assert "Algorithm 1: Greedy Column Packing" in text
    assert "end for" in text


def test_algorithm_box_unit():
    """Every body segment whose center lies inside the ruled box holding an
    "Algorithm N" line becomes kind "algorithm"."""
    box = Block(type="drawing", bbox=(50, 292, 310, 352), lines=[])
    content = _block([
        _line(54, 300, 250, 309, "Algorithm 4 Fast Column Pass"),
        _line(54, 311, 250, 320, "1: scan the blocks of the page"),
        _line(54, 322, 250, 331, "2: assign each block to a lane"),
    ])
    after = _block([
        _line(54, 380 + 12 * i, 290, 389 + 12 * i,
              f"discussion sentence number {i} after the box")
        for i in range(4)
    ])
    segments = build_segments(_page([box, content, after]))

    algo = [seg for seg in segments if "Fast Column Pass" in seg.text]
    assert algo and all(seg.kind == "algorithm" for seg in algo)
    assert all(seg.kind == "algorithm" for seg in segments
               if "assign each block" in seg.text)
    assert all(seg.kind == "body" for seg in segments
               if "discussion sentence" in seg.text)


def test_algorithm_heuristic_without_box_unit():
    """Boxless algorithm: the header plus indented/numbered lines form the
    block; the next flush-left paragraph ends it."""
    filler = _block([
        _line(54, 100 + 12 * i, 290, 109 + 12 * i,
              f"lead-in sentence number {i} for the body size")
        for i in range(4)
    ])
    header = _block([
        _line(54, 200, 260, 209, "Algorithm 2: Tier Quantization"),
    ])
    steps = _block([
        _line(62, 224, 280, 233, "1: compute the fused score"),
        _line(62, 248, 280, 257, "2: clamp the score into tiers"),
    ])
    outro = _block([
        _line(54, 290, 290, 299, "The resulting tiers drive the pipeline."),
    ])
    segments = build_segments(_page([filler, header, steps, outro]))

    head = next(seg for seg in segments if "Tier Quantization" in seg.text)
    assert head.kind == "algorithm"
    for marker in ("fused score", "clamp the score"):
        found = next(seg for seg in segments if marker in seg.text)
        assert found.kind == "algorithm", marker
    outro_seg = next(seg for seg in segments if "resulting tiers" in seg.text)
    assert outro_seg.kind == "body"


def test_algorithm_prose_mention_stays_body_unit():
    """A body paragraph merely mentioning "Algorithm 1 ..." (no ruled box,
    no punctuated title, no pseudocode cues) must stay translatable."""
    filler = _block([
        _line(54, 100 + 12 * i, 290, 109 + 12 * i,
              f"context sentence number {i} for the body size")
        for i in range(4)
    ])
    mention = _block([
        _line(54, 200, 290, 209, "Algorithm 1 summarizes the whole design"),
        _line(54, 212, 290, 221, "and clarifies the flow for each corpus."),
    ])
    following = _block([
        _line(62, 236, 280, 245, "An indented remark follows the mention."),
    ])
    segments = build_segments(_page([filler, mention, following]))

    seg = next(seg for seg in segments if "summarizes the whole" in seg.text)
    assert seg.kind == "body"
    remark = next(seg for seg in segments if "indented remark" in seg.text)
    assert remark.kind == "body"
