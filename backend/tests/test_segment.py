# Tests for segment.build_segments: columns, hyphens, kinds.
from __future__ import annotations

from app.pipeline.models import Block, DocumentLayout, Line, PageLayout, Span
from app.pipeline.segment import build_segments


def _line(x0, y0, x1, y1, text, size=9.0, font="Helvetica"):
    span = Span(text=text, bbox=(x0, y0, x1, y1), font=font, size=size, flags=0)
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


def test_two_columns_detected(sample_segments):
    for page in (0, 1):
        columns = {
            seg.column
            for seg in sample_segments
            if seg.page == page and seg.kind != "header_footer"
        }
        assert columns == {0, 1}, f"page {page}: expected 2 columns, got {columns}"


def test_each_column_has_three_plus_body_paragraphs(sample_segments):
    for page in (0, 1):
        for column in (0, 1):
            bodies = [
                seg
                for seg in sample_segments
                if seg.page == page and seg.column == column and seg.kind == "body"
            ]
            assert len(bodies) >= 3, f"page {page} column {column}: {len(bodies)}"


def test_hyphen_line_breaks_restored(sample_segments):
    all_text = " ".join(seg.text for seg in sample_segments)
    assert "neural machine translation have" in all_text
    assert "restores hyphenated words" in all_text
    assert "classic page segmentation heuristics" in all_text
    assert "transla- tion" not in all_text
    assert "seg- mentation" not in all_text


def test_segment_kinds(sample_segments):
    kinds = {seg.kind for seg in sample_segments}
    assert kinds == {
        "body", "heading", "caption", "header_footer",
        "formula", "author", "reference", "figure_text", "algorithm",
    }

    headings = [seg for seg in sample_segments if seg.kind == "heading"]
    heading_text = " | ".join(seg.text for seg in headings)
    assert "Introduction" in heading_text
    assert "Experimental Setup" in heading_text
    assert "Layout-Preserving" in heading_text  # title picked up as heading

    captions = [seg for seg in sample_segments if seg.kind == "caption"]
    assert len(captions) == 3  # figure caption + two table captions
    assert any(seg.text.startswith("Figure 1:") for seg in captions)
    assert any(seg.text.startswith("TABLE 1.") for seg in captions)

    header_footer = [seg for seg in sample_segments if seg.kind == "header_footer"]
    assert len(header_footer) == 4  # header + footer on both pages


def test_indented_block_does_not_bridge_columns():
    """Regression: transitive x0 chaining must not merge two real columns.

    An indented block (x0=250) nested inside the left column's x-extent used
    to chain with the right column (x0=312, gap < width*0.15), polluting the
    right column's text and making its union bbox invade the left column.
    """
    left = _block(
        [_line(54, 100 + 11 * i, 290, 109 + 11 * i, f"left line {i}")
         for i in range(5)]
    )
    indented = _block([_line(250, 122, 300, 131, "indented note")])
    right = _block(
        [_line(312, 100 + 11 * i, 558, 109 + 11 * i, f"right line {i}")
         for i in range(5)]
    )
    segments = build_segments(_page([left, indented, right]))

    right_segs = [seg for seg in segments if "right line" in seg.text]
    assert right_segs, "right column paragraph missing"
    for seg in right_segs:
        assert "indented" not in seg.text, "right column text polluted"
        assert seg.bbox[0] >= 312.0, "right column bbox invaded the left column"

    indented_segs = [seg for seg in segments if "indented" in seg.text]
    assert indented_segs, "indented block lost"
    left_columns = {seg.column for seg in segments if "left line" in seg.text}
    right_columns = {seg.column for seg in right_segs}
    assert {seg.column for seg in indented_segs} == left_columns
    assert right_columns.isdisjoint(left_columns)
    for seg in indented_segs:
        assert seg.bbox[2] <= 312.0, "indented segment bbox crossed the gutter"


def test_front_matter_bands_do_not_collapse_body_columns(sample_segments):
    """Regression: full-width front matter (title/authors/funding/abstract)
    must not make page-0 look single-column and cross-merge the two-column
    body below it (real-paper page-0 symptom)."""
    page0 = [seg for seg in sample_segments if seg.page == 0]
    # The inline "ABSTRACT" label is its own heading segment (symptom F);
    # the abstract body stays a full-width body segment.
    label = next(seg for seg in page0 if seg.text == "ABSTRACT")
    assert label.kind == "heading"
    abstract = next(seg for seg in page0
                    if seg.text.startswith("Document translation systems"))
    assert abstract.kind == "body"
    assert abstract.bbox[2] - abstract.bbox[0] > 400.0  # genuinely full-width

    # Every body segment below the INDEX TERMS band belongs to exactly one
    # column: its bbox never exceeds the column width (~ half of the page).
    terms = next(seg for seg in page0
                 if seg.text.startswith("Document structure analysis"))
    body_below = [seg for seg in page0
                  if seg.kind == "body" and seg.bbox[1] >= terms.bbox[3]]
    assert body_below, "two-column body paragraphs below the abstract missing"
    for seg in body_below:
        width = seg.bbox[2] - seg.bbox[0]
        assert width <= 260.0, (seg.id, width, seg.text[:50])

    # Left-column and right-column sentences never share a segment.
    left_marker = "treat the page as a first-class object"
    right_marker = "derives a column map"
    left_seg = next(seg for seg in page0 if left_marker in seg.text)
    right_seg = next(seg for seg in page0 if right_marker in seg.text)
    assert right_marker not in left_seg.text
    assert left_marker not in right_seg.text
    assert left_seg.bbox[2] <= 318.0, "left-column bbox crossed the gutter"
    assert right_seg.bbox[0] >= 300.0, "right-column bbox crossed the gutter"
    assert left_seg.column == 0 and right_seg.column == 1


def test_front_matter_reading_order(sample_segments):
    """Reading order: bands top-to-bottom (title, authors, funding,
    abstract), then the body band left column before the right column."""
    page0 = [seg for seg in sample_segments
             if seg.page == 0 and seg.kind != "header_footer"]

    def index_of(predicate):
        return next(i for i, seg in enumerate(page0) if predicate(seg))

    title_i = index_of(lambda s: "Layout-Preserving" in s.text)
    funding_i = index_of(lambda s: "Imaginary Research Council" in s.text)
    label_i = index_of(lambda s: s.text == "ABSTRACT")
    abstract_i = index_of(
        lambda s: s.text.startswith("Document translation systems"))
    intro_i = index_of(lambda s: "Introduction" in s.text)
    left_i = index_of(lambda s: "treat the page as a first-class object" in s.text)
    right_i = index_of(lambda s: "derives a column map" in s.text)
    assert (title_i < funding_i < label_i < abstract_i
            < intro_i <= left_i < right_i)


def test_full_width_bands_isolate_columns_unit():
    """Regression: a page mixing full-width blocks with two-column runs is
    split into vertical bands; column clustering runs per band, paragraph
    merging never crosses a band or column boundary, and reading order is
    band-major (top-to-bottom), column-minor (left-to-right)."""
    title = _block([_line(120, 70, 490, 92, "A Centered Grand Title", size=20.0)])
    note = _block(
        [_line(54, 110, 558, 120,
               "Manuscript received 1 January 2026; funded by grant 42.")]
    )
    upper_left = _block(
        [_line(54, 150 + 11 * i, 290, 159 + 11 * i, f"upper left sentence {i}")
         for i in range(4)]
    )
    upper_right = _block(
        [_line(318, 150 + 11 * i, 558, 159 + 11 * i, f"upper right sentence {i}")
         for i in range(4)]
    )
    divider = _block(
        [_line(54, 210, 558, 220,
               "A full width divider note between the column runs.")]
    )
    lower_left = _block(
        [_line(54, 240 + 11 * i, 290, 249 + 11 * i, f"lower left sentence {i}")
         for i in range(4)]
    )
    lower_right = _block(
        [_line(318, 240 + 11 * i, 558, 249 + 11 * i, f"lower right sentence {i}")
         for i in range(4)]
    )
    segments = build_segments(_page(
        [title, note, upper_left, upper_right, divider, lower_left, lower_right]
    ))
    content = [seg for seg in segments if seg.kind != "header_footer"]

    # No cross-column merging anywhere.
    for seg in content:
        assert not ("left sentence" in seg.text and "right sentence" in seg.text)
        if "left sentence" in seg.text:
            assert seg.bbox[2] <= 318.0 and seg.column == 0
        if "right sentence" in seg.text:
            assert seg.bbox[0] >= 318.0 and seg.column == 1

    # Full-width blocks are their own single-column bands.
    note_seg = next(seg for seg in content if "Manuscript received" in seg.text)
    divider_seg = next(seg for seg in content if "divider note" in seg.text)
    assert note_seg.column == 0 and divider_seg.column == 0
    assert "left sentence" not in note_seg.text
    assert "left sentence" not in divider_seg.text

    # Reading order: title, note, upper band L->R, divider, lower band L->R.
    def index_of(marker):
        return next(i for i, seg in enumerate(content) if marker in seg.text)

    assert (
        index_of("Grand Title")
        < index_of("Manuscript received")
        < index_of("upper left sentence")
        < index_of("upper right sentence")
        < index_of("divider note")
        < index_of("lower left sentence")
        < index_of("lower right sentence")
    )


def test_equation_only_lines_excluded_from_segments():
    """Display-equation lines (all-placeholder) must not become segments."""
    para1 = _block(
        [_line(54, 100, 290, 109, "first paragraph line one"),
         _line(54, 111, 290, 120, "first paragraph line two")]
    )
    equation = _block([_line(78, 140, 250, 152, "⟦EQ0⟧", size=10.0, font="CMMI10")])
    para2 = _block([_line(54, 170, 290, 179, "second paragraph line one")])
    segments = build_segments(_page([para1, equation, para2]))

    joined = " | ".join(seg.text for seg in segments)
    assert "⟦EQ0⟧" not in joined, "equation line must not enter any segment"
    assert "first paragraph" in joined
    assert "second paragraph" in joined


def test_segment_ids_and_geometry(sample_segments):
    ids = [seg.id for seg in sample_segments]
    assert len(ids) == len(set(ids))
    for seg in sample_segments:
        assert seg.id == f"p{seg.page}_s{int(seg.id.split('_s')[1])}"
        x0, y0, x1, y1 = seg.bbox
        assert x0 < x1 and y0 < y1
        assert seg.font_size > 0


def test_nature_author_list_and_byline():
    from app.pipeline.segment import _BYLINE_RE, _looks_like_author_list
    assert _looks_like_author_list(
        "Wei Xu1,14, Ana de Souza Braz4*, Li Shi5† & John Roe2")
    assert _BYLINE_RE.match("BY MONYA BAKER")
    assert not _looks_like_author_list(
        "We trained the model, tuned it, and evaluated it on held-out data")
    assert not _BYLINE_RE.match("By contrast, the second model failed to converge")
