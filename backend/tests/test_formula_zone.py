# Tests for the formula-zone banding (display equations typeset with
# regular italic/roman faces, sub/superscript fragment lines, right-aligned
# "(n)" numbers) and for paragraph-style "(n)" enumerations whose
# continuation lines are flush left (no hanging indent):
# (1) the whole italic equation becomes ONE kind="formula" segment and no
#     variable/subscript fragment leaks into a body segment,
# (2) neighbouring prose paragraphs keep their full text (zone guards),
# (3) paragraph-style enum items absorb their flush continuation lines and
#     stop at the next marker-shaped line / paragraph gap,
# (4) the untranslated equation survives verbatim in the stub output while
#     the enum items are translated with their markers preserved,
# (5) the font-agnostic zone rules (sample symptom (R), real ACL "(5)" PCC
#     equation shape): an equation typeset ONLY with text faces — hyphen
#     compounds ("L-score"), big parens, a fraction and a right-aligned
#     "(n)" number — still becomes one formula segment; the "(n)" seed
#     aligns against the PROSE right edge (stray wide lines must not
#     disable it) and centered fragment clusters seed without any number.
from __future__ import annotations

import re

import pymupdf
import pytest

from app.pipeline.models import Block, DocumentLayout, Line, PageLayout, Span
from app.pipeline.segment import build_segments


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
def zone_output(sample_pdf, tmp_path_factory):
    """One stub pipeline run shared by the output-side tests."""
    from app.pipeline.runner import run_pipeline
    from app.pipeline.translate import StubTranslator

    out = tmp_path_factory.mktemp("zone") / "zone_out.pdf"
    run_pipeline(sample_pdf, str(out), StubTranslator())
    return str(out)


def _full_text(path: str) -> str:
    doc = pymupdf.open(path)
    try:
        return "\n".join(
            doc.load_page(i).get_text() for i in range(doc.page_count)
        )
    finally:
        doc.close()


# ------------------------------------------------- (M) italic equation zone


def test_italic_equation_zone_is_one_formula_segment(sample_segments):
    zones = [seg for seg in sample_segments
             if seg.page == 0 and seg.kind == "formula" and "(9)" in seg.text]
    assert len(zones) == 1, "italic display equation not zoned as formula"
    text = zones[0].text
    # Main line, number, numerator and every sub/superscript fragment line
    # belong to the single zone segment.
    for piece in ("MAE = y(i) yref(i)", "Nhit", "offset", "i=1"):
        assert piece in text, piece


def test_italic_equation_fragments_never_leak_into_body(sample_segments):
    bodies = [seg for seg in sample_segments
              if seg.page == 0 and seg.kind == "body"]
    joined = " | ".join(seg.text for seg in bodies)
    for residue in ("yref", "Nhit", "i=1", "(9)"):
        assert residue not in joined, f"formula residue in body: {residue!r}"


def test_zone_neighbour_paragraph_keeps_full_text(sample_segments):
    para = next(seg for seg in sample_segments
                if "A rendering report records" in seg.text)
    assert para.kind == "body"
    # The zone directly below must not eat the paragraph's closing words.
    assert "fragile regions early in the process." in para.text


def test_italic_equation_survives_untranslated_in_output(zone_output):
    text = _full_text(zone_output)
    assert "MAE = y(i) yref(i)" in text, "italic equation glyphs destroyed"
    assert "(9)" in text, "equation number destroyed"


def test_italic_zone_unit():
    """A display equation with NO math-font span (italic variables, tiny
    sub/superscript lines, right-aligned number) forms one formula zone."""
    para = _block([
        _line(54, 100 + 12 * i, 290, 110 + 12 * i,
              f"leading prose sentence number {i} of the page")
        for i in range(4)
    ])
    eq = _block([
        _line(150, 160, 165, 166, "Nhit", size=6.5, font="Times-Italic"),
        _line(140, 165, 145, 174, "1", size=9.0, font="Times-Roman"),
        _line(90, 172, 210, 182, "E = y(i) yref(i)", size=9.0,
              font="Times-Italic"),
        _line(270, 172, 290, 182, "(7)", size=9.0, font="Times-Roman"),
        _line(120, 180, 140, 187, "offset", size=6.5, font="Times-Italic"),
    ])
    after = _block([
        _line(54, 210 + 12 * i, 290, 220 + 12 * i,
              f"closing prose sentence number {i} of the page")
        for i in range(4)
    ])
    segments = build_segments(_page([para, eq, after]))

    formulas = [seg for seg in segments if seg.kind == "formula"]
    assert len(formulas) == 1
    for piece in ("Nhit", "1", "y(i)", "(7)", "offset"):
        assert piece in formulas[0].text, piece
    lead = next(seg for seg in segments if "leading prose" in seg.text)
    assert lead.kind == "body"
    assert "y(i)" not in lead.text and "Nhit" not in lead.text
    closing = next(seg for seg in segments if "closing prose" in seg.text)
    assert closing.kind == "body"


def test_zone_never_absorbs_adjacent_prose_line_unit():
    """A prose sentence line touching the zone vertically stays body."""
    filler = _block([
        _line(54, 100 + 12 * i, 290, 110 + 12 * i,
              f"plain filler sentence number {i} sets the body size")
        for i in range(4)
    ])
    eq_line = Line(
        spans=[
            Span(text="⟦EQ0⟧", bbox=(90, 150, 130, 162), font="CMMI10",
                 size=9.0, flags=0),
            Span(text=" q(i)", bbox=(130, 150, 160, 162),
                 font="Times-Italic", size=9.0, flags=0),
        ],
        bbox=(90, 150, 160, 162),
    )
    number = _line(270, 150, 290, 162, "(2)", size=9.0, font="Times-Roman")
    eq = Block(type="text", bbox=(90, 150, 290, 162),
               lines=[eq_line, number])
    tail = _block([
        _line(54, 163, 290, 173,
              "where the mean of the sample denotes the level"),
    ])
    segments = build_segments(_page([filler, eq, tail]))

    formulas = [seg for seg in segments if seg.kind == "formula"]
    assert len(formulas) == 1
    assert "q(i)" in formulas[0].text and "(2)" in formulas[0].text
    prose = next(seg for seg in segments if "denotes the level" in seg.text)
    assert prose.kind == "body", "adjacent prose line swallowed by the zone"


def test_flush_math_wrap_line_stays_body_unit():
    """A flush-left prose wrap line carrying one inline math span (real
    shape: "evaluation (⟦EQn⟧ 45).") must not seed a zone."""
    wrap = Line(
        spans=[
            Span(text="evaluation (", bbox=(54, 124, 110, 134),
                 font="Helvetica", size=9.0, flags=0),
            Span(text="⟦EQ0⟧", bbox=(110, 124, 130, 134), font="CMMI10",
                 size=9.0, flags=0),
            Span(text=" 45).", bbox=(130, 124, 155, 134),
                 font="Helvetica", size=9.0, flags=0),
        ],
        bbox=(54, 124, 155, 134),
    )
    para = Block(type="text", bbox=(54, 100, 290, 134), lines=[
        _line(54, 100, 290, 110, "the ablation study evaluates four distinct"),
        _line(54, 112, 290, 122, "variants against the proxy tiers used for"),
        wrap,
    ])
    filler = _block([
        _line(54, 200 + 12 * i, 290, 210 + 12 * i,
              f"plain body sentence number {i} for the size mode")
        for i in range(4)
    ])
    segments = build_segments(_page([para, filler]))

    seg = next(seg for seg in segments if "ablation study" in seg.text)
    assert seg.kind == "body"
    assert "evaluation (" in seg.text and "45)." in seg.text
    assert not any(s.kind == "formula" for s in segments)


# ------------------------------------ (R) font-agnostic ACL-style equation


def test_acl_equation_zone_is_one_formula_segment(sample_segments):
    """The text-face-only fraction equation (hyphen compounds, big parens,
    denominator, right-aligned number below) is ONE formula segment."""
    zones = [seg for seg in sample_segments
             if seg.page == 2 and seg.kind == "formula" and "(8)" in seg.text]
    assert len(zones) == 1, "ACL-style text-face equation not zoned"
    text = zones[0].text
    for piece in ("PCC = min", "L-score + M-score", "2", "(", ")"):
        assert piece in text, piece


def test_acl_equation_fragments_never_leak_into_body(sample_segments):
    others = [seg for seg in sample_segments
              if seg.page == 2 and seg.kind != "formula"]
    joined = " | ".join(seg.text for seg in others)
    for residue in ("-score", "= min", "(8)"):
        assert residue not in joined, f"equation residue leaked: {residue!r}"


def test_acl_equation_survives_untranslated_in_output(zone_output):
    text = _full_text(zone_output)
    assert "PCC = min" in text, "ACL equation main fragment destroyed"
    assert "L-score + M-score" in text, "ACL equation numerator destroyed"


def _right_column_block():
    """Right-column prose block: makes the left blocks form a two-column
    band (as in a real paper) instead of degenerating into per-block
    full-width bands, so the left column keeps its prose flow."""
    return _block([
        _line(318, 100 + 12 * i, 558, 110 + 12 * i,
              f"right column filler sentence number {i} of the page")
        for i in range(8)
    ])


def test_number_seed_uses_prose_right_edge_unit():
    """A stray wide line (page number, overfull row) inflating the column
    extent must not disable the right-aligned "(n)" seed: alignment is
    measured against the PROSE right edge. The equation line sits left of
    the centered band on purpose, so only the "(n)" seed can open the
    zone."""
    para = _block([
        _line(54, 100 + 12 * i, 290, 110 + 12 * i,
              f"leading prose sentence number {i} of the page")
        for i in range(4)
    ])
    eq = _block([
        _line(80, 160, 160, 172, "E(q) = r + s", font="Times-Italic"),
        _line(272, 160, 290, 172, "(4)", font="Times-Roman"),
    ])
    stray = _block([_line(260, 700, 340, 710, "28946")])
    after = _block([
        _line(54, 200 + 12 * i, 290, 210 + 12 * i,
              f"closing prose sentence number {i} of the page")
        for i in range(4)
    ])
    segments = build_segments(
        _page([para, eq, stray, after, _right_column_block()]))

    formulas = [seg for seg in segments if seg.kind == "formula"]
    assert len(formulas) == 1, "(n) seed lost to the inflated column edge"
    assert "E(q)" in formulas[0].text and "(4)" in formulas[0].text
    joined = " | ".join(s.text for s in segments if s.kind != "formula")
    assert "E(q)" not in joined and "(4)" not in joined


def test_centered_cluster_seeds_zone_unit():
    """A centered fragment cluster (>= 15% margin per side vs the prose
    extent, short, no sentence end) with a math symbol seeds a zone even
    without any math font, italic face or "(n)" number."""
    para = _block([
        _line(54, 100 + 12 * i, 290, 110 + 12 * i,
              f"leading prose sentence number {i} of the page")
        for i in range(4)
    ])
    eq = _block([
        _line(100, 160, 190, 170, "L-score + M-score"),
        _line(140, 172, 148, 182, "2"),
    ])
    after = _block([
        _line(54, 210 + 12 * i, 290, 220 + 12 * i,
              f"closing prose sentence number {i} of the page")
        for i in range(4)
    ])
    segments = build_segments(_page([para, eq, after, _right_column_block()]))

    formulas = [seg for seg in segments if seg.kind == "formula"]
    assert len(formulas) == 1, "centered cluster did not seed a zone"
    assert "L-score + M-score" in formulas[0].text
    assert "2" in formulas[0].text
    joined = " | ".join(s.text for s in segments if s.kind != "formula")
    assert "-score" not in joined


def test_centered_prose_title_never_seeds_unit():
    """A centered short line WITHOUT math evidence (subtitle shape) must
    not become a formula zone."""
    para = _block([
        _line(54, 100 + 12 * i, 290, 110 + 12 * i,
              f"leading prose sentence number {i} of the page")
        for i in range(4)
    ])
    title = _block([_line(130, 160, 214, 170, "A Short Motto")])
    segments = build_segments(_page([para, title, _right_column_block()]))
    assert not any(seg.kind == "formula" for seg in segments)


def test_hyphen_compound_math_row_absorbed_unit():
    """Real ACL PCC shape: a flush-left row mixing masked math spans with
    text-face "-score" compounds joins the zone opened by the "(n)" line
    below (hyphen-attached words are exempt from the long-word guard)."""
    para = _block([
        _line(54, 100 + 12 * i, 290, 110 + 12 * i,
              f"leading prose sentence number {i} of the page")
        for i in range(4)
    ])
    row = Line(
        spans=[
            Span(text="PCC(", bbox=(54, 150, 75, 162), font="Helvetica",
                 size=9.0, flags=0),
            Span(text="⟦EQ0⟧", bbox=(75, 150, 84, 162), font="CMMI10",
                 size=9.0, flags=0),
            Span(text=") = min ", bbox=(84, 150, 120, 162),
                 font="Helvetica", size=9.0, flags=0),
            Span(text="⟦EQ1⟧", bbox=(120, 150, 128, 162), font="CMMI10",
                 size=9.0, flags=0),
            Span(text="-score + ", bbox=(128, 150, 166, 162),
                 font="Helvetica", size=9.0, flags=0),
            Span(text="⟦EQ2⟧", bbox=(166, 150, 176, 162), font="CMMI10",
                 size=9.0, flags=0),
            Span(text="-score", bbox=(176, 150, 202, 162),
                 font="Helvetica", size=9.0, flags=0),
        ],
        bbox=(54, 150, 202, 162),
    )
    denom = _line(150, 160, 158, 170, "2", font="Times-Roman")
    number = _line(272, 166, 290, 178, "(5)", font="Times-Roman")
    eq = Block(type="text", bbox=(54, 150, 290, 178),
               lines=[row, denom, number])
    after = _block([
        _line(54, 210 + 12 * i, 290, 220 + 12 * i,
              f"closing prose sentence number {i} of the page")
        for i in range(4)
    ])
    segments = build_segments(_page([para, eq, after]))

    formulas = [seg for seg in segments if seg.kind == "formula"]
    assert len(formulas) == 1
    text = formulas[0].text
    for piece in ("PCC(", "-score", "(5)", "2"):
        assert piece in text, piece
    joined = " | ".join(s.text for s in segments if s.kind != "formula")
    assert "-score" not in joined and "min" not in joined


# ------------------------------------- (L) paragraph-style enum continuation


def test_paragraph_enum_items_absorb_flush_continuations(sample_segments):
    one = [seg for seg in sample_segments
           if seg.page == 0 and seg.text.startswith("(5) Proxy")]
    two = [seg for seg in sample_segments
           if seg.page == 0 and seg.text.startswith("(6) Sample")]
    assert len(one) == 1 and len(two) == 1
    assert one[0].kind == "body" and two[0].kind == "body"
    # The flush continuation line belongs to its item: whole sentences.
    assert "scope of the reported study conclusions." in one[0].text
    assert "validity of the reported correlations." in two[0].text
    assert "(6)" not in one[0].text
    # The flush paragraph after the enumeration never glues onto item (2).
    follow = next(seg for seg in sample_segments
                  if "remaining sections quantify" in seg.text)
    assert follow.id != two[0].id
    assert "remaining sections" not in two[0].text


def test_paragraph_enum_translated_with_markers_in_output(zone_output):
    text = _full_text(zone_output)
    for sentinel in ("Proxy label dependence limits",
                     "Sample size constrains the external"):
        assert sentinel not in text, f"enum item survived: {sentinel!r}"
    # Markers stay at the front of the translated items.
    doc = pymupdf.open(zone_output)
    try:
        page_text = doc.load_page(0).get_text()
    finally:
        doc.close()
    for marker in ("(5)", "(6)"):
        assert any(
            ln.strip().startswith(marker) and re.search(r"[가-힣]", ln)
            for ln in page_text.splitlines()
        ), marker


def test_paragraph_enum_unit():
    """Marker line indented past its flush continuation lines: the item
    absorbs them until the next fired marker / gap break."""
    intro = _block([
        _line(54, 100, 290, 109, "the items below constrain the study scope"),
    ])
    items = _block([
        _line(64, 121, 280, 130, "(1) Proxy dependence narrows all findings"),
        _line(54, 132, 285, 141, "and it is reported here in full detail."),
        _line(64, 143, 282, 152, "(2) Sample size limits the external scope"),
        _line(54, 154, 288, 163, "of every correlation reported in the paper."),
    ])
    follow = _block([
        _line(54, 178, 290, 187, "the closing paragraph stands on its own"),
    ])
    segments = build_segments(_page([intro, items, follow]))

    one = next(seg for seg in segments if seg.text.startswith("(1)"))
    two = next(seg for seg in segments if seg.text.startswith("(2)"))
    assert "reported here in full detail." in one.text
    assert "(2)" not in one.text
    assert "of every correlation reported in the paper." in two.text
    closing = next(seg for seg in segments if "closing paragraph" in seg.text)
    assert closing.id != two.id


def test_paragraph_enum_stops_at_unfired_marker_line_unit():
    """An unfired marker-shaped line ("(7) ..." with no aligned sibling in
    the window) still ends the running item's absorption."""
    lines = [
        _line(64, 100, 280, 109, "(5) Single game context limits the study"),
        _line(54, 111, 285, 120, "because the target game has not shipped."),
        _line(64, 122, 282, 131, "(6) Industrial impact claims were removed"),
    ]
    y = 133
    for i in range(9):  # push "(7)" out of the marker-neighbour window
        lines.append(_line(54, y, 288, y + 9,
                           f"flush continuation sentence number {i} here."))
        y += 11
    lines.append(_line(64, y, 280, y + 9,
                       "(7) Transfer of the offset terms is limited"))
    lines.append(_line(54, y + 11, 288, y + 20,
                       "and only the first term moved across domains."))
    segments = build_segments(_page([_block(lines)]))

    six = next(seg for seg in segments if seg.text.startswith("(6)"))
    assert "flush continuation sentence number 8" in six.text
    assert "(7)" not in six.text, "unfired marker line absorbed into item"
    seven = next(seg for seg in segments if seg.text.startswith("(7)"))
    assert "moved across domains." in seven.text
