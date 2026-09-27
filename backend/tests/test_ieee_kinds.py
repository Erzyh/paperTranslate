# Regression tests for the IEEE-paper classification defects (A)-(D):
# (A) numbered display equations -> kind "formula", kept verbatim
# (B) author blocks -> kind "author", kept verbatim
# (C) References section -> kind "reference", kept verbatim
# (D) "VII. CONCLUSION"-style headings never merge with the next paragraph
from __future__ import annotations

import re
from pathlib import Path

import pymupdf
import pytest

from app.pipeline.models import Block, DocumentLayout, Line, PageLayout, Span
from app.pipeline.segment import UNTRANSLATED_KINDS, build_segments

MATH_FONT_AVAILABLE = Path(r"C:\Windows\Fonts\seguisym.ttf").exists()


def _line(x0, y0, x1, y1, text, size=10.0, font="Helvetica", flags=0):
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


def _layout(pages_blocks):
    return DocumentLayout(
        pages=[
            PageLayout(number=i, width=612.0, height=792.0, blocks=blocks)
            for i, blocks in enumerate(pages_blocks)
        ]
    )


@pytest.fixture(scope="module")
def stub_output_pdf(sample_pdf, tmp_path_factory) -> str:
    """Full stub pipeline run over the extended sample (module scope)."""
    from app.pipeline.runner import run_pipeline
    from app.pipeline.translate import StubTranslator

    out = tmp_path_factory.mktemp("ieee_out") / "out.pdf"
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


# ---------------------------------------------------------------- (A) formula


@pytest.mark.skipif(not MATH_FONT_AVAILABLE, reason="needs seguisym math font")
def test_a_numbered_equation_becomes_formula_segment(sample_segments):
    formulas = [seg for seg in sample_segments if seg.kind == "formula"]
    assert formulas, "numbered display equation not classified as formula"
    assert any("(1)" in seg.text for seg in formulas)
    # The equation must not leak into any translatable segment. Pages 0-1
    # only: pages 2-3 carry the deliberate "(1)/(2)/(3)" paragraph
    # enumeration (make_sample construct N) whose items are body segments.
    for seg in sample_segments:
        if seg.kind not in UNTRANSLATED_KINDS and seg.page <= 1:
            assert "(1)" not in seg.text


@pytest.mark.skipif(not MATH_FONT_AVAILABLE, reason="needs seguisym math font")
def test_a_numbered_equation_survives_in_output(stub_output_pdf):
    text = _full_text(stub_output_pdf)
    assert "(1)" in text, "equation number destroyed"
    assert "Ω" in text and "∑" in text, "equation glyphs destroyed"
    assert "⟦EQ" not in text, "literal placeholder typeset into the output"


def test_formula_line_classification_unit():
    """Mixed math-font/variable lines are formulas; prose with inline math
    is not (real-paper shapes from the IEEE sample)."""
    body = _block([
        _line(317, 381, 550, 393, "where "),
    ])
    body.lines[0].spans.append(
        Span(text="⟦EQ0⟧", bbox=(360, 381, 372, 393), font="RMTMI",
             size=10.0, flags=4))
    body.lines[0].spans.append(
        Span(text=" denotes the 5th percentile of the predicted",
             bbox=(372, 381, 550, 393), font="TimesLTStd-Roman",
             size=10.0, flags=4))
    filler = _block([
        _line(317, 100 + 12 * i, 550, 110 + 12 * i,
              f"plain body sentence number {i} keeps the mode size")
        for i in range(6)
    ])
    equation = Block(type="text", bbox=(317, 420, 545, 434), lines=[Line(
        spans=[
            Span(text="S", bbox=(317, 420, 324, 434),
                 font="TimesLTStd-Italic", size=10.0, flags=6),
            Span(text="hyb", bbox=(324, 424, 334, 434),
                 font="TimesLTStd-Roman", size=7.6, flags=4),
            Span(text="⟦EQ1⟧", bbox=(338, 420, 350, 434),
                 font="MTSYN", size=10.0, flags=4),
            Span(text=" w", bbox=(352, 420, 362, 434),
                 font="TimesLTStd-Italic", size=10.0, flags=6),
            Span(text="1", bbox=(362, 424, 366, 434),
                 font="TimesLTStd-Roman", size=7.6, flags=4),
            Span(text="(3)", bbox=(530, 420, 545, 434),
                 font="TimesLTStd-Roman", size=10.0, flags=4),
        ],
        bbox=(317, 420, 545, 434),
    )])
    segments = build_segments(_layout([[filler, body, equation]]))

    formula_segs = [seg for seg in segments if seg.kind == "formula"]
    assert len(formula_segs) == 1
    assert "(3)" in formula_segs[0].text

    prose = [seg for seg in segments if "denotes the 5th percentile" in seg.text]
    assert prose and all(seg.kind == "body" for seg in prose)


# ----------------------------------------------------------------- (B) author


def test_b_author_block_classified(sample_segments):
    authors = [seg for seg in sample_segments if seg.kind == "author"]
    assert authors, "author block not classified"
    joined = " ".join(seg.text for seg in authors)
    assert "Alice Kim" in joined
    assert "@docint.example.org" in joined
    assert all(seg.page == 0 for seg in authors)


def test_b_author_block_survives_in_output(stub_output_pdf):
    text = _full_text(stub_output_pdf)
    assert "Alice Kim, Bob Lee, and Carol Park" in text
    assert "@docint.example.org" in text


def test_b_funding_note_between_title_and_abstract_is_author(sample_segments):
    """The full-width funding/editor note sits between the title and the
    Abstract, so Route 1 marks it author (kept verbatim) even after the
    band-based column detection split it into its own band."""
    notes = [seg for seg in sample_segments
             if "Imaginary Research Council" in seg.text]
    assert notes, "funding note segment missing"
    assert all(seg.kind == "author" for seg in notes)
    assert all(seg.page == 0 for seg in notes)


def test_b_funding_note_survives_in_output(stub_output_pdf):
    text = _full_text(stub_output_pdf)
    assert "Imaginary Research Council under Grant IRC-2026-042" in text
    assert "Prof. Erasmus Placeholder" in text


def test_b_author_between_title_and_abstract_positional_unit():
    """Route 1: segments between the page-0 title and the Abstract become
    author even without affiliation cues."""
    title = _block([_line(150, 80, 460, 104, "A Grand Unified Paper Title",
                          size=21.0)])
    plain_names = _block([_line(200, 130, 410, 142, "John Doe and Jane Roe")])
    abstract = _block([
        _line(54, 170, 558, 182,
              "ABSTRACT This paper studies segmentation of documents."),
        _line(54, 182, 558, 194,
              "It reports results on a corpus of scholarly articles."),
    ])
    body = _block([
        _line(54, 220 + 12 * i, 558, 230 + 12 * i,
              f"body paragraph sentence {i} with ordinary running prose")
        for i in range(6)
    ])
    segments = build_segments(_layout([[title, plain_names, abstract, body]]))

    names = [seg for seg in segments if "John Doe" in seg.text]
    assert names and names[0].kind == "author"
    # The inline label splits off (text-pattern fallback: same font, no
    # bold); the abstract body itself stays a translatable body segment.
    labels = [seg for seg in segments if seg.text == "ABSTRACT"]
    assert labels and labels[0].kind == "heading"
    abstract_segs = [seg for seg in segments
                     if seg.text.startswith("This paper studies")]
    assert abstract_segs and abstract_segs[0].kind == "body"


def test_b_body_paragraph_mentioning_university_stays_body():
    """Conservative rule: a cue word deep in the page must not reclassify
    real body text."""
    title = _block([_line(150, 80, 460, 104, "A Grand Unified Paper Title",
                          size=21.0)])
    body = _block([
        _line(54, 500 + 12 * i, 558, 510 + 12 * i,
              "Researchers at Example University proposed this method")
        for i in range(6)
    ])
    segments = build_segments(_layout([[title, body]]))
    deep = [seg for seg in segments if "Example University" in seg.text]
    assert deep and all(seg.kind == "body" for seg in deep)


# -------------------------------------------------------------- (C) reference


def test_c_reference_segments_classified(sample_segments):
    refs = [seg for seg in sample_segments if seg.kind == "reference"]
    assert len(refs) >= 2, "reference entries not classified"
    joined = " ".join(seg.text for seg in refs)
    assert "[1]" in joined and "[2]" in joined
    assert "vol. 12" in joined

    heads = [seg for seg in sample_segments
             if seg.kind == "heading" and seg.text.strip() == "REFERENCES"]
    assert heads, "REFERENCES title must stay a (translatable) heading"


def test_c_reference_entries_survive_in_output(stub_output_pdf):
    text = _full_text(stub_output_pdf)
    assert "Layout aware" in text
    assert "vol. 12, no. 3, pp. 45-67," in text
    assert "Column detection for" in text


def test_c_everything_after_references_heading_marked_unit():
    """Reference kind propagates past the heading and across pages."""
    body = _block([
        _line(54, 100 + 12 * i, 300, 110 + 12 * i,
              f"closing discussion sentence number {i} of the paper")
        for i in range(6)
    ])
    refs_head = _block([_line(54, 200, 130, 212, "References",
                              font="Helvetica-Bold", flags=16)])
    item1 = _block([
        _line(54, 216, 300, 226, "[1] F. Last, \"Some paper title,\" Journal"),
        _line(54, 227, 300, 237, "of Examples, vol. 1, pp. 1-10, 2021."),
    ])
    next_page_item = _block([
        _line(54, 100, 300, 110, "[2] G. Author, \"Another title,\" in Proc."),
        _line(54, 111, 300, 121, "Example Conf., 2019, pp. 11-22."),
    ])
    segments = build_segments(_layout([[body, refs_head, item1],
                                       [next_page_item]]))

    closing = [seg for seg in segments if "closing discussion" in seg.text]
    assert closing and closing[0].kind == "body"
    head = [seg for seg in segments if seg.text.strip() == "References"]
    assert head and head[0].kind == "heading"
    for marker in ("[1]", "[2]"):
        found = [seg for seg in segments if seg.text.startswith(marker)]
        assert found and found[0].kind == "reference", marker


def test_c_bracket_item_shape_is_reference_anywhere_unit():
    """Auxiliary rule: '[n] ... vol./pp./year' bodies are references even
    without a References heading (e.g. continuation pages)."""
    item = _block([
        _line(54, 300, 300, 310, "[17] H. Writer, \"Standalone entry,\" IEEE"),
        _line(54, 311, 300, 321, "Trans. Tests, vol. 3, no. 2, pp. 5-9, 2018."),
    ])
    filler = _block([
        _line(54, 100 + 12 * i, 300, 110 + 12 * i,
              f"ordinary paragraph number {i} about the method")
        for i in range(6)
    ])
    segments = build_segments(_layout([[filler, item]]))
    found = [seg for seg in segments if seg.text.startswith("[17]")]
    assert found and found[0].kind == "reference"


# ---------------------------------------------------------------- (D) heading


def test_d_conclusion_heading_is_own_segment(sample_segments):
    heads = [seg for seg in sample_segments
             if seg.text.strip() == "VII. CONCLUSION"]
    assert heads, "VII. CONCLUSION must be an independent segment"
    assert heads[0].kind == "heading"

    following = [seg for seg in sample_segments
                 if "This research presented" in seg.text]
    assert following, "paragraph after the heading missing"
    for seg in following:
        assert "CONCLUSION" not in seg.text
        assert seg.kind == "body"


def test_d_caps_heading_at_body_size_split_unit():
    """An all-caps numbered heading at body size glued to the next paragraph
    (2pt gap) must still be split off — size alone must not be required."""
    filler = _block([
        _line(54, 100 + 12 * i, 300, 110 + 12 * i,
              f"introductory sentence number {i} with plain prose")
        for i in range(6)
    ])
    glued = _block([
        _line(54, 300, 160, 310, "VII. CONCLUSION"),
        _line(54, 312, 300, 322, "This work described a translation system"),
        _line(54, 324, 300, 334, "that preserves the layout of every page."),
    ])
    segments = build_segments(_layout([[filler, glued]]))

    heads = [seg for seg in segments if seg.text.strip() == "VII. CONCLUSION"]
    assert heads and heads[0].kind == "heading"
    bodies = [seg for seg in segments if "This work described" in seg.text]
    assert bodies and "CONCLUSION" not in bodies[0].text


def test_d_bold_subsection_heading_split_unit():
    """IEEE 'A. SUBSECTION' bold headings at body size split as well."""
    filler = _block([
        _line(54, 100 + 12 * i, 300, 110 + 12 * i,
              f"filler sentence number {i} to set the body font size")
        for i in range(6)
    ])
    glued = _block([
        _line(54, 300, 220, 310, "E. Composite Loss Formulation",
              font="FormataOTF-Bold", flags=20),
        _line(54, 312, 300, 322, "Training the network necessitates a dual"),
        _line(54, 324, 300, 334, "objective optimization strategy in practice."),
    ])
    segments = build_segments(_layout([[filler, glued]]))

    heads = [seg for seg in segments
             if seg.text.strip() == "E. Composite Loss Formulation"]
    assert heads and heads[0].kind == "heading"
    bodies = [seg for seg in segments if "Training the network" in seg.text]
    assert bodies and "Composite" not in bodies[0].text


# ------------------------------------------------- untranslated-kind plumbing


def test_untranslated_kinds_skipped_by_runner(sample_pdf, tmp_path):
    from app.pipeline.runner import run_pipeline
    from app.pipeline.translate import StubTranslator

    seen: dict[str, str | None] = {}
    kinds: dict[str, str] = {}

    def segment_cb(seg, translated=None, **kwargs):
        seen[seg.id] = translated
        kinds[seg.id] = seg.kind

    out = tmp_path / "skip_out.pdf"
    run_pipeline(sample_pdf, str(out), StubTranslator(), segment_cb=segment_cb)

    assert set(kinds.values()) >= {"formula", "author", "reference",
                                   "header_footer"}
    for seg_id, translated in seen.items():
        if kinds[seg_id] in UNTRANSLATED_KINDS:
            assert translated is None, f"{seg_id} ({kinds[seg_id]}) translated"
        else:
            assert translated is not None
            assert re.search(r"[가-힣]", translated)


def test_retypeset_ignores_stray_translations_for_untranslated_kinds(
        sample_pdf, sample_segments, tmp_path):
    """Defensive guard: even if a translation is supplied for an untranslated
    kind, retypeset must not redact or re-insert it."""
    from app.pipeline.retypeset import render_translated_pdf

    victims = [seg for seg in sample_segments
               if seg.kind in ("author", "reference")]
    assert victims
    translations = {seg.id: "무단 번역" for seg in victims}
    out = tmp_path / "stray_out.pdf"
    render_translated_pdf(sample_pdf, sample_segments, translations, str(out))

    text = _full_text(str(out))
    assert "무단 번역" not in text
    assert "Alice Kim, Bob Lee, and Carol Park" in text
    assert "vol. 12, no. 3, pp. 45-67," in text
