# Regression tests for symptom (F): IEEE Access inline front-matter labels
# ("ABSTRACT body ...", "INDEX TERMS terms ...") must split into an
# independent heading segment plus a body segment with disjoint bboxes, so
# the translator never paraphrases the label into a fake preamble.
from __future__ import annotations

from app.pipeline.models import Block, DocumentLayout, Line, PageLayout, Span
from app.pipeline.segment import build_segments

BOLD_FLAG = 1 << 4


def _line(x0, y0, x1, y1, text, size=10.0, font="Helvetica", flags=0):
    span = Span(text=text, bbox=(x0, y0, x1, y1), font=font, size=size,
                flags=flags)
    return Line(spans=[span], bbox=(x0, y0, x1, y1))


def _multi_span_line(spans):
    bbox = (
        min(s.bbox[0] for s in spans),
        min(s.bbox[1] for s in spans),
        max(s.bbox[2] for s in spans),
        max(s.bbox[3] for s in spans),
    )
    return Line(spans=list(spans), bbox=bbox)


def _block(lines):
    bbox = (
        min(ln.bbox[0] for ln in lines),
        min(ln.bbox[1] for ln in lines),
        max(ln.bbox[2] for ln in lines),
        max(ln.bbox[3] for ln in lines),
    )
    return Block(type="text", bbox=bbox, lines=lines)


def _layout(blocks):
    return DocumentLayout(
        pages=[PageLayout(number=0, width=612.0, height=792.0, blocks=blocks)]
    )


def _overlap_area(a, b) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return max(width, 0.0) * max(height, 0.0)


def _filler_block():
    """Six plain lines pinning the document body font size at 10pt."""
    return _block([
        _line(54, 500 + 12 * i, 300, 510 + 12 * i,
              f"plain filler sentence number {i} with ordinary prose")
        for i in range(6)
    ])


# ------------------------------------------------------------- sample-based


def test_sample_abstract_label_is_heading_segment(sample_segments):
    labels = [seg for seg in sample_segments if seg.text == "ABSTRACT"]
    assert labels, "inline ABSTRACT label must be its own segment"
    label = labels[0]
    assert label.kind == "heading"
    assert label.page == 0

    bodies = [seg for seg in sample_segments
              if seg.text.startswith("Document translation systems")]
    assert bodies, "abstract body segment missing"
    body = bodies[0]
    assert body.kind == "body"
    assert "ABSTRACT" not in body.text
    assert _overlap_area(label.bbox, body.bbox) == 0.0


def test_sample_index_terms_label_split_single_line(sample_segments):
    labels = [seg for seg in sample_segments if seg.text == "INDEX TERMS"]
    assert labels, "inline INDEX TERMS label must be its own segment"
    label = labels[0]
    assert label.kind == "heading"

    bodies = [seg for seg in sample_segments
              if seg.text.startswith("Document structure analysis")]
    assert bodies, "index terms body segment missing"
    body = bodies[0]
    assert body.kind == "body"
    assert "INDEX TERMS" not in body.text
    # Single-line split: the label rect ends before the body rect starts on
    # the same visual line.
    assert label.bbox[2] <= body.bbox[0]
    assert _overlap_area(label.bbox, body.bbox) == 0.0


def test_sample_label_bboxes_stay_inside_paragraph(sample_segments):
    """Label + body rects of the abstract cover the paragraph area without
    leaking outside it (redaction stays within the original glyph area)."""
    label = next(seg for seg in sample_segments if seg.text == "ABSTRACT")
    body = next(seg for seg in sample_segments
                if seg.text.startswith("Document translation systems"))
    assert label.bbox[1] < body.bbox[1], "label sits above the body rect"
    assert label.bbox[3] <= body.bbox[1] + 0.1, "rects must not interleave"


# -------------------------------------------------------------- unit: spans


def test_bold_label_multiline_split_unit():
    """Span-based path: bold caps label glued to a two-line paragraph."""
    first = _multi_span_line([
        Span(text="ABSTRACT", bbox=(54, 200, 110, 212),
             font="Helvetica-Bold", size=10.0, flags=BOLD_FLAG),
        Span(text=" Neural systems translate documents with",
             bbox=(114, 200, 558, 212), font="Helvetica", size=10.0, flags=0),
    ])
    second = _line(54, 213, 558, 225,
                   "high fidelity across many domains and layouts.")
    segments = build_segments(_layout([_filler_block(),
                                       _block([first, second])]))

    labels = [seg for seg in segments if seg.text == "ABSTRACT"]
    assert labels and labels[0].kind == "heading"
    label = labels[0]
    # Multi-line policy: the label rect spans the whole first line so its
    # redaction erases the first-line remainder glyphs too.
    assert label.bbox == (54, 200, 558, 212)

    bodies = [seg for seg in segments
              if seg.text.startswith("Neural systems translate")]
    assert bodies and bodies[0].kind == "body"
    body = bodies[0]
    assert "high fidelity" in body.text  # remainder + second line joined
    assert "ABSTRACT" not in body.text
    assert body.bbox[1] >= label.bbox[3]
    assert _overlap_area(label.bbox, body.bbox) == 0.0


def test_bold_two_word_label_single_line_split_unit():
    """Span-based path: two-word bold label on a single line."""
    line = _multi_span_line([
        Span(text="INDEX TERMS", bbox=(54, 300, 120, 312),
             font="Helvetica-Bold", size=10.0, flags=BOLD_FLAG),
        Span(text=" Layout analysis, document translation.",
             bbox=(124, 300, 500, 312), font="Helvetica", size=10.0, flags=0),
    ])
    segments = build_segments(_layout([_filler_block(), _block([line])]))

    labels = [seg for seg in segments if seg.text == "INDEX TERMS"]
    assert labels and labels[0].kind == "heading"
    bodies = [seg for seg in segments
              if seg.text.startswith("Layout analysis")]
    assert bodies and bodies[0].kind == "body"
    assert labels[0].bbox[2] <= bodies[0].bbox[0]
    assert _overlap_area(labels[0].bbox, bodies[0].bbox) == 0.0


def test_generic_bold_caps_label_split_unit():
    """The span-based rule is generic: any bold caps run splits, not only
    the known label words."""
    line = _multi_span_line([
        Span(text="HIGHLIGHTS", bbox=(54, 300, 118, 312),
             font="Helvetica-Bold", size=10.0, flags=BOLD_FLAG),
        Span(text=" The pipeline preserves two-column layouts.",
             bbox=(122, 300, 500, 312), font="Helvetica", size=10.0, flags=0),
    ])
    segments = build_segments(_layout([_filler_block(), _block([line])]))
    labels = [seg for seg in segments if seg.text == "HIGHLIGHTS"]
    assert labels and labels[0].kind == "heading"


# ----------------------------------------------------------- unit: fallback


def test_text_pattern_fallback_without_font_signal_unit():
    """Fallback path: label and body share one span (same font, no bold);
    the known label words still split, with a proportional bbox cut."""
    para = _block([
        _line(54, 200, 558, 212,
              "ABSTRACT This study examines layout preserving translation"),
        _line(54, 213, 558, 225,
              "and reports results on synthetic and real documents."),
    ])
    segments = build_segments(_layout([_filler_block(), para]))

    labels = [seg for seg in segments if seg.text == "ABSTRACT"]
    assert labels and labels[0].kind == "heading"
    bodies = [seg for seg in segments
              if seg.text.startswith("This study examines")]
    assert bodies and bodies[0].kind == "body"
    assert "ABSTRACT" not in bodies[0].text
    assert _overlap_area(labels[0].bbox, bodies[0].bbox) == 0.0


def test_keywords_with_colon_fallback_unit():
    """"KEYWORDS: a, b." splits and the body loses the separator colon."""
    line = _line(54, 300, 500, 312,
                 "KEYWORDS: deep learning, document analysis.")
    segments = build_segments(_layout([_filler_block(), _block([line])]))

    labels = [seg for seg in segments if seg.text == "KEYWORDS"]
    assert labels and labels[0].kind == "heading"
    bodies = [seg for seg in segments
              if seg.text.startswith("deep learning")]
    assert bodies and bodies[0].kind == "body"
    assert labels[0].bbox[2] <= bodies[0].bbox[0]


# ---------------------------------------------------------- unit: negatives


def test_plain_caps_acronym_does_not_split_unit():
    """A non-bold caps acronym in the body font must not be mistaken for a
    label (no style signal, not a known label word)."""
    line = _multi_span_line([
        Span(text="MODELS", bbox=(54, 300, 95, 312),
             font="Helvetica", size=10.0, flags=0),
        Span(text=" of this family translate well in practice.",
             bbox=(99, 300, 500, 312), font="Helvetica", size=10.0, flags=0),
    ])
    segments = build_segments(_layout([_filler_block(), _block([line])]))
    matching = [seg for seg in segments if "MODELS" in seg.text]
    assert len(matching) == 1
    assert matching[0].kind == "body"
    assert matching[0].text.startswith("MODELS of this family")


def test_short_bold_acronym_does_not_split_unit():
    """Bold caps runs shorter than four letters ("CNN") never split."""
    line = _multi_span_line([
        Span(text="CNN", bbox=(54, 300, 78, 312),
             font="Helvetica-Bold", size=10.0, flags=BOLD_FLAG),
        Span(text=" backbones dominate the benchmark suites.",
             bbox=(82, 300, 500, 312), font="Helvetica", size=10.0, flags=0),
    ])
    segments = build_segments(_layout([_filler_block(), _block([line])]))
    matching = [seg for seg in segments if "CNN" in seg.text]
    assert len(matching) == 1
    assert matching[0].kind == "body"


def test_bold_table_caption_does_not_split_unit():
    """"TABLE 1. ..." caption shapes keep the caption rule, no label split."""
    line = _multi_span_line([
        Span(text="TABLE", bbox=(54, 300, 90, 312),
             font="Helvetica-Bold", size=10.0, flags=BOLD_FLAG),
        Span(text=" 1. Overview of the experimental results.",
             bbox=(94, 300, 480, 312), font="Helvetica", size=10.0, flags=0),
    ])
    segments = build_segments(_layout([_filler_block(), _block([line])]))
    matching = [seg for seg in segments if "TABLE" in seg.text]
    assert len(matching) == 1
    assert matching[0].kind == "caption"


def test_bold_caps_biography_name_does_not_split_unit():
    """IEEE Access biography paragraphs start with a bold caps name that
    continues the sentence ("NAME received the B.S. degree ..."); the
    lowercase continuation must block the generic split."""
    first = _multi_span_line([
        Span(text="HYEONWOO GIL", bbox=(54, 300, 130, 312),
             font="Helvetica-Bold", size=10.0, flags=BOLD_FLAG),
        Span(text=" received the B.S. degree in computer science",
             bbox=(134, 300, 540, 312), font="Helvetica", size=10.0, flags=0),
    ])
    second = _line(54, 313, 540, 325,
                   "from an imaginary university, in 2020.")
    segments = build_segments(_layout([_filler_block(),
                                       _block([first, second])]))
    matching = [seg for seg in segments if "HYEONWOO GIL" in seg.text]
    assert len(matching) == 1
    assert matching[0].text.startswith("HYEONWOO GIL received")


def test_lone_caps_line_without_body_does_not_split_unit():
    """A lone all-caps label line with no body text stays one segment."""
    line = _line(54, 300, 110, 312, "ABSTRACT",
                 font="Helvetica-Bold", flags=BOLD_FLAG)
    segments = build_segments(_layout([_filler_block(), _block([line])]))
    matching = [seg for seg in segments if seg.text == "ABSTRACT"]
    assert len(matching) == 1
