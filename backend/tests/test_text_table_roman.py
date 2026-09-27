# Regression tests for (S) lowercase Roman-numeral enumerations "(i)/(ii)"
# and (T) text-only table exclusion (TABLE 2 without ruled lines), plus the
# Roman caption-number key ("TABLE IV" -> table-4).
from __future__ import annotations

from app.pipeline.models import Segment
from app.pipeline.regions import select_captions
from app.pipeline.segment import split_list_marker


def test_roman_enum_items_split(sample_segments):
    """(i)/(ii) items each become their own body segment with the marker."""
    first = [s for s in sample_segments if s.text.startswith("(i) ")]
    second = [s for s in sample_segments if s.text.startswith("(ii) ")]
    assert len(first) == 1, [s.text for s in first]
    assert len(second) == 1, [s.text for s in second]
    assert first[0].kind == "body"
    assert second[0].kind == "body"
    # Continuation lines stay inside their item (hanging-indent absorption).
    assert "across repeated pipeline executions." in first[0].text
    assert "page count for every tested corpus." in second[0].text
    prefix, rest = split_list_marker(first[0].text)
    assert prefix.startswith("(i)") and rest.startswith("Anchor")
    prefix, rest = split_list_marker(second[0].text)
    assert prefix.startswith("(ii)") and rest.startswith("Rendered")


def test_text_table_cells_excluded(sample_segments):
    """Text-only table cells are figure_text; caption and prose stay."""
    cells = [s for s in sample_segments
             if "0.91" in s.text or "Physics" in s.text]
    assert cells
    assert all(s.kind == "figure_text" for s in cells), \
        [(s.text, s.kind) for s in cells]
    caption = next(s for s in sample_segments
                   if s.text.startswith("TABLE 2"))
    assert caption.kind == "caption"
    # False-positive guards: surrounding prose and the References section
    # must not be swallowed by the table region.
    follow = next(s for s in sample_segments if "rank drifts" in s.text)
    assert follow.kind == "body"
    refhead = next(s for s in sample_segments if s.text == "REFERENCES")
    assert refhead.kind == "heading"
    refs = [s for s in sample_segments if s.text.startswith("[1]")]
    assert refs and refs[0].kind == "reference"


def test_roman_caption_key():
    """"TABLE IV." maps to the table-4 key like "Table 4." does."""
    seg = Segment(id="s0", page=0, column=0, bbox=(0, 0, 100, 10),
                  text="TABLE IV. RESULTS OF THE AUDIT.", kind="caption",
                  font_size=8.5)
    picked = select_captions([seg])
    assert picked == [(seg, "table", 4)]
