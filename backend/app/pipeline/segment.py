# Column detection, paragraph segmentation and segment-kind classification.
#
# Column handling is band-based: pages that mix full-width front matter
# (title, authors, funding note, abstract) with a two-column body are first
# split into vertical bands, and x0 column clustering runs per band. This
# prevents the page-level "one column" misdetection that used to cross-merge
# left/right column lines of the body into a single full-width paragraph.
from __future__ import annotations

import re
from collections import defaultdict

from .extract import _is_math_font
from .models import DocumentLayout, Line, PageLayout, Region, Segment, Span
from .regions import detect_figures, h_overlap, select_captions

HEADER_FOOTER_BAND = 0.05  # top/bottom page-height fraction
# Left/right page-width fraction treated as the outer margin. A line that
# starts inside the right strip (or ends inside the left one) is margin
# furniture: vertical download stamps ("Downloaded from science.org"),
# whose upright CJK glyphs survive the rotated-line filter in extract.py.
# No body line starts beyond 92% of the page width.
SIDE_MARGIN_BAND = 0.08
COLUMN_GAP_RATIO = 0.15  # x0 gap larger than width*ratio splits columns
PARA_GAP_FACTOR = 0.8  # y gap below font_size*factor merges lines
HEADING_SIZE_DELTA = 1.5  # pt above body size that marks a heading
FULL_WIDTH_RATIO = 0.6  # block wider than content-width*ratio is full-width
CENTER_OVERHANG_RATIO = 0.15  # min per-side overhang across the center line

# Segment kinds that must never be translated nor redacted: their original
# glyphs stay in the output PDF untouched (page furniture, display formulas,
# author blocks, bibliography entries, figure/table-internal text and
# algorithm/pseudocode blocks).
UNTRANSLATED_KINDS = frozenset({
    "header_footer", "formula", "author", "reference",
    "figure_text", "algorithm",
})

_CAPTION_RE = re.compile(
    r"^\s*(?:fig(?:ure)?\.?|table)\s*(?:\d|[IVXLCDM]+\b)", re.IGNORECASE
)
# A line made only of equation placeholders (plus whitespace) is a display
# equation: it must stay untouched in the source PDF (DESIGN 1.3-4), so it is
# excluded from segmentation and therefore never redacted or translated.
_EQ_ONLY_RE = re.compile(r"^(?:\s|⟦EQ\d+⟧)+$")
_EQ_TOKEN_RE = re.compile(r"⟦EQ\d+⟧")

# (D) numbered-heading prefix: Roman numeral, decimal number (possibly
# dotted) or a single capital letter followed by "."/")" — IEEE style
# "VII. CONCLUSION", "2.3 Results", "A. MOTIVATION".
_HEAD_PREFIX_RE = re.compile(
    r"^\s*(?:(?:[IVXLCDM]+|\d+(?:\.\d+)*|[A-Z])[.)]|(?:[IVXLCDM]+|\d+(?:\.\d+)*))\s+(?=\S)"
)
# (C) references-section title line, optional numbering prefix.
_REFS_TITLE_RE = re.compile(
    r"^\s*(?:(?:[IVXLCDM]+|\d+(?:\.\d+)*|[A-Z])[.)]?\s+)?"
    r"(?:references|bibliography|참고문헌)\s*:?\s*$",
    re.IGNORECASE,
)
# (C) auxiliary bibliography-entry shape: "[7] ..." plus vol./pp./no./year.
_REF_ITEM_HEAD_RE = re.compile(r"^\s*\[\d+\]\s")
_REF_ITEM_BODY_RE = re.compile(r"\bvol\.|\bpp\.|\bno\.\s*\d|\b(?:19|20)\d{2}\b")
# (A) display-equation number: "... (3)" tail or a standalone "(3)" line.
_EQ_NUMBER_TAIL_RE = re.compile(r"\(\d+\)\s*$")
_EQ_NUMBER_ONLY_RE = re.compile(r"^\s*\(\d+\)\s*$")
# (B) author-block cues (affiliations, e-mails, IEEE membership notes).
_AUTHOR_CUE_RE = re.compile(
    r"@|\buniversit|\binstitut|\bdepartment\b|\blaborator|\bschool of\b|"
    r"\bcollege of\b|\bieee\s+(?:member|senior member|student member|fellow)\b",
    re.IGNORECASE,
)
_ABSTRACT_RE = re.compile(r"^\s*abstract\b", re.IGNORECASE)
# (B') Nature/Science-style front matter carries none of the cues above:
# the author list is bare names with superscript affiliation numbers, the
# article metadata sits in a side column, and affiliations are a small-type
# footnote at the bottom of page 0. These shapes are recognised directly.
#
# Superscript affiliation marks glued to a name: "Xu1,14", "Braz4*", "Shi5†".
_AFFIL_MARK_RE = re.compile(r"(?<=[A-Za-z.])[\d*†‡§¶#✉]+(?:,[\d*†‡§¶#✉]+)*")
# Lower-case name particles that may sit inside a person's name.
_NAME_PARTICLES = frozenset({
    "de", "van", "der", "von", "da", "di", "la", "le", "du", "del", "dos",
    "bin", "al",
})
# Article metadata block (DOI / received / accepted / published dates).
_ARTICLE_META_RE = re.compile(
    r"https?://doi\.org/|\bReceived:\s|\bAccepted:\s|\bPublished online:",
    re.IGNORECASE,
)
# Magazine byline: "BY MONYA BAKER", "By Jane Doe and John Roe".
_BYLINE_RE = re.compile(
    r"^\s*by\s+[A-Za-z][A-Za-z.'’-]*(?:\s+(?:and\s+)?[A-Za-z][A-Za-z.'’-]*){0,5}\s*$",
    re.IGNORECASE,
)
# An affiliation footnote names several institutions at once.
_AFFIL_FOOTNOTE_MIN_CUES = 2


def _looks_like_author_list(text: str) -> bool:
    """True for a bare list of person names ("A B1, C D2 & E F3").

    Every comma/"&"/"and"-separated part must look like a name: 2-5 words,
    each capitalised (initials allowed) or a lower-case particle, no digits
    once affiliation marks are removed. At least three names are required so
    a short phrase with two capitalised words never qualifies.
    """
    cleaned = _AFFIL_MARK_RE.sub("", text)
    parts = [p.strip() for p in re.split(r",|&|\band\b", cleaned) if p.strip()]
    if len(parts) < 3:
        return False
    named = 0
    for part in parts:
        words = part.split()
        if not 2 <= len(words) <= 5 or any(ch.isdigit() for ch in part):
            continue
        if all(w[0].isupper() or w.lower() in _NAME_PARTICLES for w in words):
            named += 1
    return named / len(parts) >= 0.8
# (F) IEEE Access glues all-caps front-matter labels to the paragraph they
# announce ("ABSTRACT Document translation ...", "INDEX TERMS Terms ...").
# Such a label is split off as an independent heading segment so the
# translator never sees "label + body" as one text. Shape of a label run:
# at most two all-caps words, letters only.
_INLINE_LABEL_SHAPE_RE = re.compile(r"^[A-Z]+(?: [A-Z]+)?$")
# Text-pattern fallback when font style carries no signal (label not bold
# and same font as the body): only well-known front-matter labels split.
_INLINE_LABEL_FALLBACK_RE = re.compile(r"^(ABSTRACT|INDEX TERMS|KEYWORDS)(?=[\s:])")
_INLINE_LABEL_MIN_LETTERS = 4  # rejects stray short acronyms ("CNN", "A")
# Separators between an inline label and its body ("KEYWORDS: a, b").
_INLINE_LABEL_SEPARATORS = " \t:;–—"
# Theorem-environment head: a line opening with "Theorem 1.", "Definition 2
# (Transition).", "Proof." etc. forces a new paragraph so the statement never
# glues onto the preceding prose paragraph. The trailing "."/":" requirement
# rejects ordinary words like "Proofread"; mid-sentence mentions ("proof of
# Theorem 1") never match because the pattern is anchored at the line start.
_THEOREM_HEAD_RE = re.compile(
    r"^(?:Theorem|Lemma|Definition|Corollary|Proposition|Assumption|Remark"
    r"|Proof)(?: \d+)?(?: \([^)]{1,40}\))?[.:]"
)
# Fraction of the page height below which author-cue segments may live.
_AUTHOR_TOP_BAND = 0.45
# Minimum size delta (pt) above body text for the page-0 title segment.
_TITLE_SIZE_DELTA = 2.0

# --- list-item markers -------------------------------------------------
# A line-leading marker forces a new segment so every list item is
# translated on its own and keeps its marker + line break (the marker is
# stripped before translation and re-prefixed deterministically by the
# runner; see split_list_marker). Bullet characters (•/◦/▪/*) fire by
# themselves; the en dash "–" (which also opens ordinary wrap lines after a
# parenthetical dash), enumeration markers "1)" / "(1)" / "a." and masked
# math-font bullet glyphs (⟦EQn⟧, e.g. a "•" from MTSYN) additionally need
# a nearby sibling line with the same marker family at the same left edge,
# so a lone "1) ..." or "– ..." line stays inside its paragraph.
_MARKER_TEXT_RE = re.compile(
    r"^\s*(?P<marker>[•◦–▪*]"
    r"|\(?(?:viii|vii|iii|vi|iv|ix|ii|i|v|x)[.)]"
    r"|\(?\d{1,2}[.)]|\(?[a-z][.)])\s+(?=\S)"
)
# Letters forming a lowercase Roman-numeral enumeration marker ("(i)",
# "(ii)", "iv."). Single letters i/v/x are classified Roman as well so a
# lone "(i)" item fires as soon as an "(ii)" sibling sits in the window.
_ROMAN_MARKER_CHARS = frozenset("ivx")
_EQ_MARKER_RE = re.compile(r"^\s*(?P<marker>⟦EQ\d+⟧)\s+(?=\S)")
_SOLO_BULLET_CHARS = "•◦▪*"
_DASH_CHARS = "–"
# A masked marker glyph is a tiny math-font span; anything wider is a real
# leading inline formula, not a bullet.
_EQ_MARKER_MAX_WIDTH_EM = 0.75
# Enumeration adjacency: sibling marker lines must sit within this many
# lines of each other in the same column and share the left edge.
_MARKER_NEIGHBOR_WINDOW = 8
_MARKER_X_TOL = 3.0
# Continuation lines of an item are indented past the marker line's edge.
_ITEM_INDENT_MIN = 1.0


def split_list_marker(text: str) -> tuple[str, str]:
    """Split a leading list-marker prefix off a segment text.

    Returns (prefix, rest) with prefix + rest == text; prefix is "" when the
    text does not start with a marker shape. The prefix keeps the marker's
    original separator whitespace, so re-prefixing it verbatim after
    translation reproduces the source exactly on translator fallback. Used
    by the runner to keep markers out of the translator's hands.
    """
    for pattern in (_MARKER_TEXT_RE, _EQ_MARKER_RE):
        match = pattern.match(text)
        if match is not None:
            return text[:match.end()], text[match.end():]
    return "", text


def _line_marker_family(line: Line) -> str | None:
    """Marker family: "bullet"|"dash"|"number"|"alpha"|"eqmark"|None."""
    text = _line_text(line)
    match = _MARKER_TEXT_RE.match(text)
    if match is not None:
        marker = match.group("marker")
        if marker[0] in _SOLO_BULLET_CHARS:
            return "bullet"
        if marker[0] in _DASH_CHARS:
            return "dash"
        if any(ch.isdigit() for ch in marker):
            return "number"
        letters = [ch for ch in marker if ch.isalpha()]
        if letters and all(ch in _ROMAN_MARKER_CHARS for ch in letters):
            return "roman"
        return "alpha"
    match = _EQ_MARKER_RE.match(text)
    if match is None:
        return None
    first = next((s for s in line.spans if s.text.strip()), None)
    if first is None or not _is_math_font(first.font):
        return None
    width = first.bbox[2] - first.bbox[0]
    if width > _EQ_MARKER_MAX_WIDTH_EM * _line_size(line):
        return None  # leading inline formula, not a masked bullet glyph
    return "eqmark"


def _fired_markers(
    lines: list[Line], forced: frozenset[int] | set[int] = frozenset()
) -> list[bool]:
    """Per-line flag: does this line start a list item?

    Literal bullet characters (•/◦/▪/*) are unambiguous and fire alone.
    Dash markers, enumeration markers and masked bullet glyphs only fire
    when another line with the same family and the same left edge sits
    within _MARKER_NEIGHBOR_WINDOW lines, which keeps wrap lines like
    "2) is the best..." or "– are not supported..." inside their paragraph.
    ``forced`` holds id()s of lines fired by the document-wide enumeration
    sequence pass (_sequence_marker_ids): those fire regardless of the
    neighbor window, so "(1) ... (7) ..." items whose paragraphs are longer
    than the window still split.
    """
    families = [_line_marker_family(line) for line in lines]
    fired: list[bool] = []
    for i, family in enumerate(families):
        if family is None:
            fired.append(False)
        elif family == "bullet" or id(lines[i]) in forced:
            fired.append(True)
        else:
            lo = max(0, i - _MARKER_NEIGHBOR_WINDOW)
            hi = min(len(families), i + _MARKER_NEIGHBOR_WINDOW + 1)
            fired.append(any(
                j != i and families[j] == family
                and abs(lines[j].bbox[0] - lines[i].bbox[0]) <= _MARKER_X_TOL
                for j in range(lo, hi)
            ))
    return fired


# --- paragraph-leading "(n)" enumeration sequences ---------------------
# LIMITATIONS-style paragraph enumerations put "(1) Label. ..." at the head
# of consecutive paragraphs that are far longer than the ±8-line neighbor
# window, and the sequence may continue in the next column or on the next
# page. Candidates are paragraph-leading marker lines: the line opens with
# "(n)" (1-2 digits — four-digit years like "(2026)" never match) followed
# by whitespace and a capital, and the previous line of the column flow
# ends a sentence (or the line opens the column). Candidates whose numbers
# form a consecutive run (n, n+1, ...) of length >= 2 in document reading
# order all fire as list markers, with no line-distance limit.
_PARA_ENUM_RE = re.compile(r"^\s*\((\d{1,2})\)\s+(?=[A-Z])")
# Roman analogue "(i) Label. ..." for lowercase Roman-numeral enumerations;
# processed as its own run stream so "(2)" never continues "(i)".
_PARA_ROMAN_RE = re.compile(
    r"^\s*\((viii|vii|iii|vi|iv|ix|ii|i|v|x)\)\s+(?=[A-Z])"
)
_ROMAN_ORDINALS = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6,
                   "vii": 7, "viii": 8, "ix": 9, "x": 10}
_PARA_END_CHARS = ".!?:"
_PARA_END_TRAILERS = "'\"’”»)]"


def _ends_paragraph(text: str) -> bool:
    """True when the line text ends a sentence (candidate pre-condition)."""
    tail = text.rstrip()
    while tail and tail[-1] in _PARA_END_TRAILERS:
        tail = tail[:-1].rstrip()
    return bool(tail) and tail[-1] in _PARA_END_CHARS


def _sequence_marker_ids(
    page_flows: list[list[tuple[int, int, list[Line]]]],
) -> set[int]:
    """id()s of paragraph-leading "(n)" lines in consecutive-number runs.

    ``page_flows`` holds one flow per page: (band, column, sorted lines)
    triples in reading order (see _page_flow). Runs may span columns and
    pages; a number that does not continue the previous run starts a new
    one, and only runs with at least two members fire.
    """
    # Decimal "(n)" and Roman "(i)" candidates form separate run streams.
    streams: dict[str, list[tuple[int, int]]] = {"number": [], "roman": []}
    for flow in page_flows:
        for _band, _column, lines in flow:
            for i, line in enumerate(lines):
                text = _line_text(line)
                match = _PARA_ENUM_RE.match(text)
                if match is not None:
                    number = int(match.group(1))
                    stream = "number"
                else:
                    match = _PARA_ROMAN_RE.match(text)
                    if match is None:
                        continue
                    number = _ROMAN_ORDINALS[match.group(1)]
                    stream = "roman"
                if i > 0 and not _ends_paragraph(_line_text(lines[i - 1])):
                    continue
                streams[stream].append((number, id(line)))
    fired: set[int] = set()
    for candidates in streams.values():
        run: list[tuple[int, int]] = []
        for number, line_id in candidates:
            if run and number == run[-1][0] + 1:
                run.append((number, line_id))
                continue
            if len(run) >= 2:
                fired.update(line_id for _n, line_id in run)
            run = [(number, line_id)]
        if len(run) >= 2:
            fired.update(line_id for _n, line_id in run)
    return fired


# --- visual-row reassembly ---------------------------------------------
# Inline math splits one visual text row into several extraction lines
# with ~1pt baseline offsets ("dation is restricted to a Likert pilot (N"
# + "= 5 annotators,"), which used to interleave as independent lines and
# scramble the paragraph text. Lines chained by strong vertical overlap
# form one visual row; a row whose left edge sits at (or a first-line
# indent away from) the column's PROSE left edge belongs to the prose flow
# and is re-assembled in x order into ONE logical line before paragraph
# merging. Display-equation rows are indented further into the column —
# and a column (band) without any prose line at all is skipped entirely —
# so equation fragment lines stay intact for the formula-zone pass.
_ROW_OVERLAP_RATIO = 0.5
_ROW_MAX_INDENT_RATIO = 0.08  # of the column width ...
_ROW_MIN_INDENT_PT = 12.0  # ... with a floor for narrow columns


def _join_row(group: list[Line]) -> Line:
    """Concatenate a visual row's fragment lines (x order) into one Line.

    A single-space span is synthesized between fragments whose boundary
    carries no whitespace, so "...pilot (N" + "= 5 annotators," reads
    "...pilot (N = 5 annotators,". Fragment spans are reused as-is.
    """
    ordered = sorted(group, key=lambda ln: ln.bbox[0])
    spans: list[Span] = []
    for line in ordered:
        if spans and not spans[-1].text.endswith((" ", "\t")) \
                and not line.spans[0].text.startswith((" ", "\t")):
            prev = spans[-1]
            spans.append(Span(
                text=" ",
                bbox=(prev.bbox[2], line.bbox[1], line.bbox[0], line.bbox[3]),
                font=prev.font, size=prev.size, flags=prev.flags,
            ))
        spans.extend(line.spans)
    return Line(spans=spans, bbox=_union_bbox([ln.bbox for ln in ordered]))


def _merge_visual_rows(
    lines: list[Line], body_size: float, regions: list[Region] = (),
) -> list[Line]:
    """Re-assemble prose rows split into fragment lines (see block comment).

    ``lines`` must be sorted by (y0, x0); the result preserves that order.
    Lines of different layout-model regions never join one row (a pull
    quote beside a column line is not the rest of that line).
    """
    if len(lines) < 2:
        return lines
    prose_edges = [line.bbox[0] for line in lines
                   if _is_zone_prose_line(line)]
    if not prose_edges:
        return lines  # no prose flow in this column (e.g. an equation band)
    prose_x0 = min(prose_edges)
    column_width = max(
        max(line.bbox[2] for line in lines)
        - min(line.bbox[0] for line in lines), 1.0)
    indent_tol = max(_ROW_MIN_INDENT_PT, column_width * _ROW_MAX_INDENT_RATIO)
    merged: list[Line] = []
    group: list[Line] = []
    group_y0 = group_y1 = 0.0

    def flush() -> None:
        if not group:
            return
        if len(group) == 1:
            merged.append(group[0])
            return
        if len({_region_key(line, regions) for line in group} - {-1}) > 1:
            merged.extend(group)
            return
        row_x0 = min(line.bbox[0] for line in group)
        flow_row = (
            row_x0 <= prose_x0 + indent_tol
            and not any(_is_heading_line(line, body_size) for line in group)
        )
        if flow_row:
            merged.append(_join_row(group))
        else:
            merged.extend(group)

    for line in lines:
        height = max(line.bbox[3] - line.bbox[1], 0.1)
        if group:
            group_h = max(group_y1 - group_y0, 0.1)
            overlap = min(line.bbox[3], group_y1) - max(line.bbox[1], group_y0)
            if overlap >= _ROW_OVERLAP_RATIO * min(height, group_h):
                group.append(line)
                group_y0 = min(group_y0, line.bbox[1])
                group_y1 = max(group_y1, line.bbox[3])
                continue
            flush()
        group = [line]
        group_y0, group_y1 = line.bbox[1], line.bbox[3]
    flush()
    return merged


# --- formula zones -----------------------------------------------------
# Display equations often mix math-font glyphs with REGULAR italic faces
# (variables, sub/superscripts like an italic "offset", numerators, sum
# limits). Line-level formula detection misses those regular-font fragment
# lines, so they used to leak into the neighbouring body paragraph. A
# formula ZONE is a vertical band seeded by unmistakable equation lines
# (math-font lines without prose, or a right-aligned "(n)" equation-number
# line) that then absorbs adjacent fragment lines; every line of a zone
# becomes one kind="formula" segment (zone-union bbox).
_ZONE_GAP = 3.0  # max vertical gap (pt) between a zone and an absorbed line
_ZONE_SUB_SIZE_RATIO = 0.85  # size <= body*ratio marks a sub/superscript
_ZONE_FRAG_MAX_CHARS = 30  # short-fragment absorption cap (visible chars)
_ZONE_SEED_MAX_CHARS = 40  # visible-length cap for math-font seed lines
_ZONE_SEED_MAX_LONG_WORDS = 1  # lowercase words >= 5 chars allowed in a seed
_ZONE_NUMBER_EDGE_RATIO = 0.05  # "(n)" seed: x1 within 5% of the column edge
_ZONE_SEED_MIN_INDENT = 2.0  # math-font seeds start past the column left edge
_ZONE_PROSE_MIN_CHARS = 15  # prose-line guard: min length ...
_ZONE_PROSE_MIN_STOPWORDS = 2  # ... and min stopword count
_ZONE_SENTENCE_END = ".!?"  # fragments never end a sentence
_ITALIC_FLAG = 1 << 1  # PyMuPDF span flag bit for italic faces

# Function words that mark running prose; display-formula lines contain none.
_PROSE_STOPWORDS = frozenset(
    """
    the a an of and or is are was were be been being this that these those it
    its we our us in on for with as by to from at into over under between
    during which where when while then than if because since through denotes
    denote given let holds defined follows following using implies such
    """.split()
)

_MATH_SYMBOLS = set("=+−×÷·∑∏∫√∞≈≠≤≥±→←↔⇒⇐∈∉⊂⊆∪∩∧∨¬∀∃∂∇^_/<>|‖")

_WORD_RE = re.compile(r"[A-Za-z]+")
_BOLD_FLAG = 1 << 4  # PyMuPDF span flag bit for bold faces


def _line_text(line: Line) -> str:
    return "".join(span.text for span in line.spans)


def _is_equation_line(line: Line) -> bool:
    text = _line_text(line)
    return bool(text.strip()) and bool(_EQ_ONLY_RE.fullmatch(text))


def _line_size(line: Line) -> float:
    """Representative font size of a line: size covering the most characters."""
    weights: dict[float, int] = defaultdict(int)
    for span in line.spans:
        weights[round(span.size, 1)] += max(len(span.text), 1)
    return max(weights.items(), key=lambda item: item[1])[0]


def _spans_are_bold(spans: list[Span]) -> bool:
    """True when the majority of the spans' characters use a bold face."""
    bold_chars = 0
    total = 0
    for span in spans:
        n = len(span.text.strip())
        if n == 0:
            continue
        total += n
        name = span.font.upper()
        if span.flags & _BOLD_FLAG or "BOLD" in name or "BLACK" in name \
                or "HEAVY" in name:
            bold_chars += n
    return total > 0 and bold_chars / total >= 0.6


def _line_is_bold(line: Line) -> bool:
    """True when the majority of the line's characters use a bold face."""
    return _spans_are_bold(line.spans)


def _caps_only(text: str) -> bool:
    """True when the text has letters and none of them is lowercase."""
    letters = [ch for ch in text if ch.isalpha()]
    return bool(letters) and all(not ch.islower() for ch in letters)


def _visible_text(text: str) -> str:
    """Text with equation placeholders removed."""
    return _EQ_TOKEN_RE.sub("", text)


def _is_formula_line(line: Line) -> bool:
    """Display-equation line: math-font spans mixed with variable fragments.

    Requires at least one math-font span and a placeholder-free remainder
    that contains no prose (no function words, no long lowercase words), so
    body sentences with inline math ("where q denotes ...") stay body. A
    standalone equation-number line "(3)" also counts.
    """
    text = _line_text(line).strip()
    if not text:
        return False
    if _EQ_NUMBER_ONLY_RE.fullmatch(text):
        return True
    if not any(span.text.strip() and _is_math_font(span.font)
               for span in line.spans):
        return False
    words = _WORD_RE.findall(_visible_text(text))
    if any(word.lower() in _PROSE_STOPWORDS for word in words):
        return False
    if any(word.islower() and len(word) >= 5 for word in words):
        return False
    return True


def _is_formula_fragment(line: Line, formula_lines: list[Line]) -> bool:
    """Short symbol fragment sitting on the same visual row as a formula.

    Equation layouts split sub/superscripts and delimiters into separate
    extraction lines without math fonts (e.g. an italic "offset" subscript);
    such a fragment is absorbed into the open formula paragraph when it
    vertically overlaps one of its lines.
    """
    text = _line_text(line).strip()
    if not text:
        return False
    visible = "".join(_visible_text(text).split())
    if len(visible) > 20:
        return False
    words = _WORD_RE.findall(visible)
    if any(word.lower() in _PROSE_STOPWORDS for word in words):
        return False
    y0, y1 = line.bbox[1], line.bbox[3]
    for other in formula_lines:
        oy0, oy1 = other.bbox[1], other.bbox[3]
        overlap = min(y1, oy1) - max(y0, oy0)
        if overlap > 0.5 * max(min(y1 - y0, oy1 - oy0), 0.1):
            return True
    return False


def _zone_stopword_count(text: str) -> int:
    return sum(1 for word in _WORD_RE.findall(text)
               if word.lower() in _PROSE_STOPWORDS)


def _zone_visible(line: Line) -> str:
    """Whitespace-normalized line text with equation placeholders removed."""
    return " ".join(_visible_text(_line_text(line)).split())


def _is_zone_prose_line(line: Line) -> bool:
    """Ordinary body sentence line: never seeds nor joins a formula zone."""
    visible = _zone_visible(line)
    return (len(visible) >= _ZONE_PROSE_MIN_CHARS
            and _zone_stopword_count(visible) >= _ZONE_PROSE_MIN_STOPWORDS)


def _line_has_math_span(line: Line) -> bool:
    return any(span.text.strip() and _is_math_font(span.font)
               for span in line.spans)


def _line_is_italic(line: Line) -> bool:
    """True when the majority of the line's characters use an italic face."""
    italic = 0
    total = 0
    for span in line.spans:
        n = len(span.text.strip())
        if n == 0:
            continue
        total += n
        name = span.font.upper()
        if span.flags & _ITALIC_FLAG or "ITALIC" in name or "OBLIQUE" in name:
            italic += n
    return total > 0 and italic / total >= 0.6


def _zone_line_excluded(line: Line, body_size: float) -> bool:
    """Lines that must never belong to a formula zone.

    Prose sentence lines (>= 2 stopwords and >= 15 visible chars), captions,
    references titles and heading lines stay out of every zone, which keeps
    a zone from swallowing multi-line body text (false-positive guard).
    """
    text = _line_text(line).strip()
    return (
        _is_zone_prose_line(line)
        or _CAPTION_RE.match(text) is not None
        or _REFS_TITLE_RE.match(text) is not None
        or _is_heading_line(line, body_size)
    )


# Sentinel replacing ⟦EQn⟧ tokens so word-adjacency survives the substitution
# (a bare removal would glue neighbouring words together or lose the glue
# information entirely).
_EQ_ATTACH_SENTINEL = "￼"


def _detached_long_words(text: str) -> int:
    """Long lowercase words (>= 5 chars) not glued to formula material.

    ACL-style equations mix text-font words into math: "⟦EQ⟧-score" (the
    hyphen compound after a masked variable) or "⟦EQ⟧score" (a word-shaped
    subscript). Such ATTACHED words are formula vocabulary, not prose, so
    they are exempt from the long-word count; only detached words like
    "empirical" mark running text. ``text`` is the RAW line text (before
    placeholder removal) — attachment is invisible afterwards.
    """
    marked = _EQ_TOKEN_RE.sub(_EQ_ATTACH_SENTINEL, text)
    count = 0
    for match in _WORD_RE.finditer(marked):
        word = match.group()
        if not (word.islower() and len(word) >= 5):
            continue
        before = marked[match.start() - 1] if match.start() else ""
        after = marked[match.end()] if match.end() < len(marked) else ""
        if before in ("-", _EQ_ATTACH_SENTINEL) or after == _EQ_ATTACH_SENTINEL:
            continue
        count += 1
    return count


def _zone_word_shape_ok(text: str) -> bool:
    """Fragment word shape: no stopwords, at most one long lowercase word.

    A single long lowercase word ("offset") is a typical italic subscript;
    several of them mark running text (e.g. "empirical human-limit constant
    ..." wrap lines carrying one inline math span) that must stay body.
    Hyphen compounds and words glued to a masked math run ("-score") do not
    count (see _detached_long_words). ``text`` is the raw line text.
    """
    words = _WORD_RE.findall(_visible_text(text))
    if any(word.lower() in _PROSE_STOPWORDS for word in words):
        return False
    return _detached_long_words(text) <= _ZONE_SEED_MAX_LONG_WORDS


def _is_zone_seed(
    line: Line,
    body_size: float,
    column_x0: float,
    column_x1: float,
    column_width: float,
) -> bool:
    """Zone seed: an unmistakable display-equation line.

    Either (a) an indented body-sized line carrying a math-font span and no
    prose (short visible remainder, no stopwords, at most one long
    lowercase word, not ending a sentence when it has a long word), or (b)
    a standalone equation-number line "(n)" right-aligned to the column
    edge. Flush-left lines never seed: a wrap line of running prose whose
    only words happen to be non-stopwords ("evaluation (n⟦EQm⟧ 45).")
    starts at the column edge, display-equation content does not.
    """
    text = _line_text(line).strip()
    if not text:
        return False
    if _EQ_NUMBER_ONLY_RE.fullmatch(text):
        return line.bbox[2] >= column_x1 - column_width * _ZONE_NUMBER_EDGE_RATIO
    if not _line_has_math_span(line):
        return False
    if line.bbox[0] <= column_x0 + _ZONE_SEED_MIN_INDENT:
        return False
    if _zone_line_excluded(line, body_size):
        return False
    if _line_size(line) < body_size * _ZONE_SUB_SIZE_RATIO:
        return False  # tiny math-carrying lines join by absorption instead
    visible = _zone_visible(line)
    if len(visible) > _ZONE_SEED_MAX_CHARS:
        return False
    if not _zone_word_shape_ok(_line_text(line)):
        return False
    if _detached_long_words(_line_text(line)) \
            and visible[-1] in _ZONE_SENTENCE_END:
        return False  # "... evaluation (n 45)." ends a prose sentence
    return True


# --- centered zone-seed cluster ----------------------------------------
# Display equations without ANY math-font span and without an "(n)" number
# (or whose number sits on a fragment line the extractor split off) still
# follow display GEOMETRY: short lines centered in the column, clear of the
# prose margins on both sides. A cluster of vertically chained centered
# candidate lines seeds a zone when it shows math evidence (a math symbol,
# a digit or a masked ⟦EQn⟧ run somewhere in the cluster) so that centered
# subtitles or epigraphs ("A Short Motto") never qualify.
_ZONE_CENTER_MARGIN_RATIO = 0.15  # min per-side margin vs the prose extent


def _is_centered_zone_candidate(
    line: Line,
    body_size: float,
    prose_x0: float,
    prose_x1: float,
) -> bool:
    """Short centered non-sentence line: may join a centered seed cluster."""
    text = _line_text(line).strip()
    if not text:
        return False
    if _zone_line_excluded(line, body_size):
        return False
    visible = _zone_visible(line)
    if not visible or len(visible) > _ZONE_SEED_MAX_CHARS:
        return False
    if visible[-1] in _ZONE_SENTENCE_END:
        return False
    if not _zone_word_shape_ok(_line_text(line)):
        return False
    margin = (prose_x1 - prose_x0) * _ZONE_CENTER_MARGIN_RATIO
    return (line.bbox[0] >= prose_x0 + margin
            and line.bbox[2] <= prose_x1 - margin)


def _cluster_has_math_evidence(cluster: list[Line]) -> bool:
    """Math symbol or masked math run anywhere in the cluster.

    Bare digits deliberately do NOT count: centered chart tick labels
    ("0.2", "0.4") or a lone year line must never seed a zone.
    """
    for line in cluster:
        text = _line_text(line)
        if _EQ_TOKEN_RE.search(text):
            return True
        if any(ch in _MATH_SYMBOLS for ch in _visible_text(text)):
            return True
    return False


def _is_zone_fragment_line(line: Line, body_size: float) -> bool:
    """Line a formula zone may absorb once it is vertically adjacent.

    Requires the fragment word shape plus one of: a math-font span, a
    sub/superscript font size, an italic-majority face, or a short text
    that does not end a sentence.
    """
    text = _line_text(line).strip()
    if not text:
        return False
    if _zone_line_excluded(line, body_size):
        return False
    visible = _zone_visible(line)
    if len(visible) > _ZONE_SEED_MAX_CHARS:
        return False
    if not _zone_word_shape_ok(_line_text(line)):
        return False
    if _line_has_math_span(line):
        return True
    ends_sentence = bool(visible) and visible[-1] in _ZONE_SENTENCE_END
    if ends_sentence:
        return False
    if _line_size(line) <= body_size * _ZONE_SUB_SIZE_RATIO:
        return True
    if _line_is_italic(line):
        return True
    return len(visible) <= _ZONE_FRAG_MAX_CHARS


def _formula_zone_ids(
    lines: list[Line], body_size: float, fired: list[bool]
) -> list[int | None]:
    """Per-line formula-zone id (None = not in a zone).

    Seeds open zones; zones with overlapping/adjacent (<= _ZONE_GAP) y-spans
    merge; fragment lines that vertically touch a zone are absorbed and
    extend it, to a fixpoint. Fired list-marker lines never participate:
    a masked-bullet item line ("⟦EQn⟧ NPC baseline term:") is a list item,
    not equation content.
    """
    ids: list[int | None] = [None] * len(lines)
    if not lines:
        return ids
    column_x0 = min(line.bbox[0] for line in lines)
    column_x1 = max(line.bbox[2] for line in lines)

    # A line sharing its visual row with a prose sentence line is a piece of
    # running text split by extraction ("iment (⟦EQm⟧", "⟦EQm⟧ 5 raters,"),
    # never display-equation content: it must stay in the body flow.
    prose_boxes = [line.bbox for line in lines if _is_zone_prose_line(line)]

    def overlaps_prose(line: Line) -> bool:
        y0, y1 = line.bbox[1], line.bbox[3]
        for box in prose_boxes:
            overlap = min(y1, box[3]) - max(y0, box[1])
            if overlap > 0.5 * max(min(y1 - y0, box[3] - box[1]), 0.1):
                return True
        return False

    # Right-alignment reference for "(n)" number seeds: stray wide lines
    # (page numbers, overfull rows) inflate the all-lines column extent and
    # used to push the "(n)" line out of the 5% edge band, so the PROSE
    # right edge is the reference whenever the column has a prose flow.
    if prose_boxes:
        seed_x1 = max(box[2] for box in prose_boxes)
    else:
        seed_x1 = column_x1
    seed_width = max(seed_x1 - column_x0, 1.0)

    zones: list[list[float]] = []  # [y0, y1] per zone id (index = id)
    alive: list[bool] = []
    for i, line in enumerate(lines):
        if (not fired[i] and not overlaps_prose(line)
                and _is_zone_seed(line, body_size, column_x0, seed_x1,
                                  seed_width)):
            ids[i] = len(zones)
            zones.append([line.bbox[1], line.bbox[3]])
            alive.append(True)

    # Centered-cluster seeds: display-equation GEOMETRY without any math
    # font (see _is_centered_zone_candidate). Chains of vertically adjacent
    # centered candidate lines seed one zone per chain when the chain shows
    # math evidence; columns without a prose flow have no trustworthy
    # center reference and are skipped.
    if prose_boxes:
        prose_x0 = min(box[0] for box in prose_boxes)
        prose_x1 = max(box[2] for box in prose_boxes)
        candidates = [
            i for i, line in enumerate(lines)
            if ids[i] is None and not fired[i] and not overlaps_prose(line)
            and _is_centered_zone_candidate(line, body_size, prose_x0,
                                            prose_x1)
        ]
        cluster: list[int] = []

        def flush_cluster() -> None:
            group = [lines[i] for i in cluster]
            if not group or not _cluster_has_math_evidence(group):
                return
            zid = len(zones)
            zones.append([min(ln.bbox[1] for ln in group),
                          max(ln.bbox[3] for ln in group)])
            alive.append(True)
            for i in cluster:
                ids[i] = zid

        for i in candidates:
            if cluster and lines[i].bbox[1] <= \
                    max(lines[j].bbox[3] for j in cluster) + _ZONE_GAP:
                cluster.append(i)
                continue
            flush_cluster()
            cluster = [i]
        flush_cluster()

    if not zones:
        return ids
    absorbable = [
        ids[i] is None and not fired[i] and not overlaps_prose(line)
        and _is_zone_fragment_line(line, body_size)
        for i, line in enumerate(lines)
    ]

    def merge_zones() -> None:
        changed = True
        while changed:
            changed = False
            for a in range(len(zones)):
                if not alive[a]:
                    continue
                for b in range(a + 1, len(zones)):
                    if not alive[b]:
                        continue
                    if (zones[a][0] <= zones[b][1] + _ZONE_GAP
                            and zones[a][1] >= zones[b][0] - _ZONE_GAP):
                        zones[a][0] = min(zones[a][0], zones[b][0])
                        zones[a][1] = max(zones[a][1], zones[b][1])
                        alive[b] = False
                        for k, zid in enumerate(ids):
                            if zid == b:
                                ids[k] = a
                        changed = True

    merge_zones()
    changed = True
    while changed:
        changed = False
        for i, line in enumerate(lines):
            if ids[i] is not None or not absorbable[i]:
                continue
            for z in range(len(zones)):
                if not alive[z]:
                    continue
                if (line.bbox[1] <= zones[z][1] + _ZONE_GAP
                        and line.bbox[3] >= zones[z][0] - _ZONE_GAP):
                    ids[i] = z
                    zones[z][0] = min(zones[z][0], line.bbox[1])
                    zones[z][1] = max(zones[z][1], line.bbox[3])
                    changed = True
                    break
        if changed:
            merge_zones()
    return ids


# A one-line "formula" at the column PROSE edge directly under an open body
# paragraph is the literalized tail of the paragraph's sentence ("p < 10−5."
# closing "...rejecting random choice at"), not a display equation: display
# equations are indented/centered, sentence tails are flush continuation
# lines. Such a line is absorbed into the body paragraph so its characters
# translate with the sentence instead of stranding as an English fragment.
_TAIL_X_TOL = 2.0
_TAIL_END_CHARS = ".,;:!?"


def _is_body_formula_tail(line: Line, prose_x0: float | None) -> bool:
    """Flush short formula-classified line that closes a body sentence."""
    if prose_x0 is None or line.bbox[0] > prose_x0 + _TAIL_X_TOL:
        return False
    visible = _zone_visible(line)
    return (0 < len(visible) <= _ZONE_SEED_MAX_CHARS
            and visible[-1] in _TAIL_END_CHARS)


# Unnumbered run-in subheadings of magazine layouts (Nature features,
# Science news): "THE SCALE OF REPRODUCIBILITY", "WHAT CAN BE DONE?". They
# are bold all-caps lines at roughly body size, so neither the numbering
# prefix nor the size delta catches them and they used to be glued onto the
# paragraph below.
_UNNUMBERED_HEAD_MAX_WORDS = 8


def _is_unnumbered_heading_line(line: Line, body_size: float) -> bool:
    """Bold all-caps standalone line that reads like a subheading.

    A bare caps name ("MONYA BAKER") has the same shape, so a heading must
    either be a question or contain at least one function word ("THE",
    "OF", "CAN", ...) -- person names never do.
    """
    text = _line_text(line).strip()
    if not text or len(text) > 80 or text[-1] in ".,;:":
        return False
    if not (_caps_only(text) and _line_is_bold(line)):
        return False
    if _line_size(line) < body_size - 0.5:
        return False  # figure labels, footnote-sized caps
    words = _WORD_RE.findall(text)
    if not 2 <= len(words) <= _UNNUMBERED_HEAD_MAX_WORDS:
        return False
    return text.endswith("?") or any(w.lower() in _PROSE_STOPWORDS for w in words)


def _is_heading_line(line: Line, body_size: float) -> bool:
    """Numbered-heading line: "VII. CONCLUSION", "A. MOTIVATION", "2.3 Foo".

    IEEE renders section titles in caps/bold at body size, so all-caps OR
    bold OR a size delta each suffice after the numbering-prefix match.
    """
    text = _line_text(line).strip()
    if not text or len(text) > 120:
        return False
    match = _HEAD_PREFIX_RE.match(text)
    if match is None:
        return _is_unnumbered_heading_line(line, body_size)
    rest = text[match.end():].strip()
    if not any(ch.isalpha() for ch in rest):
        return False
    return (
        _caps_only(rest)
        or _line_is_bold(line)
        or _line_size(line) - body_size >= HEADING_SIZE_DELTA
    )


def _is_heading_continuation(prev: Line, line: Line) -> bool:
    """All-caps wrap line continuing a heading paragraph (same size)."""
    text = _line_text(line).strip()
    return (
        bool(text)
        and len(text) <= 80
        and _caps_only(text)
        and abs(_line_size(line) - _line_size(prev)) <= 1.0
    )


def _classify_line(line: Line, body_size: float) -> str:
    """Line class driving paragraph breaks: refs_title|formula|heading|normal."""
    if _REFS_TITLE_RE.match(_line_text(line).strip()):
        return "refs_title"
    if _is_formula_line(line):
        return "formula"
    if _is_heading_line(line, body_size):
        return "heading"
    return "normal"


def _looks_like_formula_paragraph(text: str) -> bool:
    """Segment-level fallback for display equations the line pass missed.

    Short text ending in an equation number "(n)" with a high share of math
    symbols/placeholders and almost no prose words.
    """
    if not _EQ_NUMBER_TAIL_RE.search(text):
        return False
    visible = "".join(_visible_text(text).split())
    if not visible or len(visible) > 160:
        return False
    words = _WORD_RE.findall(_visible_text(text))
    prose = [w for w in words
             if w.lower() in _PROSE_STOPWORDS or (w.islower() and len(w) >= 5)]
    if len(prose) >= 2:
        return False
    if _EQ_TOKEN_RE.search(text):
        return True
    symbols = sum(1 for ch in visible if ch in _MATH_SYMBOLS)
    return symbols / len(visible) >= 0.15


def _cluster_columns(
    intervals: list[tuple[float, float]], page_width: float
) -> list[int]:
    """Assign a column index to each block x-interval (1D clustering on x0).

    Plain chain clustering on sorted x0 values is transitive: an indented
    block (e.g. an equation start) sitting between two real columns would
    bridge them into a single cluster. To prevent that, a block whose x0
    falls inside the x-extent already established by a column joins that
    column directly and does not take part in the gap chain; only blocks
    starting at a genuinely new left edge extend a column (gap <= page
    width * COLUMN_GAP_RATIO) or open the next one.
    """
    order = sorted(range(len(intervals)), key=lambda i: intervals[i][0])
    # Each column tracks the extent defined by its edge-aligned blocks and
    # the last x0 that participated in the gap chain.
    columns: list[dict[str, float]] = []
    assignment = [0] * len(intervals)
    for i in order:
        x0, x1 = intervals[i]
        contained = None
        for ci in range(len(columns) - 1, -1, -1):
            col = columns[ci]
            if col["x0"] <= x0 <= col["x1"] - 2.0:
                contained = ci
                break
        if contained is not None:
            col = columns[contained]
            if x0 - col["x0"] <= 2.0:
                # Starts at the column edge: a defining member, may widen it.
                col["x1"] = max(col["x1"], x1)
            # Indented members join without widening the defining extent, so
            # they can never pull a farther column into this one.
            assignment[i] = contained
            continue
        if columns and x0 - columns[-1]["last_x0"] <= page_width * COLUMN_GAP_RATIO:
            col = columns[-1]
            col["last_x0"] = x0
            col["x1"] = max(col["x1"], x1)
            assignment[i] = len(columns) - 1
        else:
            columns.append({"x0": x0, "x1": x1, "last_x0": x0})
            assignment[i] = len(columns) - 1
    return assignment


def _is_full_width(
    interval: tuple[float, float], content_x0: float, content_x1: float
) -> bool:
    """True when a block spans the page's whole body-text width.

    Either the block is wider than FULL_WIDTH_RATIO of the content width,
    or it straddles the content center line with at least
    CENTER_OVERHANG_RATIO of the content width on each side (catches
    centered title/author lines narrower than the width threshold).
    """
    width = content_x1 - content_x0
    if width <= 0:
        return True
    x0, x1 = interval
    if x1 - x0 > width * FULL_WIDTH_RATIO:
        return True
    center = (content_x0 + content_x1) / 2.0
    overhang = width * CENTER_OVERHANG_RATIO
    return x0 <= center - overhang and x1 >= center + overhang


def _partition_bands(
    intervals: list[tuple[float, float]], tops: list[float]
) -> list[tuple[bool, list[int]]]:
    """Split block indices into vertical bands via a top-to-bottom y scan.

    A full-width block forms a single-column band of its own; a run of
    consecutive non-full-width blocks forms one band whose columns are
    detected by x0 clustering. Returns (is_full_width, block_indices)
    pairs in top-to-bottom order.
    """
    if not intervals:
        return []
    content_x0 = min(iv[0] for iv in intervals)
    content_x1 = max(iv[1] for iv in intervals)
    order = sorted(range(len(intervals)),
                   key=lambda i: (tops[i], intervals[i][0]))
    bands: list[tuple[bool, list[int]]] = []
    for i in order:
        full = _is_full_width(intervals[i], content_x0, content_x1)
        if full or not bands or bands[-1][0]:
            bands.append((full, [i]))
        else:
            bands[-1][1].append(i)
    return bands


def _join_paragraph(texts: list[str]) -> str:
    """Join line texts into one paragraph, restoring hyphen line breaks."""
    merged = ""
    for text in texts:
        piece = text.strip()
        if not piece:
            continue
        if not merged:
            merged = piece
        elif merged.endswith("-") and len(merged) > 1 and merged[-2].isalpha():
            merged = merged[:-1] + piece
        else:
            merged = merged + " " + piece
    return merged


def _union_bbox(bboxes: list[tuple]) -> tuple[float, float, float, float]:
    return (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


def _majority_size(spans: list[Span], default: float) -> float:
    """Char-count weighted mode of the spans' font sizes."""
    weights: dict[float, int] = defaultdict(int)
    for span in spans:
        if span.text.strip():
            weights[round(span.size, 1)] += max(len(span.text), 1)
    if not weights:
        return default
    return max(weights.items(), key=lambda item: item[1])[0]


def _leading_caps_span_count(line: Line) -> int:
    """Length of the leading all-caps span run of a line (0 = none).

    Whitespace-only spans inside the run are absorbed; the run ends at the
    first span carrying lowercase letters, digits or punctuation.
    """
    count = 0
    for span in line.spans:
        text = span.text.strip()
        if not text:
            count += 1  # neutral whitespace span, may sit between label words
            continue
        if _caps_only(text) and _INLINE_LABEL_SHAPE_RE.fullmatch(
                " ".join(text.split())):
            count += 1
            continue
        break
    while count and not line.spans[count - 1].text.strip():
        count -= 1  # trailing whitespace spans belong to the body side
    return count


def _split_first_line_boxes(
    line: Line, label_len: int
) -> tuple[list[tuple], list[tuple]]:
    """Split the first line's span bboxes at a character offset.

    Returns (label_boxes, rest_boxes). A span straddling the offset is cut
    proportionally to its character count (used by the text-pattern fallback
    where the label shares one span with the body).
    """
    label_boxes: list[tuple] = []
    rest_boxes: list[tuple] = []
    consumed = label_len
    for span in line.spans:
        n = len(span.text)
        if consumed >= n:
            if span.text.strip():
                label_boxes.append(span.bbox)
            consumed -= n
            continue
        if consumed > 0:
            x0, y0, x1, y1 = span.bbox
            cut = x0 + (x1 - x0) * (consumed / n)
            label_boxes.append((x0, y0, cut, y1))
            rest_boxes.append((cut, y0, x1, y1))
            consumed = 0
            continue
        if span.text.strip():
            rest_boxes.append(span.bbox)
    return label_boxes, rest_boxes


def _split_inline_label(
    lines: list[Line],
) -> list[tuple[str, tuple, float, str]] | None:
    """Split a paragraph-leading inline label off the paragraph (F).

    IEEE Access renders "ABSTRACT body ...", "INDEX TERMS terms ..." as one
    paragraph whose first spans are a bold all-caps label. Returns
    [(label_text, bbox, font_size, "heading"), (body_text, bbox, font_size,
    "body")] or None when the paragraph does not start with such a label.

    bbox policy: the two segments must never overlap after retypesetting
    while their union must still cover every original glyph (both rects are
    redacted). For a single-line paragraph the line is cut vertically at the
    label boundary. For a multi-line paragraph the label segment takes the
    whole first line (its rect redacts the label AND the first line's
    remainder glyphs) and the body segment takes the remaining lines; the
    body text still starts with the first line's remainder words.
    """
    first = lines[0]
    line_text = _line_text(first)
    if _CAPTION_RE.match(line_text.strip()):
        return None  # "TABLE 1. ..." caption shapes keep their own rule
    label_len = 0
    label_text = ""
    count = _leading_caps_span_count(first)
    if count:
        label_spans = first.spans[:count]
        candidate = " ".join("".join(s.text for s in label_spans).split())
        words = candidate.split(" ")
        if (
            1 <= len(words) <= 2
            and _INLINE_LABEL_SHAPE_RE.fullmatch(candidate)
            and sum(len(w) for w in words) >= _INLINE_LABEL_MIN_LETTERS
        ):
            body_span = next(
                (s for line in (first.spans[count:], *[ln.spans for ln in lines[1:]])
                 for s in line if s.text.strip()),
                None,
            )
            label_font = next(
                (s.font for s in label_spans if s.text.strip()), "")
            style_ok = _spans_are_bold(label_spans) or (
                body_span is not None and body_span.font != label_font)
            # A label announces a new sentence, so the body must not start
            # with a lowercase letter. This keeps bold-caps biography names
            # ("HYEONWOO GIL received the B.S. degree ...") from splitting.
            candidate_len = sum(len(s.text) for s in label_spans)
            rest_preview = line_text[candidate_len:].strip()
            if not rest_preview and len(lines) > 1:
                rest_preview = _line_text(lines[1]).strip()
            rest_preview = rest_preview.lstrip(_INLINE_LABEL_SEPARATORS)
            continues_sentence = bool(rest_preview) and rest_preview[0].islower()
            if style_ok and not continues_sentence:
                label_len = candidate_len
                label_text = candidate
    if label_len == 0:
        # Text-pattern fallback: well-known labels split even when the font
        # gives no signal (same face, not bold).
        stripped = line_text.lstrip()
        match = _INLINE_LABEL_FALLBACK_RE.match(stripped)
        if match is None:
            return None
        label_len = (len(line_text) - len(stripped)) + match.end()
        label_text = match.group(1)
    remainder = line_text[label_len:].strip()
    if not remainder and len(lines) == 1:
        return None  # a lone label line is a plain heading, nothing to split
    label_boxes, rest_boxes = _split_first_line_boxes(first, label_len)
    if not label_boxes:
        return None
    if len(lines) == 1:
        if not rest_boxes:
            return None
        label_bbox = _union_bbox(label_boxes)
        rest = _union_bbox(rest_boxes)
        body_bbox = (max(rest[0], label_bbox[2]), rest[1], rest[2], rest[3])
        if body_bbox[0] >= body_bbox[2]:
            return None
        body_spans = [s for s in first.spans[count:]] if count else first.spans
    else:
        # The label rect covers the whole first line so its redaction also
        # erases the first-line remainder glyphs; the body rect starts below
        # and the two rects stay disjoint.
        label_bbox = tuple(first.bbox)
        rest = _union_bbox([ln.bbox for ln in lines[1:]])
        y0 = max(rest[1], label_bbox[3])
        if y0 >= rest[3]:
            return None
        body_bbox = (rest[0], y0, rest[2], rest[3])
        body_spans = ([s for s in first.spans[count:]] if count else
                      list(first.spans))
        body_spans = body_spans + [s for ln in lines[1:] for s in ln.spans]
    body_text = _join_paragraph(
        [line_text[label_len:]] + [_line_text(ln) for ln in lines[1:]]
    ).lstrip(_INLINE_LABEL_SEPARATORS)
    if not body_text:
        return None
    label_size = _majority_size(
        first.spans[:count] if count else first.spans, _line_size(first))
    body_size = _majority_size(body_spans, label_size)
    return [
        (label_text, label_bbox, label_size, "heading"),
        (body_text, body_bbox, body_size, "body"),
    ]


def _body_font_size(layout: DocumentLayout) -> float:
    """Document-level representative body size (char-count weighted mode)."""
    weights: dict[float, int] = defaultdict(int)
    for page in layout.pages:
        top = page.height * HEADER_FOOTER_BAND
        bottom = page.height * (1.0 - HEADER_FOOTER_BAND)
        for block in page.blocks:
            if block.type != "text":
                continue
            for line in block.lines:
                if line.bbox[3] <= top or line.bbox[1] >= bottom:
                    continue
                for span in line.spans:
                    weights[round(span.size, 1)] += len(span.text)
    if not weights:
        return 10.0
    return max(weights.items(), key=lambda item: item[1])[0]


def _merge_column_paragraphs(
    lines: list[Line], body_size: float,
    forced_markers: frozenset[int] | set[int] = frozenset(),
    regions: list[Region] = (),
) -> list[tuple[list[Line], str]]:
    """Merge one column's sorted lines into (lines, forced_kind) paragraphs.

    Formula zones run first: each zone's lines leave the normal flow and
    become one forced "formula" paragraph (zone-union bbox downstream).
    Besides the vertical-gap rule, paragraph breaks are forced at class
    transitions so that heading lines (D) and display-formula lines (A)
    never merge with surrounding body text. A line starting with a fired
    list marker (``forced_markers``: sequence-fired "(n)" lines, see
    _sequence_marker_ids) always opens a new "list" paragraph (one segment
    per item). An item absorbs its continuation lines in one of two styles,
    decided by the first line after the marker line: an indented line marks
    a hanging-indent item (only indented lines join, so a flush paragraph
    after the list never glues onto the last item), while a line the marker
    line is indented PAST (paragraph-style "(n)" enumeration without a
    hanging indent) opens flush-continuation absorption until the next
    marker-shaped line, gap break or class change.
    """
    all_fired = _fired_markers(lines, forced_markers)
    zone_ids = _formula_zone_ids(lines, body_size, all_fired)
    prose_edges = [line.bbox[0] for line in lines
                   if _is_zone_prose_line(line)]
    prose_x0 = min(prose_edges) if prose_edges else None
    paragraphs: list[tuple[list[Line], str]] = []
    zone_lines: dict[int, list[Line]] = defaultdict(list)
    plain: list[Line] = []
    fired: list[bool] = []
    for line, fired_flag, zid in zip(lines, all_fired, zone_ids):
        if zid is None:
            plain.append(line)
            fired.append(fired_flag)
        else:
            zone_lines[zid].append(line)
    for zid in sorted(zone_lines):
        zlines = sorted(zone_lines[zid], key=lambda ln: (ln.bbox[1], ln.bbox[0]))
        paragraphs.append((zlines, "formula"))
    # Layout-model regions: the remaining lines are regrouped per region
    # (a pull quote mid-column no longer interleaves with the body lines)
    # and a paragraph never continues into another region. Formula zones
    # were taken out first, so equation fragments are never split apart.
    keys = [_region_key(line, regions) for line in plain]
    if regions:
        first_seen: dict[int, int] = {}
        for i, key in enumerate(keys):
            first_seen.setdefault(key, i)
        order = sorted(range(len(plain)),
                       key=lambda i: (first_seen[keys[i]], i))
        plain = [plain[i] for i in order]
        fired = [fired[i] for i in order]
        keys = [keys[i] for i in order]

    current: list[Line] = []
    cur_kind = ""  # "", "heading", "formula" or "list"
    list_style = ""  # "", "hanging" or "paragraph" (kind "list" only)
    for idx, line in enumerate(plain):
        cls = _classify_line(line, body_size)
        is_marker = fired[idx] and cls == "normal"
        if current:
            gap = line.bbox[1] - current[-1].bbox[3]
            fits_gap = gap < _line_size(current[-1]) * PARA_GAP_FACTOR
            if (keys[idx] != keys[idx - 1]
                    and not _continues_across_regions(
                        current[-1], line, keys[idx - 1], keys[idx], regions)):
                absorb = False  # another layout region: new paragraph
            elif is_marker:
                absorb = False  # a marker line always starts a new item
            elif cur_kind == "heading":
                absorb = (fits_gap and cls == "normal"
                          and _is_heading_continuation(current[-1], line))
            elif cur_kind == "formula":
                absorb = fits_gap and (
                    cls == "formula"
                    or (cls == "normal" and _is_formula_fragment(line, current))
                )
            elif cur_kind == "list":
                absorb = fits_gap and cls == "normal"
                if absorb and len(current) == 1:
                    # First continuation candidate decides the item style.
                    marker_x0 = current[0].bbox[0]
                    if line.bbox[0] >= marker_x0 + _ITEM_INDENT_MIN:
                        list_style = "hanging"
                    elif (marker_x0 >= line.bbox[0] + _ITEM_INDENT_MIN
                            and _line_marker_family(line) is None):
                        list_style = "paragraph"
                    else:
                        absorb = False
                elif absorb and list_style == "paragraph":
                    # Flush continuation lines join until a marker-shaped
                    # line ends the item (unfired "(n)" lines included).
                    absorb = _line_marker_family(line) is None
                elif absorb:
                    # Hanging item: only indented lines belong to it.
                    absorb = (line.bbox[0] >= current[0].bbox[0]
                              + _ITEM_INDENT_MIN)
            else:
                absorb = fits_gap and (
                    cls == "normal"
                    # Sentence tail typeset as a one-line formula (see
                    # _is_body_formula_tail): stays in the body flow.
                    or (cls == "formula"
                        and _is_body_formula_tail(line, prose_x0))
                )
            # A line opening a theorem environment ("Theorem 1.", "Proof.",
            # "Definition 2 (Transition).") starts its own paragraph when
            # the previous line closed a sentence: IEEE sets these headers
            # inline with their body, so the LINE stays one segment with the
            # statement text, but it must never glue onto the preceding
            # prose paragraph. The guard keeps wrap lines that merely start
            # with "Theorem 1." mid-sentence in their paragraph: the split
            # only fires when the previous line closed a sentence or is a
            # numbered (sub)heading line ("A. Consistency") that the style
            # heuristics missed (e.g. an italic run-in subheading).
            if absorb and cls == "normal" \
                    and _THEOREM_HEAD_RE.match(_line_text(line).lstrip()):
                prev_text = _line_text(current[-1]).strip()
                if (_ends_paragraph(prev_text)
                        or _HEAD_PREFIX_RE.match(prev_text) is not None):
                    absorb = False
            if absorb:
                current.append(line)
                continue
            paragraphs.append((current, cur_kind))
        current = [line]
        list_style = ""
        cur_kind = ("list" if is_marker else
                    "heading" if cls in ("heading", "refs_title") else
                    "formula" if cls == "formula" else "")
    if current:
        paragraphs.append((current, cur_kind))
    return paragraphs


def _page_flow(
    page: PageLayout, body_size: float
) -> tuple[list[tuple[int, int, list[Line]]], list[Line], int]:
    """Column flow of a page: (band, column, lines) triples in reading order.

    Placeholder-only display-equation lines are dropped (their glyphs stay
    in place untouched), header/footer lines are split off, vertical bands
    over the text-block x-intervals separate full-width front matter from
    multi-column runs, and each column's sorted lines get their split
    visual rows re-assembled (_merge_visual_rows). Returns (flow,
    header_footer_lines, band_count).
    """
    top = page.height * HEADER_FOOTER_BAND
    bottom = page.height * (1.0 - HEADER_FOOTER_BAND)
    left = page.width * SIDE_MARGIN_BAND
    right = page.width * (1.0 - SIDE_MARGIN_BAND)
    # A line in the side band is margin furniture only when it also lies
    # wholly outside the text area: a justified body line can end inside the
    # band ("... learning,'' in") and its last word must not be torn off.
    inner = [ln.bbox for b in page.blocks if b.type == "text"
             for ln in b.lines
             if ln.bbox[0] < right and ln.bbox[2] > left]
    text_right = max((bb[2] for bb in inner), default=right)
    text_left = min((bb[0] for bb in inner), default=left)

    header_footer_lines: list[Line] = []
    content_blocks: list[list[Line]] = []  # content lines grouped per block
    for block in page.blocks:
        if block.type != "text":
            continue
        content: list[Line] = []
        for line in block.lines:
            if _is_equation_line(line):
                # Placeholder-only display-equation lines never become
                # segments: the original glyphs stay in place untouched.
                continue
            if (line.bbox[3] <= top or line.bbox[1] >= bottom
                    or (line.bbox[0] >= right and line.bbox[0] >= text_right)
                    or (line.bbox[2] <= left and line.bbox[2] <= text_left)):
                header_footer_lines.append(line)
            else:
                content.append(line)
        if content:
            content_blocks.append(content)

    # Vertical bands over text-block x-intervals (header/footer excluded):
    # full-width blocks (title/authors/abstract) become single-column bands,
    # runs of narrower blocks form multi-column bands clustered on x0.
    intervals = [
        (
            min(line.bbox[0] for line in lines),
            max(line.bbox[2] for line in lines),
        )
        for lines in content_blocks
    ]
    tops = [min(line.bbox[1] for line in lines) for lines in content_blocks]
    bands = _partition_bands(intervals, tops)

    flow: list[tuple[int, int, list[Line]]] = []
    for band_idx, (full, block_ids) in enumerate(bands):
        if full:
            assignment = [0] * len(block_ids)
        else:
            assignment = _cluster_columns(
                [intervals[i] for i in block_ids], page.width
            )
        column_lines: dict[int, list[Line]] = defaultdict(list)
        for i, column in zip(block_ids, assignment):
            column_lines[column].extend(content_blocks[i])
        for column in sorted(column_lines):
            lines = sorted(column_lines[column],
                           key=lambda ln: (ln.bbox[1], ln.bbox[0]))
            flow.append((band_idx, column,
                         _merge_visual_rows(lines, body_size, page.regions)))
    return flow, header_footer_lines, len(bands)


def _page_segments(
    page: PageLayout,
    body_size: float,
    flow: list[tuple[int, int, list[Line]]],
    header_footer_lines: list[Line],
    band_count: int,
    forced_markers: frozenset[int] | set[int] = frozenset(),
) -> list[Segment]:
    # Merge lines into paragraphs per band and column (class-aware, (A)/(D)).
    # Paragraph merging never crosses a band or column boundary.
    paragraphs: list[tuple[int, int, list[Line], str]] = []
    for band_idx, column, lines in flow:
        for para_lines, forced_kind in _merge_column_paragraphs(
                lines, body_size, forced_markers, page.regions):
            paragraphs.append((band_idx, column, para_lines, forced_kind))

    # Reading order: bands top-to-bottom, columns left-to-right inside a
    # band, paragraphs top-to-bottom inside a column; header/footer last.
    entries: list[tuple[int, int, list[Line], str]] = list(paragraphs)
    entries.sort(key=lambda item: (item[0], item[1], item[2][0].bbox[1]))
    for line in sorted(header_footer_lines, key=lambda ln: ln.bbox[1]):
        entries.append((band_count, 0, [line], "header_footer"))

    segments: list[Segment] = []
    idx = 0
    for _band, column, lines, forced_kind in entries:
        text = _join_paragraph([_line_text(line) for line in lines])
        if not text:
            continue
        # (F) inline front-matter labels ("ABSTRACT body ...") split into a
        # heading segment plus a body segment with disjoint bboxes.
        pieces = None if forced_kind else _split_inline_label(lines)
        if pieces is not None:
            for piece_text, piece_bbox, piece_size, piece_kind in pieces:
                segments.append(
                    Segment(
                        id=f"p{page.number}_s{idx}",
                        page=page.number,
                        column=column,
                        bbox=piece_bbox,
                        text=piece_text,
                        kind=piece_kind,
                        font_size=piece_size,
                    )
                )
                idx += 1
            continue
        font_size = _majority_size(
            [span for line in lines for span in line.spans], 10.0)
        if forced_kind:
            # List items are ordinary translatable body text; the "list"
            # forcing only controls the paragraph split above.
            kind = "body" if forced_kind == "list" else forced_kind
        elif _CAPTION_RE.match(text):
            kind = "caption"
        elif font_size - body_size >= HEADING_SIZE_DELTA:
            kind = "heading"
        elif _looks_like_formula_paragraph(text):
            kind = "formula"
        else:
            kind = "body"
        segments.append(
            Segment(
                id=f"p{page.number}_s{idx}",
                page=page.number,
                column=column,
                bbox=_union_bbox([line.bbox for line in lines]),
                text=text,
                kind=kind,
                font_size=font_size,
            )
        )
        idx += 1
    return segments


# --- multi-line heading merge -------------------------------------------
# A multi-line paper title is typeset as consecutive full-width lines; each
# line becomes its own band (see _partition_bands), so paragraph merging can
# never join them and every line used to translate as half a sentence with
# its own font shrink. Adjacent heading segments with (near-)equal font size
# and the same alignment merge into ONE heading segment. A numbered section
# title ("IV. GUARANTEES", "A. MOTIVATION") never joins a previous heading,
# and body text is untouched, so numbered headings still never absorb the
# paragraph below them.
_HEADING_MERGE_SIZE_TOL = 0.5  # pt: font sizes must be near-identical
_HEADING_MERGE_OVERLAP = 2.0  # pt of allowed vertical overlap
_HEADING_ALIGN_CENTER_RATIO = 0.1  # center offset tol vs the wider line


def _headings_aligned(a: Segment, b: Segment) -> bool:
    """Same alignment: shared left edge or shared center line."""
    if abs(a.bbox[0] - b.bbox[0]) <= _MARKER_X_TOL:
        return True
    width = max(a.bbox[2] - a.bbox[0], b.bbox[2] - b.bbox[0], 1.0)
    center_a = (a.bbox[0] + a.bbox[2]) / 2.0
    center_b = (b.bbox[0] + b.bbox[2]) / 2.0
    return abs(center_a - center_b) <= width * _HEADING_ALIGN_CENTER_RATIO


def _same_region(a: Segment, b: Segment,
                 regions_by_page: dict[int, list[Region]]) -> bool:
    """Both segments sit mostly inside one layout-model region."""
    regions = regions_by_page.get(a.page) or []
    ia, sa = _best_region(a.bbox, regions)
    ib, sb = _best_region(b.bbox, regions)
    return (ia is not None and ia == ib
            and sa >= _REGION_MIN_SHARE and sb >= _REGION_MIN_SHARE)


def _merge_adjacent_headings(
    segments: list[Segment],
    regions_by_page: dict[int, list[Region]] | None = None,
) -> list[Segment]:
    """Merge consecutive same-style heading segments (multi-line titles).

    Headings the layout model put in ONE title region (a pull quote whose
    first word is set larger) merge even when their sizes differ.
    """
    regions_by_page = regions_by_page or {}
    merged: list[Segment] = []
    for seg in segments:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.kind == "heading" and seg.kind == "heading"
            and prev.page == seg.page
            and _same_region(prev, seg, regions_by_page)
            and _HEAD_PREFIX_RE.match(seg.text) is None
        ):
            prev.text = _join_paragraph([prev.text, seg.text])
            prev.bbox = _union_bbox([prev.bbox, seg.bbox])
            continue
        if (
            prev is not None
            and prev.kind == "heading" and seg.kind == "heading"
            and prev.page == seg.page
            and abs(prev.font_size - seg.font_size) <= _HEADING_MERGE_SIZE_TOL
            and seg.bbox[1] - prev.bbox[3] >= -_HEADING_MERGE_OVERLAP
            and seg.bbox[1] - prev.bbox[3]
                < max(prev.font_size, seg.font_size) * PARA_GAP_FACTOR
            and _headings_aligned(prev, seg)
            # A numbered section title or the references title always keeps
            # its own segment: it is a NEW heading, not a wrap line.
            and _HEAD_PREFIX_RE.match(seg.text) is None
            and _REFS_TITLE_RE.match(seg.text) is None
        ):
            prev.text = _join_paragraph([prev.text, seg.text])
            prev.bbox = _union_bbox([prev.bbox, seg.bbox])
            continue
        merged.append(seg)
    return merged


def _apply_author_kind(
    layout: DocumentLayout, segments: list[Segment], body_size: float
) -> None:
    """Mark front-matter author blocks on the first page as kind "author".

    Route 1: body segments between the title (largest-font segment in the
    top half of page 0) and the Abstract. Route 2: body segments in the top
    band of page 0, below the title, carrying author cues (e-mail "@",
    University/Institute/Department/..., IEEE membership). Anything not
    matched stays body: mistranslating an author line is better than losing
    a real body paragraph.
    """
    if not layout.pages:
        return
    page0 = layout.pages[0]
    first = [seg for seg in segments
             if seg.page == page0.number and seg.kind != "header_footer"]
    if not first:
        return
    title: Segment | None = max(first, key=lambda seg: seg.font_size)
    if (title.font_size < body_size + _TITLE_SIZE_DELTA
            or title.bbox[1] > page0.height * 0.5):
        title = None
    abstract = next((seg for seg in first if _ABSTRACT_RE.match(seg.text)), None)
    for seg in first:
        if seg.kind != "body" or seg is title or seg is abstract:
            continue
        below_title = title is not None and seg.bbox[1] >= title.bbox[3] - 2.0
        if (title is not None and abstract is not None and below_title
                and seg.bbox[3] <= abstract.bbox[1] + 2.0):
            seg.kind = "author"
        elif (_AUTHOR_CUE_RE.search(seg.text)
                and seg.bbox[3] <= page0.height * _AUTHOR_TOP_BAND
                and (title is None or below_title)):
            seg.kind = "author"
        # Nature/Science front matter (see _looks_like_author_list).
        elif (seg.bbox[3] <= page0.height * _AUTHOR_TOP_BAND
                and _looks_like_author_list(seg.text)):
            seg.kind = "author"
        # A byline ("BY MONYA BAKER") often sits under a half-page cover
        # headline, below the top band; its shape alone is specific enough.
        elif _BYLINE_RE.match(seg.text):
            seg.kind = "author"
        elif _ARTICLE_META_RE.search(seg.text) and len(seg.text) < 400:
            seg.kind = "author"
        # Small-type affiliation footnote, usually at the bottom of page 0.
        elif (seg.font_size < body_size - 0.5
                and len(_AUTHOR_CUE_RE.findall(seg.text))
                >= _AFFIL_FOOTNOTE_MIN_CUES):
            seg.kind = "author"


def _apply_reference_kind(segments: list[Segment]) -> None:
    """Mark bibliography segments as kind "reference".

    Every segment after the References/Bibliography heading (across the
    remaining pages, header/footer and formulas excluded) becomes a
    reference; the heading itself stays a heading and may be translated.
    Independently, "[n] ... vol./pp./year" shaped body segments are
    references wherever they appear.
    """
    in_refs = False
    for seg in segments:
        if (not in_refs and seg.kind == "heading"
                and _REFS_TITLE_RE.match(seg.text)):
            in_refs = True
            continue
        if in_refs:
            if seg.kind in ("body", "caption", "heading"):
                seg.kind = "reference"
        elif (seg.kind == "body" and _REF_ITEM_HEAD_RE.match(seg.text)
                and _REF_ITEM_BODY_RE.search(seg.text)):
            seg.kind = "reference"


def _bbox_center(bbox: tuple) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _center_inside(bbox: tuple, box: tuple) -> bool:
    cx, cy = _bbox_center(bbox)
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _apply_figure_text_kind(
    layout: DocumentLayout, segments: list[Segment]
) -> None:
    """Mark non-caption text inside figure/table regions as "figure_text".

    Regions come from the same caption-adjacency logic the assets API uses
    (regions.detect_figures), so axis labels, table cells and legends living
    inside a detected graphic region are never translated nor redacted. The
    caption itself keeps kind "caption" and stays translatable; membership
    is decided by the segment's center point.
    """
    figures, _boxes = detect_figures(layout, segments, HEADER_FOOTER_BAND)
    if not figures:
        return
    regions_by_page: dict[int, list[list[float]]] = defaultdict(list)
    for fig in figures:
        regions_by_page[fig["page"]].append(fig["bbox"])
    for seg in segments:
        if seg.kind not in ("body", "heading"):
            continue
        if any(_center_inside(seg.bbox, box)
               for box in regions_by_page.get(seg.page, ())):
            seg.kind = "figure_text"


# --- text-only tables ---------------------------------------------------
# IEEE tables (TABLE IV...) are often typeset with plain text alignment and
# few or no rules, so the graphic-block region detector misses them and
# their cell text used to be translated (overlapping, broken output). A
# text-table region is detected around a "TABLE n" caption: the contiguous
# line block below (or above) the caption qualifies when it is made of many
# short cell texts with a high digit share and shows table STRUCTURE
# (>= 3 x-aligned span-edge clusters or a horizontal rule drawing). Prose
# guard: a long line carrying >= 2 stopwords ends the block before it.
_TABLE_ROW_GAP = 20.0  # max vertical gap (pt) between caption/rows
_TABLE_MIN_LINES = 4  # min extraction lines in a table block
_TABLE_SHORT_CELL_MAX = 32  # visible chars of a "short cell" span
_TABLE_MIN_SHORT_RATIO = 0.6  # share of short cells required
_TABLE_DIGIT_MIN_RATIO = 0.12  # digit share of non-space chars required
_TABLE_COL_X_TOL = 3.0  # span-x0 cluster tolerance (pt)
_TABLE_MIN_COLUMNS = 3  # x-aligned clusters (>= 2 members) required
_TABLE_PROSE_MIN_CHARS = 35  # prose-guard line length ...
_TABLE_RULE_MAX_H = 3.0  # horizontal-rule drawing: max height ...
_TABLE_RULE_MIN_W = 30.0  # ... and min width (pt)
_TABLE_EDGE_SLOP = 2.0
# Horizontal slack when growing the table x-window: rightmost cells of a
# wide table need not overlap the caption's x-range, but the slack stays
# near the two-column gutter scale so a neighbouring column's prose is not
# pulled into the block.
_TABLE_X_SLACK = 36.0


def _is_table_prose_line(line: Line) -> bool:
    """Long running-prose line: terminates a table block (false-positive guard)."""
    text = " ".join(_line_text(line).split())
    return (len(text) >= _TABLE_PROSE_MIN_CHARS
            and _zone_stopword_count(text) >= 2)


def _table_block_lines(
    caption_bbox: tuple, lines: list[Line], side: str, body_size: float
) -> list[Line]:
    """Contiguous candidate-line run adjacent to a table caption on one side.

    Lines must horizontally overlap the growing region (seeded by the
    caption's x-range), chain within _TABLE_ROW_GAP vertically, and stop at
    the first long prose line.
    """
    if side == "below":
        pool = sorted(
            (ln for ln in lines
             if ln.bbox[1] >= caption_bbox[3] - _TABLE_EDGE_SLOP),
            key=lambda ln: ln.bbox[1])
        edge = caption_bbox[3]
    else:
        pool = sorted(
            (ln for ln in lines
             if ln.bbox[3] <= caption_bbox[1] + _TABLE_EDGE_SLOP),
            key=lambda ln: -ln.bbox[3])
        edge = caption_bbox[1]
    x_range = [caption_bbox[0], caption_bbox[2]]
    block: list[Line] = []
    for ln in pool:
        if (min(ln.bbox[2], x_range[1] + _TABLE_X_SLACK)
                - max(ln.bbox[0], x_range[0] - _TABLE_X_SLACK)) <= 0:
            continue  # far outside the table window: another column
        gap = (ln.bbox[1] - edge) if side == "below" else (edge - ln.bbox[3])
        if gap > _TABLE_ROW_GAP:
            break
        # Block terminators: running prose, another caption, a section
        # heading or the references title never belong to a table body.
        text = _line_text(ln).strip()
        if (_is_table_prose_line(ln)
                or _CAPTION_RE.match(text) is not None
                or _REFS_TITLE_RE.match(text) is not None
                or _is_heading_line(ln, body_size)):
            break
        block.append(ln)
        edge = max(edge, ln.bbox[3]) if side == "below" \
            else min(edge, ln.bbox[1])
        x_range[0] = min(x_range[0], ln.bbox[0])
        x_range[1] = max(x_range[1], ln.bbox[2])
    return block


def _has_horizontal_rule(region: tuple, drawings: list[tuple]) -> bool:
    """Thin wide drawing inside the region's vertical extent."""
    for box in drawings:
        if box[3] - box[1] > _TABLE_RULE_MAX_H:
            continue
        if box[2] - box[0] < _TABLE_RULE_MIN_W:
            continue
        if h_overlap(box, region) <= 0:
            continue
        if (box[1] >= region[1] - _TABLE_EDGE_SLOP
                and box[3] <= region[3] + _TABLE_EDGE_SLOP):
            return True
    return False


def _looks_like_table_block(
    block: list[Line], drawings: list[tuple]
) -> bool:
    """Table-ness of a candidate line block (see section comment)."""
    if len(block) < _TABLE_MIN_LINES:
        return False
    cells = [" ".join(span.text.split())
             for ln in block for span in ln.spans if span.text.strip()]
    if not cells:
        return False
    short = sum(1 for cell in cells if len(cell) <= _TABLE_SHORT_CELL_MAX)
    if short < len(cells) * _TABLE_MIN_SHORT_RATIO:
        return False
    chars = [ch for cell in cells for ch in cell if not ch.isspace()]
    if not chars:
        return False
    digits = sum(1 for ch in chars if ch.isdigit())
    if digits / len(chars) < _TABLE_DIGIT_MIN_RATIO:
        return False
    # Structure: x-aligned span-edge clusters (text-alignment columns) or a
    # horizontal rule drawing inside the block.
    edges = sorted(span.bbox[0]
                   for ln in block for span in ln.spans if span.text.strip())
    clusters: list[list[float]] = []  # [base_x, member_count]
    for x in edges:
        if clusters and x - clusters[-1][0] <= _TABLE_COL_X_TOL:
            clusters[-1][1] += 1
        else:
            clusters.append([x, 1])
    aligned = sum(1 for _x, n in clusters if n >= 2)
    if aligned >= _TABLE_MIN_COLUMNS:
        return True
    return _has_horizontal_rule(_union_bbox([ln.bbox for ln in block]),
                                drawings)


def _apply_text_table_kind(
    layout: DocumentLayout, segments: list[Segment], body_size: float
) -> None:
    """Mark text-only table bodies near "TABLE n" captions as "figure_text".

    Complements _apply_figure_text_kind for tables typeset without graphic
    blocks. The caption keeps kind "caption" (translated); every body or
    heading segment whose center lies inside the detected table region is
    excluded from translation and redaction.
    """
    table_captions = [
        (seg, num) for seg, kind, num in select_captions(segments)
        if kind == "table"
    ]
    if not table_captions:
        return
    pages = {page.number: page for page in layout.pages}
    segs_by_page: dict[int, list[Segment]] = defaultdict(list)
    for seg in segments:
        segs_by_page[seg.page].append(seg)
    for caption, _num in table_captions:
        page = pages.get(caption.page)
        if page is None:
            continue
        page_lines = [
            line for block in page.blocks if block.type == "text"
            for line in block.lines if _line_text(line).strip()
        ]
        drawings = [tuple(block.bbox) for block in page.blocks
                    if block.type == "drawing"]
        region = None
        for side in ("below", "above"):
            block = _table_block_lines(tuple(caption.bbox), page_lines, side,
                                       body_size)
            if _looks_like_table_block(block, drawings):
                region = _union_bbox([ln.bbox for ln in block])
                break
        if region is None:
            continue
        for seg in segs_by_page[caption.page]:
            if seg.kind not in ("body", "heading"):
                continue
            if _center_inside(seg.bbox, region):
                seg.kind = "figure_text"


# "Algorithm N" header line (number required, case-insensitive).
_ALGO_HEAD_RE = re.compile(r"^\s*algorithm\s+\d+\b", re.IGNORECASE)
# Punctuated header title "Algorithm 3: ..." / "Algorithm 3. ...".
_ALGO_TITLE_RE = re.compile(r"^\s*algorithm\s+\d+\s*[:.]", re.IGNORECASE)
# Numbered pseudocode step "12: ..." (colon excluded from list markers).
_ALGO_STEP_RE = re.compile(r"\b\d+\s*:\s")
# Strong pseudocode cues for the boxless header guard. Deliberately narrow
# (Require:/Ensure:/Input:/Output: labels or numbered steps): loose words
# like "for each" appear in ordinary prose and must not reclassify a body
# paragraph that merely mentions "Algorithm 1 ...".
_ALGO_GUARD_RE = re.compile(
    r"\b(?:require|ensure|input|output)\s*:|\b\d+\s*:\s",
    re.IGNORECASE,
)
_ALGO_BOX_SLOP = 2.0  # pt tolerance for the ruled-box enclosure test
_ALGO_INDENT_MIN = 2.0  # pt indent marking a pseudocode continuation


def _algo_box_for_line(
    line: Line, drawings: list[tuple]
) -> tuple | None:
    """Smallest drawing rect ruling an Algorithm header line.

    The rect must vertically enclose the line (with slop) and horizontally
    cover at least half of it, so thin horizontal rules above/below the
    block and vertical border strokes never qualify on their own.
    """
    lb = line.bbox
    min_overlap = 0.5 * max(lb[2] - lb[0], 1.0)
    best: tuple[float, tuple] | None = None
    for box in drawings:
        if min(lb[2], box[2]) - max(lb[0], box[0]) < min_overlap:
            continue
        if box[1] > lb[1] + _ALGO_BOX_SLOP or box[3] < lb[3] - _ALGO_BOX_SLOP:
            continue
        area = (box[2] - box[0]) * (box[3] - box[1])
        if best is None or area < best[0]:
            best = (area, box)
    return None if best is None else best[1]


def _mark_algorithm_span(page_segs: list[Segment], line: Line) -> None:
    """Boxless fallback: mark the header segment and its pseudocode run.

    Starting at the segment containing the Algorithm line, subsequent
    segments in reading order stay in the block while they are indented
    past the header's left edge or carry numbered-step/pseudocode cues;
    the first ordinary paragraph ends the block. A prose mention such as
    "Algorithm 1 summarizes ..." (no punctuated title, no cues) is left
    untouched.
    """
    cx, cy = _bbox_center(line.bbox)
    header = next(
        (seg for seg in page_segs
         if seg.kind in ("body", "heading")
         and seg.bbox[0] <= cx <= seg.bbox[2]
         and seg.bbox[1] <= cy <= seg.bbox[3]),
        None,
    )
    if header is None:
        return
    if not (_ALGO_TITLE_RE.match(header.text) or _line_is_bold(line)
            or _ALGO_GUARD_RE.search(header.text)):
        return
    header.kind = "algorithm"
    started = False
    for seg in page_segs:
        if seg is header:
            started = True
            continue
        if not started:
            continue
        if seg.kind == "formula":
            continue  # formulas inside the block already stay verbatim
        if seg.kind != "body":
            break
        below = seg.bbox[1] >= header.bbox[1] - 1.0
        h_over = (min(seg.bbox[2], header.bbox[2])
                  - max(seg.bbox[0], header.bbox[0]))
        if not below or h_over <= 0:
            break
        indented = seg.bbox[0] >= header.bbox[0] + _ALGO_INDENT_MIN
        if indented or _ALGO_STEP_RE.search(seg.text[:12]):
            seg.kind = "algorithm"
            continue
        break


def _apply_algorithm_kind(
    layout: DocumentLayout, segments: list[Segment]
) -> None:
    """Mark algorithm/pseudocode blocks as kind "algorithm".

    An "Algorithm N" line inside a ruled drawing box marks every body or
    heading segment whose center lies in the box; without a box the
    indentation/step heuristic (_mark_algorithm_span) bounds the block.
    """
    segs_by_page: dict[int, list[Segment]] = defaultdict(list)
    for seg in segments:
        segs_by_page[seg.page].append(seg)
    for page in layout.pages:
        page_segs = segs_by_page.get(page.number, [])
        if not page_segs:
            continue
        drawings = [tuple(block.bbox) for block in page.blocks
                    if block.type == "drawing"]
        for block in page.blocks:
            if block.type != "text":
                continue
            for line in block.lines:
                if not _ALGO_HEAD_RE.match(_line_text(line).strip()):
                    continue
                box = _algo_box_for_line(line, drawings)
                if box is None:
                    _mark_algorithm_span(page_segs, line)
                    continue
                for seg in page_segs:
                    if (seg.kind in ("body", "heading")
                            and _center_inside(seg.bbox, box)):
                        seg.kind = "algorithm"


# --- layout-model regions ------------------------------------------------
# When the layout model ran (layout_model.py), every page carries labelled
# regions. They are used twice: a paragraph never spans two regions (this
# separates a standfirst from its headline, chart labels from the body next
# to them, and paragraphs the gap rule glued together), and after all rule
# passes the region label decides what the rules could not see. Author
# detection, list markers and formula zones stay rule-based; the model only
# overrides a kind where its label is unambiguous.
_REGION_MIN_SHARE = 0.5  # share of a line/segment area inside its region
_REGION_UNCOVERED_SHARE = 0.2  # below this share a segment has no region
_FURNITURE_LABELS = frozenset({
    "header", "footer", "number", "header_image", "footer_image", "seal",
    "vertical_text",
})
_GRAPHIC_LABELS = frozenset({"image", "chart", "table"})
_TITLE_LABELS = frozenset({"doc_title", "paragraph_title"})
_CAPTION_LABELS = frozenset({"figure_title", "vision_footnote"})
_PROSE_LABELS = frozenset({"text", "abstract", "content", "footnote",
                           "aside_text"})
_GRAPHIC_TEXT_MAX_WORDS = 25  # longer text inside a graphic stays prose
_TITLE_MAX_WORDS = 20  # a "heading" longer than this is prose
_STRAY_LABEL_MAX_WORDS = 12  # short text outside every region = graphic label
_RESCUE_MIN_WORDS = 12  # figure_text this long inside a text region = prose
_RESCUE_MIN_SCORE = 0.8


def _area(bbox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _best_region(bbox, regions: list[Region]) -> tuple[int | None, float]:
    """(index, covered share) of the region overlapping bbox the most."""
    best, best_inter = None, 0.0
    for i, region in enumerate(regions):
        rb = region.bbox
        ix = min(bbox[2], rb[2]) - max(bbox[0], rb[0])
        iy = min(bbox[3], rb[3]) - max(bbox[1], rb[1])
        if ix <= 0 or iy <= 0:
            continue
        if ix * iy > best_inter:
            best, best_inter = i, ix * iy
    area = _area(bbox)
    return best, (best_inter / area if area > 0 else 0.0)


_REGION_ATTACH_GAP = 1.5  # line heights: stray line joins an adjacent region


def _region_key(line: Line, regions: list[Region]) -> int:
    """Index of the layout region a line belongs to (-1 = none).

    A line the model left outside every region joins the region directly
    above or below it (the model's box clipped the paragraph's last line).
    """
    if not regions:
        return -1
    idx, share = _best_region(line.bbox, regions)
    if share < _REGION_MIN_SHARE:
        idx = _adjacent_region(line.bbox, regions)
    return -1 if idx is None else idx


def _continues_across_regions(prev: Line, line: Line, prev_key: int,
                              key: int, regions: list[Region]) -> bool:
    """A sentence running on into the next text region (a model mis-split).

    The model sometimes cuts one paragraph into two text boxes; when the
    previous line leaves its sentence open and the next line starts in
    lower case, and both regions hold prose, the paragraph continues.
    """
    for k in (prev_key, key):
        if k >= 0 and regions[k].label not in _PROSE_LABELS:
            return False
    prev_text = _line_text(prev).rstrip()
    next_text = _line_text(line).lstrip()
    if not prev_text or not next_text or _ends_paragraph(prev_text):
        return False
    first = next((ch for ch in next_text if ch.isalpha()), "")
    return first.islower()


def _adjacent_region(bbox, regions: list[Region]) -> int | None:
    """Region vertically adjacent to a stray line (same x-range), if any."""
    reach = max(bbox[3] - bbox[1], 1.0) * _REGION_ATTACH_GAP
    best, best_gap = None, reach
    for i, region in enumerate(regions):
        rb = region.bbox
        h_over = min(bbox[2], rb[2]) - max(bbox[0], rb[0])
        if h_over < 0.5 * (bbox[2] - bbox[0]):
            continue
        gap = max(rb[1] - bbox[3], bbox[1] - rb[3], 0.0)
        if gap <= best_gap:
            best, best_gap = i, gap
    return best


_PROSE_MIN_STOPWORD_RATIO = 0.1


def _reads_as_prose(text: str) -> bool:
    """Long text with a normal share of function words (not a data dump)."""
    words = _WORD_RE.findall(text)
    if len(text.split()) <= _GRAPHIC_TEXT_MAX_WORDS or not words:
        return False
    return _zone_stopword_count(text) / len(words) >= _PROSE_MIN_STOPWORD_RATIO


def _layout_kind(seg: Segment, region: Region | None, share: float) -> str:
    """The kind a segment gets from its layout region (or its own kind)."""
    kind = seg.kind
    words = len(seg.text.split())
    if region is None or share < _REGION_UNCOVERED_SHARE:
        # Short text the model did not see as any region: a label drawn in
        # a graphic (donut-chart percentages, infographic callouts).
        if kind == "body" and words <= _STRAY_LABEL_MAX_WORDS:
            return "figure_text"
        return kind
    if share < _REGION_MIN_SHARE:
        return kind
    label = region.label
    if label in _FURNITURE_LABELS:
        return "header_footer" if kind in ("body", "heading", "caption") else kind
    if label in _GRAPHIC_LABELS:
        # Chart/table text is always graphic-internal; an "image" box is
        # sometimes drawn over real text, so there only text that does not
        # read as prose (short, or few function words: heatmap values,
        # axis ticks) counts as graphic.
        if kind in ("body", "heading") and (
                label != "image" or not _reads_as_prose(seg.text)):
            return "figure_text"
        return kind
    if label == "algorithm":
        return "algorithm" if kind in ("body", "heading") else kind
    if label == "reference_content":
        return "reference" if kind == "body" else kind
    if label in ("display_formula", "formula_number"):
        if kind == "body" and ("⟦EQ" in seg.text or words <= 6):
            return "formula"
        return kind
    if label in _TITLE_LABELS:
        if kind == "body" and words <= _TITLE_MAX_WORDS:
            return "heading"
        return kind
    if label in _CAPTION_LABELS:
        if kind not in ("body", "heading", "caption"):
            return kind
        # A lone panel letter ("b", "d") is a label, not a caption.
        return "figure_text" if len(seg.text.strip()) <= 2 else "caption"
    if label in _PROSE_LABELS:
        if kind == "heading" and words > _TITLE_MAX_WORDS:
            return "body"
        if (kind == "figure_text" and words >= _RESCUE_MIN_WORDS
                and region.score >= _RESCUE_MIN_SCORE):
            return "body"
    return kind


def _apply_layout_kinds(
    layout: DocumentLayout, segments: list[Segment]
) -> None:
    """Let the layout model's region labels settle segment kinds."""
    regions_by_page = {page.number: page.regions for page in layout.pages}
    for seg in segments:
        regions = regions_by_page.get(seg.page)
        if not regions or seg.kind in ("author", "header_footer"):
            continue
        idx, share = _best_region(seg.bbox, regions)
        region = regions[idx] if idx is not None else None
        seg.kind = _layout_kind(seg, region, share)


def build_segments(layout: DocumentLayout) -> list[Segment]:
    """Turn a DocumentLayout into translation-ready paragraph segments."""
    body_size = _body_font_size(layout)
    # Two passes: page flows first (row-merged column lines), so the
    # document-wide enumeration-sequence scan can fire "(n)" paragraph
    # markers across columns and pages before paragraph merging runs.
    flows = [_page_flow(page, body_size) for page in layout.pages]
    forced_markers = _sequence_marker_ids([flow for flow, _hf, _nb in flows])
    segments: list[Segment] = []
    for page, (flow, hf_lines, band_count) in zip(layout.pages, flows):
        segments.extend(
            _page_segments(page, body_size, flow, hf_lines, band_count,
                           forced_markers))
    segments = _merge_adjacent_headings(
        segments, {page.number: page.regions for page in layout.pages})
    _apply_author_kind(layout, segments, body_size)
    _apply_figure_text_kind(layout, segments)
    # Algorithm blocks claim their segments BEFORE the text-table pass:
    # pseudocode is made of short digit-heavy lines and a nearby "TABLE n"
    # caption used to swallow it as a text-table region.
    _apply_algorithm_kind(layout, segments)
    _apply_text_table_kind(layout, segments, body_size)
    _apply_reference_kind(segments)
    _apply_layout_kinds(layout, segments)
    return segments
