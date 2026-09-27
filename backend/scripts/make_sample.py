# Generate a two-column academic-paper style sample PDF for pipeline tests.
#
# Contents: large title, IEEE-style author block (names + affiliation +
# e-mails), a full-width funding/editor note and a full-width abstract
# (IEEE Access-style front matter), two-column English body (3+ paragraphs
# per column with hyphen line breaks), one raster image with caption, fake
# equation lines rendered with a font whose BaseFont is renamed to CMMI10
# (so the math-font heuristic in extract.py triggers), a numbered display
# equation "(1)", an all-caps "VII. CONCLUSION" heading immediately followed
# by body text, a References section with [1]/[2] entries, and page
# headers/footers. These reproduce the real-paper symptoms (A) formula,
# (B) author, (C) reference, (D) heading, (E) full-width front matter
# above a two-column body (band-based column detection regression), and
# (F) IEEE Access inline front-matter labels: a bold "ABSTRACT" label glued
# to the abstract body on the same line, and a bold "INDEX TERMS" label
# followed by the terms on one line, (G) a mixed math/regular-font display
# formula whose bbox overlaps the paragraph right above it (whole-bbox
# protection target), and (H) a body paragraph carrying two inline math
# runs (inline-flow insertion target), (I) a three-item bullet list and a
# "1)/2)" enumeration whose items must each become their own segment with
# the marker preserved, (J) axis-label text inside the Figure 1 region
# (kind "figure_text", never translated), (K) a ruled Algorithm 1 box
# with numbered pseudocode (kind "algorithm", never translated), (L) a
# paragraph-style "(1)/(2)" enumeration whose continuation lines are flush
# left (no hanging indent) and must still be absorbed into their items,
# and (M) a display equation typeset only with regular italic/roman faces
# (sub/superscript fragment lines, right-aligned "(9)" number) that only
# the formula-zone banding can classify as a formula.
#
# Pages 3-4 (no header/footer) reproduce the boundary/micro-token symptoms:
# (N) a far-apart paragraph-style "(1)/(2)/(3)" enumeration whose items sit
# in different columns AND on different pages (only the document-wide
# consecutive-sequence rule can split them), (O) decimal points typeset
# with the math font inside running prose ("0.949 and 0.227", micro-token
# literalization target), (P) a visual text row split into fragment lines
# with offset baselines by an inline math relation, following a hyphenated
# line (visual-row reassembly target), (Q) a sentence broken at the
# page 3 -> page 4 boundary whose tail paragraph starts lowercase
# (continuation-prompt target), and (R) an ACL-style fraction equation
# typeset ONLY with regular text faces — hyphen-compound words, big
# parentheses, a denominator line and a right-aligned "(8)" number below
# the fraction — that only the font-agnostic zone rules can classify as
# one formula segment. The References section lives at the end of page 4
# so the after-the-heading reference rule keeps matching the real paper
# shape (bibliography last).
#
# For the assets feature the body additionally carries in-text mentions
# ("as shown in Fig. 1", "see Table 1", a "[2]" citation) and page 2 holds a
# drawn table grid below a "TABLE 1." caption (figure/table region and
# mention detection targets).
from __future__ import annotations

import os
import sys

import pymupdf

PAGE_W, PAGE_H = 612.0, 792.0  # US Letter
MARGIN = 54.0
COL_W = 240.0
GUTTER = 24.0
LEFT_X = MARGIN
RIGHT_X = MARGIN + COL_W + GUTTER  # 318

BODY_SIZE = 9.0
BODY_STEP = 11.0
PARA_STEP = 24.0
HEADING_SIZE = 12.0
CAPTION_SIZE = 8.5
HF_SIZE = 8.0
FUNDING_SIZE = 8.5

MATH_FONT_FILE = r"C:\Windows\Fonts\seguisym.ttf"
MATH_FONT_ALIAS = "mathf"
# Font carrying the "•" bullet glyph (the base-14 helv has no extractable
# bullet); machines without it degrade to an ASCII "*" marker.
LIST_FONT_FILE = r"C:\Windows\Fonts\arial.ttf"
LIST_FONT_ALIAS = "listf"

# (I) bullet list: (marker line, indented continuation lines) per item; the
# 11pt line step between items means only the marker rule can split them.
# Every item spans two lines so the translated text fits its own bbox.
BULLET_ITEMS = [
    ["Delta timing preserves the source rhythm",
     "of every simultaneous playback lane."],
    ["Spatial lanes anchor blocks to columns",
     "and keep the gutter separation intact."],
    ["Note topology survives the translation",
     "and keeps anchors attached to pages."],
]
# (I) numbered enumeration: two adjacent "n)" items (adjacency requirement),
# each with an indented continuation line.
ENUM_ITEMS = [
    ["1) Extract every span from the source",
     "layout with its font and position."],
    ["2) Group the spans into column-aware",
     "paragraph segments for translation."],
]
ENUM_INDENT = 10.0  # hanging indent of enumeration continuation lines
# (L) paragraph-style "(n)" enumeration WITHOUT a hanging indent: the marker
# line carries a first-line paragraph indent and the continuation lines are
# flush left, so only the flush-continuation absorption rule can keep each
# full sentence inside its item segment (real-paper p13 "(5)/(6)" shape).
# The markers deliberately reuse the real paper's "(5)/(6)" numbers: "(1)"
# would collide with the leak assertions for the numbered display equation.
PARA_ENUM_ITEMS = [
    ["(5) Proxy label dependence limits the",
     "scope of the reported study conclusions."],
    ["(6) Sample size constrains the external",
     "validity of the reported correlations."],
]
PARA_ENUM_INDENT = 10.0  # first-line indent of the marker lines
# Flush paragraph following the enumeration (separated by a paragraph gap):
# it must never glue onto the last item.
PARA_ENUM_FOLLOW = [
    "The remaining sections quantify these",
    "constraints against the corpus data.",
]
# (M) display equation typeset entirely in REGULAR italic/roman faces (no
# math font): variables and sub/superscript fragment lines plus a
# right-aligned "(9)" equation number. Line-level formula detection cannot
# see it (no math-font span, "offset" is a 5+ char lowercase word), so only
# the formula-zone banding keeps its fragments out of the body text
# (real-paper p6 equation "(5)" shape).
ITALIC_EQ_UPPER = "Nhit"     # superscript sum limit line
ITALIC_EQ_NUMER = "1"        # fraction numerator line
ITALIC_EQ_MAIN = "MAE = y(i) yref(i)"  # italic main line
ITALIC_EQ_NUMBER = "(9)"     # right-aligned equation number
ITALIC_EQ_SUB = "offset"     # subscript line under the main line
ITALIC_EQ_LOWER = "i=1"      # lower sum limit line
ITALIC_EQ_SMALL = 6.5        # sub/superscript font size
# (N) far-apart paragraph-style "(1)/(2)/(3)" enumeration: item (1) closes
# the left column of page 3, item (2) opens its right column and item (3)
# sits on page 4, so the ±8-line neighbor rule can never fire them — only
# the document-wide consecutive-sequence rule splits the items.
SEQ_INTRO = [
    "Three chart statistics summarize the",
    "corpus geometry used by every ranking",
    "experiment reported in this section and",
    "motivate the item split described next.",
]
SEQ_ITEM_1 = [
    "(1) Delta lane density measures the",
    "local pressure that simultaneous notes",
    "exert on a single hand during the fast",
    "sections, and it grows with every added",
    "lane until the chart saturates the span.",
]
SEQ_ITEM_2 = [
    "(2) Spatial migration length captures",
    "how far consecutive notes travel across",
    "the lane axis between adjacent steps,",
    "which penalizes the wide jumps forcing",
    "full wrist relocation between beats.",
]
SEQ_ITEM_3 = [
    "(3) Vertical release timing counts the",
    "holds whose release lands off the grid,",
    "because a late release blocks the next",
    "press on the same lane column.",
]
SEQ_FOLLOW = [
    "The remaining paragraphs relate these",
    "statistics to the observed rank drifts.",
]
# (S) lowercase Roman-numeral enumeration "(i)/(ii)": two adjacent items,
# each with a hanging-indent continuation line — the markers must split the
# items exactly like the "1)/2)" enumeration does (real IEEE "(i)" shape).
ROMAN_ITEMS = [
    ["(i) Anchor extraction is deterministic",
     "across repeated pipeline executions."],
    ["(ii) Rendered output keeps the source",
     "page count for every tested corpus."],
]
# (T) text-only table: "TABLE 2." caption above a caption-less text grid
# (no ruled lines, no images) — the x-aligned short numeric cells are the
# only structural signal, so only the text-table detector can exclude the
# cells from translation (real IEEE TABLE IV-VII shape).
TEXT_TABLE_CAPTION = [
    "TABLE 2. Text-only per-domain accuracy",
    "figures without any ruled grid lines.",
]
TEXT_TABLE_ROWS = [
    ("Domain", "Acc", "Err"),
    ("Physics", "0.91", "0.09"),
    ("Biology", "0.87", "0.13"),
    ("Layout", "0.93", "0.07"),
]
TEXT_TABLE_COL_XS = [0.0, 100.0, 160.0]  # x offsets of the three columns
TEXT_TABLE_ROW_STEP = 12.0
# (R) ACL-style display equation typeset ONLY with regular text faces (no
# math font anywhere, unlike (A)/(G)): "PCC = min", big text-face
# parentheses around a fraction whose numerator carries hyphen-compound
# words ("L-score + M-score"), a denominator "2", and a "(8)" equation
# number below the fraction, right-aligned to the column's PROSE right
# edge (justified papers align it to the column edge; the sample's ragged
# lines are narrower than COL_W, so the prose edge is the honest analogue
# and keeps the number inside the clustered column extent). Neither the
# math-font list nor italic detection can see any of it — only the
# font-agnostic zone rules (right-aligned "(n)" number seed + hyphen-
# compound fragment absorption) turn it into one formula segment (real
# ACL paper "(5)" PCC equation shape).
ACL_EQ_MAIN = "PCC = min"           # roman main fragment (text face)
ACL_EQ_NUMER = "L-score + M-score"  # numerator with hyphen compounds
ACL_EQ_DENOM = "2"                  # fraction denominator line
ACL_EQ_NUMBER = "(8)"               # right-aligned equation number
ACL_EQ_PAREN_SIZE = 22.0            # big parens: text face at display size
# Main-line indent: must exceed the visual-row flow tolerance
# (max(12pt, 8% of the column width)) so the fragments stay separate
# extraction lines instead of joining the prose flow.
ACL_EQ_INDENT = 15.0
ACL_EQ_GAP = 2.0                    # horizontal gap between fragments
# (Q) sentence broken at the page 3 -> page 4 boundary: the head paragraph
# ends mid-sentence and the tail paragraph opens lowercase on page 4.
CONT_HEAD = [
    "The appendix aggregates the raw counts",
    "behind these statistics and lists the",
]
CONT_TAIL = [
    "remaining charts with their ranks kept",
    "intact for the follow-up audit study.",
]
# (O) decimal points typeset with the math font inside running prose: the
# "." spans are 1-character math-font micro tokens that must stay literal
# ("0.949 and 0.227") instead of masking into ⟦EQn⟧ placeholders.
DECIMAL_HEAD = "The two proxy metrics reach accuracy"
DECIMAL_PARTS = ["levels of 0", "949 and 0", "227 on the corpus."]
# (P) visual row split by an inline math relation: the row after a
# hyphenated line is drawn as two prose fragments with slightly different
# baselines plus a math-font "=" between them, reproducing the extraction
# shape "dation ... (N" / "=" / "5 raters) ..." that the visual-row
# reassembly must re-join (hyphen restore included).
SPLIT_HYPHEN_LINE = "External raters conducted an extra vali-"
SPLIT_LEFT = "dation pass on a shared pilot (N"
SPLIT_RIGHT = "5 raters) before release."
SPLIT_GAP = 4.0  # gap around the drawn "=" relation
# Baseline lifts of the "=" and right fragments: large enough that the
# right fragment's bbox top precedes the left fragment's in the (y0, x0)
# line sort — the real-paper shape where the naive sort scrambles the
# reading order — while keeping the vertical overlap far above the row
# threshold so the reassembly re-joins them.
SPLIT_MATH_LIFT = 4.0
SPLIT_RIGHT_LIFT = 4.6
# (J) figure-internal axis labels: (text, dx, dy) offsets inside the
# Figure 1 image rect; classified figure_text and kept in English.
AXIS_LABELS = [
    ("Accuracy", 12.0, 31.0),
    ("Epochs", 96.0, 106.0),
]
# (K) ruled algorithm box: bold header + numbered pseudocode steps.
ALGO_LINES = [
    "Algorithm 1: Greedy Column Packing",
    "1: for each block b in the page do",
    "2: assign b to the nearest lane",
    "3: end for",
]
ALGO_INDENT = 8.0  # pseudocode indent below the header line
ALGO_BOX_RIGHT_INSET = 26.0  # box right edge: x + COL_W - inset

TITLE_LINES = [
    "Layout-Preserving Neural Translation",
    "of Two-Column Academic Documents",
]
# IEEE-style author block: names, affiliation, e-mails (symptom B target).
AUTHOR_BLOCK = [
    ("Alice Kim, Bob Lee, and Carol Park", 10.0),
    ("Institute for Document Intelligence, Seoul National Campus", 9.0),
    ("{alice.kim, bob.lee, carol.park}@docint.example.org", 9.0),
]
HEADERS = [
    "Proceedings of the Imaginary Conference on Document Intelligence (ICDI 2026)",
    "ICDI 2026 — Preprint",
]
# Full-width funding/editor note, IEEE Access front-matter style (symptom E:
# the real paper's editor-metadata sentence used to cross-merge into the
# two-column introduction below it).
FUNDING_NOTE = [
    "This work was supported in part by the Imaginary Research Council under"
    " Grant IRC-2026-042 and in part by the Document",
    "Intelligence Initiative. The associate editor coordinating the review of"
    " this manuscript was Prof. Erasmus Placeholder.",
]
# Full-width abstract paragraph spanning both body columns (symptom E).
# The bold all-caps label sits inline before the body on the first line,
# IEEE Access style (symptom F: label used to enter the translator glued to
# the abstract body).
ABSTRACT_LABEL = "ABSTRACT"
ABSTRACT_BODY_LINES = [
    "Document translation systems that ignore page geometry produce"
    " outputs that are hard to compare",
    "with their sources. This paper describes a pipeline that preserves"
    " two-column layouts, restores hyphenated",
    "words, and keeps figures, equations, and references anchored to their"
    " original positions. Experiments on",
    "synthetic and real papers show that rendered translations stay within"
    " the source bounding boxes.",
]
# Single-line INDEX TERMS block: bold label + terms on one line (symptom F,
# single-line split path).
INDEX_TERMS_LABEL = "INDEX TERMS"
INDEX_TERMS_BODY = (
    "Document structure analysis, hyphen restoration, layout preservation,"
    " neural translation."
)
LABEL_GAP = 4.0  # gap between an inline label and the following body text

# Hand-wrapped body lines; a trailing "-" marks an intentional hyphen break.
P1_LEFT = [
    ("heading", "1  Introduction"),
    ("para", [
        "Recent advances in neural machine transla-",
        "tion have enabled high quality document",
        "conversion across many languages [12]. Yet",
        "most systems discard the visual structure",
        "of the source and emit plain text streams.",
    ]),
    ("para", [
        "When scholarly PDFs are processed naively,",
        "two-column layouts collapse into a single",
        "unstructured flow, and figure references",
        "lose their anchors within the page.",
    ]),
    ("equation", "∑ α ‖x‖² ≈ ∫ f(x) dx − λθ"),
    ("para", [
        "We argue that a translation system should",
        "treat the page as a first-class object. Our",
        "pipeline detects columns, restores hyphen-",
        "ated words, and rewrites each paragraph in",
        "place while keeping figures untouched.",
    ]),
    # (I) "1)/2)" enumeration: two adjacent single-line items.
    ("enum", ENUM_ITEMS),
    # (L) paragraph-style "(1)/(2)" enumeration with flush continuations.
    ("para_enum", PARA_ENUM_ITEMS),
    ("para", PARA_ENUM_FOLLOW),
]
P1_RIGHT = [
    ("para", [
        "The proposed method analyzes every text",
        "block, clusters horizontal positions, and",
        "derives a column map for each page. The",
        "map guides both segmentation and the",
        "final typesetting stage.",
    ]),
    ("image", None),
    ("caption", [
        "Figure 1: Synthetic diagram used to verify",
        "that raster content survives retypesetting.",
    ]),
    ("para", [
        "Masked entities such as citations, URLs,",
        "and inline equations are carefully pro-",
        "tected during decoding, then restored",
        "verbatim in the output document.",
    ]),
    ("para", [
        "A rendering report records every overflow",
        "and every scale factor, as shown in Fig. 1,",
        "allowing the user interface to flag",
        "fragile regions early in the process.",
    ]),
    # (M) italic-face display equation with a right-aligned "(9)" number.
    ("italic_equation", None),
]
P2_LEFT = [
    ("heading", "2  Experimental Setup"),
    ("para", [
        "Our corpus contains twelve open access",
        "papers drawn from a public mirror at",
        "https://example.org/dataset and covers",
        "three scientific domains with distinct",
        "layout conventions and font families.",
    ]),
    ("para", [
        "Baselines follow the classic page seg-",
        "mentation heuristics reported in [3, 7],",
        "which operate on projection profiles and",
        "ignore reading order entirely.",
    ]),
    ("para", [
        "Every run is repeated five times to bound",
        "the variance of the extractor; see Table 1",
        "for the per-domain figures, since even",
        "small drifts can cause irreversible reflow",
        "errors in the final rendered document.",
    ]),
    # Numbered display equation, IEEE style (symptom A target).
    ("equation_numbered", ("Ω = ∑ β ‖x‖ ≈ ∫ g(x) dx", "(1)")),
    # IEEE-style table: caption line above a drawn grid (assets target: the
    # table region is the union of the grid's drawing blocks below the
    # caption).
    ("caption", [
        "TABLE 1. Segmentation accuracy across",
        "three document domains.",
    ]),
    ("table", None),
    # (G) mixed display formula whose bbox overlaps the paragraph right above
    # it (symptom A target: redacting the paragraph bbox used to erase the
    # formula's regular-font glyphs "= 0.42 (3)", because only math-font
    # runs were snapshot-restored).
    ("para", [
        "The bullet criteria below compare the",
        "resulting score against a threshold",
        "that the ablation study fixed early.",
    ]),
    ("overlap_equation", ("∑ γ ‖x‖", "= 0.42 (3)")),
    # (H) body paragraph with two inline formulas (symptom B target: the
    # runs must flow with the retypeset Korean sentence instead of being
    # restored at their original page coordinates).
    ("inline_para", [
        ("A subsequent sensitivity analysis with ", "ρ < 0.01", ""),
        ("confirms the effect there, and a paired", None, ""),
        ("test with ", "δ ≥ 3σ", " supports the conclusion."),
    ]),
    # (I) bullet list: three items at plain 11pt line steps; splitting must
    # come from the marker rule, not from vertical gaps.
    ("bullets", BULLET_ITEMS),
]
P2_RIGHT = [
    ("heading", "3  Results and Discussion"),
    ("para", [
        "Across all documents the column detector",
        "recovers the expected structure, and the",
        "hyphen restorer rejoins broken tokens",
        "without corrupting genuine compounds.",
    ]),
    ("equation", "θ⋆ = argmin ∑ ℓ(x, y; θ) + λ‖θ‖"),
    ("para", [
        "Translated pages preserve figure counts",
        "exactly, while textual overflow remains",
        "below five percent of segments under the",
        "default scaling policy [2].",
    ]),
    ("para", [
        "A verbatim placeholder audit confirms that",
        "masked spans reappear unchanged, which we",
        "consider essential for scholarly integrity",
        "and reproducible citation graphs.",
    ]),
    # (K) ruled Algorithm 1 box with numbered pseudocode: every segment
    # inside the drawn rect is kind "algorithm" and stays untranslated.
    ("algorithm", ALGO_LINES),
    # All-caps section title at body size, immediately followed by body text
    # with a normal line step (symptom D target: the heading used to merge
    # with the following paragraph).
    ("heading_caps", "VII. CONCLUSION"),
    ("tight_para", [
        "This research presented a layout preserving",
        "translation pipeline and verified that head-",
        "ings stay separated from following text.",
    ]),
]
P3_LEFT = [
    ("para", SEQ_INTRO),
    # (N) item (1): sole "(n)" marker of this column.
    ("para_enum", [SEQ_ITEM_1]),
    # (R) text-face-only ACL-style fraction equation with "(8)" number.
    ("acl_equation", None),
]
P3_RIGHT = [
    # (N) item (2): opens the right column, far from item (1).
    ("para_enum", [SEQ_ITEM_2]),
    # (O) math-font decimal points inside prose.
    ("decimal_para", None),
    # (P) hyphenated line + split visual row with an inline "=" relation.
    ("split_para", None),
    # (S) "(i)/(ii)" Roman enumeration: adjacent two-line hanging items.
    ("enum", ROMAN_ITEMS),
    # (Q) head of the page-boundary sentence (ends mid-sentence).
    ("para", CONT_HEAD),
]
P4_LEFT = [
    # (Q) lowercase tail of the sentence broken at the page boundary.
    ("para", CONT_TAIL),
    # (N) item (3): continues the (1)/(2) sequence across the page break.
    ("para_enum", [SEQ_ITEM_3]),
    ("para", SEQ_FOLLOW),
    # (U) theorem-environment paragraphs at plain 11pt line steps: the
    # "Definition 2 ..." and "Proof." lines start right below the preceding
    # prose with NO paragraph gap, so only the theorem-head rule can split
    # them off the previous paragraph (real-paper Definition/Theorem/Proof
    # shape). Each header line continues into its statement body on the
    # same segment (IEEE run-in style).
    ("theorem_paras", [
        ["The gate invariants hold for every",
         "session trace generated by the engine."],
        ["Definition 2 (Transition). A transition",
         "maps responses to committed states."],
        ["Proof. The base state satisfies the",
         "invariant by construction of the rules."],
    ]),
    # (T) text-only table: caption + unruled text grid (figure_text target).
    ("caption", TEXT_TABLE_CAPTION),
    ("text_table", TEXT_TABLE_ROWS),
    # References section: bold title + [n] entries (symptom C target). The
    # bibliography sits on the LAST page so the "everything after the
    # References heading is a reference" rule keeps matching the sample.
    ("refhead", "REFERENCES"),
    ("refitem", [
        "[1] A. Author and B. Writer, \"Layout aware",
        "translation of scholarly documents,\" IEEE",
        "Trans. Doc. Anal., vol. 12, no. 3, pp. 45-67,",
        "2020.",
    ]),
    ("refitem", [
        "[2] C. Kim and D. Lee, \"Column detection for",
        "academic typesetting,\" J. Doc. Eng., vol. 8,",
        "no. 1, pp. 10-24, 2019.",
    ]),
]


def _math_text_length(text: str, fontsize: float) -> float:
    """Width of a math-font run (helv approximation without the font file)."""
    if os.path.exists(MATH_FONT_FILE):
        return pymupdf.Font(fontfile=MATH_FONT_FILE).text_length(
            text, fontsize=fontsize)
    return pymupdf.get_text_length(text, fontname="helv", fontsize=fontsize)


def _p3_left_prose_width() -> float:
    """Widest prose line width of the page-3 left column (x-relative).

    The (R) equation's "(8)" number right-aligns to this edge, mirroring a
    justified paper column where equation numbers align with the text edge.
    """
    widths = [
        pymupdf.get_text_length(line, fontname="helv", fontsize=BODY_SIZE)
        for line in SEQ_INTRO
    ]
    widths += [
        (PARA_ENUM_INDENT if i == 0 else 0.0)
        + pymupdf.get_text_length(line, fontname="helv", fontsize=BODY_SIZE)
        for i, line in enumerate(SEQ_ITEM_1)
    ]
    return max(widths)


def _bullet_marker_width() -> float:
    """Width of the drawn bullet marker ("• " or the ASCII fallback "* ")."""
    if os.path.exists(LIST_FONT_FILE):
        return pymupdf.Font(fontfile=LIST_FONT_FILE).text_length(
            "• ", fontsize=BODY_SIZE)
    return pymupdf.get_text_length("* ", fontname="helv", fontsize=BODY_SIZE)


def _draw_bullet_marker(page: pymupdf.Page, x: float, y: float) -> float:
    """Draw one bullet marker at (x, y); returns its advance width."""
    if os.path.exists(LIST_FONT_FILE):
        page.insert_text((x, y), "• ", fontname=LIST_FONT_ALIAS,
                         fontfile=LIST_FONT_FILE, fontsize=BODY_SIZE)
    else:  # degraded ASCII fallback on machines without the font
        page.insert_text((x, y), "* ", fontname="helv", fontsize=BODY_SIZE)
    return _bullet_marker_width()


def _check_widths() -> None:
    """Fail fast if any hand-wrapped line exceeds its column width."""
    for column in (P1_LEFT, P1_RIGHT, P2_LEFT, P2_RIGHT,
                   P3_LEFT, P3_RIGHT, P4_LEFT):
        for kind, payload in column:
            if kind in ("para", "tight_para", "refitem"):
                for line in payload:
                    width = pymupdf.get_text_length(
                        line, fontname="helv", fontsize=BODY_SIZE
                    )
                    assert width <= COL_W, f"line too wide ({width:.1f}pt): {line}"
            elif kind == "theorem_paras":
                for chunk in payload:
                    for line in chunk:
                        width = pymupdf.get_text_length(
                            line, fontname="helv", fontsize=BODY_SIZE
                        )
                        assert width <= COL_W, f"line too wide: {line}"
            elif kind == "enum":
                for item in payload:
                    for i, line in enumerate(item):
                        indent = 0.0 if i == 0 else ENUM_INDENT
                        width = indent + pymupdf.get_text_length(
                            line, fontname="helv", fontsize=BODY_SIZE
                        )
                        assert width <= COL_W, f"enum line too wide: {line}"
            elif kind == "para_enum":
                for item in payload:
                    for i, line in enumerate(item):
                        indent = PARA_ENUM_INDENT if i == 0 else 0.0
                        width = indent + pymupdf.get_text_length(
                            line, fontname="helv", fontsize=BODY_SIZE
                        )
                        assert width <= COL_W, f"para_enum line too wide: {line}"
            elif kind == "italic_equation":
                width = 40.0 + pymupdf.get_text_length(
                    ITALIC_EQ_MAIN, fontname="tiit", fontsize=BODY_SIZE
                )
                assert width <= COL_W, "italic equation main line too wide"
            elif kind == "acl_equation":
                prose_w = _p3_left_prose_width()
                paren_w = pymupdf.get_text_length(
                    "(", fontname="helv", fontsize=ACL_EQ_PAREN_SIZE)
                close_x = (
                    ACL_EQ_INDENT
                    + pymupdf.get_text_length(ACL_EQ_MAIN, fontname="helv",
                                              fontsize=BODY_SIZE)
                    + ACL_EQ_GAP + paren_w + ACL_EQ_GAP
                    + pymupdf.get_text_length(ACL_EQ_NUMER, fontname="helv",
                                              fontsize=BODY_SIZE)
                    + ACL_EQ_GAP
                )
                # Containment: every fragment must START inside the column
                # extent defined by the prose lines, so x0 clustering keeps
                # the pieces in the left column (no bridge to the right one).
                assert close_x <= prose_w - 2.0, "acl equation too wide"
                # The main line's indent must exceed the visual-row flow
                # tolerance so the fragments stay separate extraction lines.
                assert ACL_EQ_INDENT > max(12.0, prose_w * 0.08) + 1.0, \
                    "acl equation indent below the flow-row tolerance"
            elif kind == "decimal_para":
                width = pymupdf.get_text_length(
                    "".join(DECIMAL_PARTS), fontname="helv", fontsize=BODY_SIZE
                ) + 2.0 * _math_text_length(".", BODY_SIZE)
                assert width <= COL_W, "decimal paragraph line too wide"
            elif kind == "split_para":
                width = (
                    pymupdf.get_text_length(SPLIT_LEFT + SPLIT_RIGHT,
                                            fontname="helv", fontsize=BODY_SIZE)
                    + 2.0 * SPLIT_GAP + _math_text_length("=", BODY_SIZE)
                )
                assert width <= COL_W, "split row too wide"
            elif kind == "bullets":
                marker_w = _bullet_marker_width()
                for item in payload:
                    for line in item:
                        width = marker_w + pymupdf.get_text_length(
                            line, fontname="helv", fontsize=BODY_SIZE
                        )
                        assert width <= COL_W, f"bullet line too wide: {line}"
            elif kind == "algorithm":
                box_w = COL_W - ALGO_BOX_RIGHT_INSET - 4.0
                for i, line in enumerate(payload):
                    fname = "hebo" if i == 0 else "helv"
                    indent = 0.0 if i == 0 else ALGO_INDENT
                    width = indent + pymupdf.get_text_length(
                        line, fontname=fname, fontsize=BODY_SIZE
                    )
                    assert width <= box_w - 4.0, f"algorithm line too wide: {line}"
            elif kind == "inline_para":
                for prefix, math_text, suffix in payload:
                    width = pymupdf.get_text_length(
                        prefix + suffix, fontname="helv", fontsize=BODY_SIZE
                    )
                    if math_text:
                        width += _math_text_length(math_text, BODY_SIZE)
                    assert width <= COL_W, f"inline line too wide: {prefix!r}"
            elif kind == "text_table":
                for row in payload:
                    for col_x, cell in zip(TEXT_TABLE_COL_XS, row):
                        width = col_x + pymupdf.get_text_length(
                            cell, fontname="helv", fontsize=BODY_SIZE
                        )
                        assert width <= COL_W, f"table cell too wide: {cell}"
            elif kind == "caption":
                for line in payload:
                    width = pymupdf.get_text_length(
                        line, fontname="helv", fontsize=CAPTION_SIZE
                    )
                    assert width <= COL_W, f"caption too wide: {line}"
    full_width = PAGE_W - 2.0 * MARGIN
    for lines, size in ((FUNDING_NOTE, FUNDING_SIZE),
                        (ABSTRACT_BODY_LINES[1:], BODY_SIZE)):
        for line in lines:
            width = pymupdf.get_text_length(line, fontname="helv", fontsize=size)
            assert width <= full_width, f"front-matter line too wide: {line}"
            # Full-width classification needs > 60% of the content width.
            assert width > full_width * 0.62, f"front-matter line too narrow: {line}"
    # Inline-label lines: bold label + gap + body must stay full-width.
    for label, body in ((ABSTRACT_LABEL, ABSTRACT_BODY_LINES[0]),
                        (INDEX_TERMS_LABEL, INDEX_TERMS_BODY)):
        width = (
            pymupdf.get_text_length(label, fontname="hebo", fontsize=BODY_SIZE)
            + LABEL_GAP
            + pymupdf.get_text_length(body, fontname="helv", fontsize=BODY_SIZE)
        )
        assert width <= full_width, f"label line too wide: {label}"
        assert width > full_width * 0.62, f"label line too narrow: {label}"


def _make_figure_pixmap() -> pymupdf.Pixmap:
    """Draw a small vector scene and rasterize it into a Pixmap."""
    tmp = pymupdf.open()
    page = tmp.new_page(width=240, height=120)
    page.draw_rect(pymupdf.Rect(0, 0, 240, 120), color=None, fill=(0.93, 0.95, 0.98))
    page.draw_rect(pymupdf.Rect(14, 18, 92, 100), color=(0.1, 0.2, 0.5),
                   fill=(0.31, 0.51, 0.74), width=1.5)
    page.draw_circle(pymupdf.Point(160, 58), 34, color=(0.5, 0.15, 0.1),
                     fill=(0.85, 0.44, 0.32))
    page.draw_line(pymupdf.Point(92, 58), pymupdf.Point(126, 58),
                   color=(0.2, 0.2, 0.2), width=2)
    pix = page.get_pixmap(dpi=144)
    tmp.close()
    return pix


def _draw_header_footer(page: pymupdf.Page, header: str, footer: str) -> None:
    header_w = pymupdf.get_text_length(header, fontname="helv", fontsize=HF_SIZE)
    page.insert_text(
        ((PAGE_W - header_w) / 2.0, 30.0), header, fontname="helv", fontsize=HF_SIZE
    )
    footer_w = pymupdf.get_text_length(footer, fontname="helv", fontsize=HF_SIZE)
    page.insert_text(
        ((PAGE_W - footer_w) / 2.0, 775.0), footer, fontname="helv", fontsize=HF_SIZE
    )


def _draw_column(page: pymupdf.Page, x: float, start_y: float,
                 items: list[tuple], pix: pymupdf.Pixmap | None) -> float:
    """Draw one column's content; returns the final baseline y."""
    y = start_y
    for kind, payload in items:
        if kind == "heading":
            y += 6.0
            page.insert_text((x, y), payload, fontname="hebo", fontsize=HEADING_SIZE)
            y += PARA_STEP
        elif kind == "para":
            for line in payload:
                page.insert_text((x, y), line, fontname="helv", fontsize=BODY_SIZE)
                y += BODY_STEP
            y += PARA_STEP - BODY_STEP
        elif kind == "caption":
            for line in payload:
                page.insert_text((x, y), line, fontname="helv", fontsize=CAPTION_SIZE)
                y += BODY_STEP - 1.0
            y += PARA_STEP - BODY_STEP
        elif kind == "equation":
            y += 8.0
            if os.path.exists(MATH_FONT_FILE):
                page.insert_text((x + 24.0, y), payload, fontname=MATH_FONT_ALIAS,
                                 fontfile=MATH_FONT_FILE, fontsize=10.0)
            else:  # degraded fallback on machines without the symbol font
                page.insert_text((x + 24.0, y), "sum a x = int f(x) dx",
                                 fontname="helv", fontsize=10.0)
            y += PARA_STEP + 8.0
        elif kind == "equation_numbered":
            # Display equation with an IEEE-style "(n)" number at the column
            # right edge, sharing the baseline with the math run.
            formula, number = payload
            y += 8.0
            if os.path.exists(MATH_FONT_FILE):
                page.insert_text((x + 24.0, y), formula, fontname=MATH_FONT_ALIAS,
                                 fontfile=MATH_FONT_FILE, fontsize=10.0)
            else:  # degraded fallback on machines without the symbol font
                page.insert_text((x + 24.0, y), "O = sum b x = int g(x) dx",
                                 fontname="helv", fontsize=10.0)
            # Keep the number inside the column's text x-extent (the sample's
            # ragged-right lines end near x+160, unlike justified IEEE text).
            page.insert_text((x + 150.0, y), number,
                             fontname="helv", fontsize=10.0)
            y += PARA_STEP + 8.0
        elif kind == "overlap_equation":
            # Display formula mixing math-font and regular-font glyphs whose
            # bbox overlaps the paragraph right above (symptom A): the
            # formula baseline sits only 9pt below the last body baseline,
            # so the two segment bboxes intersect and redacting the body
            # paragraph would erase the formula's glyph tops without
            # whole-bbox protection.
            formula, tail = payload
            y = y - PARA_STEP + 9.0
            if os.path.exists(MATH_FONT_FILE):
                page.insert_text((x + 24.0, y), formula,
                                 fontname=MATH_FONT_ALIAS,
                                 fontfile=MATH_FONT_FILE, fontsize=10.0)
            else:  # degraded fallback on machines without the symbol font
                page.insert_text((x + 24.0, y), "sum g x",
                                 fontname="helv", fontsize=10.0)
            tail_x = x + 24.0 + _math_text_length(formula, 10.0) + 6.0
            page.insert_text((tail_x, y), tail, fontname="helv", fontsize=10.0)
            y += PARA_STEP + 8.0
        elif kind == "inline_para":
            # Body lines carrying inline math runs (prefix, math, suffix):
            # the math span sits between prose spans on the same baseline,
            # so the line stays a body line and the paragraph keeps kind
            # "body" (symptom B target).
            for prefix, math_text, suffix in payload:
                cx = x
                if prefix:
                    page.insert_text((cx, y), prefix, fontname="helv",
                                     fontsize=BODY_SIZE)
                    cx += pymupdf.get_text_length(prefix, fontname="helv",
                                                  fontsize=BODY_SIZE)
                if math_text:
                    if os.path.exists(MATH_FONT_FILE):
                        page.insert_text((cx, y), math_text,
                                         fontname=MATH_FONT_ALIAS,
                                         fontfile=MATH_FONT_FILE,
                                         fontsize=BODY_SIZE)
                    else:  # degraded fallback without the symbol font
                        page.insert_text((cx, y), "p<1", fontname="helv",
                                         fontsize=BODY_SIZE)
                    cx += _math_text_length(math_text, BODY_SIZE)
                if suffix:
                    page.insert_text((cx, y), suffix, fontname="helv",
                                     fontsize=BODY_SIZE)
                y += BODY_STEP
            y += PARA_STEP - BODY_STEP
        elif kind == "heading_caps":
            # All-caps title at body size followed by a near-normal line step
            # (gap stays far below the paragraph-break threshold), so only
            # the heading-pattern rule can separate it from the next
            # paragraph (symptom D). The extra 3pt keeps the insert_text
            # bboxes from overlapping across the segment boundary.
            page.insert_text((x, y), payload, fontname="helv", fontsize=BODY_SIZE)
            y += BODY_STEP + 3.0
        elif kind == "tight_para":
            # Paragraph starting right below the previous line (no extra gap).
            for line in payload:
                page.insert_text((x, y), line, fontname="helv", fontsize=BODY_SIZE)
                y += BODY_STEP
            y += PARA_STEP - BODY_STEP
        elif kind == "theorem_paras":
            # (U) theorem-environment chunks at near-body line steps: the
            # 3pt extra between chunks stays far below the paragraph-break
            # threshold (only the theorem-head rule can split them) while
            # keeping the rendered Korean lines from colliding across the
            # forced segment boundary.
            for i, chunk in enumerate(payload):
                if i:
                    y += 3.0
                for line in chunk:
                    page.insert_text((x, y), line, fontname="helv",
                                     fontsize=BODY_SIZE)
                    y += BODY_STEP
            y += PARA_STEP - BODY_STEP
        elif kind == "bullets":
            # (I) bullet list: marker + first line, continuation lines
            # indented to the text edge, tight 11pt steps between items.
            for item in payload:
                marker_w = _draw_bullet_marker(page, x, y)
                page.insert_text((x + marker_w, y), item[0], fontname="helv",
                                 fontsize=BODY_SIZE)
                y += BODY_STEP
                for cont in item[1:]:
                    page.insert_text((x + marker_w, y), cont, fontname="helv",
                                     fontsize=BODY_SIZE)
                    y += BODY_STEP
            y += PARA_STEP - BODY_STEP
        elif kind == "enum":
            # (I) "n)" enumeration: adjacent two-line items with a hanging
            # indent on the continuation line.
            for item in payload:
                page.insert_text((x, y), item[0], fontname="helv",
                                 fontsize=BODY_SIZE)
                y += BODY_STEP
                for cont in item[1:]:
                    page.insert_text((x + ENUM_INDENT, y), cont,
                                     fontname="helv", fontsize=BODY_SIZE)
                    y += BODY_STEP
            y += PARA_STEP - BODY_STEP
        elif kind == "para_enum":
            # (L) paragraph-style "(n)" enumeration: indented marker line,
            # flush-left continuation lines, plain 11pt steps throughout.
            for item in payload:
                page.insert_text((x + PARA_ENUM_INDENT, y), item[0],
                                 fontname="helv", fontsize=BODY_SIZE)
                y += BODY_STEP
                for cont in item[1:]:
                    page.insert_text((x, y), cont, fontname="helv",
                                     fontsize=BODY_SIZE)
                    y += BODY_STEP
            y += PARA_STEP - BODY_STEP
        elif kind == "decimal_para":
            # (O) prose line whose decimal points are 1-char math-font
            # spans ("0.949 and 0.227"); the rest is regular body text.
            page.insert_text((x, y), DECIMAL_HEAD, fontname="helv",
                             fontsize=BODY_SIZE)
            y += BODY_STEP
            cx = x
            for i, part in enumerate(DECIMAL_PARTS):
                page.insert_text((cx, y), part, fontname="helv",
                                 fontsize=BODY_SIZE)
                cx += pymupdf.get_text_length(part, fontname="helv",
                                              fontsize=BODY_SIZE)
                if i < len(DECIMAL_PARTS) - 1:
                    if os.path.exists(MATH_FONT_FILE):
                        page.insert_text((cx, y), ".",
                                         fontname=MATH_FONT_ALIAS,
                                         fontfile=MATH_FONT_FILE,
                                         fontsize=BODY_SIZE)
                    else:  # degraded fallback without the symbol font
                        page.insert_text((cx, y), ".", fontname="helv",
                                         fontsize=BODY_SIZE)
                    cx += _math_text_length(".", BODY_SIZE)
            y += PARA_STEP
        elif kind == "split_para":
            # (P) hyphenated line, then one visual row drawn as two prose
            # fragments with lifted baselines around a math-font "=". The
            # fragments are drawn right-to-left (reverse content-stream
            # order, the shape TeX overlays produce), so extraction emits
            # separate line entries that the visual-row reassembly must
            # re-join in x order.
            page.insert_text((x, y), SPLIT_HYPHEN_LINE, fontname="helv",
                             fontsize=BODY_SIZE)
            y += BODY_STEP
            left_w = pymupdf.get_text_length(SPLIT_LEFT, fontname="helv",
                                             fontsize=BODY_SIZE)
            eq_x = x + left_w + SPLIT_GAP
            right_x = eq_x + _math_text_length("=", BODY_SIZE) + SPLIT_GAP
            page.insert_text((right_x, y - SPLIT_RIGHT_LIFT), SPLIT_RIGHT,
                             fontname="helv", fontsize=BODY_SIZE)
            page.insert_text((x, y), SPLIT_LEFT, fontname="helv",
                             fontsize=BODY_SIZE)
            if os.path.exists(MATH_FONT_FILE):
                page.insert_text((eq_x, y - SPLIT_MATH_LIFT), "=",
                                 fontname=MATH_FONT_ALIAS,
                                 fontfile=MATH_FONT_FILE, fontsize=BODY_SIZE)
            else:  # degraded fallback without the symbol font
                page.insert_text((eq_x, y - SPLIT_MATH_LIFT), "=",
                                 fontname="helv", fontsize=BODY_SIZE)
            y += PARA_STEP
        elif kind == "italic_equation":
            # (M) display equation drawn only with regular Times faces:
            # superscript limit, numerator, italic main line, right-aligned
            # "(9)" number, subscript "offset" and lower limit. Every
            # fragment line chains to the zone via overlap or a <=3pt gap.
            y += 14.0
            page.insert_text((x + 150.0, y - 17.0), ITALIC_EQ_UPPER,
                             fontname="tiit", fontsize=ITALIC_EQ_SMALL)
            page.insert_text((x + 120.0, y - 9.0), ITALIC_EQ_NUMER,
                             fontname="tiro", fontsize=BODY_SIZE)
            page.insert_text((x + 40.0, y), ITALIC_EQ_MAIN,
                             fontname="tiit", fontsize=BODY_SIZE)
            num_w = pymupdf.get_text_length(ITALIC_EQ_NUMBER, fontname="tiro",
                                            fontsize=BODY_SIZE)
            page.insert_text((x + COL_W - num_w, y), ITALIC_EQ_NUMBER,
                             fontname="tiro", fontsize=BODY_SIZE)
            page.insert_text((x + 78.0, y + 6.0), ITALIC_EQ_SUB,
                             fontname="tiit", fontsize=ITALIC_EQ_SMALL)
            page.insert_text((x + 148.0, y + 14.0), ITALIC_EQ_LOWER,
                             fontname="tiit", fontsize=ITALIC_EQ_SMALL)
            y += PARA_STEP + 14.0
        elif kind == "acl_equation":
            # (R) text-face-only fraction equation: "PCC = min", big parens
            # spanning the fraction, "L-score + M-score" numerator, "2"
            # denominator and a "(8)" number BELOW the fraction, right-
            # aligned to the prose edge (real ACL "(5)" layout: the number
            # sits on its own extraction line under the equation band).
            y += 16.0
            right_edge = x + _p3_left_prose_width()
            main_x = x + ACL_EQ_INDENT
            page.insert_text((main_x, y), ACL_EQ_MAIN, fontname="helv",
                             fontsize=BODY_SIZE)
            open_x = main_x + pymupdf.get_text_length(
                ACL_EQ_MAIN, fontname="helv", fontsize=BODY_SIZE) + ACL_EQ_GAP
            page.insert_text((open_x, y + 5.0), "(", fontname="helv",
                             fontsize=ACL_EQ_PAREN_SIZE)
            numer_x = open_x + pymupdf.get_text_length(
                "(", fontname="helv", fontsize=ACL_EQ_PAREN_SIZE) + ACL_EQ_GAP
            page.insert_text((numer_x, y - 4.0), ACL_EQ_NUMER,
                             fontname="helv", fontsize=BODY_SIZE)
            numer_w = pymupdf.get_text_length(ACL_EQ_NUMER, fontname="helv",
                                              fontsize=BODY_SIZE)
            denom_w = pymupdf.get_text_length(ACL_EQ_DENOM, fontname="helv",
                                              fontsize=BODY_SIZE)
            page.insert_text((numer_x + (numer_w - denom_w) / 2.0, y + 8.0),
                             ACL_EQ_DENOM, fontname="helv", fontsize=BODY_SIZE)
            page.insert_text((numer_x + numer_w + ACL_EQ_GAP, y + 5.0), ")",
                             fontname="helv", fontsize=ACL_EQ_PAREN_SIZE)
            num_w = pymupdf.get_text_length(ACL_EQ_NUMBER, fontname="helv",
                                            fontsize=BODY_SIZE)
            page.insert_text((right_edge - num_w, y + 14.0), ACL_EQ_NUMBER,
                             fontname="helv", fontsize=BODY_SIZE)
            y += PARA_STEP + 26.0
        elif kind == "algorithm":
            # (K) ruled algorithm box: bold header + numbered pseudocode
            # inside one drawn rect (the rect vertically encloses the
            # header line, which is the box-detection cue).
            box = pymupdf.Rect(
                x - 4.0, y - BODY_SIZE - 3.0,
                x + COL_W - ALGO_BOX_RIGHT_INSET,
                y + BODY_STEP * (len(payload) - 1) + 4.0,
            )
            page.draw_rect(box, color=(0.15, 0.15, 0.15), width=0.8)
            for i, line in enumerate(payload):
                fname = "hebo" if i == 0 else "helv"
                indent = 0.0 if i == 0 else ALGO_INDENT
                page.insert_text((x + indent, y), line, fontname=fname,
                                 fontsize=BODY_SIZE)
                y += BODY_STEP
            y += PARA_STEP - BODY_STEP + 8.0
        elif kind == "refhead":
            page.insert_text((x, y), payload, fontname="hebo", fontsize=BODY_SIZE)
            y += BODY_STEP + 3.0
        elif kind == "refitem":
            for line in payload:
                page.insert_text((x, y), line, fontname="helv", fontsize=BODY_SIZE)
                y += BODY_STEP
            y += PARA_STEP - BODY_STEP
        elif kind == "text_table":
            # (T) text-only table: three x-aligned text columns, no rules.
            for row in payload:
                for col_x, cell in zip(TEXT_TABLE_COL_XS, row):
                    page.insert_text((x + col_x, y), cell, fontname="helv",
                                     fontsize=BODY_SIZE)
                y += TEXT_TABLE_ROW_STEP
            y += PARA_STEP - TEXT_TABLE_ROW_STEP
        elif kind == "table":
            # Ruled grid drawn with vector paths only (no raster image, so
            # page-level image counts stay unchanged): outer rect plus inner
            # row/column lines, all becoming "drawing" blocks in extraction.
            table_w, row_h, n_rows, n_cols = 200.0, 16.0, 4, 3
            top = y - 6.0
            rect = pymupdf.Rect(x, top, x + table_w, top + row_h * n_rows)
            page.draw_rect(rect, color=(0.15, 0.15, 0.15), width=0.8)
            for r in range(1, n_rows):
                yy = top + row_h * r
                page.draw_line(pymupdf.Point(x, yy),
                               pymupdf.Point(x + table_w, yy),
                               color=(0.15, 0.15, 0.15), width=0.5)
            for c in range(1, n_cols):
                xx = x + table_w * c / n_cols
                page.draw_line(pymupdf.Point(xx, top),
                               pymupdf.Point(xx, top + row_h * n_rows),
                               color=(0.15, 0.15, 0.15), width=0.5)
            y = rect.y1 + 18.0
        elif kind == "image":
            rect = pymupdf.Rect(x, y - 4.0, x + COL_W, y - 4.0 + 120.0)
            page.insert_image(rect, pixmap=pix)
            # (J) figure-internal axis labels: text whose center lies inside
            # the figure region must become kind "figure_text" and stay in
            # the source language.
            for label, dx, dy in AXIS_LABELS:
                page.insert_text((rect.x0 + dx, rect.y0 + dy), label,
                                 fontname="helv", fontsize=7.5)
            y = rect.y1 + 16.0
    return y


def _rename_math_font(doc: pymupdf.Document) -> None:
    """Rename the embedded symbol font's BaseFont to CMMI10.

    Rendering and text extraction are unaffected (font program and ToUnicode
    stay intact); only the reported font name changes, which makes the sample
    exercise the math-font heuristic exactly like a TeX-generated paper.
    """
    seen: set[int] = set()
    for pno in range(doc.page_count):
        for entry in doc.load_page(pno).get_fonts(full=True):
            xref, refname = entry[0], entry[4]
            if refname != MATH_FONT_ALIAS or xref in seen:
                continue
            seen.add(xref)
            doc.xref_set_key(xref, "BaseFont", "/CMMI10")
            kind, value = doc.xref_get_key(xref, "DescendantFonts")
            if kind == "array":
                inner = value.strip("[] ")
                if inner.endswith("0 R"):
                    doc.xref_set_key(int(inner.split()[0]), "BaseFont", "/CMMI10")


def make_sample(out_path: str) -> str:
    """Create the sample PDF at out_path and return the path."""
    _check_widths()
    doc = pymupdf.open()

    # --- page 1 ---
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    _draw_header_footer(page, HEADERS[0], "1")
    y = 88.0
    for line in TITLE_LINES:
        width = pymupdf.get_text_length(line, fontname="hebo", fontsize=20.0)
        page.insert_text(((PAGE_W - width) / 2.0, y), line, fontname="hebo",
                         fontsize=20.0)
        y += 24.0
    # Author block: 11pt line steps keep the three lines merged into a single
    # front-matter paragraph (names + affiliation + e-mails, symptom B).
    author_y = 146.0
    for author_line, size in AUTHOR_BLOCK:
        line_w = pymupdf.get_text_length(author_line, fontname="helv",
                                         fontsize=size)
        page.insert_text(((PAGE_W - line_w) / 2.0, author_y), author_line,
                         fontname="helv", fontsize=size)
        author_y += 11.0
    # Full-width funding/editor note and abstract between the author block
    # and the two-column body (symptom E: IEEE Access front matter).
    note_y = 196.0
    for line in FUNDING_NOTE:
        page.insert_text((MARGIN, note_y), line, fontname="helv",
                         fontsize=FUNDING_SIZE)
        note_y += 10.0
    # Abstract: bold inline label + body on the first line (symptom F).
    abstract_y = 224.0
    label_w = pymupdf.get_text_length(ABSTRACT_LABEL, fontname="hebo",
                                      fontsize=BODY_SIZE)
    page.insert_text((MARGIN, abstract_y), ABSTRACT_LABEL, fontname="hebo",
                     fontsize=BODY_SIZE)
    page.insert_text((MARGIN + label_w + LABEL_GAP, abstract_y),
                     ABSTRACT_BODY_LINES[0], fontname="helv", fontsize=BODY_SIZE)
    abstract_y += BODY_STEP
    for line in ABSTRACT_BODY_LINES[1:]:
        page.insert_text((MARGIN, abstract_y), line, fontname="helv",
                         fontsize=BODY_SIZE)
        abstract_y += BODY_STEP
    # INDEX TERMS: bold inline label + terms on one line (symptom F). The
    # vertical gap above (>= 2 body line steps) keeps it out of the abstract
    # paragraph even if extraction merges both into one block.
    terms_y = 279.0
    terms_label_w = pymupdf.get_text_length(INDEX_TERMS_LABEL, fontname="hebo",
                                            fontsize=BODY_SIZE)
    page.insert_text((MARGIN, terms_y), INDEX_TERMS_LABEL, fontname="hebo",
                     fontsize=BODY_SIZE)
    page.insert_text((MARGIN + terms_label_w + LABEL_GAP, terms_y),
                     INDEX_TERMS_BODY, fontname="helv", fontsize=BODY_SIZE)
    page.draw_line(pymupdf.Point(MARGIN, 288.0), pymupdf.Point(PAGE_W - MARGIN, 288.0),
                   color=(0.25, 0.25, 0.25), width=0.8)
    pix = _make_figure_pixmap()
    _draw_column(page, LEFT_X, 300.0, P1_LEFT, None)
    _draw_column(page, RIGHT_X, 300.0, P1_RIGHT, pix)

    # --- page 2 ---
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    _draw_header_footer(page, HEADERS[1], "2")
    _draw_column(page, LEFT_X, 92.0, P2_LEFT, None)
    _draw_column(page, RIGHT_X, 92.0, P2_RIGHT, None)

    # --- pages 3-4 (no header/footer): boundary/micro-token symptoms ---
    # (N) far-apart "(1)/(2)/(3)" enumeration across columns and the page
    # break, (O) math-font decimals, (P) split visual row, (Q) sentence
    # broken at the page boundary, plus the References section (last page).
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    _draw_column(page, LEFT_X, 92.0, P3_LEFT, None)
    _draw_column(page, RIGHT_X, 92.0, P3_RIGHT, None)

    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    _draw_column(page, LEFT_X, 92.0, P4_LEFT, None)

    _rename_math_font(doc)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    doc.save(out_path, garbage=3, deflate=True)
    doc.close()
    return out_path


if __name__ == "__main__":
    default_out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "sample.pdf",
    )
    target = sys.argv[1] if len(sys.argv) > 1 else default_out
    print(f"샘플 PDF 생성: {make_sample(target)}")
