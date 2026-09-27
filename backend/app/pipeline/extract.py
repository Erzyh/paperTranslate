# PDF layout extraction based on PyMuPDF.
from __future__ import annotations

import pymupdf

from .models import Block, DocumentLayout, Line, PageLayout, Span

# Font-name markers that indicate math glyph runs. The list is grounded in
# fonts actually observed in real papers:
# - TeX Computer Modern math families (CMMI*, CMSY*, CMEX*),
# - MathTime families used by IEEE productions (RMTMI/RMTMIB -> "MTMI",
#   MTSYN/MTSYB -> "MTSY", plus the MTEX/BLEX extension fonts),
# - AMS symbol fonts (MSAM10, MSBM10),
# - anything advertising itself as a math font (CambriaMath, STIXMath,
#   *-Math, ...). Bare "STIX" is intentionally NOT a marker: STIXGeneral is
#   a body-text font in some journals and must never be masked.
#
# Measured against an ACL 2025 production (NimbusRomNo9L body text): its
# display equations use exactly CMMI10/8/9, CMSY10/6/8/9, CMEX10 (all
# covered above) PLUS CMR10/8/9 for roman math material ("(", ") = min",
# fraction digits). "CMR" is deliberately NOT a marker: Computer Modern
# Roman is the BODY font of plain-TeX papers, so masking it would destroy
# running prose there (same reasoning as STIX). Equations therefore mix
# masked math-font spans with literal text-font spans ("-score" words,
# CMR operators); grouping those into one untouched segment is the job of
# the font-agnostic formula-zone pass in segment.py, not of this list.
_MATH_FONT_MARKERS = (
    "CMMI", "CMSY", "CMEX",
    "MTMI", "MTSY", "MTEX", "BLEX",
    "MSAM", "MSBM",
    "MATH",
)


def _is_math_font(font_name: str) -> bool:
    """Return True when the span font looks like a math font."""
    upper = font_name.upper()
    return any(marker in upper for marker in _MATH_FONT_MARKERS)


# --- micro-token literalization -----------------------------------------
# Real papers render single punctuation/relation glyphs of running prose
# with math fonts: the decimal point of "0.949" (RMTMI "."), the "=" of
# "N = 5" (MTSYN), exponents like the "6" of "10^-6". Masking those as
# ⟦EQn⟧ makes the visible characters vanish from the translated text
# ("0 949") and floods the placeholder guard with micro tokens. A math-font
# span whose text is such a micro token (1-2 visible characters, all from
# the whitelist below) therefore keeps its literal text and flows with the
# surrounding prose instead of being masked. Letters (Greek variables, big
# operators with broken ToUnicode like BLEX "\x11") are never literalized.
# Hyphen/dash characters are excluded on purpose: a literal line-final "-"
# would trigger the hyphenation-restore join in segmentation.
_LITERAL_MAX_CHARS = 2
_LITERAL_CHARS = frozenset(
    "0123456789"
    ".,;:!?()[]{}|/\\%"
    "=<>+±−×÷·∗*≤≥≈≃≅∼~≠≲≳"
    "•◦▪"
)


def math_span_literal(text: str) -> str | None:
    """Literal replacement for a micro math-font span, or None to mask.

    Returns ``text`` unchanged (whitespace preserved) when its visible core
    is 1-2 characters long and consists only of whitelisted punctuation,
    digits and relation/operator symbols; None means "mask as ⟦EQn⟧".
    """
    core = "".join(text.split())
    if 0 < len(core) <= _LITERAL_MAX_CHARS \
            and all(ch in _LITERAL_CHARS for ch in core):
        return text
    return None


def is_masked_math_span(text: str, font_name: str) -> bool:
    """True when analyze_pdf masks this span as an ⟦EQn⟧ placeholder.

    Shared with retypeset: only masked runs take part in snapshot
    restoration and inline-flow matching; literalized micro tokens are
    ordinary text (redacted and re-typeset with the translation).
    """
    return bool(text.strip()) and _is_math_font(font_name) \
        and math_span_literal(text) is None


def analyze_pdf(path: str) -> DocumentLayout:
    """Read a PDF and return its block/line/span layout.

    Spans rendered with a math font keep their bbox/font/size/flags but their
    text is replaced by an ``(EQn)`` placeholder (U+27E6/U+27E7 brackets) so
    that downstream translation never touches formula content. Micro tokens
    (single punctuation/digit/relation glyphs such as the decimal point of
    "0.949" or the "=" of "N = 5", see math_span_literal) are exempt: they
    keep their literal text and stay part of the prose.
    """
    doc = pymupdf.open(path)
    eq_counter = 0
    pages: list[PageLayout] = []
    try:
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            raw = page.get_text("dict")
            blocks: list[Block] = []
            for raw_block in raw.get("blocks", []):
                if raw_block.get("type") == 1:  # raster image block
                    blocks.append(
                        Block(type="image", bbox=tuple(raw_block["bbox"]), lines=[])
                    )
                    continue
                lines: list[Line] = []
                for raw_line in raw_block.get("lines", []):
                    spans: list[Span] = []
                    for raw_span in raw_line.get("spans", []):
                        text = raw_span["text"]
                        if is_masked_math_span(text, raw_span["font"]):
                            text = f"⟦EQ{eq_counter}⟧"
                            eq_counter += 1
                        spans.append(
                            Span(
                                text=text,
                                bbox=tuple(raw_span["bbox"]),
                                font=raw_span["font"],
                                size=raw_span["size"],
                                flags=raw_span["flags"],
                            )
                        )
                    if spans:
                        lines.append(Line(spans=spans, bbox=tuple(raw_line["bbox"])))
                if lines:
                    blocks.append(
                        Block(type="text", bbox=tuple(raw_block["bbox"]), lines=lines)
                    )
            for drawing in page.get_drawings():
                rect = drawing["rect"]
                blocks.append(
                    Block(
                        type="drawing",
                        bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                        lines=[],
                    )
                )
            pages.append(
                PageLayout(
                    number=pno,
                    width=page.rect.width,
                    height=page.rect.height,
                    blocks=blocks,
                )
            )
    finally:
        doc.close()
    return DocumentLayout(pages=pages)
