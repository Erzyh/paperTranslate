# Dataclasses shared by the pipeline modules.
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Span:
    text: str
    bbox: tuple[float, float, float, float]
    font: str
    size: float
    flags: int


@dataclass
class Line:
    spans: list[Span]
    bbox: tuple


@dataclass
class Block:
    type: str  # "text" | "image" | "drawing"
    bbox: tuple
    lines: list[Line] = field(default_factory=list)  # text blocks only


@dataclass
class Region:
    """A page region found by the layout model (see layout_model.py)."""
    label: str  # "text" | "paragraph_title" | "chart" | "footer" | ...
    score: float
    bbox: tuple[float, float, float, float]  # PDF points
    order: int | None = None  # reading order; None for floating regions


@dataclass
class PageLayout:
    number: int
    width: float
    height: float
    blocks: list[Block]
    # Layout-model regions; empty when the model is unavailable/disabled.
    regions: list[Region] = field(default_factory=list)
    # True when the page had no text layer and its lines came from OCR.
    ocr: bool = False


@dataclass
class DocumentLayout:
    pages: list[PageLayout]


@dataclass
class Segment:
    id: str  # "p{page}_s{idx}"
    page: int  # 0-based
    column: int  # 0-based, 0 for single-column pages
    bbox: tuple[float, float, float, float]  # union bbox of the paragraph
    text: str  # merged text with hyphenation restored
    kind: str  # "body" | "heading" | "caption" | "header_footer"
    font_size: float  # representative font size


@dataclass
class RenderReport:
    overflow_segments: list[str]
    scaled_segments: dict[str, float]  # segment id -> applied scale factor
    # segment id -> rect the translation actually occupies in the output
    # (paragraphs flow within their column, so it can differ from the bbox)
    placed_rects: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict)
