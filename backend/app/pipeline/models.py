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
class PageLayout:
    number: int
    width: float
    height: float
    blocks: list[Block]


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
