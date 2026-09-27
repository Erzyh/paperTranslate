# Pipeline orchestration: analyze -> segment -> translate -> retypeset.
from __future__ import annotations

import inspect
import re
from typing import Callable

from .extract import analyze_pdf
from .models import RenderReport, Segment
from .retypeset import render_translated_pdf
from .segment import UNTRANSLATED_KINDS, build_segments, split_list_marker
from .translate import Translator, mask, unmask

ProgressCB = Callable[[dict], None]

_PREVIEW_LEN = 80
# Matches the translator-side context cap (translate._CONTEXT_MAX_CHARS).
_CONTEXT_LEN = 400
# Character budget for the glossary sample sent to build_glossary(): spread
# over the whole paper (headings, opening, each section's first paragraph,
# captions) so terms introduced late are covered too.
_GLOSSARY_SAMPLE_MAX_CHARS = 6000
_GLOSSARY_OPENING_CHARS = 2000   # first-page body (abstract + intro opening)
_GLOSSARY_SECTION_CHARS = 400    # first body paragraph after each heading
_GLOSSARY_CAPTION_CHARS = 200
# Document brief (title + abstract excerpt) handed to the translator.
_BRIEF_ABSTRACT_MIN_CHARS = 150
_BRIEF_ABSTRACT_CHARS = 500

# --- mid-sentence segment starts ----------------------------------------
# A body paragraph continuing a sentence broken at a column/page boundary
# starts with a lowercase word ("truth, and a pairwise ...") or with a
# leftover math/number tail ("≈ −0.05. The mean rater ..."). Such segments
# get the translator's continuation instruction so the model does not open
# a fresh sentence (S2). Placeholder tokens are ignored for the check.
_ANY_TOKEN_RE = re.compile(r"⟦(?:EQ|M)\d+⟧")
_LEADING_NONWORD_RE = re.compile(r"^[^A-Za-z가-힣]+")


def is_continuation_start(text: str) -> bool:
    """True when a segment text starts mid-sentence (see block comment)."""
    head = _ANY_TOKEN_RE.sub(" ", text).strip()
    if not head:
        return False
    if head[0].isalpha():
        return head[0].islower()
    match = _LEADING_NONWORD_RE.match(head)
    run = match.group(0)
    rest = head[match.end():]
    # A lowercase word after the leading non-word run continues a sentence
    # ("0.949) that did not ...", "= 5 raters, before ...") — unless the run
    # is a bare integer, which legitimately opens sentences ("2026 marks
    # the corpus cutoff year").
    if rest and rest[0].islower() and not run.strip().isdigit():
        return True
    # A leading math/number run that closes a sentence ("≈ −0.05. The ...")
    # is the tail of the previous paragraph's last sentence.
    return "." in run


def _preview(text: str) -> str:
    flat = " ".join(text.split())
    return flat[:_PREVIEW_LEN]


def _glossary_sample(translatable: list[Segment]) -> str:
    """Glossary sample spread over the paper, most informative parts first.

    Order: all headings, the first-page body opening (abstract/intro), the
    first body paragraph after every heading, then figure/table captions.
    Capped at _GLOSSARY_SAMPLE_MAX_CHARS.
    """
    parts = [seg.text for seg in translatable if seg.kind == "heading"]

    opening, size = [], 0
    for seg in translatable:
        if seg.kind == "body" and seg.page == 0 and size < _GLOSSARY_OPENING_CHARS:
            opening.append(seg.text[:_GLOSSARY_OPENING_CHARS - size])
            size += len(opening[-1])
    parts += opening

    after_heading = False
    for seg in translatable:
        if seg.kind == "heading":
            after_heading = True
        elif seg.kind == "body" and after_heading:
            if seg.page > 0:  # page-0 body is already in the opening
                parts.append(seg.text[:_GLOSSARY_SECTION_CHARS])
            after_heading = False

    parts += [seg.text[:_GLOSSARY_CAPTION_CHARS]
              for seg in translatable if seg.kind == "caption"]
    return "\n".join(parts)[:_GLOSSARY_SAMPLE_MAX_CHARS]


def _document_brief(translatable: list[Segment]) -> str:
    """Title and abstract excerpt: tells the model the paper's field."""
    title = next((s.text for s in translatable if s.kind == "heading"), "")
    abstract = next((s.text for s in translatable
                     if s.kind == "body" and s.page == 0
                     and len(s.text) >= _BRIEF_ABSTRACT_MIN_CHARS), "")
    lines = []
    if title:
        lines.append("제목: " + " ".join(title.split()))
    if abstract:
        lines.append("초록 일부: " + " ".join(abstract.split())[:_BRIEF_ABSTRACT_CHARS])
    return "\n".join(lines)


def _prepare_glossary(translator: Translator, translatable: list[Segment]) -> None:
    """Build and attach a document glossary when the translator supports it.

    Translators without build_glossary (e.g. the stub) are skipped. Glossary
    extraction is best-effort: every failure is swallowed so it can never
    abort the translation pipeline.
    """
    build = getattr(translator, "build_glossary", None)
    if not callable(build):
        return
    sample = _glossary_sample(translatable)
    if not sample.strip():
        return
    try:
        translator.glossary = build(sample) or {}
    except Exception:
        pass


def run_pipeline(
    src_path: str,
    out_path: str,
    translator: Translator,
    on_progress: ProgressCB | None = None,
    segment_cb: Callable[[Segment, str | None], None] | None = None,
    on_phase: Callable[[str], None] | None = None,
) -> RenderReport:
    """Run the full translation pipeline.

    Segments whose kind is in UNTRANSLATED_KINDS (header_footer, formula,
    author, reference, figure_text, algorithm) are kept in the original
    language: they are neither translated nor redacted. List-item segments
    keep their leading marker deterministically (stripped before, re-prefixed
    after translation). ``on_progress`` receives one dict per translated
    segment with keys: segment_id, done, total, page, num_pages,
    source_preview, translated_preview. ``segment_cb(segment, translated)`` is
    invoked for every segment; ``translated`` is None for skipped untranslated
    segments. For translated segments a keyword ``status`` is passed:
    "fallback" when the translator gave up on placeholder repair and returned
    the source text (translator.last_fallback), "done" otherwise.

    Before translating, a document glossary is built from headings and the
    first-page body opening when the translator exposes ``build_glossary``;
    failures there never abort the pipeline. Unmarked body segments whose
    text starts mid-sentence (is_continuation_start) are translated with
    ``continuation=True`` when the translator's signature accepts the flag.

    ``on_phase(name)`` is called when the run enters each stage, in order:
    "analyzing" (layout + segmentation), "glossary", "translating",
    "rendering". Batch UIs use it to show where each paper is.
    """
    def phase(name: str) -> None:
        if on_phase is not None:
            on_phase(name)

    phase("analyzing")
    layout = analyze_pdf(src_path)
    segments = build_segments(layout)
    num_pages = len(layout.pages)

    translatable = [seg for seg in segments
                    if seg.kind not in UNTRANSLATED_KINDS]
    total = len(translatable)
    translations: dict[str, str] = {}

    # Document glossary pass (before any translation; best-effort, no-op for
    # translators without build_glossary such as the stub).
    phase("glossary")
    _prepare_glossary(translator, translatable)
    if hasattr(translator, "document_brief"):
        translator.document_brief = _document_brief(translatable)

    # Translators predating the continuation flag (e.g. test fakes with a
    # (text, context) signature) keep working: the kwarg is only passed when
    # the translate signature accepts it.
    try:
        params = inspect.signature(translator.translate).parameters
    except (TypeError, ValueError):
        params = {}
    supports_continuation = "continuation" in params
    supports_previous = "previous_translation" in params

    for segment in segments:
        if segment.kind in UNTRANSLATED_KINDS and segment_cb is not None:
            segment_cb(segment, None)

    phase("translating")
    previous_source: str | None = None
    # How the previous paragraph was translated (None after a fallback, which
    # is English and would teach the model nothing).
    previous_translation: str | None = None
    for done, segment in enumerate(translatable, start=1):
        # List markers ("•", "1)", "(a)", masked bullet glyphs) are never
        # entrusted to the model: the marker prefix (marker + its original
        # separator) is stripped before masking and deterministically
        # re-prefixed afterwards, so every list item keeps its marker at the
        # line start and a translator fallback reproduces the source text.
        marker, item_text = split_list_marker(segment.text)
        masked, mapping = mask(item_text)
        context = previous_source[:_CONTEXT_LEN] if previous_source else None
        # Mid-sentence starts (S2): only unmarked body paragraphs can be the
        # tail of a sentence broken at a column/page boundary.
        continuation = (
            segment.kind == "body" and not marker
            and is_continuation_start(item_text)
        )
        kwargs = {}
        if supports_continuation:
            kwargs["continuation"] = continuation
        if supports_previous:
            kwargs["previous_translation"] = previous_translation
        raw_translated = translator.translate(masked, context=context, **kwargs)
        translated = unmask(raw_translated, mapping)
        if marker:
            translated = marker + translated.lstrip()
        # Translators without placeholder repair (e.g. the stub) have no
        # last_fallback attribute; treat them as always successful.
        fallback = bool(getattr(translator, "last_fallback", False))
        translations[segment.id] = translated
        previous_source = segment.text
        previous_translation = None if fallback else translated
        if segment_cb is not None:
            segment_cb(segment, translated,
                       status="fallback" if fallback else "done")
        if on_progress is not None:
            on_progress(
                {
                    "segment_id": segment.id,
                    "done": done,
                    "total": total,
                    "page": segment.page,
                    "num_pages": num_pages,
                    "source_preview": _preview(segment.text),
                    "translated_preview": _preview(translated),
                }
            )

    phase("rendering")
    return render_translated_pdf(src_path, segments, translations, out_path)
