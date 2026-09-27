# End-to-end tests: CLI run + verification criteria 1(a)(b)(c), runner callbacks.
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pymupdf
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]

# Distinctive body phrases planted by make_sample; none may survive retypesetting.
BODY_SENTINELS = [
    "Recent advances in neural machine",
    "two-column layouts collapse",
    "irreversible reflow",
    "verbatim placeholder audit",
    "Layout-Preserving Neural Translation",
    "Figure 1: Synthetic diagram",
    # full-width abstract is body: it must be translated away
    "systems that ignore page geometry",
    # paragraphs around the formula-preservation constructs (G)/(H)
    "bullet criteria below",
    "subsequent sensitivity analysis",
    # pages 3-4 boundary/micro-token constructs (N)/(P)/(Q)
    "Delta lane density measures",
    "External raters conducted",
    "remaining charts with their ranks",
]
HEADER_PHRASE = "Imaginary Conference"  # header_footer stays untranslated
MATH_FONT_AVAILABLE = Path(r"C:\Windows\Fonts\seguisym.ttf").exists()


@pytest.fixture(scope="session")
def translated_pdf(sample_pdf, tmp_path_factory) -> str:
    out_path = str(tmp_path_factory.mktemp("out") / "out.pdf")
    # Pin the translator explicitly: an ambient PAPERTRANSLATE_TRANSLATOR=ollama
    # in the shell must never make tests call Ollama (absolute rule).
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               PAPERTRANSLATE_TRANSLATOR="stub")
    result = subprocess.run(
        [sys.executable, "-m", "app.cli", sample_pdf, out_path],
        cwd=str(BACKEND_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stdout}\n{result.stderr}"
    assert "완료" in result.stdout
    assert os.path.isfile(out_path)
    return out_path


def _page_texts(path: str) -> list[str]:
    doc = pymupdf.open(path)
    try:
        return [doc.load_page(i).get_text() for i in range(doc.page_count)]
    finally:
        doc.close()


def test_criterion_a_image_count_preserved(sample_pdf, translated_pdf):
    src = pymupdf.open(sample_pdf)
    out = pymupdf.open(translated_pdf)
    try:
        assert out.page_count == src.page_count
        for pno in range(src.page_count):
            src_images = len(src.load_page(pno).get_images(full=True))
            out_images = len(out.load_page(pno).get_images(full=True))
            # Formula preservation may legitimately ADD images (inline-flow
            # runs, whole-bbox formula snapshots) but must never lose one.
            assert out_images >= src_images
            assert (
                len(out.load_page(pno).get_image_info())
                >= len(src.load_page(pno).get_image_info())
            )
        # Page 0 carries the raster figure and no formula-restoration
        # constructs: its image count must stay exactly unchanged.
        assert (
            len(out.load_page(0).get_images(full=True))
            == len(src.load_page(0).get_images(full=True))
        )
        assert (
            len(out.load_page(0).get_image_info())
            == len(src.load_page(0).get_image_info())
        )
    finally:
        src.close()
        out.close()


def test_criterion_b_korean_extractable(translated_pdf):
    for pno, text in enumerate(_page_texts(translated_pdf)):
        assert re.search(r"[가-힣]", text), f"page {pno}: no Hangul extracted"


def test_criterion_c_english_body_removed(translated_pdf):
    full_text = "\n".join(_page_texts(translated_pdf))
    for sentinel in BODY_SENTINELS:
        assert sentinel not in full_text, f"body text survived: {sentinel!r}"
    # headers/footers are intentionally kept in the source language
    assert HEADER_PHRASE in full_text


@pytest.mark.skipif(not MATH_FONT_AVAILABLE, reason="math font sample needs seguisym")
def test_criterion_d_equations_preserved(translated_pdf):
    # Formulas must keep their position AND content (DESIGN 1.3-4): the
    # original math glyphs survive and no literal placeholder is typeset.
    full_text = "\n".join(_page_texts(translated_pdf))
    assert "⟦EQ" not in full_text, "literal equation placeholder leaked into output"
    assert "argmin" in full_text, "equation content destroyed"
    assert "∑" in full_text, "equation content destroyed"


def test_runner_progress_and_segment_callbacks(sample_pdf, tmp_path):
    from app.pipeline.extract import analyze_pdf
    from app.pipeline.models import RenderReport
    from app.pipeline.runner import run_pipeline
    from app.pipeline.segment import UNTRANSLATED_KINDS, build_segments
    from app.pipeline.translate import StubTranslator

    events: list[dict] = []
    seen: list[tuple[str, str | None]] = []
    out_path = str(tmp_path / "runner_out.pdf")
    report = run_pipeline(
        sample_pdf,
        out_path,
        StubTranslator(),
        on_progress=events.append,
        segment_cb=lambda seg, translated, **kw: seen.append((seg.id, translated)),
    )

    assert isinstance(report, RenderReport)
    assert os.path.isfile(out_path)

    assert events, "expected progress events"
    total = events[0]["total"]
    assert [event["done"] for event in events] == list(range(1, total + 1))
    for event in events:
        assert set(event.keys()) == {
            "segment_id", "done", "total", "page", "num_pages",
            "source_preview", "translated_preview",
        }
        assert event["num_pages"] == 4
        assert event["page"] in (0, 1, 2, 3)
        assert re.search(r"[가-힣]", event["translated_preview"])

    translated_ids = {seg_id for seg_id, translated in seen if translated is not None}
    skipped_ids = {seg_id for seg_id, translated in seen if translated is None}
    assert len(translated_ids) == total
    # Untranslated kinds (header_footer/formula/author/reference) are all
    # reported through the callback with translated=None.
    expected_skipped = {
        seg.id
        for seg in build_segments(analyze_pdf(sample_pdf))
        if seg.kind in UNTRANSLATED_KINDS
    }
    assert skipped_ids == expected_skipped
    assert len(skipped_ids) >= 4  # at least header/footer on both pages
    assert {event["segment_id"] for event in events} == translated_ids
