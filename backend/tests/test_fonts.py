# Tests for the Korean serif body font injection (retypeset).
from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from app.pipeline.models import RenderReport, Segment
from app.pipeline.retypeset import (
    _insert_textbox,
    _load_body_font,
    render_translated_pdf,
)

HANBATANG = Path(r"C:\Windows\Fonts\HANBatang.ttf")
HANBATANG_BOLD = Path(r"C:\Windows\Fonts\HANBatangB.ttf")


def _clear_font_env(monkeypatch):
    monkeypatch.delenv("PAPERTRANSLATE_FONT_PATH", raising=False)
    monkeypatch.delenv("PAPERTRANSLATE_FONT_BOLD_PATH", raising=False)


def _make_source(path: Path) -> str:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Heading Title", fontsize=14)
    page.insert_text((72, 130), "Body text to be replaced.", fontsize=10)
    doc.save(str(path))
    doc.close()
    return str(path)


_SEGMENTS = [
    Segment(id="p0_s0", page=0, column=0, bbox=(70.0, 86.0, 540.0, 112.0),
            text="Heading Title", kind="heading", font_size=14.0),
    Segment(id="p0_s1", page=0, column=0, bbox=(70.0, 118.0, 540.0, 160.0),
            text="Body text to be replaced.", kind="body", font_size=10.0),
]
_TRANSLATIONS = {
    "p0_s0": "한글 제목",
    "p0_s1": "한글 본문 문장입니다. 세리프 폰트로 렌더링되어야 한다.",
}


def _render(tmp_path: Path) -> tuple[str, list[tuple], RenderReport]:
    """Render the fixed translations; return (extracted text, fonts, report)."""
    src = _make_source(tmp_path / "font_src.pdf")
    out = tmp_path / "font_out.pdf"
    report = render_translated_pdf(src, _SEGMENTS, _TRANSLATIONS, str(out))
    doc = pymupdf.open(str(out))
    try:
        text = doc.load_page(0).get_text()
        fonts = doc.load_page(0).get_fonts()
    finally:
        doc.close()
    return text, fonts, report


def _font_names(fonts: list[tuple]) -> str:
    return " | ".join(f[3] for f in fonts)


@pytest.mark.skipif(not HANBATANG.exists(), reason="needs HANBatang.ttf")
def test_default_render_embeds_batang_serif(tmp_path, monkeypatch):
    """Without env overrides the Batang serif is embedded and Hangul stays
    extractable (subset cmap intact)."""
    _clear_font_env(monkeypatch)
    text, fonts, _report = _render(tmp_path)
    assert "Batang" in _font_names(fonts), _font_names(fonts)
    assert "한글 본문" in text
    assert "한글 제목" in text


@pytest.mark.skipif(not HANBATANG_BOLD.exists(), reason="needs HANBatangB.ttf")
def test_env_font_path_override(tmp_path, monkeypatch):
    """PAPERTRANSLATE_FONT_PATH replaces the auto-detected regular face."""
    _clear_font_env(monkeypatch)
    monkeypatch.setenv("PAPERTRANSLATE_FONT_PATH", str(HANBATANG_BOLD))
    text, fonts, _report = _render(tmp_path)
    # The bold file doubles as the regular face under an explicit override,
    # so its own name must appear in the output.
    assert "Batang Bold" in _font_names(fonts), _font_names(fonts)
    assert "한글 본문" in text


def test_missing_font_path_falls_back_to_sans(tmp_path, monkeypatch):
    """An explicit but nonexistent font path disables the custom font: the
    render must still succeed with the previous sans behaviour."""
    _clear_font_env(monkeypatch)
    monkeypatch.setenv(
        "PAPERTRANSLATE_FONT_PATH", str(tmp_path / "does_not_exist.ttf"))
    assert _load_body_font() is None
    text, fonts, _report = _render(tmp_path)
    assert "Batang" not in _font_names(fonts), _font_names(fonts)
    assert "한글 본문" in text


@pytest.mark.skipif(not HANBATANG.exists(), reason="needs HANBatang.ttf")
def test_textbox_fallback_uses_body_font(tmp_path, monkeypatch):
    """The insert_textbox fallback path embeds the same serif font."""
    _clear_font_env(monkeypatch)
    doc = pymupdf.open()
    try:
        page = doc.new_page(width=612, height=792)
        report = RenderReport(overflow_segments=[], scaled_segments={})
        assert _insert_textbox(
            page, _SEGMENTS[1], "폴백 경로 한글", report, [])
        assert "Batang" in _font_names(page.get_fonts())
    finally:
        doc.close()


def test_font_cache_reads_file_once(tmp_path, monkeypatch):
    """Buffers are cached per env pair: the file is not re-read per call."""
    _clear_font_env(monkeypatch)
    fake = tmp_path / "fake.ttf"
    fake.write_bytes(HANBATANG.read_bytes() if HANBATANG.exists() else b"")
    if not HANBATANG.exists():
        pytest.skip("needs HANBatang.ttf as a real font to copy")
    monkeypatch.setenv("PAPERTRANSLATE_FONT_PATH", str(fake))
    first = _load_body_font()
    assert first is not None
    fake.unlink()  # a second call must hit the cache, not the file
    assert _load_body_font() is first
