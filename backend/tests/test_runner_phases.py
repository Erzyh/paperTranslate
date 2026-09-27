"""run_pipeline reports its stages in order (batch job board)."""

from app.pipeline.runner import run_pipeline
from app.pipeline.translate import StubTranslator


def test_on_phase_reports_stages_in_order(sample_pdf, tmp_path):
    phases = []
    run_pipeline(sample_pdf, str(tmp_path / "out.pdf"), StubTranslator(),
                 on_phase=phases.append)
    assert phases == ["analyzing", "glossary", "translating", "rendering"]
