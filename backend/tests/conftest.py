# Shared fixtures: build the sample PDF once per session.
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> str:
    from scripts.make_sample import make_sample

    path = tmp_path_factory.mktemp("sample") / "sample.pdf"
    return make_sample(str(path))


@pytest.fixture(scope="session")
def sample_layout(sample_pdf):
    from app.pipeline.extract import analyze_pdf

    return analyze_pdf(sample_pdf)


@pytest.fixture(scope="session")
def sample_segments(sample_layout):
    from app.pipeline.segment import build_segments

    return build_segments(sample_layout)
