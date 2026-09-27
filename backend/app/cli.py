# Command line entry point: python -m app.cli input.pdf output.pdf
from __future__ import annotations

import argparse
import os
import sys

from app.pipeline.runner import run_pipeline
from app.pipeline.translate import get_translator


def main(argv: list[str] | None = None) -> int:
    # Keep console output robust on cp949 terminals (replace unknown glyphs).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(
        prog="app.cli",
        description="레이아웃을 보존하며 영어 PDF를 한국어 PDF로 번역합니다.",
    )
    parser.add_argument("input", help="입력 PDF 경로")
    parser.add_argument("output", help="출력 PDF 경로")
    parser.add_argument(
        "--translator",
        default=os.environ.get("PAPERTRANSLATE_TRANSLATOR", "stub"),
        help="번역기 이름 (stub | ollama, 기본값: 환경변수 또는 stub)",
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"오류: 입력 파일을 찾을 수 없습니다: {args.input}", file=sys.stderr)
        return 2

    try:
        translator = get_translator(args.translator)
    except ValueError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    def on_progress(event: dict) -> None:
        print(
            f"[{event['done']}/{event['total']}] "
            f"{event['page'] + 1}/{event['num_pages']}쪽 "
            f"{event['segment_id']} 번역 완료"
        )

    try:
        report = run_pipeline(
            args.input, args.output, translator, on_progress=on_progress
        )
    except Exception as exc:  # surface a readable Korean error, non-zero exit
        print(f"오류: 파이프라인 실행에 실패했습니다: {exc}", file=sys.stderr)
        return 1

    print(f"완료: {args.output}")
    if report.overflow_segments:
        print(f"경고: 넘침이 발생한 문단 {len(report.overflow_segments)}개: "
              f"{', '.join(report.overflow_segments)}")
    if report.scaled_segments:
        print(f"안내: 축소된 문단 {len(report.scaled_segments)}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
