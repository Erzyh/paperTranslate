"""Download the layout-detection and OCR models into backend/models.

Usage (repo root):  backend\\.venv\\Scripts\\python.exe backend\\scripts\\fetch_models.py

The models are not committed (see .gitignore); desktop/build.ps1 runs this
script before packaging. Every download is checked against its published
SHA-256. The layout model is quantized to int8 (weights of MatMul/Gemm/Conv),
which shrinks it from 214 MB to about 59 MB; on our paper set its boxes agree
with the fp32 model on 98% of regions.

Sources (Apache-2.0, PaddlePaddle models converted to ONNX by RapidAI):
- PP-DocLayoutV2 page layout detector (25 region classes, reading order)
- PP-OCRv6 small text detector / recognizer (used only for scanned pages)
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_LAYOUT_URL = ("https://www.modelscope.cn/models/RapidAI/RapidLayout/resolve/"
               "v1.2.0/onnx/pp_doc_layout/pp_doc_layoutv2.onnx")
_LAYOUT_SHA = "0bd2ea0997fe0789f0300292291f8bbf897d890b44a9a3bd5be72afd6198aa90"

_OCR_BASE = ("https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/"
             "v3.9.2/onnx/PP-OCRv6")
DOWNLOADS = {
    "ocr_det.onnx": (
        f"{_OCR_BASE}/det/PP-OCRv6_det_small.onnx",
        "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
    ),
    "ocr_rec.onnx": (
        f"{_OCR_BASE}/rec/PP-OCRv6_rec_small.onnx",
        "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
    ),
}
LAYOUT_FILE = "layout_int8.onnx"


def _download(url: str, sha256: str, dest: Path) -> None:
    print(f"다운로드: {url.rsplit('/', 1)[-1]}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 불일치: {url}")
    tmp.replace(dest)


def _fetch_layout(dest: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "layout_fp32.onnx"
        _download(_LAYOUT_URL, _LAYOUT_SHA, fp32)
        print("레이아웃 모델을 int8로 줄이는 중...", flush=True)
        part = dest.with_suffix(".part.onnx")
        quantize_dynamic(
            str(fp32), str(part),
            weight_type=QuantType.QUInt8,
            op_types_to_quantize=["MatMul", "Gemm", "Conv"],
        )
        part.replace(dest)


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    layout = MODELS_DIR / LAYOUT_FILE
    if not layout.exists():
        _fetch_layout(layout)
    for name, (url, sha) in DOWNLOADS.items():
        dest = MODELS_DIR / name
        if not dest.exists():
            _download(url, sha, dest)
    print(f"모델 준비 완료: {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
