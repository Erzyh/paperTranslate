# PyInstaller spec for the paperTranslate desktop app (one-folder build).
# Build with desktop/build.ps1 (it builds the frontend first).
# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent  # noqa: F821 - SPECPATH is injected by PyInstaller

a = Analysis(
    [str(ROOT / "desktop" / "launcher.py")],
    pathex=[str(ROOT / "backend")],
    datas=[
        (str(ROOT / "frontend" / "dist"), "frontend_dist"),
        # Layout-detection / OCR models (backend/scripts/fetch_models.py).
        (str(ROOT / "backend" / "models"), "models"),
    ],
    # uvicorn picks its loop/protocol implementations by name at runtime, and
    # the pipeline is imported lazily inside the request handlers.
    hiddenimports=collect_submodules("uvicorn") + collect_submodules("app"),
    excludes=["pytest", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="paperTranslate",
    console=False,  # windowed app: no console window
    upx=False,      # UPX-packed binaries trigger antivirus false positives
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="paperTranslate",
    upx=False,
)
