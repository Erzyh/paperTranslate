"""Download many translated PDFs at once as one zip.

POST /api/archives builds the zip from the finished documents the client
lists (with the relative path each should get inside the zip, so papers
added from a folder keep their folder structure) and returns a URL;
GET /api/archives/{name} serves it like the single-PDF download, so the
desktop app's download handling works the same way.
"""
from __future__ import annotations

import asyncio
import re
import secrets
import time
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import config, db

router = APIRouter()

# Built zips are kept briefly so the download can be retried, then removed.
_KEEP_SECONDS = 3600
_MAX_ITEMS = 500
_NAME_RE = re.compile(r"[0-9a-f]{32}\.zip")
# Characters Windows does not allow in file names.
_BAD_CHARS_RE = re.compile(r'[<>:"|?*\x00-\x1f]')


class ArchiveItem(BaseModel):
    id: str
    # Path inside the zip, e.g. "papers/2025/attention.pdf"; defaults to the
    # uploaded file name.
    path: str | None = None


class ArchiveRequest(BaseModel):
    items: list[ArchiveItem]


def _archives_dir() -> Path:
    return config.get_data_dir() / "archives"


def _safe_member(path: str) -> str | None:
    """A relative, Windows-safe path inside the zip (no "..", no drive)."""
    parts = []
    for part in PurePosixPath(path.replace("\\", "/")).parts:
        part = _BAD_CHARS_RE.sub("_", part).strip().rstrip(".")
        if part and part not in (".", "..", "/"):
            parts.append(part)
    if not parts:
        return None
    if not parts[-1].lower().endswith(".pdf"):
        parts[-1] += ".pdf"
    return "/".join(parts)


def _unique(member: str, used: set[str]) -> str:
    """"a.pdf", "a (2).pdf", ... so two papers never overwrite each other."""
    if member.lower() not in used:
        used.add(member.lower())
        return member
    stem, dot, ext = member.rpartition(".")
    n = 2
    while f"{stem} ({n}).{ext}".lower() in used:
        n += 1
    unique = f"{stem} ({n}).{ext}"
    used.add(unique.lower())
    return unique


def _cleanup_old() -> None:
    cutoff = time.time() - _KEEP_SECONDS
    for old in _archives_dir().glob("*.zip"):
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except OSError:
            pass


def _build(items: list[ArchiveItem]) -> dict:
    archives = _archives_dir()
    archives.mkdir(parents=True, exist_ok=True)
    _cleanup_old()
    docs = db.get_documents([item.id for item in items])
    name = f"{secrets.token_hex(16)}.zip"
    target = archives / name
    used: set[str] = set()
    added = 0
    skipped: list[str] = []
    # PDFs are already compressed: store them as-is, which is much faster.
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as zf:
        for item in items:
            doc = docs.get(item.id)
            output = config.get_outputs_dir() / f"{item.id}.pdf"
            member = _safe_member(item.path or (doc or {}).get("filename", ""))
            if doc is None or doc["status"] != "done" or not output.exists() or not member:
                skipped.append(item.id)
                continue
            zf.write(output, _unique(member, used))
            added += 1
    if not added:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=404, detail="내려받을 번역 PDF가 없습니다.")
    return {"url": f"/api/archives/{name}", "count": added, "skipped": skipped}


@router.post("/archives")
async def create_archive(request: ArchiveRequest):
    """Zip the finished translations of ``items``; returns a download URL."""
    if not request.items:
        raise HTTPException(status_code=400, detail="내려받을 논문을 골라 주세요.")
    if len(request.items) > _MAX_ITEMS:
        raise HTTPException(status_code=400,
                            detail=f"한 번에 {_MAX_ITEMS}편까지 묶을 수 있습니다.")
    return await asyncio.get_running_loop().run_in_executor(None, _build, request.items)


@router.get("/archives/{name}")
async def get_archive(name: str):
    if not _NAME_RE.fullmatch(name):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    path = _archives_dir() / name
    if not path.exists():
        raise HTTPException(status_code=404,
                            detail="파일이 만료되었습니다. 다시 내려받아 주세요.")
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    return FileResponse(str(path), media_type="application/zip",
                        filename=f"paperTranslate_번역_{stamp}.zip")
