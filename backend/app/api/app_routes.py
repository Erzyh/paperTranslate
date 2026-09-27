"""App-level routes: version and self-update (see app/updater.py)."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app import __version__, updater

router = APIRouter()


@router.get("/version")
async def get_version():
    return {"version": __version__}


@router.get("/update")
async def check_update(force: bool = False):
    """Latest release vs this build; ``skipped`` when the user declined it."""
    return await asyncio.get_running_loop().run_in_executor(None, updater.check, force)


class SkipRequest(BaseModel):
    version: str


@router.post("/update/skip")
async def skip_update(request: SkipRequest):
    updater.skip(request.version)
    return {"skipped": request.version}


@router.post("/update/download")
async def download_update():
    """Start downloading in the background; poll GET /update/status."""
    return await asyncio.get_running_loop().run_in_executor(None, updater.start_download)


@router.get("/update/status")
async def update_status():
    return updater.status()


@router.post("/restart")
async def restart(request: Request):
    """Close the app now so a staged update installs, then reopen it.

    Only the desktop launcher can do this; it registers the callback.
    """
    callback = getattr(request.app.state, "request_restart", None)
    if callback is None or updater.pending_install() is None:
        raise HTTPException(status_code=409, detail="적용할 업데이트가 없습니다.")
    asyncio.get_running_loop().call_later(0.3, callback)
    return {"restarting": True}
