"""Self-update from GitHub Releases (desktop app only).

Flow: check() asks GitHub for the latest release; if its tag is newer than
this build and the user did not skip that version, the UI offers an update.
start_download() fetches the release zip in a background thread while the
app keeps running, verifies it against the release's .sha256 asset, and
unpacks it into <data>/updates/<version>/. The desktop launcher installs the
staged copy over the app folder after the app exits (a running .exe cannot
be overwritten on Windows) and, if asked, relaunches it.

Release assets expected per tag vX.Y.Z:
  paperTranslate-vX.Y.Z-windows-x64.zip          (top folder "paperTranslate/")
  paperTranslate-vX.Y.Z-windows-x64.zip.sha256   ("<hex>  <file name>")
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
import zipfile
from pathlib import Path

import httpx

from app import __version__, config

_API = "https://api.github.com/repos/{repo}/releases/latest"
_ASSET_RE = re.compile(r"^paperTranslate-.*-windows-x64\.zip$")
_CHECK_TTL = 3600.0          # seconds between GitHub checks
_TIMEOUT = 15.0
_EXE_NAME = "paperTranslate.exe"
_MAX_INSTALL_ATTEMPTS = 3

# Test hook: an httpx.MockTransport so tests never reach GitHub.
TRANSPORT: httpx.BaseTransport | None = None

_lock = threading.Lock()
_check_cache: dict = {"at": 0.0, "info": None}
_download: dict = {"state": "idle", "version": None, "progress": 0.0, "error": None}


def parse_version(tag: str | None) -> tuple[int, ...] | None:
    """"v1.2.3" / "1.2.3" -> (1, 2, 3); None when not a plain version."""
    match = re.fullmatch(r"v?(\d+(?:\.\d+)*)", (tag or "").strip())
    return tuple(int(p) for p in match.group(1).split(".")) if match else None


def _updates_dir() -> Path:
    return config.get_data_dir() / "updates"


def _state_path() -> Path:
    return _updates_dir() / "state.json"


def _load_state() -> dict:
    try:
        return json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    _updates_dir().mkdir(parents=True, exist_ok=True)
    _state_path().write_text(json.dumps(state), encoding="utf-8")


def enabled() -> bool:
    """Updates only apply to the packaged app (the launcher sets the folder)."""
    return config.get_install_dir() is not None


def _client() -> httpx.Client:
    return httpx.Client(timeout=_TIMEOUT, transport=TRANSPORT, follow_redirects=True,
                        headers={"User-Agent": f"paperTranslate/{__version__}"})


def _fetch_latest() -> dict:
    with _client() as client:
        response = client.get(_API.format(repo=config.get_update_repo()),
                              headers={"Accept": "application/vnd.github+json"})
        response.raise_for_status()
        release = response.json()
    assets = {a.get("name", ""): a for a in release.get("assets", [])}
    zip_asset = next((a for n, a in assets.items() if _ASSET_RE.fullmatch(n)), None)
    sha_asset = assets.get(f"{zip_asset['name']}.sha256") if zip_asset else None
    return {
        "tag": release.get("tag_name", ""),
        "notes": (release.get("body") or "")[:4000],
        "page": release.get("html_url"),
        "zip_url": zip_asset.get("browser_download_url") if zip_asset else None,
        "zip_size": zip_asset.get("size") if zip_asset else None,
        "sha_url": sha_asset.get("browser_download_url") if sha_asset else None,
    }


def check(force: bool = False) -> dict:
    """Is a newer release available (and not skipped by the user)?"""
    info = {"current": __version__, "enabled": enabled(), "available": False}
    if not enabled():
        return info
    now = time.monotonic()
    with _lock:
        cached = _check_cache["info"]
        fresh = cached is not None and now - _check_cache["at"] < _CHECK_TTL
    if force or not fresh:
        try:
            cached = _fetch_latest()
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            return {**info, "error": f"업데이트 확인 실패: {type(exc).__name__}"}
        with _lock:
            _check_cache.update(at=now, info=cached)
    latest = parse_version(cached["tag"])
    current = parse_version(__version__)
    latest_str = ".".join(map(str, latest)) if latest else None
    available = bool(latest and current and latest > current and cached["zip_url"])
    return {
        **info,
        "latest": latest_str,
        "available": available,
        "skipped": available and _load_state().get("skipped") == latest_str,
        "notes": cached["notes"],
        "page": cached["page"],
        "size": cached["zip_size"],
    }


def skip(version: str) -> None:
    """Do not offer ``version`` again; a newer release is offered as usual."""
    state = _load_state()
    state["skipped"] = version
    _save_state(state)


def status() -> dict:
    with _lock:
        snapshot = dict(_download)
    ready = pending_install()
    if ready and snapshot["state"] == "idle":
        snapshot.update(state="ready", version=ready["version"], progress=1.0)
    return snapshot


def start_download() -> dict:
    """Download + verify + stage the latest release in a background thread."""
    info = check()
    if not info.get("available"):
        return {"state": "idle", "error": "받을 업데이트가 없습니다."}
    with _lock:
        if _download["state"] in ("downloading", "ready") and \
                _download["version"] == info["latest"]:
            return dict(_download)
        _download.update(state="downloading", version=info["latest"],
                         progress=0.0, error=None)
    threading.Thread(target=_download_worker, args=(info["latest"],),
                     daemon=True).start()
    with _lock:
        return dict(_download)


def _set(**fields) -> None:
    with _lock:
        _download.update(**fields)


def _download_worker(version: str) -> None:
    try:
        with _lock:
            release = dict(_check_cache["info"])
        target = _updates_dir() / version
        archive = _updates_dir() / f"{version}.zip"
        _updates_dir().mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with _client() as client:
            expected = None
            if release.get("sha_url"):
                expected = client.get(release["sha_url"]).raise_for_status().text.split()[0].lower()
            with client.stream("GET", release["zip_url"]) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length") or release.get("zip_size") or 0)
                done = 0
                with open(archive, "wb") as fh:
                    for chunk in response.iter_bytes(1 << 16):
                        fh.write(chunk)
                        digest.update(chunk)
                        done += len(chunk)
                        if total:
                            _set(progress=min(done / total, 0.99))
        if expected and digest.hexdigest() != expected:
            raise ValueError("다운로드한 파일의 체크섬이 맞지 않습니다.")
        staged = _extract(archive, target)
        archive.unlink(missing_ok=True)
        state = _load_state()
        state["ready"] = {"version": version, "dir": str(staged), "attempts": 0}
        _save_state(state)
        _set(state="ready", progress=1.0)
    except Exception as exc:  # noqa: BLE001 - reported to the UI
        _set(state="error", error=f"업데이트를 받지 못했습니다: {exc}")


def _extract(archive: Path, target: Path) -> Path:
    """Unpack safely (no path escapes) and return the folder holding the exe."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    root = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            dest = (target / member).resolve()
            if root not in dest.parents and dest != root:
                raise ValueError(f"잘못된 경로가 포함된 압축 파일입니다: {member}")
        zf.extractall(target)
    exe = next(target.rglob(_EXE_NAME), None)
    if exe is None:
        raise ValueError("압축 파일에 paperTranslate.exe가 없습니다.")
    return exe.parent


def pending_install() -> dict | None:
    """A staged update newer than this build, still to be installed."""
    ready = _load_state().get("ready")
    if not ready:
        return None
    newer = (parse_version(ready.get("version")) or ()) > (parse_version(__version__) or ())
    if not newer or not Path(ready.get("dir", "")).is_dir() \
            or ready.get("attempts", 0) >= _MAX_INSTALL_ATTEMPTS:
        return None
    return ready


def note_install_attempt() -> None:
    state = _load_state()
    if state.get("ready"):
        state["ready"]["attempts"] = state["ready"].get("attempts", 0) + 1
        _save_state(state)


def cleanup_installed() -> None:
    """After an update is installed, drop its staged copy and the marker."""
    state = _load_state()
    ready = state.get("ready")
    if not ready:
        return
    installed = (parse_version(ready.get("version")) or ()) <= (parse_version(__version__) or ())
    if installed or ready.get("attempts", 0) >= _MAX_INSTALL_ATTEMPTS:
        shutil.rmtree(_updates_dir() / ready.get("version", "-"), ignore_errors=True)
        state.pop("ready", None)
        _save_state(state)
