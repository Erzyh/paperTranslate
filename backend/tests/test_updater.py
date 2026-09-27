"""Self-update (app/updater.py) — GitHub is faked with an httpx.MockTransport."""
from __future__ import annotations

import hashlib
import io
import time
import zipfile

import httpx
import pytest

from app import updater

ZIP_NAME = "paperTranslate-v9.0.0-windows-x64.zip"


def make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


GOOD_ZIP = make_zip({"paperTranslate/paperTranslate.exe": b"MZ new",
                     "paperTranslate/_internal/lib.dll": b"lib"})


def github(tag="v9.0.0", zip_bytes=GOOD_ZIP, sha=None, calls=None):
    sha = sha if sha is not None else hashlib.sha256(zip_bytes).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request.url.path)
        path = request.url.path
        if path.endswith("/releases/latest"):
            return httpx.Response(200, json={
                "tag_name": tag, "body": "새 기능", "html_url": "https://example.invalid/r",
                "assets": [
                    {"name": ZIP_NAME, "size": len(zip_bytes),
                     "browser_download_url": "https://dl.invalid/app.zip"},
                    {"name": ZIP_NAME + ".sha256",
                     "browser_download_url": "https://dl.invalid/app.zip.sha256"},
                ]})
        if path == "/app.zip":
            return httpx.Response(200, content=zip_bytes)
        if path == "/app.zip.sha256":
            return httpx.Response(200, text=f"{sha}  {ZIP_NAME}\n")
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def packaged_app(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERTRANSLATE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PAPERTRANSLATE_INSTALL_DIR", str(tmp_path / "app"))
    monkeypatch.setattr(updater, "__version__", "0.1.0")
    monkeypatch.setattr(updater, "_check_cache", {"at": 0.0, "info": None})
    monkeypatch.setattr(updater, "_download",
                        {"state": "idle", "version": None, "progress": 0.0, "error": None})


def wait_for(state: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = updater.status()
        if current["state"] == state:
            return current
        time.sleep(0.02)
    pytest.fail(f"download never reached {state!r}: {updater.status()}")


def test_parse_version():
    assert updater.parse_version("v1.2.10") == (1, 2, 10)
    assert updater.parse_version("0.1") == (0, 1)
    assert updater.parse_version("latest") is None


def test_newer_release_is_offered(monkeypatch):
    monkeypatch.setattr(updater, "TRANSPORT", github())
    info = updater.check()
    assert info["available"] is True and info["latest"] == "9.0.0"
    assert info["skipped"] is False and info["notes"] == "새 기능"


def test_same_or_older_release_is_not_offered(monkeypatch):
    monkeypatch.setattr(updater, "TRANSPORT", github(tag="v0.1.0"))
    assert updater.check()["available"] is False


def test_source_checkout_never_updates(monkeypatch):
    monkeypatch.delenv("PAPERTRANSLATE_INSTALL_DIR")
    calls: list[str] = []
    monkeypatch.setattr(updater, "TRANSPORT", github(calls=calls))
    assert updater.check() == {"current": "0.1.0", "enabled": False, "available": False}
    assert calls == []


def test_skipped_version_stays_quiet_until_a_newer_one(monkeypatch):
    monkeypatch.setattr(updater, "TRANSPORT", github(tag="v9.0.0"))
    updater.skip("9.0.0")
    assert updater.check()["skipped"] is True
    monkeypatch.setattr(updater, "TRANSPORT", github(tag="v9.1.0"))
    assert updater.check(force=True)["skipped"] is False


def test_check_is_cached(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(updater, "TRANSPORT", github(calls=calls))
    updater.check()
    updater.check()
    assert calls.count("/repos/Erzyh/paperTranslate/releases/latest") == 1


def test_download_verifies_and_stages(monkeypatch):
    monkeypatch.setattr(updater, "TRANSPORT", github())
    updater.start_download()
    wait_for("ready")
    pending = updater.pending_install()
    assert pending["version"] == "9.0.0"
    staged = pending["dir"]
    assert open(staged + "/paperTranslate.exe", "rb").read() == b"MZ new"


def test_checksum_mismatch_is_rejected(monkeypatch):
    monkeypatch.setattr(updater, "TRANSPORT", github(sha="0" * 64))
    updater.start_download()
    status = wait_for("error")
    assert "체크섬" in status["error"]
    assert updater.pending_install() is None


def test_zip_path_escape_is_rejected(monkeypatch):
    evil = make_zip({"../../evil.exe": b"x", "paperTranslate/paperTranslate.exe": b"MZ"})
    monkeypatch.setattr(updater, "TRANSPORT", github(zip_bytes=evil))
    updater.start_download()
    assert "잘못된 경로" in wait_for("error")["error"]


def test_installed_update_is_cleaned_up(monkeypatch):
    monkeypatch.setattr(updater, "TRANSPORT", github())
    updater.start_download()
    wait_for("ready")
    # Next start runs the new build: the staged copy is no longer pending.
    monkeypatch.setattr(updater, "__version__", "9.0.0")
    assert updater.pending_install() is None
    updater.cleanup_installed()
    assert "ready" not in updater._load_state()


def test_install_gives_up_after_repeated_failures(monkeypatch):
    monkeypatch.setattr(updater, "TRANSPORT", github())
    updater.start_download()
    wait_for("ready")
    for _ in range(3):
        updater.note_install_attempt()
    assert updater.pending_install() is None
