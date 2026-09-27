"""paperTranslate desktop app.

Starts the FastAPI backend in-process on 127.0.0.1 and shows the built
frontend in a native window (pywebview / WebView2). Closing the window stops
the server — no separate console or .bat to keep open.

Run from source:   backend/.venv/Scripts/python.exe desktop/launcher.py
Packaged build:    see desktop/build.ps1 (PyInstaller, one folder)
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path

APP_NAME = "paperTranslate"
PREFERRED_PORT = 8000
# Local models tried in order when the user has not picked one; the first one
# installed in Ollama wins (see pick_ollama_model).
PREFERRED_OLLAMA_MODELS = ("qwen3.5:9b", "qwen3:14b", "qwen3:8b")
FALLBACK_OLLAMA_MODEL = "qwen3.5:9b"


def frozen() -> bool:
    return getattr(sys, "frozen", False)


def resource_dir() -> Path:
    """Bundled files (PyInstaller) or the repository root (from source)."""
    if frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    """Per-user data: uploads, outputs, DB, settings, logs."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / APP_NAME


def load_settings() -> dict:
    try:
        return json.loads((data_dir() / "settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def free_port(preferred: int = PREFERRED_PORT) -> int:
    """The first free port of preferred..preferred+9, else any free port.

    Web storage (saved keys, job board) is per origin, i.e. per port, so a
    stable port keeps it across restarts; a random port is the last resort.
    """
    for port in (*range(preferred, preferred + 10), 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return sock.getsockname()[1]
    raise RuntimeError("사용 가능한 포트가 없습니다.")


def pick_ollama_model(settings: dict) -> str:
    """User choice, else the first preferred model installed in Ollama."""
    if settings.get("ollama_model"):
        return settings["ollama_model"]
    try:
        import httpx

        url = os.environ.get("PAPERTRANSLATE_OLLAMA_URL", "http://localhost:11434")
        tags = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=2.0).json()
        installed = {m.get("name", "") for m in tags.get("models", [])}
    except Exception:  # noqa: BLE001 - Ollama missing or down: use the fallback
        return FALLBACK_OLLAMA_MODEL
    for name in PREFERRED_OLLAMA_MODELS:
        if name in installed:
            return name
    qwen = sorted(n for n in installed if n.startswith("qwen"))
    return qwen[0] if qwen else FALLBACK_OLLAMA_MODEL


def configure_environment() -> None:
    """Point the backend at per-user data and the bundled frontend.

    Must run before app.* is imported (config reads these per call, but
    main.py resolves the frontend folder at import time). Values already set
    in the environment win, so power users can still override them.
    """
    data = data_dir()
    data.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    root = resource_dir()
    dist = root / "frontend_dist" if frozen() else root / "frontend" / "dist"
    os.environ.setdefault("PAPERTRANSLATE_DATA_DIR", str(data / "data"))
    os.environ.setdefault("PAPERTRANSLATE_FRONTEND_DIST", str(dist))
    os.environ.setdefault("PAPERTRANSLATE_TRANSLATOR", "ollama")
    os.environ.setdefault("PAPERTRANSLATE_OLLAMA_MODEL", pick_ollama_model(settings))
    if frozen():
        # Enables self-update (app/updater.py): the folder to install into.
        os.environ.setdefault("PAPERTRANSLATE_INSTALL_DIR",
                              str(Path(sys.executable).resolve().parent))
    else:
        sys.path.insert(0, str(root / "backend"))


# Installs a staged update once this process has exited (a running .exe
# cannot be overwritten on Windows): copy the new files over the app folder,
# then optionally reopen the app. robocopy /E adds and overwrites but never
# deletes, so other files next to the app are left alone.
_INSTALL_SCRIPT = r"""
param([int]$WaitPid, [string]$Source, [string]$Target, [string]$Relaunch)
$ErrorActionPreference = 'Continue'
$log = Join-Path $PSScriptRoot 'install_update.log'
function Log($msg) { Add-Content -Path $log -Value ("{0:s} {1}" -f (Get-Date), $msg) }
Log "waiting for app (pid $WaitPid) to exit"
Wait-Process -Id $WaitPid -Timeout 120 -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 500
Log "copying $Source -> $Target"
robocopy "$Source" "$Target" /E /R:10 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
Log "robocopy exit code $LASTEXITCODE"
if ($Relaunch -eq '1') {
    Start-Process -FilePath (Join-Path $Target 'paperTranslate.exe')
    Log "relaunched"
}
"""


def spawn_installer(pending: dict, relaunch: bool) -> None:
    """Hand the staged update to a detached PowerShell that installs it."""
    import subprocess

    from app import updater

    updater.note_install_attempt()
    script = data_dir() / "install_update.ps1"
    script.write_text(_INSTALL_SCRIPT, encoding="utf-8-sig")
    args = ["powershell", "-NoProfile", "-NonInteractive",
            "-ExecutionPolicy", "Bypass", "-File", str(script),
            "-WaitPid", str(os.getpid()), "-Source", pending["dir"],
            "-Target", os.environ["PAPERTRANSLATE_INSTALL_DIR"],
            "-Relaunch", "1" if relaunch else "0"]
    # A hidden console of its own (CREATE_NO_WINDOW). PowerShell started with
    # no console at all (DETACHED_PROCESS) exits before running the script.
    detached = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    # The installer must outlive this process. When the app was started
    # inside a Windows job that kills its members on exit (some launchers and
    # terminals do this), only a process that breaks away from the job
    # survives; jobs that forbid breaking away reject the flag, so retry
    # without it.
    for flags in (detached | subprocess.CREATE_BREAKAWAY_FROM_JOB, detached):
        try:
            subprocess.Popen(args, creationflags=flags, close_fds=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            logging.info("update installer started (version %s)", pending["version"])
            return
        except OSError as exc:
            logging.warning("could not start update installer: %s", exc)


def redirect_output_to_log() -> None:
    """A windowed build has no console (stdout/stderr are None); log to a file."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    log = open(data_dir() / "app.log", "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    sys.stdout = sys.stderr = log
    logging.basicConfig(stream=log, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class BackendServer:
    """uvicorn running in a background thread, stoppable from the UI thread."""

    def __init__(self, port: int) -> None:
        import uvicorn

        from app.main import app

        self.port = port
        self.server = uvicorn.Server(uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="warning",
            log_config=None))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self, timeout: float = 30.0) -> None:
        self.thread.start()
        deadline = time.monotonic() + timeout
        while not self.server.started:
            if not self.thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("번역 서버를 시작하지 못했습니다.")
            time.sleep(0.05)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5.0)


def active_translations() -> int:
    """Papers currently queued or translating (asked before closing)."""
    from app.api import routes

    return len(routes.PHASE)


def main() -> None:
    configure_environment()
    redirect_output_to_log()

    from app import updater

    updater.cleanup_installed()
    # An update downloaded earlier that did not get installed on exit (e.g.
    # the app was killed): install it now and reopen.
    pending = updater.pending_install() if frozen() else None
    if pending:
        spawn_installer(pending, relaunch=True)
        os._exit(0)

    import webview

    server = BackendServer(free_port())
    server.start()

    # "PDF" buttons are <a download> links; WebView2 blocks them unless allowed.
    webview.settings["ALLOW_DOWNLOADS"] = True
    window = webview.create_window(
        APP_NAME, f"http://127.0.0.1:{server.port}/",
        width=1280, height=860, min_size=(900, 600))

    def on_closing() -> bool:
        running = active_translations()
        if not running:
            return True
        return window.create_confirmation_dialog(
            "번역 중인 논문이 있습니다",
            f"{running}편이 아직 번역 중이거나 대기 중입니다.\n"
            "앱을 닫으면 번역이 중단됩니다. 닫을까요?")

    window.events.closing += on_closing

    # "지금 재시작" in the update notice (POST /api/app/restart).
    restart = {"requested": False}

    def request_restart() -> None:
        restart["requested"] = True
        window.destroy()

    from app.main import app

    app.state.request_restart = request_restart

    # pywebview defaults to private mode, which wipes localStorage on exit:
    # the saved API keys, model choice and job board would be lost at every
    # restart. Keep the web storage in the per-user data folder instead.
    webview.start(private_mode=False,
                  storage_path=str(data_dir() / "webview"))

    server.stop()
    # A downloaded update installs now that the app is closing; it reopens
    # only when the user asked for a restart.
    pending = updater.pending_install() if frozen() else None
    if pending:
        spawn_installer(pending, relaunch=restart["requested"])
    # Pipeline worker threads may still be inside a translation request;
    # the next start marks those runs as interrupted (db recovery).
    os._exit(0)


if __name__ == "__main__":
    main()
