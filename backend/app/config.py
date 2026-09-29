"""Application configuration.

Environment variables are read dynamically (per call) so tests can
override paths/settings without re-importing modules.

- PAPERTRANSLATE_TRANSLATOR:     "stub" (default) | "ollama"
- PAPERTRANSLATE_DATA_DIR:       overrides the data directory (default backend/data)
- PAPERTRANSLATE_FRONTEND_DIST:  built frontend to serve (default frontend/dist)
- PAPERTRANSLATE_OLLAMA_MODEL:   Ollama model name (default "qwen3:8b")
- PAPERTRANSLATE_OLLAMA_URL:     Ollama base URL (default "http://localhost:11434")
- PAPERTRANSLATE_OPENAI_BASE_URL: OpenAI-compatible endpoint (default OpenAI)
- PAPERTRANSLATE_OLLAMA_TIMEOUT: per-request timeout in seconds (default 180)
- PAPERTRANSLATE_FONT_PATH:      body font TTF for translated text
                                 (default "": auto-detect a Korean serif)
- PAPERTRANSLATE_FONT_BOLD_PATH: bold TTF for headings/bold runs
                                 (default "": pair of the auto-detected font)
- PAPERTRANSLATE_MAX_PARALLEL_OLLAMA / _OPENAI / _GEMINI / _ANTHROPIC / _STUB:
                                 how many papers of each engine are translated
                                 at the same time (defaults 2 / 4 / 4 / 4 / 4);
                                 the rest wait in the queue
- PAPERTRANSLATE_CLAUDE_EFFORT:  Claude effort level low|medium|high|xhigh|max
                                 (default "medium")
- PAPERTRANSLATE_MODELS_DIR:     layout/OCR ONNX models (default backend/models,
                                 filled by scripts/fetch_models.py)
- PAPERTRANSLATE_LAYOUT_MODEL:   "0" disables the layout model and OCR
                                 (rule-based segmentation only)
"""

import os
from pathlib import Path

# backend/ directory (parent of the app package)
BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_TRANSLATOR = "stub"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com"
DEFAULT_OLLAMA_TIMEOUT = 180.0
# API 번역에서 고를 수 있는 모델 → 제공사 (화면의 드롭다운과 같은 목록;
# frontend/src/pages/BatchPage.tsx 의 API_MODELS 와 함께 고칠 것).
API_MODELS = {
    "gpt-6-luna": "openai",
    "gpt-5.6-luna": "openai",
    "gpt-6-sol": "openai",
    "gpt-5.6-terra": "openai",
    "gpt-6-astra": "openai",
    "gemini-3.8-flash": "gemini",
    "claude-opus-5-5": "anthropic",
}
OPENAI_MODELS = tuple(m for m, p in API_MODELS.items() if p == "openai")
DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
API_PROVIDERS = ("openai", "gemini", "anthropic")
# Gemini is reached through Google's OpenAI-compatible endpoint.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
GEMINI_CHAT_PATH = "/v1beta/openai/chat/completions"
# Claude's thinking is always on; effort is the quality/latency/cost control.
DEFAULT_CLAUDE_EFFORT = "medium"
_CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")


def get_claude_effort() -> str:
    """Claude effort level (PAPERTRANSLATE_CLAUDE_EFFORT, default "medium")."""
    raw = os.environ.get("PAPERTRANSLATE_CLAUDE_EFFORT", DEFAULT_CLAUDE_EFFORT)
    return raw if raw in _CLAUDE_EFFORTS else DEFAULT_CLAUDE_EFFORT


def get_translator_name() -> str:
    """Return the configured translator name ("stub" | "ollama")."""
    return os.environ.get("PAPERTRANSLATE_TRANSLATOR", DEFAULT_TRANSLATOR)


def get_ollama_model() -> str:
    return os.environ.get("PAPERTRANSLATE_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def get_openai_base_url() -> str:
    """Base URL of the OpenAI(-compatible) API.

    Defaults to OpenAI. Point PAPERTRANSLATE_OPENAI_BASE_URL at another
    OpenAI-compatible server (a self-hosted gateway, another provider) to use
    it instead; the key format check is skipped for such servers.
    """
    return os.environ.get("PAPERTRANSLATE_OPENAI_BASE_URL",
                          DEFAULT_OPENAI_BASE_URL)


def using_custom_openai_server() -> bool:
    """True when the API base URL points somewhere other than OpenAI."""
    return get_openai_base_url().rstrip("/") != DEFAULT_OPENAI_BASE_URL


def get_ollama_url() -> str:
    return os.environ.get("PAPERTRANSLATE_OLLAMA_URL", DEFAULT_OLLAMA_URL)


def get_ollama_timeout() -> float:
    raw = os.environ.get("PAPERTRANSLATE_OLLAMA_TIMEOUT")
    if raw is None:
        return DEFAULT_OLLAMA_TIMEOUT
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_OLLAMA_TIMEOUT


# 동시에 번역하는 논문 수 (엔진별). 로컬 GPU는 두 편 정도가 적당하다:
# 한 편이 PDF 분석·조판(CPU)을 하는 동안 다른 편이 GPU를 쓰므로 처리량이 오른다.
# API는 네트워크 대기가 대부분이라 더 많이 돌려도 된다.
DEFAULT_MAX_PARALLEL = {"ollama": 2, "openai": 4, "gemini": 4,
                        "anthropic": 4, "stub": 4}


def get_max_parallel(engine: str) -> int:
    """Concurrent translation runs allowed for one engine (at least 1)."""
    default = DEFAULT_MAX_PARALLEL.get(engine, 2)
    raw = os.environ.get(f"PAPERTRANSLATE_MAX_PARALLEL_{engine.upper()}")
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def get_font_path() -> str:
    """Body font TTF path for translated text ("" = auto-detect chain)."""
    return os.environ.get("PAPERTRANSLATE_FONT_PATH", "")


def get_font_bold_path() -> str:
    """Bold body font TTF path (headings); "" pairs the auto-detected font."""
    return os.environ.get("PAPERTRANSLATE_FONT_BOLD_PATH", "")


def get_frontend_dist() -> Path:
    """Built frontend to serve (PAPERTRANSLATE_FRONTEND_DIST overrides)."""
    raw = os.environ.get("PAPERTRANSLATE_FRONTEND_DIST")
    return Path(raw) if raw else BASE_DIR.parent / "frontend" / "dist"


# Self-update (desktop app only): GitHub repository whose releases carry the
# app zip, and the folder the running app was installed in (set by the
# desktop launcher; unset when running from source, which disables updates).
DEFAULT_UPDATE_REPO = "Erzyh/paperTranslate"


def get_update_repo() -> str:
    return os.environ.get("PAPERTRANSLATE_UPDATE_REPO", DEFAULT_UPDATE_REPO)


def get_install_dir() -> Path | None:
    raw = os.environ.get("PAPERTRANSLATE_INSTALL_DIR")
    return Path(raw) if raw else None


def get_models_dir() -> Path:
    """Layout/OCR ONNX models (PAPERTRANSLATE_MODELS_DIR overrides)."""
    raw = os.environ.get("PAPERTRANSLATE_MODELS_DIR")
    return Path(raw) if raw else BASE_DIR / "models"


def layout_model_enabled() -> bool:
    """False when PAPERTRANSLATE_LAYOUT_MODEL=0 (rules-only segmentation)."""
    return os.environ.get("PAPERTRANSLATE_LAYOUT_MODEL", "1") != "0"


def get_data_dir() -> Path:
    return Path(os.environ.get("PAPERTRANSLATE_DATA_DIR", str(BASE_DIR / "data")))


def get_uploads_dir() -> Path:
    return get_data_dir() / "uploads"


def get_outputs_dir() -> Path:
    return get_data_dir() / "outputs"


def get_db_path() -> Path:
    return get_data_dir() / "app.db"


def ensure_dirs() -> None:
    """Create data/, data/uploads/, data/outputs/ if missing."""
    for d in (get_data_dir(), get_uploads_dir(), get_outputs_dir()):
        d.mkdir(parents=True, exist_ok=True)


# Snapshot at import time for convenience (contract: TRANSLATOR=stub|ollama).
TRANSLATOR = get_translator_name()
