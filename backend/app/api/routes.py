"""REST API routes.

The pipeline package is imported lazily inside the worker function so the
server (and its tests) can run before/while the pipeline module is being
written.

SSE design (broadcast): every GET /events connection owns a dedicated
asyncio.Queue registered in the per-document subscriber set. The pipeline
runs in a thread executor and fans events out to all subscriber queues via
loop.call_soon_threadsafe. When there is no subscriber, events are dropped
(only the PROGRESS snapshot is updated) so nothing accumulates. All
per-document registries are cleared when a run reaches done/error; late
subscribers synthesize the terminal event from the DB status instead.

Batch scheduling: many papers can be submitted at once. Each run waits for a
slot of its engine (config.get_max_parallel) and is "queued" until then; its
DB status is already "translating" so older clients simply keep polling.
The finer-grained stage (queued/analyzing/glossary/translating/rendering) is
exposed as "phase" in the status payloads and as SSE "phase" events.
"""

import asyncio
import inspect
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import httpx
import pymupdf
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from app import config, db
from app.api import semantic_scholar

router = APIRouter()

# Reference entries without a parsed title are queried by this many leading
# characters of the raw bibliography entry.
_REF_QUERY_MAX_LEN = 120

# Valid figure asset keys ("figure-1", "table-2"); anything else is a 404
# and never reaches the filesystem.
_FIGURE_KEY_RE = re.compile(r"^(?:figure|table)-\d+$")
# Rendering parameters for GET /figures/{key}.png.
_FIGURE_DPI = 150
_FIGURE_MARGIN = 4.0  # crop margin in pt around the detected region

# Placeholder tokens (⟦EQn⟧ from extraction, ⟦Mn⟧ from masking) must never
# reach the explain prompt; the model cannot interpret them.
_PLACEHOLDER_TOKEN_RE = re.compile(r"⟦(?:EQ|M)\d+⟧")

# Test hook: inject an httpx.MockTransport so the Ollama preflight never
# touches a real server in tests. None selects the default transport.
OLLAMA_PREFLIGHT_TRANSPORT: httpx.AsyncBaseTransport | None = None
# Preflight is a metadata-only GET; keep the timeout short.
_PREFLIGHT_TIMEOUT = 5.0

# Test hook: inject an httpx.MockTransport for the one-off OpenAI translator
# so tests never touch api.openai.com. None selects the default transport.
OPENAI_TRANSPORT: httpx.BaseTransport | None = None

# Test hook: a fake Anthropic client for the Claude translator so tests never
# touch api.anthropic.com. None builds the real SDK client.
CLAUDE_CLIENT: object | None = None

# Test hook: inject an httpx.MockTransport for the POST /explain stream so
# tests never touch a real Ollama server. None selects the default transport.
OLLAMA_EXPLAIN_TRANSPORT: httpx.BaseTransport | None = None

# Fixed single-chunk answer streamed when the stub translator is active.
STUB_EXPLAIN_MESSAGE = (
    "스텁 모드에서는 설명을 제공하지 않습니다. "
    "Ollama 번역기를 활성화하면 AI 설명을 사용할 수 있습니다."
)
_EXPLAIN_MEDIA_TYPE = "text/plain; charset=utf-8"

# Per-document sets of per-subscriber SSE queues (broadcast fan-out).
SUBSCRIBERS: dict[str, set[asyncio.Queue]] = {}
# Per-document "first subscriber connected" events for the active run.
SUBSCRIBER_ARRIVED: dict[str, asyncio.Event] = {}
# Strong references to background tasks so they are not garbage-collected.
TASKS: dict[str, asyncio.Task] = {}
# In-memory progress cache fed by the runner progress callback.
PROGRESS: dict[str, dict] = {}

# --- batch scheduling (see module docstring) ------------------------------
# Current stage of each active run ("queued" until its engine slot is free).
PHASE: dict[str, str] = {}
# FIFO of runs still waiting for a slot; index + 1 is the queue position.
WAITING: list[str] = []
# Cancel flags, checked by the pipeline callbacks in the worker thread.
CANCEL: dict[str, threading.Event] = {}
# API keys of active runs — memory only, never persisted or logged (same
# contract as TranslateRequest.openai_api_key); dropped when the run ends.
CREDENTIALS: dict[str, str] = {}
# Monotonic start of the translating stage (elapsed time for ETA display).
STARTED_AT: dict[str, float] = {}
# Engine name of each active run ("ollama" | "openai" | "stub").
ENGINE: dict[str, str] = {}
# OpenAI model chosen for each active API run.
MODELS: dict[str, str] = {}
# Per-engine semaphores. asyncio primitives bind to one event loop, so they
# are rebuilt whenever the loop changes (each TestClient runs its own loop).
_SLOTS: dict[str, asyncio.Semaphore] = {}
_SLOTS_LOOP: asyncio.AbstractEventLoop | None = None
# Pipeline runs get their own threads so a large batch can never starve the
# default executor that renders figures/segments for the viewer.
_PIPELINE_EXECUTOR = ThreadPoolExecutor(max_workers=32,
                                        thread_name_prefix="pipeline")

CANCELLED_MESSAGE = "사용자가 번역을 취소했습니다."


class _JobCancelled(Exception):
    """Raised inside the worker thread when the run's cancel flag is set."""


def _engine_slot(engine: str) -> asyncio.Semaphore:
    global _SLOTS_LOOP
    loop = asyncio.get_running_loop()
    if _SLOTS_LOOP is not loop:
        _SLOTS.clear()
        _SLOTS_LOOP = loop
    slot = _SLOTS.get(engine)
    if slot is None:
        slot = asyncio.Semaphore(config.get_max_parallel(engine))
        _SLOTS[engine] = slot
    return slot


def _engine_name(api_key: str | None, api_model: str | None = None) -> str:
    """Queue/engine name: the API provider for API jobs, else the local one."""
    if api_key is not None:
        return config.API_MODELS.get(api_model or config.DEFAULT_OPENAI_MODEL,
                                     "openai")
    name = config.get_translator_name().strip().lower()
    return name if name in ("ollama", "stub") else "stub"


# Seconds between "ping" keepalive events on an idle SSE stream.
_PING_INTERVAL = 15.0
# Seconds a fresh run waits for a first subscriber before starting, so the
# head of the event stream is not dropped for clients that connect right
# after POST /translate (events are never buffered for absent subscribers).
_FIRST_SUBSCRIBER_GRACE = 1.0


def _sse(event: str, data: dict) -> str:
    """Format one server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _upload_path(doc_id: str):
    return config.get_uploads_dir() / f"{doc_id}.pdf"


def _output_path(doc_id: str):
    return config.get_outputs_dir() / f"{doc_id}.pdf"


def _get_document_or_404(doc_id: str) -> dict:
    doc = db.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    return doc


@router.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF and analyze it synchronously (page count only)."""
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    config.ensure_dirs()
    doc_id = uuid.uuid4().hex
    path = _upload_path(doc_id)
    content = await file.read()
    path.write_bytes(content)

    # Analysis step: count pages directly with pymupdf (no pipeline dependency).
    num_pages = 0
    error_detail = None
    try:
        with pymupdf.open(str(path)) as pdf:
            if pdf.needs_pass:
                error_detail = "암호로 보호된 PDF는 지원하지 않습니다."
            else:
                num_pages = pdf.page_count
    except Exception:
        error_detail = "PDF 파일을 열 수 없습니다. 올바른 PDF인지 확인해 주세요."
    if error_detail is not None:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=error_detail)

    db.insert_document(doc_id, filename, num_pages, status="uploaded")
    return {"id": doc_id, "filename": filename, "num_pages": num_pages,
            "status": "uploaded"}


def _document_payload(doc: dict) -> dict:
    """Status payload shared by GET /documents/{id} and the batch listing.

    Beyond the original contract fields it carries the batch-board extras:
    phase, engine, queue_position (1-based, queued runs only), elapsed
    seconds in the translating stage, and the page currently in progress.
    """
    doc_id = doc["id"]
    prog = PROGRESS.get(doc_id)
    if prog is None:
        done, total = db.get_progress(doc_id)
        prog = {"done": done, "total": total}
    if doc["status"] == "translating":
        phase = PHASE.get(doc_id, "queued")
    else:
        phase = doc["status"]
    started = STARTED_AT.get(doc_id)
    progress = {"done": prog.get("done", 0), "total": prog.get("total", 0)}
    if prog.get("page") is not None:
        progress["page"] = prog["page"]
    return {
        "id": doc_id,
        "filename": doc["filename"],
        "num_pages": doc["num_pages"],
        "status": doc["status"],
        "progress": progress,
        "error": doc.get("error"),
        "phase": phase,
        "engine": ENGINE.get(doc_id),
        "model": MODELS.get(doc_id),
        "queue_position": (WAITING.index(doc_id) + 1
                           if doc_id in WAITING else None),
        "elapsed": (round(time.monotonic() - started, 1)
                    if started is not None else None),
        "created_at": doc.get("created_at"),
    }


def _queue_summary() -> dict:
    running: dict[str, int] = {}
    for doc_id, phase in PHASE.items():
        if phase != "queued":
            engine = ENGINE.get(doc_id, "stub")
            running[engine] = running.get(engine, 0) + 1
    return {
        "running": running,
        "queued": len(WAITING),
        "limits": {engine: config.get_max_parallel(engine)
                   for engine in config.DEFAULT_MAX_PARALLEL},
    }


# Upper bound on ids per batch request (keeps the SQL IN list small).
_BATCH_MAX_IDS = 200


@router.get("/documents")
async def list_documents(ids: str | None = Query(default=None),
                         limit: int = Query(default=100, ge=1, le=500)):
    """Batch status for a job board: one request instead of one per paper.

    With ``ids`` (comma-separated) the documents come back in that order;
    unknown ids are listed under "missing" so the client can drop them.
    Without it, the most recent documents are returned (newest first).
    """
    if ids is not None:
        wanted = [i for i in (x.strip() for x in ids.split(",")) if i]
        wanted = list(dict.fromkeys(wanted))[:_BATCH_MAX_IDS]
        rows = db.get_documents(wanted)
        docs = [rows[i] for i in wanted if i in rows]
        missing = [i for i in wanted if i not in rows]
    else:
        docs = db.list_documents(limit)
        missing = []
    return {
        "documents": [_document_payload(d) for d in docs],
        "missing": missing,
        "queue": _queue_summary(),
    }


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    doc = _get_document_or_404(doc_id)
    return _document_payload(doc)


def _broadcast(doc_id: str, event: str, data: dict) -> None:
    """Fan one event out to every subscriber queue (event loop thread only).

    With no subscribers this is a no-op: events are dropped, never buffered.
    """
    for queue in tuple(SUBSCRIBERS.get(doc_id, ())):
        queue.put_nowait((event, data))


def _accepts_kwarg(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


async def _run_translation(doc_id: str, src_path: str, out_path: str,
                           openai_api_key: str | None = None,
                           openai_model: str | None = None,
                           ollama_model: str | None = None) -> None:
    """Wait for an engine slot, run the pipeline in a thread, broadcast events."""
    loop = asyncio.get_running_loop()
    cancel = CANCEL.setdefault(doc_id, threading.Event())
    engine = _engine_name(openai_api_key, openai_model)
    ENGINE[doc_id] = engine
    if openai_api_key is not None:
        CREDENTIALS[doc_id] = openai_api_key
        MODELS[doc_id] = openai_model or config.DEFAULT_OPENAI_MODEL
    elif engine == "ollama":
        MODELS[doc_id] = ollama_model or config.get_ollama_model()

    def set_phase(phase: str) -> None:
        # Called from the worker thread (pipeline stages).
        PHASE[doc_id] = phase
        if phase == "translating":
            STARTED_AT[doc_id] = time.monotonic()
        loop.call_soon_threadsafe(_broadcast, doc_id, "phase", {"phase": phase})

    def check_cancel() -> None:
        if cancel.is_set():
            raise _JobCancelled()

    def on_progress(info: dict) -> None:
        # Called from the pipeline worker thread.
        check_cancel()
        PROGRESS[doc_id] = {"done": info.get("done", 0),
                            "total": info.get("total", 0),
                            "page": info.get("page")}
        loop.call_soon_threadsafe(_broadcast, doc_id, "progress", info)

    def segment_cb(segment, translated=None, *args, status=None, **kwargs) -> None:
        # Called from the pipeline worker thread; persist the segment. The
        # runner passes status="fallback" when placeholder repair failed and
        # the source text was kept; older callers omit it entirely.
        check_cancel()
        seg_id = getattr(segment, "id", None)
        if seg_id is None:
            return
        if status is None:
            status = "done" if translated is not None else "pending"
        bbox = tuple(getattr(segment, "bbox", ()) or ())
        db.upsert_segment(
            doc_id,
            seg_id,
            int(getattr(segment, "page", 0)),
            bbox,
            getattr(segment, "kind", "body"),
            getattr(segment, "text", ""),
            translated,
            status=status,
        )

    def work():
        # Lazy import: the pipeline package may still be under construction.
        from app.pipeline import translate as translate_mod
        from app.pipeline.runner import run_pipeline

        check_cancel()
        credential = CREDENTIALS.get(doc_id)
        if credential is not None:
            # API job: the key exists only in memory (CREDENTIALS and the
            # translator's client) and is released with the task — never
            # persisted to the DB, files, or logs.
            model = MODELS.get(doc_id, config.DEFAULT_OPENAI_MODEL)
            if engine == "anthropic":
                from app.pipeline.claude import ClaudeTranslator

                translator = ClaudeTranslator(
                    api_key=credential, model=model,
                    effort=config.get_claude_effort(), client=CLAUDE_CLIENT)
            elif engine == "gemini":
                translator = translate_mod.OpenAITranslator(
                    api_key=credential, model=model,
                    base_url=config.GEMINI_BASE_URL,
                    chat_path=config.GEMINI_CHAT_PATH, provider="Gemini",
                    transport=OPENAI_TRANSPORT)
            else:
                translator = translate_mod.OpenAITranslator(
                    api_key=credential, model=model,
                    base_url=config.get_openai_base_url(),
                    transport=OPENAI_TRANSPORT)
        else:
            translator = (
                translate_mod.get_translator(config.get_translator_name(),
                                             ollama_model=ollama_model)
                if ollama_model else
                translate_mod.get_translator(config.get_translator_name()))
        kwargs = {}
        if _accepts_kwarg(run_pipeline, "on_phase"):
            kwargs["on_phase"] = set_phase
        else:
            set_phase("translating")
        return run_pipeline(src_path, out_path, translator,
                            on_progress=on_progress,
                            segment_cb=segment_cb, **kwargs)

    WAITING.append(doc_id)
    PHASE[doc_id] = "queued"
    try:
        # Give a client that just POSTed /translate a moment to open its SSE
        # stream; without a subscriber the early events would be dropped.
        arrived = SUBSCRIBER_ARRIVED.get(doc_id)
        if arrived is not None and not arrived.is_set():
            try:
                await asyncio.wait_for(arrived.wait(),
                                       timeout=_FIRST_SUBSCRIBER_GRACE)
            except asyncio.TimeoutError:
                pass
        async with _engine_slot(engine):
            # No await between acquiring the slot and leaving WAITING: the
            # cancel route relies on "in WAITING" meaning "not started".
            WAITING.remove(doc_id)
            check_cancel()
            await loop.run_in_executor(_PIPELINE_EXECUTOR, work)
        db.set_document_status(doc_id, "done")
        # Invalidate AFTER the status flip: an /assets call racing the flip
        # may have cached a payload without translated-side mentions; any
        # payload cached after this point is computed against status "done".
        db.clear_doc_assets(doc_id)
        _broadcast(doc_id, "done", {})
    except (_JobCancelled, asyncio.CancelledError):
        if not cancel.is_set():
            # Server shutdown, not a user cancel: leave the status alone so
            # the startup recovery reports the interrupted run.
            raise
        db.set_document_status(doc_id, "cancelled", error=CANCELLED_MESSAGE)
        _broadcast(doc_id, "error", {"detail": CANCELLED_MESSAGE})
    except Exception as exc:  # noqa: BLE001 - report any pipeline failure via SSE
        detail = f"번역 처리 중 오류가 발생했습니다: {exc}"
        db.set_document_status(doc_id, "error", error=str(exc))
        _broadcast(doc_id, "error", {"detail": detail})
    finally:
        # Terminal cleanup: nothing about this document may linger in memory.
        # Subscribers already hold the terminal event in their own queues;
        # anyone connecting later synthesizes it from the DB status.
        if doc_id in WAITING:
            WAITING.remove(doc_id)
        for registry in (SUBSCRIBER_ARRIVED, SUBSCRIBERS, PROGRESS, TASKS,
                         PHASE, CANCEL, CREDENTIALS, STARTED_AT, ENGINE,
                         MODELS):
            registry.pop(doc_id, None)


# Ollama model names ("qwen3.5:9b", "library/qwen3:14b-q4_K_M").
_OLLAMA_MODEL_RE = re.compile(r"[\w.:/+-]{1,100}")
# Installed models that cannot translate (embedding-only models).
_NON_CHAT_MODEL_HINTS = ("embed", "bge-", "minilm")


async def _ollama_tags() -> list[dict]:
    """Installed Ollama models (GET /api/tags), or 503 when Ollama is down.

    /api/tags returns metadata only — it never loads a model into VRAM, so it
    is safe on a GPU-busy machine. Endpoints that load a model (/api/generate,
    /api/chat, ...) must not be used for checks.
    """
    base_url = config.get_ollama_url().rstrip("/")
    try:
        async with httpx.AsyncClient(
            base_url=base_url,
            timeout=_PREFLIGHT_TIMEOUT,
            transport=OLLAMA_PREFLIGHT_TRANSPORT,
        ) as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
    except (httpx.HTTPError, ValueError):
        raise HTTPException(
            status_code=503,
            detail="Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인해 주세요.",
        )


async def _ollama_preflight(model: str | None = None) -> None:
    """Fail fast (503) if the Ollama server is down or lacks the model.

    ``model`` is the job's chosen local model; None checks the configured
    default (PAPERTRANSLATE_OLLAMA_MODEL).
    """
    model = model or config.get_ollama_model()
    names = {m.get("name", "") for m in await _ollama_tags()}
    # An untagged configured name matches its ":latest" variant (Ollama rule).
    candidates = {model} if ":" in model else {model, f"{model}:latest"}
    if not (names & candidates):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Ollama에 '{model}' 모델이 없습니다. "
                f"'ollama pull {model}' 실행 후 다시 시도해 주세요."
            ),
        )


@router.get("/local-models")
async def list_local_models():
    """Local models the user can pick: the installed Ollama chat models.

    ``available`` is False (with an empty list) when Ollama is not running,
    so the UI can explain instead of failing. ``default`` is the configured
    model, used when a job does not name one.
    """
    try:
        tags = await _ollama_tags()
    except HTTPException:
        return {"available": False, "models": [],
                "default": config.get_ollama_model()}
    # Ollama "cloud" models (e.g. "gemma4:31b-cloud") run on Ollama's servers:
    # the paper text would leave this PC, so they are not offered at all.
    models = [
        {"name": m.get("name", ""),
         "size_gb": round((m.get("size") or 0) / 1e9, 1)}
        for m in tags
        if m.get("name")
        and not m["name"].endswith("-cloud") and not m.get("remote_host")
        and not any(h in m["name"].lower() for h in _NON_CHAT_MODEL_HINTS)
    ]
    models.sort(key=lambda m: m["name"])
    return {"available": True, "models": models,
            "default": config.get_ollama_model()}


class TranslateRequest(BaseModel):
    # API key for the chosen API model's provider (OpenAI, Gemini, Anthropic);
    # kept in memory for this job only and never persisted or logged (see
    # _run_translation). ``openai_api_key`` / ``openai_model`` are the older
    # names of ``api_key`` / ``api_model`` and still accepted.
    api_key: str | None = None
    api_model: str | None = None
    openai_api_key: str | None = None
    openai_model: str | None = None
    # Installed Ollama model for a local job (GET /local-models); None = the
    # configured default.
    ollama_model: str | None = None


# Cheap key-format checks per provider (a wrong key would otherwise only
# surface after the paper was analyzed).
_KEY_PREFIXES = {
    "openai": ("sk-", "OpenAI API 키 형식이 올바르지 않습니다 (sk-로 시작해야 합니다)."),
    "anthropic": ("sk-ant-",
                  "Anthropic API 키 형식이 올바르지 않습니다 (sk-ant-로 시작해야 합니다)."),
}


@router.post("/documents/{doc_id}/translate")
async def start_translation(doc_id: str, request: TranslateRequest | None = None):
    doc = _get_document_or_404(doc_id)
    if doc["status"] == "translating":
        # Already running; idempotent response.
        return {"status": "translating"}

    openai_api_key = None
    openai_model = None
    if request is not None:
        openai_api_key = request.api_key or request.openai_api_key
        openai_model = request.api_model or request.openai_model
    if openai_api_key is not None and not openai_api_key.strip():
        openai_api_key = None
    if openai_model is not None and openai_model not in config.API_MODELS:
        raise HTTPException(status_code=400,
                            detail=f"지원하지 않는 모델입니다: {openai_model}")
    if openai_api_key is not None:
        # API path: no Ollama preflight; only a cheap key-format check.
        # Other OpenAI-compatible servers use their own key formats.
        provider = _engine_name(openai_api_key, openai_model)
        prefix, message = _KEY_PREFIXES.get(provider, ("", ""))
        custom = provider == "openai" and config.using_custom_openai_server()
        if prefix and not custom and not openai_api_key.startswith(prefix):
            raise HTTPException(status_code=400, detail=message)
    ollama_model = None
    if openai_api_key is None:
        ollama_model = request.ollama_model if request is not None else None
        if ollama_model is not None and not _OLLAMA_MODEL_RE.fullmatch(ollama_model):
            raise HTTPException(status_code=400,
                                detail=f"올바르지 않은 모델 이름입니다: {ollama_model}")
        if config.get_translator_name().strip().lower() == "ollama":
            await _ollama_preflight(ollama_model)
        else:
            ollama_model = None  # stub translator: nothing to choose

    config.ensure_dirs()
    db.clear_segments(doc_id)
    # Re-translation invalidates the cached assets payload (contract).
    db.clear_doc_assets(doc_id)
    db.set_document_status(doc_id, "translating")
    PROGRESS[doc_id] = {"done": 0, "total": 0}
    arrived = asyncio.Event()
    if SUBSCRIBERS.get(doc_id):
        # A subscriber is already connected (e.g. a reconnected EventSource);
        # the run does not need to wait for one.
        arrived.set()
    SUBSCRIBER_ARRIVED[doc_id] = arrived

    task = asyncio.create_task(
        _run_translation(doc_id, str(_upload_path(doc_id)),
                         str(_output_path(doc_id)),
                         openai_api_key=openai_api_key,
                         openai_model=openai_model,
                         ollama_model=ollama_model)
    )
    TASKS[doc_id] = task
    return {"status": "translating"}


@router.post("/documents/{doc_id}/cancel")
async def cancel_translation(doc_id: str):
    """Cancel a queued or running translation (idempotent).

    A queued run is dropped at once. A running one stops at the next segment
    boundary (the pipeline thread cannot be interrupted mid-request), so its
    status flips to "cancelled" shortly after this returns.
    """
    doc = _get_document_or_404(doc_id)
    if doc["status"] != "translating":
        return {"status": doc["status"]}
    flag = CANCEL.get(doc_id)
    if flag is None:
        # Stale "translating" row without a task in this process.
        db.set_document_status(doc_id, "cancelled", error=CANCELLED_MESSAGE)
        return {"status": "cancelled"}
    flag.set()
    task = TASKS.get(doc_id)
    if doc_id in WAITING and task is not None:
        # Still waiting for a slot: nothing runs in a thread yet.
        task.cancel()
    return {"status": "cancelling"}


def _terminal_sse(doc: dict) -> str | None:
    """Return a synthesized terminal SSE frame if the document is finished."""
    if doc["status"] == "done":
        return _sse("done", {})
    if doc["status"] == "error":
        return _sse("error", {"detail": doc.get("error")
                              or "번역 처리 중 오류가 발생했습니다."})
    if doc["status"] == "cancelled":
        return _sse("error", {"detail": doc.get("error") or CANCELLED_MESSAGE})
    return None


@router.get("/documents/{doc_id}/events")
async def stream_events(doc_id: str):
    _get_document_or_404(doc_id)

    async def gen():
        # Late subscriber: the run already finished (or never happened in
        # this process) — synthesize the terminal event from the DB status.
        doc = db.get_document(doc_id)
        if doc is not None:
            terminal = _terminal_sse(doc)
            if terminal is not None:
                yield terminal
                return

        # Live subscription: register a dedicated queue for this connection.
        # No await between the status check above and the registration, so a
        # terminal broadcast cannot slip through the gap.
        queue: asyncio.Queue = asyncio.Queue()
        SUBSCRIBERS.setdefault(doc_id, set()).add(queue)
        arrived = SUBSCRIBER_ARRIVED.get(doc_id)
        if arrived is not None:
            arrived.set()
        try:
            while True:
                try:
                    event, data = await asyncio.wait_for(
                        queue.get(), timeout=_PING_INTERVAL)
                except asyncio.TimeoutError:
                    doc = db.get_document(doc_id)
                    if doc is not None:
                        terminal = _terminal_sse(doc)
                        if terminal is not None:
                            yield terminal
                            return
                    # Real ping event (comment lines are invisible to
                    # EventSource, so the client could not detect liveness).
                    yield _sse("ping", {})
                    continue
                yield _sse(event, data)
                if event in ("done", "error"):
                    return
        finally:
            subs = SUBSCRIBERS.get(doc_id)
            if subs is not None:
                subs.discard(queue)
                if not subs:
                    SUBSCRIBERS.pop(doc_id, None)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/documents/{doc_id}/segments")
async def list_segments(doc_id: str, page: int | None = None):
    _get_document_or_404(doc_id)
    rows = db.list_segments(doc_id, page)
    return [
        {
            "seg_id": row["seg_id"],
            "page": row["page"],
            "bbox": row["bbox"],
            "kind": row["kind"],
            "source": row["source"],
            "translated": row["translated"],
        }
        for row in rows
    ]


def _get_segment_or_404(doc_id: str, seg_id: str) -> dict:
    """Return one stored segment row (bbox already JSON-decoded) or 404."""
    row = next((r for r in db.list_segments(doc_id) if r["seg_id"] == seg_id),
               None)
    if row is None:
        raise HTTPException(status_code=404, detail="세그먼트를 찾을 수 없습니다.")
    return row


def _segment_clip_rect(page: pymupdf.Page, seg: dict,
                       margin: float = 0.0) -> pymupdf.Rect:
    """Segment bbox (optionally padded) clamped to the page rect."""
    rect = pymupdf.Rect(*seg["bbox"]) + (-margin, -margin, margin, margin)
    rect.intersect(page.rect)
    return rect


@router.get("/documents/{doc_id}/segments/{seg_id}.png")
async def get_segment_png(doc_id: str, seg_id: str):
    """Crop-render one segment region from the source PDF (PNG, dpi 150).

    Same rendering parameters as GET /figures/{key}.png, but with no file
    cache: segment crops are small, so rendering per request is fine.
    """
    _get_document_or_404(doc_id)
    seg = _get_segment_or_404(doc_id, seg_id)
    src = _upload_path(doc_id)
    if not src.exists():
        raise HTTPException(status_code=404, detail="원본 PDF 파일을 찾을 수 없습니다.")

    def render() -> bytes:
        with pymupdf.open(str(src)) as pdf:
            page = pdf.load_page(int(seg["page"]))
            rect = _segment_clip_rect(page, seg, margin=_FIGURE_MARGIN)
            pix = page.get_pixmap(clip=rect, dpi=_FIGURE_DPI)
            return pix.tobytes("png")

    png = await asyncio.get_running_loop().run_in_executor(None, render)
    return Response(content=png, media_type="image/png")


@router.get("/documents/{doc_id}/original.pdf")
async def get_original_pdf(doc_id: str):
    doc = _get_document_or_404(doc_id)
    path = _upload_path(doc_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="원본 PDF 파일을 찾을 수 없습니다.")
    return FileResponse(str(path), media_type="application/pdf",
                        filename=doc["filename"])


@router.get("/documents/{doc_id}/output.pdf")
async def get_output_pdf(doc_id: str):
    doc = _get_document_or_404(doc_id)
    path = _output_path(doc_id)
    if doc["status"] != "done" or not path.exists():
        raise HTTPException(status_code=404, detail="번역 PDF가 아직 준비되지 않았습니다.")
    return FileResponse(str(path), media_type="application/pdf",
                        filename=f"translated_{doc['filename']}")


def _figures_dir(doc_id: str):
    return config.get_data_dir() / "figures" / doc_id


async def _load_assets(doc_id: str, doc: dict) -> dict:
    """Return the assets payload for a document, computing and caching it.

    Translated-side mentions are only searched when the document is done and
    the output PDF exists; otherwise they stay empty per the contract. The
    cache is invalidated when a (re-)translation starts and when it finishes.
    """
    cached = db.get_doc_assets(doc_id)
    if cached is not None:
        return cached
    src = _upload_path(doc_id)
    if not src.exists():
        raise HTTPException(status_code=404, detail="원본 PDF 파일을 찾을 수 없습니다.")
    out = _output_path(doc_id)
    translated = str(out) if doc["status"] == "done" and out.exists() else None

    def work():
        # Lazy import: consistent with the worker-side pipeline imports.
        from app.pipeline.assets import compute_assets

        return compute_assets(str(src), translated)

    payload = await asyncio.get_running_loop().run_in_executor(None, work)
    # Cache only when the document status did not change during the
    # computation: a payload computed against a "translating" document could
    # otherwise land AFTER the done-transition invalidation and pin a
    # translated-empty payload in the cache.
    current = db.get_document(doc_id)
    if current is not None and current["status"] == doc["status"]:
        db.set_doc_assets(doc_id, payload)
    return payload


@router.get("/documents/{doc_id}/assets")
async def get_assets(doc_id: str):
    doc = _get_document_or_404(doc_id)
    return await _load_assets(doc_id, doc)


@router.get("/documents/{doc_id}/figures/{key}.png")
async def get_figure_png(doc_id: str, key: str):
    doc = _get_document_or_404(doc_id)
    if _FIGURE_KEY_RE.fullmatch(key) is None:
        raise HTTPException(status_code=404, detail="그림을 찾을 수 없습니다.")
    payload = await _load_assets(doc_id, doc)
    figure = next((f for f in payload["figures"] if f["key"] == key), None)
    if figure is None:
        raise HTTPException(status_code=404, detail="그림을 찾을 수 없습니다.")
    png_path = _figures_dir(doc_id) / f"{key}.png"
    if not png_path.exists():
        src = _upload_path(doc_id)
        if not src.exists():
            raise HTTPException(status_code=404,
                                detail="원본 PDF 파일을 찾을 수 없습니다.")

        def render():
            png_path.parent.mkdir(parents=True, exist_ok=True)
            with pymupdf.open(str(src)) as pdf:
                page = pdf.load_page(int(figure["page"]))
                rect = pymupdf.Rect(*figure["bbox"]) + (
                    -_FIGURE_MARGIN, -_FIGURE_MARGIN,
                    _FIGURE_MARGIN, _FIGURE_MARGIN,
                )
                rect.intersect(page.rect)
                pix = page.get_pixmap(clip=rect, dpi=_FIGURE_DPI)
                pix.save(str(png_path))

        await asyncio.get_running_loop().run_in_executor(None, render)
    return FileResponse(str(png_path), media_type="image/png")


@router.get("/documents/{doc_id}/references/{ref_num}/details")
async def get_reference_details(doc_id: str, ref_num: int):
    """Look up one bibliography entry on Semantic Scholar (cached).

    Only found=true payloads are cached, so a transient failure (timeout,
    HTTP error, no match) is retried on the next request. References come
    from the source PDF alone, so a cached hit survives re-translation.
    """
    doc = _get_document_or_404(doc_id)
    cached = db.get_ref_details(doc_id, ref_num)
    if cached is not None:
        return cached
    payload = await _load_assets(doc_id, doc)
    reference = payload["references"].get(str(ref_num))
    if reference is None:
        raise HTTPException(status_code=404, detail="참고문헌 항목을 찾을 수 없습니다.")
    query = reference.get("title") or reference["entry"][:_REF_QUERY_MAX_LEN]
    details = await semantic_scholar.fetch_paper_details(query)
    if details["found"]:
        db.set_ref_details(doc_id, ref_num, details)
    return details


class ExplainRequest(BaseModel):
    text: str
    kind: Literal["selection", "formula"]
    # When set, the model input is re-extracted from the source PDF via the
    # segment bbox (the stored source may hold ⟦EQn⟧ placeholders instead of
    # the real formula characters); the body text is ignored then.
    segment_id: str | None = None


def _first_heading_source(doc_id: str) -> str | None:
    """Document title: source text of the first heading segment, if any."""
    for row in db.list_segments(doc_id):
        if row["kind"] == "heading":
            return row["source"]
    return None


@router.post("/documents/{doc_id}/explain")
async def explain(doc_id: str, request: ExplainRequest):
    """Stream a Korean AI explanation for a selection or formula (text/plain).

    With the stub translator the stream is one fixed notice chunk. With the
    Ollama translator the /api/tags preflight runs first (503 before the
    stream starts when the server is down), then the /api/chat stream=true
    answer chunks are proxied through as plain text.

    With segment_id set, the model input is the segment region's text
    re-extracted from the source PDF (real formula characters instead of the
    stored ⟦EQn⟧ placeholders); the body text is ignored. Unknown ids: 404.
    """
    _get_document_or_404(doc_id)

    segment = (None if request.segment_id is None
               else _get_segment_or_404(doc_id, request.segment_id))

    if config.get_translator_name().strip().lower() != "ollama":
        return StreamingResponse(iter([STUB_EXPLAIN_MESSAGE]),
                                 media_type=_EXPLAIN_MEDIA_TYPE)

    # Fail fast with a 503 while headers can still carry it.
    await _ollama_preflight()

    text = request.text
    if segment is not None:
        src = _upload_path(doc_id)
        if not src.exists():
            raise HTTPException(status_code=404,
                                detail="원본 PDF 파일을 찾을 수 없습니다.")

        def extract_clip() -> str:
            with pymupdf.open(str(src)) as pdf:
                page = pdf.load_page(int(segment["page"]))
                return page.get_text(clip=_segment_clip_rect(page, segment))

        clipped = await asyncio.get_running_loop().run_in_executor(
            None, extract_clip)
        # Whitespace-normalized; empty extractions fall back to the body text.
        text = " ".join(clipped.split()) or text
    # Placeholder tokens carry no meaning for the model; strip any leftovers.
    text = _PLACEHOLDER_TOKEN_RE.sub("", text)

    # Lazy import: consistent with the worker-side pipeline imports.
    from app.pipeline.translate import (
        OllamaTranslator,
        TranslatorUnavailableError,
        build_explain_prompt,
    )

    prompt = build_explain_prompt(text, request.kind,
                                  _first_heading_source(doc_id))
    translator = OllamaTranslator(
        model=config.get_ollama_model(),
        base_url=config.get_ollama_url(),
        timeout=config.get_ollama_timeout(),
        transport=OLLAMA_EXPLAIN_TRANSPORT,
    )

    def gen():
        # Runs in a threadpool (sync generator). A failure here happens after
        # the 200 header went out, so it degrades to a Korean notice chunk.
        try:
            yield from translator.explain_stream(prompt)
        except TranslatorUnavailableError as exc:
            yield f"\n[오류] {exc}"
        finally:
            translator.close()

    return StreamingResponse(gen(), media_type=_EXPLAIN_MEDIA_TYPE)
