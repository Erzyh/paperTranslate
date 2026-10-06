"""Server-layer tests: upload -> translate -> SSE events -> state transitions.

The pipeline package is faked via sys.modules injection (monkeypatch), so
these tests never execute the real pipeline, never touch Ollama, and never
use the GPU. Upload analysis (num_pages) uses pymupdf directly.
"""

import json
import sys
import threading
import time
import types
from pathlib import Path

import httpx
import pymupdf
import pytest
from fastapi.testclient import TestClient

# Imported at module load time (before any sys.modules fake is installed) so
# the real exception class stays reachable; importing it runs no pipeline
# code and touches no network/GPU.
from app.pipeline.translate import TranslatorUnavailableError
# db.get_progress imports app.pipeline.segment lazily; load the real module
# now so it stays importable while the fake pipeline package is installed
# (otherwise this file fails when run on its own).
import app.pipeline.segment  # noqa: E402,F401

TOTAL_FAKE_SEGMENTS = 3


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    monkeypatch.setenv("PAPERTRANSLATE_DATA_DIR", str(d))
    monkeypatch.setenv("PAPERTRANSLATE_TRANSLATOR", "stub")
    return d


@pytest.fixture()
def fake_pipeline(monkeypatch):
    """Install fake app.pipeline.{runner,translate} modules in sys.modules.

    The route layer imports the pipeline lazily inside the worker function,
    so the fake modules are picked up even if the real package is absent
    or half-written.
    """
    state = {"runs": 0, "raise_error": False, "gate": None,
             "raise_exc": None, "segment_status": None}

    def fake_run_pipeline(src_path, out_path, translator,
                          on_progress=None, segment_cb=None):
        state["runs"] += 1
        if state["gate"] is not None:
            # Test-controlled gate: hold the worker thread until the test
            # has finished setting up (e.g. registering SSE subscribers).
            state["gate"].wait(timeout=10.0)
        if state["raise_exc"] is not None:
            raise state["raise_exc"]
        if state["raise_error"]:
            raise RuntimeError("boom")
        total = TOTAL_FAKE_SEGMENTS
        for i in range(total):
            seg = types.SimpleNamespace(
                id=f"p0_s{i}",
                page=0,
                column=0,
                bbox=(10.0, 20.0 + 30.0 * i, 200.0, 40.0 + 30.0 * i),
                text=f"Source sentence {i}.",
                kind="body",
                font_size=10.0,
            )
            translated = f"더미 번역 문장 {i}."
            if segment_cb is not None:
                if state["segment_status"] is not None:
                    segment_cb(seg, translated, status=state["segment_status"])
                else:
                    segment_cb(seg, translated)
            if on_progress is not None:
                on_progress({
                    "segment_id": seg.id,
                    "done": i + 1,
                    "total": total,
                    "page": seg.page,
                    "num_pages": 1,
                    "source_preview": seg.text,
                    "translated_preview": translated,
                })
        Path(out_path).write_bytes(b"%PDF-1.4\n% fake translated output\n")
        return types.SimpleNamespace(overflow_segments=[], scaled_segments={})

    class FakeTranslator:
        def translate(self, text, context=None):
            return text

    def fake_get_translator(name, ollama_model=None):
        state["translator_name"] = name
        state["ollama_model"] = ollama_model
        return FakeTranslator()

    pipeline_pkg = types.ModuleType("app.pipeline")
    pipeline_pkg.__path__ = []  # mark as package
    runner_mod = types.ModuleType("app.pipeline.runner")
    runner_mod.run_pipeline = fake_run_pipeline
    translate_mod = types.ModuleType("app.pipeline.translate")
    translate_mod.get_translator = fake_get_translator
    pipeline_pkg.runner = runner_mod
    pipeline_pkg.translate = translate_mod

    monkeypatch.setitem(sys.modules, "app.pipeline", pipeline_pkg)
    monkeypatch.setitem(sys.modules, "app.pipeline.runner", runner_mod)
    monkeypatch.setitem(sys.modules, "app.pipeline.translate", translate_mod)
    return state


@pytest.fixture()
def client(data_dir, fake_pipeline):
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def sample_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    for i in range(2):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 100), f"Hello world page {i}")
    doc.save(str(path))
    doc.close()
    return path


def _upload(client, sample_pdf) -> dict:
    resp = client.post(
        "/api/documents",
        files={"file": ("sample.pdf", sample_pdf.read_bytes(),
                        "application/pdf")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _read_sse_events(resp, max_pings=4) -> list[tuple[str, dict]]:
    """Parse SSE frames from an open stream until a terminal event.

    "ping" keepalive events are dropped (with a stall guard); SSE comment
    lines must no longer appear per the keepalive contract.
    """
    events = []
    pings = 0
    event_name = None
    for line in resp.iter_lines():
        if not line:
            continue
        assert not line.startswith(":"), (
            "comment keepalives are forbidden; use ping events")
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data = json.loads(line.split(":", 1)[1].strip())
            if event_name == "ping":
                pings += 1
                assert pings <= max_pings, "SSE stream stalled"
                continue
            events.append((event_name, data))
            if event_name in ("done", "error"):
                break
    return events


def _collect_sse_events(client, doc_id) -> list[tuple[str, dict]]:
    with client.stream("GET", f"/api/documents/{doc_id}/events") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        return _read_sse_events(resp)


def test_upload_and_analyze(client, sample_pdf):
    body = _upload(client, sample_pdf)
    assert body["id"]
    assert body["filename"] == "sample.pdf"
    assert body["num_pages"] == 2
    assert body["status"] == "uploaded"

    resp = client.get(f"/api/documents/{body['id']}")
    assert resp.status_code == 200
    info = resp.json()
    assert info["status"] == "uploaded"
    assert info["num_pages"] == 2
    assert info["progress"] == {"done": 0, "total": 0}


def test_upload_rejects_non_pdf(client):
    resp = client.post(
        "/api/documents",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400


def test_upload_rejects_broken_pdf(client):
    resp = client.post(
        "/api/documents",
        files={"file": ("bad.pdf", b"this is not a pdf",
                        "application/pdf")},
    )
    assert resp.status_code == 400


def test_unknown_document_404(client):
    assert client.get("/api/documents/nope").status_code == 404
    assert client.post("/api/documents/nope/translate").status_code == 404
    assert client.get("/api/documents/nope/segments").status_code == 404
    assert client.get("/api/documents/nope/original.pdf").status_code == 404
    assert client.get("/api/documents/nope/output.pdf").status_code == 404


def test_output_pdf_404_before_done(client, sample_pdf):
    body = _upload(client, sample_pdf)
    resp = client.get(f"/api/documents/{body['id']}/output.pdf")
    assert resp.status_code == 404


def test_translate_flow_with_sse(client, sample_pdf, fake_pipeline):
    body = _upload(client, sample_pdf)
    doc_id = body["id"]

    resp = client.post(f"/api/documents/{doc_id}/translate")
    assert resp.status_code == 200
    assert resp.json() == {"status": "translating"}

    events = _collect_sse_events(client, doc_id)

    assert fake_pipeline["runs"] == 1
    assert fake_pipeline["translator_name"] == "stub"

    progress_events = [e for e in events if e[0] == "progress"]
    assert len(progress_events) == TOTAL_FAKE_SEGMENTS
    first = progress_events[0][1]
    assert first["segment_id"] == "p0_s0"
    assert first["done"] == 1
    assert first["total"] == TOTAL_FAKE_SEGMENTS
    assert first["source_preview"] == "Source sentence 0."
    assert first["translated_preview"] == "더미 번역 문장 0."
    assert events[-1][0] == "done"

    # State transition: translating -> done, progress complete.
    info = client.get(f"/api/documents/{doc_id}").json()
    assert info["status"] == "done"
    assert info["progress"] == {"done": TOTAL_FAKE_SEGMENTS,
                                "total": TOTAL_FAKE_SEGMENTS}

    # Segments persisted through segment_cb.
    segs = client.get(f"/api/documents/{doc_id}/segments").json()
    assert len(segs) == TOTAL_FAKE_SEGMENTS
    for i, seg in enumerate(segs):
        assert seg["seg_id"] == f"p0_s{i}"
        assert seg["page"] == 0
        assert isinstance(seg["bbox"], list) and len(seg["bbox"]) == 4
        assert seg["kind"] == "body"
        assert seg["source"] == f"Source sentence {i}."
        assert seg["translated"] == f"더미 번역 문장 {i}."

    # Page filter.
    assert len(client.get(
        f"/api/documents/{doc_id}/segments", params={"page": 0}).json()) == 3
    assert client.get(
        f"/api/documents/{doc_id}/segments", params={"page": 1}).json() == []

    # PDFs downloadable.
    orig = client.get(f"/api/documents/{doc_id}/original.pdf")
    assert orig.status_code == 200
    assert orig.content.startswith(b"%PDF")

    out = client.get(f"/api/documents/{doc_id}/output.pdf")
    assert out.status_code == 200
    assert out.content.startswith(b"%PDF")

    # Late subscriber after completion gets a terminal event immediately.
    late = _collect_sse_events(client, doc_id)
    assert late == [("done", {})]


def test_translate_error_path(client, sample_pdf, fake_pipeline):
    fake_pipeline["raise_error"] = True
    body = _upload(client, sample_pdf)
    doc_id = body["id"]

    resp = client.post(f"/api/documents/{doc_id}/translate")
    assert resp.status_code == 200

    events = _collect_sse_events(client, doc_id)
    assert events[-1][0] == "error"
    assert "boom" in events[-1][1]["detail"]

    info = client.get(f"/api/documents/{doc_id}").json()
    assert info["status"] == "error"
    assert client.get(f"/api/documents/{doc_id}/output.pdf").status_code == 404


def _wait_for_status(client, doc_id, expected, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/api/documents/{doc_id}").json()["status"]
        if status == expected:
            return
        time.sleep(0.02)
    pytest.fail(f"document never reached status {expected!r}")


def test_second_run_not_poisoned_by_stale_queue(client, sample_pdf, fake_pipeline):
    """Regression: unconsumed events of a failed run must not be replayed.

    Run 1 fails with no SSE subscriber, leaving a stale error event queued.
    Run 2 then starts; its /events stream must deliver run 2's progress and
    done -- never the stale error from run 1.
    """
    body = _upload(client, sample_pdf)
    doc_id = body["id"]

    fake_pipeline["raise_error"] = True
    assert client.post(f"/api/documents/{doc_id}/translate").status_code == 200
    # Intentionally no /events subscription here.
    _wait_for_status(client, doc_id, "error")

    fake_pipeline["raise_error"] = False
    assert client.post(f"/api/documents/{doc_id}/translate").status_code == 200

    events = _collect_sse_events(client, doc_id)
    names = [name for name, _ in events]
    assert "error" not in names, f"stale error replayed: {events}"
    assert names[-1] == "done"
    assert names.count("done") == 1
    assert len([n for n in names if n == "progress"]) == TOTAL_FAKE_SEGMENTS
    assert fake_pipeline["runs"] == 2

    info = client.get(f"/api/documents/{doc_id}").json()
    assert info["status"] == "done"


def test_two_concurrent_subscribers_receive_all_events(client, sample_pdf,
                                                       fake_pipeline):
    """Broadcast fan-out: two simultaneous /events subscribers must EACH
    receive every progress event and the terminal done (no consumption
    competition on a shared queue)."""
    from app.api import routes

    body = _upload(client, sample_pdf)
    doc_id = body["id"]

    gate = threading.Event()
    fake_pipeline["gate"] = gate

    assert client.post(f"/api/documents/{doc_id}/translate").status_code == 200

    results: dict[str, list] = {}
    errors: dict[str, BaseException] = {}

    def subscribe(key: str) -> None:
        try:
            results[key] = _collect_sse_events(client, doc_id)
        except BaseException as exc:  # noqa: BLE001 - reported via errors
            errors[key] = exc

    workers = [threading.Thread(target=subscribe, args=(key,), daemon=True)
               for key in ("a", "b")]
    try:
        for worker in workers:
            worker.start()
        # Wait until both subscriber queues are registered, then release
        # the pipeline so event emission starts deterministically after.
        deadline = time.monotonic() + 5.0
        while len(routes.SUBSCRIBERS.get(doc_id, ())) < 2:
            assert time.monotonic() < deadline, "subscribers never registered"
            time.sleep(0.01)
    finally:
        gate.set()
    for worker in workers:
        worker.join(timeout=10.0)
        assert not worker.is_alive(), "subscriber thread did not finish"
    assert not errors, f"subscriber failed: {errors}"

    expected_done_counts = list(range(1, TOTAL_FAKE_SEGMENTS + 1))
    for key in ("a", "b"):
        names = [name for name, _ in results[key]]
        assert names.count("progress") == TOTAL_FAKE_SEGMENTS, (key, names)
        assert names.count("done") == 1, (key, names)
        assert names[-1] == "done", (key, names)
        progress_counts = [data["done"] for name, data in results[key]
                           if name == "progress"]
        assert progress_counts == expected_done_counts, (key, progress_counts)


def test_unsubscribed_run_cleans_registries_and_serves_late_done(
        client, sample_pdf, fake_pipeline):
    """A run with zero subscribers must not accumulate events anywhere:
    after completion all per-document registries are empty and a late
    subscriber immediately receives a synthesized done."""
    from app.api import routes

    body = _upload(client, sample_pdf)
    doc_id = body["id"]

    assert client.post(f"/api/documents/{doc_id}/translate").status_code == 200
    # Intentionally no /events subscription during the whole run.
    _wait_for_status(client, doc_id, "done")

    # Terminal cleanup runs atomically with the status transition, so by
    # the time 'done' is observable nothing may linger in memory.
    assert routes.SUBSCRIBERS == {}
    assert routes.SUBSCRIBER_ARRIVED == {}
    assert routes.TASKS == {}
    assert routes.PROGRESS == {}

    # Late subscriber: exactly one synthesized terminal event, no replay.
    events = _collect_sse_events(client, doc_id)
    assert events == [("done", {})]

    # Progress falls back to the DB after the snapshot cache was cleared.
    info = client.get(f"/api/documents/{doc_id}").json()
    assert info["progress"] == {"done": TOTAL_FAKE_SEGMENTS,
                                "total": TOTAL_FAKE_SEGMENTS}


def test_restart_marks_stale_translating_as_error(data_dir):
    """Startup recovery: a document left in 'translating' by a previous
    process is marked as error when the app boots."""
    from app import db
    from app.main import app

    db.init_db()
    db.insert_document("stale-doc", "stale.pdf", 3, status="translating")

    with TestClient(app) as client:
        info = client.get("/api/documents/stale-doc").json()
        assert info["status"] == "error"
        assert "서버가 재시작되어" in info["error"]

        # SSE subscribers get the synthesized error immediately.
        events = _collect_sse_events(client, "stale-doc")
        assert len(events) == 1
        assert events[0][0] == "error"
        assert "서버가 재시작되어" in events[0][1]["detail"]


def _install_preflight_transport(monkeypatch, handler) -> None:
    """Route the Ollama preflight through an in-process MockTransport."""
    from app.api import routes

    monkeypatch.setattr(routes, "OLLAMA_PREFLIGHT_TRANSPORT",
                        httpx.MockTransport(handler))


def test_preflight_503_when_ollama_down(client, sample_pdf, fake_pipeline,
                                        monkeypatch):
    def down(request):
        raise httpx.ConnectError("connection refused")

    _install_preflight_transport(monkeypatch, down)
    monkeypatch.setenv("PAPERTRANSLATE_TRANSLATOR", "ollama")

    doc_id = _upload(client, sample_pdf)["id"]
    resp = client.post(f"/api/documents/{doc_id}/translate")
    assert resp.status_code == 503
    assert "Ollama 서버에 연결할 수 없습니다" in resp.json()["detail"]

    # The run never started and the document was not touched.
    assert fake_pipeline["runs"] == 0
    assert client.get(f"/api/documents/{doc_id}").json()["status"] == "uploaded"


def test_preflight_503_when_model_missing(client, sample_pdf, fake_pipeline,
                                          monkeypatch):
    def tags_without_model(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "llama3:8b"}]})

    _install_preflight_transport(monkeypatch, tags_without_model)
    monkeypatch.setenv("PAPERTRANSLATE_TRANSLATOR", "ollama")
    monkeypatch.setenv("PAPERTRANSLATE_OLLAMA_MODEL", "qwen3:8b")

    doc_id = _upload(client, sample_pdf)["id"]
    resp = client.post(f"/api/documents/{doc_id}/translate")
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert "qwen3:8b" in detail
    assert "ollama pull" in detail
    assert fake_pipeline["runs"] == 0


def test_preflight_passes_and_run_completes(client, sample_pdf, fake_pipeline,
                                            monkeypatch):
    def tags_with_model(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [
            {"name": "qwen3:8b"}, {"name": "llama3:8b"},
        ]})

    _install_preflight_transport(monkeypatch, tags_with_model)
    monkeypatch.setenv("PAPERTRANSLATE_TRANSLATOR", "ollama")
    monkeypatch.setenv("PAPERTRANSLATE_OLLAMA_MODEL", "qwen3:8b")

    doc_id = _upload(client, sample_pdf)["id"]
    resp = client.post(f"/api/documents/{doc_id}/translate")
    assert resp.status_code == 200
    assert resp.json() == {"status": "translating"}

    events = _collect_sse_events(client, doc_id)
    assert events[-1][0] == "done"
    assert fake_pipeline["runs"] == 1
    assert fake_pipeline["translator_name"] == "ollama"


def test_stub_translator_skips_preflight(client, sample_pdf, fake_pipeline,
                                         monkeypatch):
    def must_not_be_called(request):
        raise AssertionError("preflight must not run for the stub translator")

    _install_preflight_transport(monkeypatch, must_not_be_called)
    # data_dir fixture already set PAPERTRANSLATE_TRANSLATOR=stub.

    doc_id = _upload(client, sample_pdf)["id"]
    assert client.post(f"/api/documents/{doc_id}/translate").status_code == 200
    events = _collect_sse_events(client, doc_id)
    assert events[-1][0] == "done"


def test_translator_unavailable_error_reaches_sse(client, sample_pdf,
                                                  fake_pipeline):
    fake_pipeline["raise_exc"] = TranslatorUnavailableError()

    doc_id = _upload(client, sample_pdf)["id"]
    assert client.post(f"/api/documents/{doc_id}/translate").status_code == 200

    events = _collect_sse_events(client, doc_id)
    assert events[-1][0] == "error"
    assert "Ollama 서버에 연결할 수 없습니다" in events[-1][1]["detail"]

    info = client.get(f"/api/documents/{doc_id}").json()
    assert info["status"] == "error"
    assert "Ollama 서버에 연결할 수 없습니다" in info["error"]


def test_segment_fallback_status_persisted(client, sample_pdf, fake_pipeline):
    from app import db

    fake_pipeline["segment_status"] = "fallback"

    doc_id = _upload(client, sample_pdf)["id"]
    assert client.post(f"/api/documents/{doc_id}/translate").status_code == 200
    events = _collect_sse_events(client, doc_id)
    assert events[-1][0] == "done"

    rows = db.list_segments(doc_id)
    assert len(rows) == TOTAL_FAKE_SEGMENTS
    assert all(row["status"] == "fallback" for row in rows)


def test_cors_allows_frontend_origin(client):
    resp = client.options(
        "/api/documents",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


# --- batch scheduling ------------------------------------------------------

def _wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("condition never became true")


def _batch(client, ids):
    resp = client.get("/api/documents", params={"ids": ",".join(ids)})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_batch_runs_respect_engine_limit_and_queue(client, sample_pdf,
                                                  fake_pipeline, monkeypatch):
    """Only MAX_PARALLEL runs execute at once; the rest wait in FIFO order
    and report phase "queued" with a 1-based queue position."""
    monkeypatch.setenv("PAPERTRANSLATE_MAX_PARALLEL_STUB", "1")
    gate = threading.Event()
    fake_pipeline["gate"] = gate
    ids = [_upload(client, sample_pdf)["id"] for _ in range(3)]
    for doc_id in ids:
        assert client.post(f"/api/documents/{doc_id}/translate").status_code == 200

    _wait_until(lambda: fake_pipeline["runs"] == 1
                and _batch(client, ids)["queue"]["queued"] == 2)
    board = _batch(client, ids)
    phases = [d["phase"] for d in board["documents"]]
    assert phases == ["translating", "queued", "queued"]
    assert [d["queue_position"] for d in board["documents"]] == [None, 1, 2]
    assert all(d["status"] == "translating" for d in board["documents"])
    assert board["queue"]["running"] == {"stub": 1}
    assert board["queue"]["limits"]["stub"] == 1
    time.sleep(0.2)
    assert fake_pipeline["runs"] == 1  # still capped

    gate.set()
    for doc_id in ids:
        _wait_for_status(client, doc_id, "done")
    assert fake_pipeline["runs"] == 3

    from app.api import routes
    assert routes.WAITING == [] and routes.PHASE == {}
    assert routes.CANCEL == {} and routes.CREDENTIALS == {}


def test_cancel_queued_and_running(client, sample_pdf, fake_pipeline,
                                   monkeypatch):
    monkeypatch.setenv("PAPERTRANSLATE_MAX_PARALLEL_STUB", "1")
    gate = threading.Event()
    fake_pipeline["gate"] = gate
    running, queued = (_upload(client, sample_pdf)["id"] for _ in range(2))
    client.post(f"/api/documents/{running}/translate")
    client.post(f"/api/documents/{queued}/translate")
    _wait_until(lambda: fake_pipeline["runs"] == 1
                and _batch(client, [queued])["documents"][0]["phase"] == "queued")

    # Queued: dropped immediately, never reaches the pipeline.
    assert client.post(f"/api/documents/{queued}/cancel").json() == {
        "status": "cancelling"}
    _wait_for_status(client, queued, "cancelled")
    info = client.get(f"/api/documents/{queued}").json()
    assert info["phase"] == "cancelled" and info["error"]
    assert _collect_sse_events(client, queued)[0][0] == "error"

    # Running: stops at the next segment boundary once the thread resumes.
    client.post(f"/api/documents/{running}/cancel")
    gate.set()
    _wait_for_status(client, running, "cancelled")
    assert fake_pipeline["runs"] == 1
    # Finished documents: cancel is a no-op that reports the status.
    assert client.post(f"/api/documents/{running}/cancel").json() == {
        "status": "cancelled"}
    # A cancelled paper can be translated again.
    fake_pipeline["gate"] = None
    client.post(f"/api/documents/{queued}/translate")
    _wait_for_status(client, queued, "done")


def test_batch_listing_reports_missing_and_recent(client, sample_pdf):
    a = _upload(client, sample_pdf)["id"]
    b = _upload(client, sample_pdf)["id"]
    board = _batch(client, [b, "nope", a])
    assert [d["id"] for d in board["documents"]] == [b, a]
    assert board["missing"] == ["nope"]
    recent = client.get("/api/documents").json()["documents"]
    assert {d["id"] for d in recent} >= {a, b}


# --- local model pool --------------------------------------------------------

def _tags(*names):
    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [
            {"name": n, "size": 6_600_000_000} for n in names]})
    return handler


def test_local_models_lists_installed_chat_models(client, monkeypatch):
    _install_preflight_transport(
        monkeypatch, _tags("qwen3:14b", "nomic-embed-text:latest", "qwen3.5:9b"))
    body = client.get("/api/local-models").json()
    assert body["available"] is True
    assert [m["name"] for m in body["models"]] == ["qwen3.5:9b", "qwen3:14b"]
    assert body["models"][0]["size_gb"] == 6.6


def test_local_models_unavailable_when_ollama_down(client, monkeypatch):
    def down(request):
        raise httpx.ConnectError("connection refused")

    _install_preflight_transport(monkeypatch, down)
    body = client.get("/api/local-models").json()
    assert body == {"available": False, "models": [],
                    "default": body["default"]}


def test_translate_with_chosen_local_model(client, sample_pdf, fake_pipeline,
                                          monkeypatch):
    _install_preflight_transport(monkeypatch, _tags("qwen3.5:9b", "qwen3:14b"))
    monkeypatch.setenv("PAPERTRANSLATE_TRANSLATOR", "ollama")
    monkeypatch.setenv("PAPERTRANSLATE_OLLAMA_MODEL", "qwen3.5:9b")
    doc_id = _upload(client, sample_pdf)["id"]
    resp = client.post(f"/api/documents/{doc_id}/translate",
                       json={"ollama_model": "qwen3:14b"})
    assert resp.status_code == 200, resp.text
    _wait_for_status(client, doc_id, "done")
    assert fake_pipeline["ollama_model"] == "qwen3:14b"


def test_translate_rejects_missing_or_malformed_local_model(
        client, sample_pdf, fake_pipeline, monkeypatch):
    _install_preflight_transport(monkeypatch, _tags("qwen3.5:9b"))
    monkeypatch.setenv("PAPERTRANSLATE_TRANSLATOR", "ollama")
    doc_id = _upload(client, sample_pdf)["id"]
    missing = client.post(f"/api/documents/{doc_id}/translate",
                          json={"ollama_model": "qwen3:14b"})
    assert missing.status_code == 503
    assert "qwen3:14b" in missing.json()["detail"]
    bad = client.post(f"/api/documents/{doc_id}/translate",
                      json={"ollama_model": "qwen; rm -rf"})
    assert bad.status_code == 400
    assert fake_pipeline["runs"] == 0


def test_local_models_excludes_ollama_cloud_models(client, monkeypatch):
    # Cloud models run on Ollama's servers (the paper would leave the PC).
    _install_preflight_transport(monkeypatch, _tags("qwen3.5:9b", "gemma4:31b-cloud"))
    names = [m["name"] for m in client.get("/api/local-models").json()["models"]]
    assert names == ["qwen3.5:9b"]


def test_translated_bbox_stored_and_migrated(data_dir):
    """Where a translation landed is stored per segment and served by
    GET /segments; databases from before the column existed are upgraded."""
    import sqlite3

    from app import config, db
    from app.main import app

    # An old database: segments table without translated_bbox.
    config.ensure_dirs()
    conn = sqlite3.connect(str(config.get_db_path()))
    conn.execute(
        "CREATE TABLE segments (doc_id TEXT NOT NULL, seg_id TEXT NOT NULL, "
        "page INTEGER NOT NULL, bbox TEXT NOT NULL, kind TEXT NOT NULL, "
        "source TEXT NOT NULL, translated TEXT, status TEXT NOT NULL, "
        "PRIMARY KEY (doc_id, seg_id))")
    conn.commit()
    conn.close()

    db.init_db()
    db.insert_document("flow-doc", "flow.pdf", 1, status="done")
    db.upsert_segment("flow-doc", "p0_s0", 0, (10, 100, 200, 150), "body",
                      "Source.", "번역.")
    db.upsert_segment("flow-doc", "p0_s1", 0, (10, 160, 200, 170), "body",
                      "Other.", "기타.")
    db.set_translated_bboxes("flow-doc", {"p0_s0": (10.0, 90.0, 200.0, 120.5)})

    with TestClient(app) as client:
        rows = {row["seg_id"]: row
                for row in client.get("/api/documents/flow-doc/segments").json()}
    assert rows["p0_s0"]["translated_bbox"] == [10.0, 90.0, 200.0, 120.5]
    assert rows["p0_s0"]["bbox"] == [10, 100, 200, 150]
    assert rows["p0_s1"]["translated_bbox"] is None
