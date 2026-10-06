"""SQLite persistence layer (stdlib sqlite3, data/app.db).

Every function opens a short-lived connection, so calls are safe from
both the asyncio event loop and pipeline worker threads.
"""

import json
import sqlite3
from datetime import datetime, timezone

from app import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    num_pages INTEGER NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS segments (
    doc_id TEXT NOT NULL,
    seg_id TEXT NOT NULL,
    page INTEGER NOT NULL,
    bbox TEXT NOT NULL,
    kind TEXT NOT NULL,
    source TEXT NOT NULL,
    translated TEXT,
    status TEXT NOT NULL,
    translated_bbox TEXT,
    PRIMARY KEY (doc_id, seg_id)
);
CREATE TABLE IF NOT EXISTS doc_assets (
    doc_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ref_details (
    doc_id TEXT NOT NULL,
    ref_num INTEGER NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (doc_id, ref_num)
);
"""


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.get_db_path()), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    config.ensure_dirs()
    conn = get_conn()
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        # Databases created before translated_bbox existed.
        columns = {row["name"] for row in
                   conn.execute("PRAGMA table_info(segments)").fetchall()}
        if "translated_bbox" not in columns:
            conn.execute("ALTER TABLE segments ADD COLUMN translated_bbox TEXT")
        conn.commit()
    finally:
        conn.close()


def insert_document(doc_id: str, filename: str, num_pages: int,
                    status: str = "uploaded") -> None:
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO documents (id, filename, num_pages, status, error, created_at) "
            "VALUES (?, ?, ?, ?, NULL, ?)",
            (doc_id, filename, num_pages, status,
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_document(doc_id: str) -> dict | None:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def get_documents(doc_ids: list[str]) -> dict[str, dict]:
    """Batch lookup for the job board: {id: document row} for known ids."""
    if not doc_ids:
        return {}
    placeholders = ",".join("?" * len(doc_ids))
    conn = get_conn()
    try:
        rows = conn.execute(
            f"SELECT * FROM documents WHERE id IN ({placeholders})",
            tuple(doc_ids),
        ).fetchall()
        return {row["id"]: dict(row) for row in rows}
    finally:
        conn.close()


def list_documents(limit: int = 100) -> list[dict]:
    """Most recent documents first (job board restore after a reload)."""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def set_document_status(doc_id: str, status: str, error: str | None = None) -> None:
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE documents SET status = ?, error = ? WHERE id = ?",
            (status, error, doc_id),
        )
        conn.commit()
    finally:
        conn.close()


def mark_interrupted_translations() -> None:
    """Mark documents stuck in 'translating' as errors (startup recovery).

    A document can only be 'translating' while a pipeline task is running in
    this process; after a restart no such task exists, so the state is stale.
    """
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE documents SET status = 'error', error = ? "
            "WHERE status = 'translating'",
            ("서버가 재시작되어 번역이 중단되었습니다. 다시 시도해 주세요.",),
        )
        conn.commit()
    finally:
        conn.close()


def clear_segments(doc_id: str) -> None:
    conn = get_conn()
    try:
        conn.execute("DELETE FROM segments WHERE doc_id = ?", (doc_id,))
        conn.commit()
    finally:
        conn.close()


def upsert_segment(doc_id: str, seg_id: str, page: int, bbox: tuple | list,
                   kind: str, source: str, translated: str | None,
                   status: str = "done") -> None:
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO segments (doc_id, seg_id, page, bbox, kind, source, translated, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(doc_id, seg_id) DO UPDATE SET "
            "page = excluded.page, bbox = excluded.bbox, kind = excluded.kind, "
            "source = excluded.source, translated = excluded.translated, "
            "status = excluded.status",
            (doc_id, seg_id, page, json.dumps(list(bbox)), kind, source,
             translated, status),
        )
        conn.commit()
    finally:
        conn.close()


def set_translated_bboxes(doc_id: str, rects: dict[str, tuple]) -> None:
    """Store where each translation landed in the output PDF (paragraphs
    flow within their column, so this can differ from the source bbox)."""
    if not rects:
        return
    conn = get_conn()
    try:
        conn.executemany(
            "UPDATE segments SET translated_bbox = ? "
            "WHERE doc_id = ? AND seg_id = ?",
            [(json.dumps([round(v, 2) for v in rect]), doc_id, seg_id)
             for seg_id, rect in rects.items()],
        )
        conn.commit()
    finally:
        conn.close()


def list_segments(doc_id: str, page: int | None = None) -> list[dict]:
    conn = get_conn()
    try:
        if page is None:
            rows = conn.execute(
                "SELECT * FROM segments WHERE doc_id = ? ORDER BY page, rowid",
                (doc_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM segments WHERE doc_id = ? AND page = ? "
                "ORDER BY page, rowid",
                (doc_id, page),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["bbox"] = json.loads(item["bbox"])
            raw = item.get("translated_bbox")
            item["translated_bbox"] = json.loads(raw) if raw else None
            result.append(item)
        return result
    finally:
        conn.close()


def get_doc_assets(doc_id: str) -> dict | None:
    """Return the cached assets payload for a document, if any."""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT payload FROM doc_assets WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return json.loads(row["payload"]) if row is not None else None
    finally:
        conn.close()


def set_doc_assets(doc_id: str, payload: dict) -> None:
    """Cache the computed assets payload for a document (JSON)."""
    conn = get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO doc_assets (doc_id, payload, created_at) "
            "VALUES (?, ?, ?)",
            (doc_id, json.dumps(payload, ensure_ascii=False),
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def clear_doc_assets(doc_id: str) -> None:
    """Invalidate the cached assets payload (re-translation, completion)."""
    conn = get_conn()
    try:
        conn.execute("DELETE FROM doc_assets WHERE doc_id = ?", (doc_id,))
        conn.commit()
    finally:
        conn.close()


def get_ref_details(doc_id: str, ref_num: int) -> dict | None:
    """Return the cached Semantic Scholar payload for one reference, if any."""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT payload FROM ref_details WHERE doc_id = ? AND ref_num = ?",
            (doc_id, ref_num),
        ).fetchone()
        return json.loads(row["payload"]) if row is not None else None
    finally:
        conn.close()


def set_ref_details(doc_id: str, ref_num: int, payload: dict) -> None:
    """Cache a successful Semantic Scholar lookup for one reference (JSON)."""
    conn = get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO ref_details "
            "(doc_id, ref_num, payload, created_at) VALUES (?, ?, ?, ?)",
            (doc_id, ref_num, json.dumps(payload, ensure_ascii=False),
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_progress(doc_id: str) -> tuple[int, int]:
    """Return (done, total) translatable segment counts from the segments table.

    Kinds in UNTRANSLATED_KINDS (header_footer/formula/author/reference) are
    never translated (translated stays NULL), so they are excluded to keep
    these counts consistent with SSE progress events.
    """
    from app.pipeline.segment import UNTRANSLATED_KINDS

    placeholders = ",".join("?" * len(UNTRANSLATED_KINDS))
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS total, "
            "COALESCE(SUM(CASE WHEN translated IS NOT NULL THEN 1 ELSE 0 END), 0) AS done "
            f"FROM segments WHERE doc_id = ? AND kind NOT IN ({placeholders})",
            (doc_id, *sorted(UNTRANSLATED_KINDS)),
        ).fetchone()
        return int(row["done"]), int(row["total"])
    finally:
        conn.close()
