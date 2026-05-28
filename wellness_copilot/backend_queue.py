"""SQLite-backed job queue and outbox for the backend MVP."""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from . import config
from .backend_metrics import (
    AGENT_JOBS,
    AGENT_JOB_FAILURES,
    AGENT_QUEUE_DEPTH,
    AGENT_QUEUE_LAG,
    OUTBOX_EVENTS,
    OUTBOX_FAILURES,
)
from .backend_telemetry import new_trace_id


JOB_TRANSIENT_STATUSES = {"pending", "retrying", "running"}
OUTBOX_TRANSIENT_STATUSES = {"pending", "retrying", "sending"}
RETRY_DELAYS_SEC = tuple(config.BACKEND_RETRY_DELAYS_SEC or (30, 60, 120))
MAX_ATTEMPTS = len(RETRY_DELAYS_SEC) + 1


def _db_path() -> str:
    return os.environ.get("BACKEND_DB_PATH") or os.environ.get("HEALTH_LOGS_DB_PATH") or config.BACKEND_DB_PATH


def _now() -> int:
    return int(time.time())


def _connect() -> sqlite3.Connection:
    path = Path(_db_path())
    if path.parent and str(path.parent) not in {"", "."}:
        path.parent.mkdir(parents=True, exist_ok=True)
    busy_timeout_ms = max(1000, int(config.BACKEND_SQLITE_BUSY_TIMEOUT_MS))
    conn = sqlite3.connect(
        str(path),
        timeout=busy_timeout_ms / 1000,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def ensure_backend_tables() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS agent_jobs (
                job_id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                input_json TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                available_at INTEGER NOT NULL,
                lease_until INTEGER,
                started_at INTEGER,
                finished_at INTEGER,
                result_json TEXT,
                error TEXT,
                trace_id TEXT NOT NULL,
                source TEXT,
                idempotency_key TEXT UNIQUE
            );
            CREATE INDEX IF NOT EXISTS ix_agent_jobs_ready
                ON agent_jobs(status, available_at, lease_until, created_at);
            CREATE INDEX IF NOT EXISTS ix_agent_jobs_trace ON agent_jobs(trace_id);
            CREATE INDEX IF NOT EXISTS ix_agent_jobs_status_created
                ON agent_jobs(status, created_at);
            CREATE INDEX IF NOT EXISTS ix_agent_jobs_thread
                ON agent_jobs(user_id, thread_id, created_at);

            CREATE TABLE IF NOT EXISTS outbox_events (
                event_id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                job_id TEXT,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                available_at INTEGER NOT NULL,
                lease_until INTEGER,
                sent_at INTEGER,
                idempotency_key TEXT UNIQUE,
                last_error TEXT,
                trace_id TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_outbox_ready
                ON outbox_events(status, available_at, lease_until, created_at);
            CREATE INDEX IF NOT EXISTS ix_outbox_job ON outbox_events(job_id);
            CREATE INDEX IF NOT EXISTS ix_outbox_trace ON outbox_events(trace_id);
            CREATE INDEX IF NOT EXISTS ix_outbox_status_created
                ON outbox_events(status, created_at);
            """
        )


def queue_depths() -> dict[str, int]:
    ensure_backend_tables()
    with _connect() as conn:
        return {
            row["status"]: int(row["n"])
            for row in conn.execute(
                "SELECT status, COUNT(*) AS n FROM agent_jobs GROUP BY status"
            ).fetchall()
        }


def queue_capacity() -> dict[str, Any]:
    depths = queue_depths()
    pending = int(depths.get("pending", 0)) + int(depths.get("retrying", 0))
    running = int(depths.get("running", 0))
    max_pending = max(0, int(config.BACKEND_MAX_PENDING_JOBS))
    max_running = max(0, int(config.BACKEND_MAX_RUNNING_JOBS))
    limited = False
    reasons: list[str] = []
    if max_pending and pending >= max_pending:
        limited = True
        reasons.append("pending_limit")
    if max_running and running >= max_running:
        limited = True
        reasons.append("running_limit")
    return {
        "limited": limited,
        "reasons": reasons,
        "pending": pending,
        "running": running,
        "max_pending": max_pending,
        "max_running": max_running,
        "depths": depths,
    }


def _json_dumps(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True, default=str)


def _json_loads(value: str | None, default: Any) -> Any:
    try:
        return json.loads(value or "")
    except Exception:
        return default


def _row_to_job(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    item = dict(row)
    input_json = item.pop("input_json", "")
    result_json = item.pop("result_json", "")
    item["input"] = _json_loads(input_json, {})
    item["result"] = _json_loads(result_json, None) if result_json else None
    return item


def _row_to_outbox(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    item = dict(row)
    item["payload"] = _json_loads(item.pop("payload_json", ""), {})
    return item


def enqueue_agent_job(
    *,
    user_id: str,
    thread_id: str,
    message: str,
    source: str = "api",
    content: Any = None,
    wechat_context: dict | None = None,
    trace_id: str = "",
    idempotency_key: str = "",
) -> dict:
    ensure_backend_tables()
    now = _now()
    job_id = uuid.uuid4().hex
    trace_id = trace_id or new_trace_id()
    payload = {
        "message": message or "",
        "content": content if content is not None else message or "",
        "source": source or "api",
        "wechat_context": wechat_context or {},
    }
    with _connect() as conn:
        if idempotency_key:
            conn.execute(
                """
                INSERT OR IGNORE INTO agent_jobs(
                    job_id, created_at, updated_at, user_id, thread_id, input_json,
                    status, attempts, available_at, trace_id, source, idempotency_key
                ) VALUES(?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    now,
                    now,
                    user_id,
                    thread_id,
                    _json_dumps(payload),
                    now,
                    trace_id,
                    source,
                    idempotency_key,
                ),
            )
            row = conn.execute(
                "SELECT * FROM agent_jobs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        else:
            conn.execute(
                """
                INSERT INTO agent_jobs(
                    job_id, created_at, updated_at, user_id, thread_id, input_json,
                    status, attempts, available_at, trace_id, source
                ) VALUES(?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)
                """,
                (job_id, now, now, user_id, thread_id, _json_dumps(payload), now, trace_id, source),
            )
            row = conn.execute("SELECT * FROM agent_jobs WHERE job_id = ?", (job_id,)).fetchone()
    AGENT_JOBS.labels("pending").inc()
    for status, depth in queue_depths().items():
        AGENT_QUEUE_DEPTH.labels(status).observe(depth)
    return _row_to_job(row) or {}


def get_job(job_id: str) -> dict | None:
    ensure_backend_tables()
    with _connect() as conn:
        return _row_to_job(conn.execute("SELECT * FROM agent_jobs WHERE job_id = ?", (job_id,)).fetchone())


def claim_next_job(lease_seconds: int | None = None, now: int | None = None) -> dict | None:
    ensure_backend_tables()
    now = int(now or _now())
    lease_seconds = int(lease_seconds or config.BACKEND_JOB_LEASE_SEC)
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT * FROM agent_jobs
            WHERE status IN ('pending', 'retrying', 'running')
              AND available_at <= ?
              AND (lease_until IS NULL OR lease_until <= ?)
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (now, now),
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        conn.execute(
            """
            UPDATE agent_jobs
            SET status = 'running',
                attempts = attempts + 1,
                updated_at = ?,
                started_at = COALESCE(started_at, ?),
                lease_until = ?,
                error = NULL
            WHERE job_id = ?
            """,
            (now, now, now + lease_seconds, row["job_id"]),
        )
        claimed = conn.execute("SELECT * FROM agent_jobs WHERE job_id = ?", (row["job_id"],)).fetchone()
        conn.execute("COMMIT")
    AGENT_QUEUE_LAG.observe(max(0, now - int(row["created_at"])))
    return _row_to_job(claimed)


def claim_job(job_id: str, lease_seconds: int | None = None, now: int | None = None) -> dict | None:
    ensure_backend_tables()
    now = int(now or _now())
    lease_seconds = int(lease_seconds or config.BACKEND_JOB_LEASE_SEC)
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT * FROM agent_jobs
            WHERE job_id = ?
              AND status IN ('pending', 'retrying', 'running')
              AND available_at <= ?
              AND (lease_until IS NULL OR lease_until <= ?)
            """,
            (job_id, now, now),
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        conn.execute(
            """
            UPDATE agent_jobs
            SET status = 'running',
                attempts = attempts + 1,
                updated_at = ?,
                started_at = COALESCE(started_at, ?),
                lease_until = ?,
                error = NULL
            WHERE job_id = ?
            """,
            (now, now, now + lease_seconds, job_id),
        )
        claimed = conn.execute("SELECT * FROM agent_jobs WHERE job_id = ?", (job_id,)).fetchone()
        conn.execute("COMMIT")
    AGENT_QUEUE_LAG.observe(max(0, now - int(row["created_at"])))
    return _row_to_job(claimed)


def claim_new_agent_job(
    *,
    user_id: str,
    thread_id: str,
    message: str,
    source: str = "api",
    content: Any = None,
    wechat_context: dict | None = None,
    trace_id: str = "",
    idempotency_key: str = "",
    lease_seconds: int | None = None,
) -> dict:
    ensure_backend_tables()
    now = _now()
    lease_seconds = int(lease_seconds or config.BACKEND_JOB_LEASE_SEC)
    job_id = uuid.uuid4().hex
    trace_id = trace_id or new_trace_id()
    payload = {
        "message": message or "",
        "content": content if content is not None else message or "",
        "source": source or "api",
        "wechat_context": wechat_context or {},
    }
    with _connect() as conn:
        if idempotency_key:
            conn.execute(
                """
                INSERT OR IGNORE INTO agent_jobs(
                    job_id, created_at, updated_at, user_id, thread_id, input_json,
                    status, attempts, available_at, lease_until, started_at,
                    trace_id, source, idempotency_key
                ) VALUES(?, ?, ?, ?, ?, ?, 'running', 1, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    now,
                    now,
                    user_id,
                    thread_id,
                    _json_dumps(payload),
                    now,
                    now + lease_seconds,
                    now,
                    trace_id,
                    source,
                    idempotency_key,
                ),
            )
            row = conn.execute(
                "SELECT * FROM agent_jobs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        else:
            conn.execute(
                """
                INSERT INTO agent_jobs(
                    job_id, created_at, updated_at, user_id, thread_id, input_json,
                    status, attempts, available_at, lease_until, started_at,
                    trace_id, source
                ) VALUES(?, ?, ?, ?, ?, ?, 'running', 1, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    now,
                    now,
                    user_id,
                    thread_id,
                    _json_dumps(payload),
                    now,
                    now + lease_seconds,
                    now,
                    trace_id,
                    source,
                ),
            )
            row = conn.execute("SELECT * FROM agent_jobs WHERE job_id = ?", (job_id,)).fetchone()
    AGENT_JOBS.labels("running").inc()
    AGENT_QUEUE_LAG.observe(0)
    for status, depth in queue_depths().items():
        AGENT_QUEUE_DEPTH.labels(status).observe(depth)
    return _row_to_job(row) or {}


def complete_job(job_id: str, result: dict) -> None:
    now = _now()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE agent_jobs
            SET status = 'succeeded',
                updated_at = ?,
                finished_at = ?,
                lease_until = NULL,
                result_json = ?,
                error = NULL
            WHERE job_id = ?
            """,
            (now, now, _json_dumps(result), job_id),
        )
    AGENT_JOBS.labels("succeeded").inc()


def fail_job(job_id: str, error: str, now: int | None = None) -> dict | None:
    now = int(now or _now())
    with _connect() as conn:
        row = conn.execute("SELECT attempts FROM agent_jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        attempts = int(row["attempts"] or 0)
        if attempts <= len(RETRY_DELAYS_SEC):
            delay = RETRY_DELAYS_SEC[attempts - 1]
            status = "retrying"
            available_at = now + delay
            finished_at = None
        else:
            status = "dead"
            available_at = now
            finished_at = now
        conn.execute(
            """
            UPDATE agent_jobs
            SET status = ?, updated_at = ?, available_at = ?, lease_until = NULL,
                finished_at = ?, error = ?
            WHERE job_id = ?
            """,
            (status, now, available_at, finished_at, str(error)[:1000], job_id),
        )
        updated = conn.execute("SELECT * FROM agent_jobs WHERE job_id = ?", (job_id,)).fetchone()
    AGENT_JOB_FAILURES.inc()
    AGENT_JOBS.labels(status).inc()
    return _row_to_job(updated)


def enqueue_outbox_event(
    *,
    kind: str,
    payload: dict,
    idempotency_key: str,
    job_id: str = "",
    trace_id: str = "",
) -> dict:
    ensure_backend_tables()
    now = _now()
    event_id = uuid.uuid4().hex
    trace_id = trace_id or new_trace_id()
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO outbox_events(
                event_id, created_at, updated_at, job_id, kind, payload_json,
                status, attempts, available_at, idempotency_key, trace_id
            ) VALUES(?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)
            """,
            (
                event_id,
                now,
                now,
                job_id or "",
                kind,
                _json_dumps(payload),
                now,
                idempotency_key,
                trace_id,
            ),
        )
        row = conn.execute(
            "SELECT * FROM outbox_events WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
    OUTBOX_EVENTS.labels(kind, "pending").inc()
    return _row_to_outbox(row) or {}


def get_outbox_event(event_id: str) -> dict | None:
    ensure_backend_tables()
    with _connect() as conn:
        return _row_to_outbox(conn.execute("SELECT * FROM outbox_events WHERE event_id = ?", (event_id,)).fetchone())


def claim_next_outbox(lease_seconds: int | None = None, now: int | None = None) -> dict | None:
    ensure_backend_tables()
    now = int(now or _now())
    lease_seconds = int(lease_seconds or config.BACKEND_OUTBOX_LEASE_SEC)
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT * FROM outbox_events
            WHERE status IN ('pending', 'retrying', 'sending')
              AND available_at <= ?
              AND (lease_until IS NULL OR lease_until <= ?)
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (now, now),
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        conn.execute(
            """
            UPDATE outbox_events
            SET status = 'sending',
                attempts = attempts + 1,
                updated_at = ?,
                lease_until = ?,
                last_error = NULL
            WHERE event_id = ?
            """,
            (now, now + lease_seconds, row["event_id"]),
        )
        claimed = conn.execute("SELECT * FROM outbox_events WHERE event_id = ?", (row["event_id"],)).fetchone()
        conn.execute("COMMIT")
    return _row_to_outbox(claimed)


def complete_outbox_event(event_id: str) -> None:
    now = _now()
    with _connect() as conn:
        row = conn.execute("SELECT kind FROM outbox_events WHERE event_id = ?", (event_id,)).fetchone()
        kind = str(row["kind"]) if row else "unknown"
        conn.execute(
            """
            UPDATE outbox_events
            SET status = 'sent', updated_at = ?, sent_at = ?, lease_until = NULL, last_error = NULL
            WHERE event_id = ?
            """,
            (now, now, event_id),
        )
    OUTBOX_EVENTS.labels(kind, "sent").inc()


def fail_outbox_event(event_id: str, error: str, now: int | None = None) -> dict | None:
    now = int(now or _now())
    with _connect() as conn:
        row = conn.execute("SELECT attempts, kind FROM outbox_events WHERE event_id = ?", (event_id,)).fetchone()
        if row is None:
            return None
        attempts = int(row["attempts"] or 0)
        kind = str(row["kind"] or "unknown")
        if attempts <= len(RETRY_DELAYS_SEC):
            delay = RETRY_DELAYS_SEC[attempts - 1]
            status = "retrying"
            available_at = now + delay
        else:
            status = "dead"
            available_at = now
        conn.execute(
            """
            UPDATE outbox_events
            SET status = ?, updated_at = ?, available_at = ?, lease_until = NULL, last_error = ?
            WHERE event_id = ?
            """,
            (status, now, available_at, str(error)[:1000], event_id),
        )
        updated = conn.execute("SELECT * FROM outbox_events WHERE event_id = ?", (event_id,)).fetchone()
    OUTBOX_FAILURES.labels(kind).inc()
    OUTBOX_EVENTS.labels(kind, status).inc()
    return _row_to_outbox(updated)


def enqueue_due_reminder_outbox(limit: int = 50) -> int:
    from .integrations.local_logs import due_reminders

    count = 0
    for row in due_reminders(limit=limit):
        payload = {
            "reminder_id": int(row["id"]),
            "target_wxid": row.get("target_wxid") or row.get("user_id"),
            "context_token": row.get("context_token") or "",
            "text": row.get("text") or "",
            "user_id": row.get("user_id") or "",
        }
        enqueue_outbox_event(
            kind="reminder_push",
            payload=payload,
            idempotency_key=f"reminder_push:{row['id']}",
            trace_id=new_trace_id(),
        )
        count += 1
    return count


def backend_counts() -> dict:
    ensure_backend_tables()
    with _connect() as conn:
        jobs = {
            row["status"]: int(row["n"])
            for row in conn.execute("SELECT status, COUNT(*) AS n FROM agent_jobs GROUP BY status").fetchall()
        }
        outbox = {
            row["status"]: int(row["n"])
            for row in conn.execute("SELECT status, COUNT(*) AS n FROM outbox_events GROUP BY status").fetchall()
        }
        oldest_pending = conn.execute(
            """
            SELECT MIN(created_at) AS oldest
            FROM agent_jobs
            WHERE status IN ('pending', 'retrying')
            """
        ).fetchone()
        dead_jobs = [
            _row_to_job(row)
            for row in conn.execute(
                """
                SELECT *
                FROM agent_jobs
                WHERE status = 'dead'
                ORDER BY updated_at DESC
                LIMIT 5
                """
            ).fetchall()
        ]
        pending_outbox = conn.execute(
            """
            SELECT MIN(created_at) AS oldest
            FROM outbox_events
            WHERE status IN ('pending', 'retrying')
            """
        ).fetchone()
    now = _now()
    return {
        "jobs": jobs,
        "outbox": outbox,
        "db_path": _db_path(),
        "queue": {
            "pending": int(jobs.get("pending", 0)) + int(jobs.get("retrying", 0)),
            "running": int(jobs.get("running", 0)),
            "oldest_pending_age_sec": max(0, now - int(oldest_pending["oldest"]))
            if oldest_pending and oldest_pending["oldest"]
            else 0,
        },
        "outbox_queue": {
            "pending": int(outbox.get("pending", 0)) + int(outbox.get("retrying", 0)),
            "sending": int(outbox.get("sending", 0)),
            "oldest_pending_age_sec": max(0, now - int(pending_outbox["oldest"]))
            if pending_outbox and pending_outbox["oldest"]
            else 0,
        },
        "dead_jobs_sample": dead_jobs,
    }
