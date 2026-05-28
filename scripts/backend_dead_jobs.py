"""Inspect dead Agent jobs and outbox events."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wellness_copilot.backend_queue import _db_path, ensure_backend_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List dead backend jobs/outbox events")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def _connect() -> sqlite3.Connection:
    ensure_backend_tables()
    conn = sqlite3.connect(str(Path(_db_path())), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _loads(text: str | None) -> dict:
    try:
        return json.loads(text or "{}")
    except Exception:
        return {}


def main() -> None:
    args = parse_args()
    with _connect() as conn:
        jobs = [
            dict(row)
            for row in conn.execute(
                """
                SELECT job_id, user_id, thread_id, status, attempts, updated_at,
                       error, trace_id, input_json
                FROM agent_jobs
                WHERE status = 'dead'
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (args.limit,),
            ).fetchall()
        ]
        outbox = [
            dict(row)
            for row in conn.execute(
                """
                SELECT event_id, job_id, kind, status, attempts, updated_at,
                       last_error, trace_id, payload_json
                FROM outbox_events
                WHERE status = 'dead'
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (args.limit,),
            ).fetchall()
        ]

    for job in jobs:
        payload = _loads(job.pop("input_json", ""))
        job["message_preview"] = str(payload.get("message") or "")[:160]
    for event in outbox:
        payload = _loads(event.pop("payload_json", ""))
        event["payload_preview"] = str(payload)[:200]

    result = {"db_path": _db_path(), "jobs": jobs, "outbox": outbox}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"DB: {result['db_path']}")
    print(f"Dead jobs: {len(jobs)}")
    for job in jobs:
        print(
            f"- {job['job_id']} attempts={job['attempts']} "
            f"trace={job['trace_id']} error={job.get('error') or ''}"
        )
        if job.get("message_preview"):
            print(f"  message: {job['message_preview']}")
    print(f"Dead outbox events: {len(outbox)}")
    for event in outbox:
        print(
            f"- {event['event_id']} kind={event['kind']} attempts={event['attempts']} "
            f"trace={event['trace_id']} error={event.get('last_error') or ''}"
        )


if __name__ == "__main__":
    main()
