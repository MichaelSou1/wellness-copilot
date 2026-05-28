"""Dispatch durable outbox events and due reminders."""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wellness_copilot import config
from wellness_copilot.backend_outbox import deliver_outbox_event
from wellness_copilot.backend_queue import (
    claim_next_outbox,
    complete_outbox_event,
    enqueue_due_reminder_outbox,
    fail_outbox_event,
)
from wellness_copilot.backend_telemetry import json_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wellness Copilot outbox dispatcher")
    parser.add_argument("--once", action="store_true", help="Scan once and exit")
    parser.add_argument("--limit", type=int, default=50, help="Max outbox events per scan")
    parser.add_argument("--interval", type=float, default=config.BACKEND_OUTBOX_IDLE_SEC)
    parser.add_argument("--dry-run", action="store_true", help="Mark events sent without calling external APIs")
    return parser.parse_args()


def scan_once(limit: int = 50, dry_run: bool = False) -> int:
    enqueue_due_reminder_outbox(limit=limit)
    processed = 0
    while processed < limit:
        event = claim_next_outbox()
        if not event:
            break
        try:
            deliver_outbox_event(event, dry_run=dry_run)
            complete_outbox_event(event["event_id"])
            json_log(
                "outbox_sent",
                trace_id=event.get("trace_id"),
                event_id=event.get("event_id"),
                kind=event.get("kind"),
            )
        except Exception as exc:
            updated = fail_outbox_event(event["event_id"], f"{type(exc).__name__}: {exc}")
            json_log(
                "outbox_failed",
                trace_id=event.get("trace_id"),
                event_id=event.get("event_id"),
                kind=event.get("kind"),
                status=(updated or {}).get("status"),
                error=type(exc).__name__,
                detail=str(exc)[:300],
            )
        processed += 1
    return processed


def main() -> None:
    args = parse_args()
    while True:
        processed = scan_once(limit=args.limit, dry_run=args.dry_run)
        if args.once:
            return
        if processed == 0:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
