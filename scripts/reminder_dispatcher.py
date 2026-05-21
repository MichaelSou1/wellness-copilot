"""Dispatch due reminders from SQLite to WeChat."""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wellness_copilot.integrations.local_logs import due_reminders, mark_reminder_delivered
from wellness_copilot.integrations.wechat_ilink import WeChatILinkError, get_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reminder dispatcher")
    parser.add_argument("--once", action="store_true", help="Scan once and exit")
    parser.add_argument(
        "--interval",
        type=float,
        default=float(os.environ.get("REMINDER_POLL_INTERVAL_SEC", "60")),
        help="Polling interval seconds",
    )
    parser.add_argument(
        "--dry-run-mark-delivered",
        action="store_true",
        help="Mark reminders delivered even when WECHAT_BOT_TOKEN is absent",
    )
    return parser.parse_args()


def _deliver(row: dict) -> bool:
    target = row.get("target_wxid") or row.get("user_id")
    text = row.get("text") or ""
    if not os.environ.get("WECHAT_BOT_TOKEN"):
        print(f"[reminder_dispatcher] would push to wxid={target}: {text}")
        return False
    client = get_client()
    client.push_to_user(str(target), str(text), context_token=str(row.get("context_token") or ""))
    print(f"[reminder_dispatcher] pushed reminder id={row.get('id')} to wxid={target}")
    return True


def scan_once(mark_dry_run: bool = False) -> int:
    count = 0
    for row in due_reminders():
        try:
            delivered = _deliver(row)
            if delivered or mark_dry_run:
                mark_reminder_delivered(int(row["id"]))
            count += 1
        except WeChatILinkError as exc:
            print(f"[reminder_dispatcher] delivery failed id={row.get('id')}: {exc}")
    return count


def main() -> None:
    args = parse_args()
    while True:
        scan_once(mark_dry_run=args.dry_run_mark_delivered)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
