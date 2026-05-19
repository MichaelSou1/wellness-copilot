"""Stretch helper: send a one-shot morning briefing from recent local logs."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from health_guide.integrations.local_logs import summarize_recent_logs
from health_guide.integrations.wechat_ilink import get_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a daily Health Guide briefing")
    parser.add_argument("--user-id", default=os.environ.get("HEALTH_GUIDE_USER_ID", "default_user"))
    parser.add_argument("--wxid", default=os.environ.get("WECHAT_TARGET_WXID", ""))
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_recent_logs(user_id=args.user_id, days_back=args.days)
    text = f"早安复盘：{summary or '最近还没有结构化日志，今天可以先记录一餐或一次训练。'}"
    if args.dry_run or not os.environ.get("WECHAT_BOT_TOKEN"):
        print(text)
        return
    if not args.wxid:
        raise ValueError("--wxid or WECHAT_TARGET_WXID is required")
    get_client().push_to_user(args.wxid, text)


if __name__ == "__main__":
    main()
