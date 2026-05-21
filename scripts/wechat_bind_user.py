"""Manage WeChat wxid -> project user_id bindings."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wellness_copilot.integrations.local_logs import (  # noqa: E402
    bind_wechat_user,
    default_wechat_project_user_id,
    get_wechat_binding,
    list_wechat_bindings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bind a WeChat wxid to a Wellness Copilot project user_id")
    parser.add_argument("--wxid", help="WeChat user_wxid from worker logs or wechat_inbox")
    parser.add_argument("--user-id", help="Project user_id used by profile/memory/logs/checkpoints")
    parser.add_argument("--display-name", default="", help="Optional human label for this binding")
    parser.add_argument("--list", action="store_true", help="List existing bindings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        rows = list_wechat_bindings()
        if not rows:
            print("No WeChat bindings yet.")
            return
        for row in rows:
            label = f" ({row['display_name']})" if row.get("display_name") else ""
            print(f"{row['wechat_wxid']} -> {row['project_user_id']}{label}")
        return

    if not args.wxid:
        raise SystemExit("--wxid is required unless --list is used")
    user_id = args.user_id or default_wechat_project_user_id(args.wxid)
    previous = get_wechat_binding(args.wxid)
    bound = bind_wechat_user(args.wxid, user_id, display_name=args.display_name)
    if previous:
        print(f"Updated: {args.wxid} {previous['project_user_id']} -> {bound}")
    else:
        print(f"Created: {args.wxid} -> {bound}")


if __name__ == "__main__":
    main()
