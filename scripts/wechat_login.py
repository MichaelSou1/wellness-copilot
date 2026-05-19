"""First-time WeChat iLink login helper.

Prints a QR-code URL, polls until the user confirms in WeChat, then persists
WECHAT_BOT_TOKEN into .env.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from health_guide.integrations.wechat_ilink import WeChatILinkClient


def _pick(data: dict, *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value:
            return str(value)
        nested = data.get("data")
        if isinstance(nested, dict) and nested.get(key):
            return str(nested[key])
    return ""


def _write_env_value(path: Path, key: str, value: str) -> None:
    lines = []
    found = False
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        if line.startswith(f"{key}="):
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{key}={value}")
    path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bind WeChat iLink bot token")
    parser.add_argument("--env", default=".env", help="Path to .env file to update")
    parser.add_argument("--timeout", type=int, default=180, help="Max seconds to wait for QR confirmation")
    parser.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = WeChatILinkClient(bot_token="")
    qr = client.get_bot_qrcode()
    qrcode_id = _pick(qr, "qrcode_id", "qr_id", "ticket", "id")
    qrcode_url = _pick(qr, "qrcode_url", "qr_url", "url")
    if not qrcode_id:
        raise RuntimeError(f"iLink QR response did not contain qrcode_id: {qr}")
    print(f"Open this URL and confirm in WeChat:\n{qrcode_url or qr}")

    deadline = time.time() + args.timeout
    token = ""
    while time.time() < deadline:
        status = client.poll_qrcode_status(qrcode_id)
        state = _pick(status, "status", "state")
        token = _pick(status, "bot_token", "token", "access_token")
        if token:
            break
        if state:
            print(f"[wechat_login] status={state}")
        time.sleep(args.interval)
    if not token:
        raise TimeoutError("Timed out waiting for WeChat QR confirmation")

    env_path = Path(args.env)
    _write_env_value(env_path, "WECHAT_BOT_TOKEN", token)
    print(f"WECHAT_BOT_TOKEN written to {env_path}. Restart worker to use it.")


if __name__ == "__main__":
    main()
