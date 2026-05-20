"""First-time WeChat iLink login helper.

Prints a QR-code URL, polls until the user confirms in WeChat, then persists
WECHAT_BOT_TOKEN into .env.
"""
from __future__ import annotations

import argparse
import base64
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from health_guide.integrations.wechat_ilink import WeChatILinkClient, WeChatILinkError


def _pick(data: dict, *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value:
            return str(value)
        nested = data.get("data")
        if isinstance(nested, dict) and nested.get(key):
            return str(nested[key])
    return ""


def _looks_like_data_image(value: str) -> bool:
    return value.strip().lower().startswith("data:image/")


def _looks_like_base64_image(value: str) -> bool:
    text = value.strip()
    return len(text) > 120 and bool(re.fullmatch(r"[A-Za-z0-9+/=\s]+", text))


def _save_image_b64(value: str, path: Path) -> bool:
    text = value.strip()
    if _looks_like_data_image(text):
        text = text.split(",", 1)[-1]
    if not _looks_like_base64_image(text):
        return False
    try:
        raw = base64.b64decode("".join(text.split()), validate=False)
    except Exception:
        return False
    if not raw:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return True


def _wsl_windows_path(path: Path) -> str:
    try:
        result = subprocess.run(
            ["wslpath", "-w", str(path.resolve())],
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip() or str(path)
    except Exception:
        return str(path.resolve())


def _open_target(target: str) -> bool:
    commands = []
    if os.name == "nt":
        commands.append(["cmd", "/c", "start", "", target])
    else:
        commands.extend(
            [
                ["wslview", target],
                ["explorer.exe", target],
                ["powershell.exe", "-NoProfile", "-Command", "Start-Process", target],
                ["xdg-open", target],
                ["open", target],
            ]
        )
    for command in commands:
        try:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return False


def _render_terminal_qr(payload: str) -> bool:
    try:
        import qrcode
    except Exception:
        return False
    qr = qrcode.QRCode(border=2)
    qr.add_data(payload)
    qr.make(fit=True)
    matrix = qr.get_matrix()
    black = "\033[40m  \033[0m"
    white = "\033[47m  \033[0m"
    for row in matrix:
        print("".join(black if cell else white for cell in row))
    return True


def _save_payload_qr(payload: str, path: Path) -> bool:
    try:
        import qrcode
        path.parent.mkdir(parents=True, exist_ok=True)
        img = qrcode.make(payload)
        img.save(path)
        return True
    except Exception:
        return False


def _show_qrcode(qr: dict, args: argparse.Namespace) -> None:
    qrcode_url = _pick(qr, "qrcode_img_content", "qrcode_url", "qr_url", "url")
    image_value = _pick(
        qr,
        "qrcode_base64",
        "qr_base64",
        "image_base64",
        "qrcode_image",
        "qr_image",
        "image",
    )
    payload = _pick(qr, "qrcode", "qr_code", "qr", "content", "payload")
    image_path = Path(args.qr_path)

    if image_value and _save_image_b64(image_value, image_path):
        win_path = _wsl_windows_path(image_path)
        print(f"二维码图片已保存：{image_path}")
        print(f"Windows 路径：{win_path}")
        if not args.no_open and _open_target(win_path):
            print("已尝试用 Windows 图片查看器打开二维码。")
        else:
            print("请手动打开上面的图片路径并用微信扫码。")
        return

    if qrcode_url:
        print("二维码 / 登录 URL：")
        print(qrcode_url)
        if _save_payload_qr(qrcode_url, image_path):
            win_path = _wsl_windows_path(image_path)
            print(f"二维码图片已生成：{image_path}")
            print(f"Windows 路径：{win_path}")
            if not args.no_open and _open_target(win_path):
                print("已尝试用 Windows 图片查看器打开二维码。")
            else:
                print("请手动打开上面的图片路径并用微信扫码。")
        if args.terminal_qr and _render_terminal_qr(qrcode_url):
            print("也可以直接用微信扫描上方终端二维码。")
        elif not image_path.exists() and not args.no_open and _open_target(qrcode_url):
            print("已尝试用 Windows 浏览器打开登录 URL。")
        elif not image_path.exists():
            print("请把上面的 URL 复制到浏览器或二维码生成器打开。")
        return

    if payload:
        print("服务返回 QR payload。")
        if _save_payload_qr(payload, image_path):
            win_path = _wsl_windows_path(image_path)
            print(f"二维码图片已生成：{image_path}")
            print(f"Windows 路径：{win_path}")
            if not args.no_open and _open_target(win_path):
                print("已尝试用 Windows 图片查看器打开二维码。")
        if args.terminal_qr and _render_terminal_qr(payload):
            print("请直接用微信扫描上方终端二维码。")
        elif not image_path.exists():
            print("当前环境未安装 qrcode 包，无法在终端渲染二维码。")
            print("可运行：python -m pip install qrcode")
            print(f"QR payload:\n{payload}")
        return

    print("未识别二维码展示字段，原始响应如下：")
    print(qr)


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
    parser.add_argument("--qr-path", default="tmp/wechat_qrcode.png", help="Where to save QR images when API returns base64")
    parser.add_argument("--no-open", action="store_true", help="Do not open QR URL/image with the host OS")
    parser.add_argument("--terminal-qr", action="store_true", help="Render a QR code directly in the terminal")
    parser.add_argument("--print-response", action="store_true", help="Print raw QR/status responses for endpoint debugging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = WeChatILinkClient(bot_token="")
    qr = client.get_bot_qrcode()
    if args.print_response:
        print(f"[wechat_login] QR response: {qr}")
    qrcode_id = _pick(qr, "qrcode", "qrcode_id", "qr_id", "ticket", "id")
    if not qrcode_id:
        raise RuntimeError(f"iLink QR response did not contain qrcode: {qr}")
    _show_qrcode(qr, args)
    print("扫码后请在手机微信里确认登录；脚本会继续轮询 token。")

    deadline = time.time() + args.timeout
    token = ""
    poll_base_url = ""
    account_id = ""
    user_id = ""
    while time.time() < deadline:
        try:
            status = client.poll_qrcode_status(qrcode_id, base_url=poll_base_url or None)
        except WeChatILinkError as exc:
            print(f"[wechat_login] polling retry after transient error: {exc}")
            time.sleep(args.interval)
            continue
        if args.print_response:
            print(f"[wechat_login] status response: {status}")
        state = _pick(status, "status", "state")
        token = _pick(status, "bot_token", "token", "access_token")
        account_id = _pick(status, "ilink_bot_id", "bot_id", "account_id")
        user_id = _pick(status, "ilink_user_id", "user_id")
        redirect_host = _pick(status, "redirect_host")
        if state == "scaned_but_redirect" and redirect_host:
            poll_base_url = f"https://{redirect_host}"
            print(f"[wechat_login] IDC redirect -> {redirect_host}")
        if token:
            break
        if state:
            print(f"[wechat_login] status={state}")
        time.sleep(args.interval)
    if not token:
        raise TimeoutError("Timed out waiting for WeChat QR confirmation")

    env_path = Path(args.env)
    _write_env_value(env_path, "WECHAT_BOT_TOKEN", token)
    base_url = _pick(status, "baseurl", "base_url")
    if base_url:
        _write_env_value(env_path, "WECHAT_ILINK_BASE_URL", base_url)
    if account_id:
        _write_env_value(env_path, "WECHAT_ACCOUNT_ID", account_id)
    if user_id:
        _write_env_value(env_path, "WECHAT_LOGIN_USER_ID", user_id)
    print(f"WECHAT_BOT_TOKEN written to {env_path}. Restart worker to use it.")


if __name__ == "__main__":
    main()
