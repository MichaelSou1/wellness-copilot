"""Long-poll WeChat iLink updates and pass them into the Wellness Copilot graph."""
from __future__ import annotations

import argparse
import base64
import os
import re
import sys
import time
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wellness_copilot import config as cfg
from wellness_copilot.backend_queue import enqueue_agent_job
from wellness_copilot.backend_telemetry import new_trace_id
from wellness_copilot.detail import display_role, set_detail
from wellness_copilot.integrations.wechat_ilink import (
    WeChatILinkError,
    get_client,
    get_last_offset,
    get_runtime_bot_token,
    normalize_update,
    set_last_offset,
)
from wellness_copilot.integrations.local_logs import (
    bind_wechat_user,
    enqueue_wechat_message,
    get_or_create_wechat_user_id,
    mark_wechat_messages_processed,
    pending_wechat_messages,
    pending_wechat_users,
)

_QUESTION_OR_COMMAND = re.compile(
    r"[？?]|怎么|如何|能不能|能否|可以吗|可不可以|吗\b|呢\b|"
    r"帮我|给我|建议|推荐|安排|计划|记录|提醒|分析|看看|判断|测试|test|在吗|在不在|"
    r"多少|够不够|要不要|该不该|是不是|行不行|怎么办",
    re.IGNORECASE,
)
_TEXT_EXPECTS_IMAGE = re.compile(
    r"图|图片|照片|截图|这张|拍的|发的图|这幅|这页|这个动作|姿势|化验单|报告单|包装|标签|配料表|营养成分表",
    re.IGNORECASE,
)
_STANDALONE_CHAT = re.compile(
    r"^(你好|您好|hello|hi|嗨|测试|test|在吗|在不在|谢谢|感谢|多谢|再见|拜拜|早安|晚安|收到|好的|好)[。.!！?？~～]*$",
    re.IGNORECASE,
)
_BIND_COMMAND = re.compile(r"^\s*(?:/bind|绑定用户|绑定user|绑定 user)\s+([^\s/\\]{1,80})\s*$", re.IGNORECASE)


@contextmanager
def _worker_env(user_id: str, target_wxid: str, context_token: str):
    old = {
        "WELLNESS_COPILOT_USER_ID": os.environ.get("WELLNESS_COPILOT_USER_ID"),
        "WECHAT_TARGET_WXID": os.environ.get("WECHAT_TARGET_WXID"),
        "WECHAT_CONTEXT_TOKEN": os.environ.get("WECHAT_CONTEXT_TOKEN"),
    }
    os.environ["WELLNESS_COPILOT_USER_ID"] = user_id
    os.environ["WECHAT_TARGET_WXID"] = target_wxid
    os.environ["WECHAT_CONTEXT_TOKEN"] = context_token
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _merge_inbox_messages(rows: list[dict]) -> tuple[str, list[str]]:
    texts = [str(row.get("text") or "").strip() for row in rows if str(row.get("text") or "").strip()]
    media_ids: list[str] = []
    for row in rows:
        media_ids.extend(row.get("media_ids") or [])
    return "\n".join(texts).strip(), media_ids


def _is_standalone_text(row: dict) -> bool:
    text = str(row.get("text") or "").strip()
    return bool(text and not (row.get("media_ids") or []) and _STANDALONE_CHAT.search(text))


def _is_bind_command_row(row: dict) -> bool:
    text = str(row.get("text") or "").strip()
    return bool(text and not (row.get("media_ids") or []) and _bind_command_user_id(text))


def _inbox_complete(text: str, media_ids: list[str]) -> tuple[bool, str]:
    text = (text or "").strip()
    if not text:
        return False, "waiting_for_text"
    if not media_ids and _STANDALONE_CHAT.search(text):
        return True, "standalone_chat"
    if _TEXT_EXPECTS_IMAGE.search(text) and not media_ids:
        return False, "waiting_for_image"
    if _QUESTION_OR_COMMAND.search(text):
        return True, "question_or_command"
    if not media_ids and len(text) <= 20:
        return True, "short_text"
    return False, "waiting_for_question_or_command"


def _bind_command_user_id(text: str) -> str:
    match = _BIND_COMMAND.search(text or "")
    return match.group(1).strip() if match else ""


def _handle_bind_command(client, rows: list[dict], text: str, detail: bool = False) -> bool:
    project_user_id = _bind_command_user_id(text)
    if not project_user_id:
        return False
    ids = [int(row["id"]) for row in rows]
    wxid = rows[-1].get("user_wxid") or ""
    context_token = rows[-1].get("context_token") or ""
    bind_wechat_user(wxid, project_user_id)
    if context_token:
        client.remember_context(wxid, context_token)
        client.send_message(
            context_token,
            text=f"已绑定当前微信到项目用户：{project_user_id}",
            user_id=wxid,
        )
    elif wxid:
        client.push_to_user(wxid, f"已绑定当前微信到项目用户：{project_user_id}")
    mark_wechat_messages_processed(ids)
    if detail:
        print(
            f"[wechat_worker] bound wxid={display_role(wxid)} project_user_id={display_role(project_user_id)} inbox_ids={ids}",
            flush=True,
        )
    return True


def _build_content(client, text: str, media_ids: list[str]):
    if not media_ids:
        return text
    parts = []
    if text:
        parts.append({"type": "text", "text": text})
    for media_id in media_ids:
        try:
            image_bytes = client.download_media(media_id)
            encoded = base64.b64encode(image_bytes).decode("ascii")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    "media_id": media_id,
                }
            )
        except Exception as exc:
            parts.append(
                {
                    "type": "text",
                    "text": f"[图片 media_id={media_id} 下载失败：{type(exc).__name__}]",
                }
            )
    return parts


def _process_inbox_turn(client, rows: list[dict], detail: bool = False) -> bool:
    if not rows:
        return False
    if len(rows) > 1 and (_is_standalone_text(rows[-1]) or _is_bind_command_row(rows[-1])):
        stale_ids = [int(row["id"]) for row in rows[:-1]]
        mark_wechat_messages_processed(stale_ids, status="superseded")
        if detail:
            print(
                f"[wechat_worker] superseded stale inbox_ids={stale_ids} "
                f"before standalone text={rows[-1].get('text')!r}",
                flush=True,
            )
        rows = [rows[-1]]

    text, media_ids = _merge_inbox_messages(rows)
    if not media_ids and _handle_bind_command(client, rows, text, detail=detail):
        return True
    complete, reason = _inbox_complete(text, media_ids)
    if not complete:
        if detail:
            print(f"[wechat_worker] inbox waiting user={rows[0].get('user_wxid')} reason={reason} pending={len(rows)}", flush=True)
        return False

    ids = [int(row["id"]) for row in rows]
    wxid = rows[-1].get("user_wxid") or os.environ.get("WECHAT_TARGET_WXID", "")
    project_user_id = get_or_create_wechat_user_id(wxid)
    context_token = rows[-1].get("context_token") or ""
    chat_type = rows[-1].get("chat_type") or "private"
    if context_token:
        client.remember_context(wxid, context_token)
    thread_id = f"wechat:{project_user_id}"
    content = _build_content(client, text, media_ids)
    trace_id = new_trace_id()
    with _worker_env(project_user_id, wxid, context_token):
        job = enqueue_agent_job(
            user_id=project_user_id,
            thread_id=thread_id,
            message=text or "[image]",
            content=content,
            source="wechat",
            trace_id=trace_id,
            wechat_context={
                "context_token": context_token,
                "chat_type": chat_type,
                "user_wxid": wxid,
                "project_user_id": project_user_id,
                "pre_accumulated": True,
            },
            idempotency_key=f"wechat_inbox:{ids[0]}:{ids[-1]}",
        )
    if detail:
        print(
            f"[wechat_worker] queued job={job.get('job_id')} trace={job.get('trace_id')} "
            f"wxid={display_role(wxid)} "
            f"project_user_id={display_role(project_user_id)} "
            f" inbox_ids={ids}"
            ,
            flush=True,
        )
    mark_wechat_messages_processed(ids)
    return True


def _enqueue_update(client, update: dict, detail: bool = False) -> None:
    msg = normalize_update(update)
    if not msg.text and not msg.media_ids:
        return
    user_id = msg.user_wxid or os.environ.get("WELLNESS_COPILOT_USER_ID", "wechat_user")
    if msg.context_token:
        client.remember_context(user_id, msg.context_token)
    project_user_id = get_or_create_wechat_user_id(user_id)
    inserted = enqueue_wechat_message(
        update_id=msg.update_id,
        user_wxid=user_id,
        context_token=msg.context_token,
        chat_type=msg.chat_type,
        text=msg.text,
        media_ids=msg.media_ids,
        raw=msg.raw,
    )
    if detail:
        print(
            f"[wechat_worker] queued update={msg.update_id} inserted={inserted} "
            f"wxid={display_role(user_id)} project_user_id={display_role(project_user_id)} "
            f"text={msg.text[:60]!r} media={len(msg.media_ids)}",
            flush=True,
        )


def _drain_inbox(client, detail: bool = False) -> int:
    processed = 0
    for user_id in pending_wechat_users(limit=20):
        rows = pending_wechat_messages(user_id, limit=20)
        if _process_inbox_turn(client, rows, detail=detail):
            processed += 1
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WeChat iLink long-poll worker")
    parser.add_argument("--once", action="store_true", help="Process one polling batch and exit")
    parser.add_argument("--detail", action="store_true", help="Print node/tool details")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_detail(args.detail)
    client = get_client()
    if not get_runtime_bot_token():
        print("WECHAT_BOT_TOKEN is not configured. Run scripts/wechat_login.py first.")
        if args.once:
            return

    backoff = 1.0
    while True:
        try:
            if not get_runtime_bot_token():
                time.sleep(min(30, backoff))
                backoff = min(60, backoff * 2)
                if args.once:
                    return
                continue
            client = get_client()
            updates = client.get_updates(timeout=cfg.WECHAT_POLL_TIMEOUT_SEC, offset=get_last_offset())
            backoff = 1.0
            for update in updates:
                _enqueue_update(client, update, detail=args.detail)
            if client.last_updates_cursor:
                set_last_offset(client.last_updates_cursor)
            _drain_inbox(client, detail=args.detail)
            if args.once:
                return
            time.sleep(cfg.WECHAT_WORKER_IDLE_SEC)
        except (WeChatILinkError, OSError) as exc:
            print(f"[wechat_worker] transient error: {type(exc).__name__}: {exc}")
            time.sleep(backoff)
            backoff = min(60, backoff * 2)
            if args.once:
                return
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
