"""Reminder scheduling tool.

The tool only writes a durable reminder row. A separate dispatcher process
delivers due reminders through WeChat, which keeps LLM/tool retries idempotent.
"""
from __future__ import annotations

import os

from langchain_core.tools import tool

from .local_logs import _actuation_response, _target_user_id, create_reminder


@tool
def push_reminder(
    remind_at_iso: str,
    text: str,
    idempotency_key: str = "",
    user_id: str = "",
    target_wxid: str = "",
    context_token: str = "",
    priority: str = "normal",
) -> str:
    """创建定时提醒。remind_at_iso 必须是 ISO 时间，例如 2026-05-19T20:00:00+08:00。"""
    user_id = _target_user_id(user_id)
    target_wxid = target_wxid or os.environ.get("WECHAT_TARGET_WXID", "")
    context_token = context_token or os.environ.get("WECHAT_CONTEXT_TOKEN", "")
    event = create_reminder(
        user_id=user_id,
        remind_at_iso=remind_at_iso,
        text=text,
        idempotency_key=idempotency_key,
        target_wxid=target_wxid,
        context_token=context_token,
        priority=priority,
    )
    if event.get("ok"):
        text_out = "提醒已写入本地队列。" if not event.get("duplicate") else "提醒已存在，未重复写入。"
    else:
        text_out = f"提醒创建失败：{event.get('error', '未知错误')}"
    return _actuation_response(event, text_out)
