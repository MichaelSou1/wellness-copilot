"""Outbox delivery handlers."""
from __future__ import annotations

from typing import Any

from .backend_telemetry import json_log
from .integrations.local_logs import mark_reminder_delivered
from .integrations.wechat_ilink import get_client


def _text_preview(text: str) -> str:
    return (text or "").replace("\n", " ")[:120]


def deliver_outbox_event(event: dict[str, Any], *, dry_run: bool = False) -> None:
    kind = str(event.get("kind") or "")
    payload = event.get("payload") or {}
    trace_id = event.get("trace_id") or ""
    event_id = event.get("event_id") or ""

    if kind in {"wechat_reply", "reminder_push"}:
        target = payload.get("target_wxid") or payload.get("user_id") or ""
        context_token = payload.get("context_token") or ""
        text = payload.get("text") or ""
        if dry_run:
            json_log(
                "outbox_dry_run",
                trace_id=trace_id,
                event_id=event_id,
                kind=kind,
                target_wxid=target,
                text_preview=_text_preview(text),
            )
        else:
            client = get_client()
            if context_token:
                client.send_message(str(context_token), text=str(text), user_id=str(target))
            else:
                client.push_to_user(str(target), str(text))
        if kind == "reminder_push" and payload.get("reminder_id"):
            mark_reminder_delivered(int(payload["reminder_id"]))
        return

    if kind in {"apple_calendar_event", "apple_workout"}:
        if dry_run:
            json_log("outbox_dry_run", trace_id=trace_id, event_id=event_id, kind=kind, payload_keys=sorted(payload))
            return
        if kind == "apple_workout":
            from .integrations.apple_calendar import schedule_workout

            result = schedule_workout.invoke(payload)
        else:
            from .integrations.apple_calendar import schedule_calendar_event

            result = schedule_calendar_event.invoke(payload)
        if '"ok": false' in str(result).lower():
            raise RuntimeError(str(result)[:300])
        return

    raise ValueError(f"unsupported outbox kind: {kind}")
