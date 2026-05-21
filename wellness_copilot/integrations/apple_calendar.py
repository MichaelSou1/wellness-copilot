"""Apple Calendar / iCloud CalDAV integration.

Mutating tools return the same ``[ACTUATION]`` envelope used by local logs so
Aggregator and Critic can verify that calendar side effects really happened.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from ..config import DEFAULT_TIMEZONE
from .local_logs import _actuation_response, _now_epoch, _target_user_id


DEFAULT_ICLOUD_CALDAV_URL = "https://caldav.icloud.com"


def _setting(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _event_payload(
    ok: bool,
    action: str,
    *,
    uid: str = "",
    duplicate: bool | None = None,
    event_url: str = "",
    calendar_name: str = "",
    user_id: str = "",
    start_iso: str = "",
    end_iso: str = "",
    idempotency_key: str = "",
    error: str = "",
    exception: str = "",
    available_calendars: list[str] | None = None,
) -> dict:
    event: dict[str, Any] = {
        "ok": bool(ok),
        "action": action,
        "table": "apple_calendar",
        "row_id": None,
        "ts": _now_epoch(),
    }
    extras = {
        "uid": uid,
        "duplicate": duplicate,
        "event_url": event_url,
        "calendar_name": calendar_name,
        "user_id": user_id,
        "start_iso": start_iso,
        "end_iso": end_iso,
        "idempotency_key": idempotency_key,
        "error": error,
        "exception": exception,
        "available_calendars": available_calendars,
    }
    event.update({key: value for key, value in extras.items() if value not in ("", None)})
    return event


def _response(event: dict, success_text: str) -> str:
    if event.get("ok"):
        human = success_text if not event.get("duplicate") else "Apple Calendar 日程已存在，未重复创建。"
    else:
        human = f"Apple Calendar 写入失败：{event.get('error', '未知错误')}"
    return _actuation_response(event, human)


def _timezone() -> timezone | ZoneInfo:
    name = _setting("DEFAULT_TIMEZONE", DEFAULT_TIMEZONE) or "UTC"
    try:
        return ZoneInfo(name)
    except Exception:
        return timezone.utc


def _parse_start(start_iso: str) -> datetime:
    text = (start_iso or "").strip()
    if not text:
        raise ValueError("start_iso 必填")
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_timezone())
    return dt


def _stable_uid(
    *,
    action: str,
    user_id: str,
    title: str,
    start_iso: str,
    duration_min: int,
    description: str,
    idempotency_key: str,
) -> str:
    raw = idempotency_key or f"{action}:{user_id}:{title}:{start_iso}:{duration_min}:{description}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"wellness-copilot-{digest}@wellness-copilot"


def _calendar_name(calendar: Any) -> str:
    for attr in ("name", "calendar_name"):
        value = getattr(calendar, attr, "")
        if callable(value):
            try:
                value = value()
            except Exception:
                value = ""
        if value:
            return str(value).strip()
    url = str(getattr(calendar, "url", "") or "").rstrip("/")
    return url.rsplit("/", 1)[-1] if url else ""


def _calendar_url(value: Any) -> str:
    return str(getattr(value, "url", "") or "").strip()


def _event_url_for_uid(calendar: Any, uid: str) -> str:
    base = _calendar_url(calendar).rstrip("/")
    if not base or not uid:
        return ""
    return f"{base}/{quote(uid, safe='')}.ics"


def _select_calendar(calendars: list[Any], requested_name: str) -> tuple[Any | None, list[str]]:
    named = [(calendar, _calendar_name(calendar)) for calendar in calendars]
    names = [name for _, name in named if name]
    if not calendars:
        return None, names

    requested = (requested_name or "").strip().casefold()
    if requested:
        for calendar, name in named:
            if name.casefold() == requested:
                return calendar, names
        for calendar, name in named:
            if requested in name.casefold():
                return calendar, names
        return None, names

    read_only_hints = ("birthday", "birthdays", "holiday", "holidays", "生日", "节假日", "假日")
    for calendar, name in named:
        lowered = name.casefold()
        if name and not any(hint in lowered for hint in read_only_hints):
            return calendar, names
    return calendars[0], names


def _find_existing_event(calendar: Any, uid: str) -> Any | None:
    event_by_url = getattr(calendar, "event_by_url", None)
    direct_url = _event_url_for_uid(calendar, uid)
    if callable(event_by_url) and direct_url:
        try:
            existing = event_by_url(direct_url)
            if existing:
                return existing
        except Exception:
            pass

    event_by_uid = getattr(calendar, "event_by_uid", None)
    if callable(event_by_uid):
        try:
            existing = event_by_uid(uid)
            if existing:
                return existing
        except Exception:
            pass

    search = getattr(calendar, "search", None)
    if callable(search):
        for kwargs in ({"uid": uid}, {"event": True, "uid": uid}):
            try:
                matches = search(**kwargs)
            except Exception:
                continue
            if matches:
                return matches[0]
    return None


def _icalendar_payload(
    *,
    uid: str,
    title: str,
    start_dt: datetime,
    end_dt: datetime,
    description: str,
    category: str,
) -> str:
    from icalendar import Calendar, Event

    cal = Calendar()
    cal.add("prodid", "-//Wellness Copilot//Apple Calendar//CN")
    cal.add("version", "2.0")
    event = Event()
    event.add("uid", uid)
    event.add("summary", title)
    event.add("dtstart", start_dt)
    event.add("dtend", end_dt)
    event.add("dtstamp", datetime.now(timezone.utc))
    if description:
        event.add("description", description)
    if category:
        event.add("categories", ["HEALTH", category.upper()])
    event.add("status", "CONFIRMED")
    event.add("transp", "OPAQUE")
    cal.add_component(event)
    return cal.to_ical().decode("utf-8")


def _connect_and_select_calendar_with_names(action: str = "schedule_calendar_event"):
    try:
        import caldav
    except ImportError as exc:
        return None, [], _event_payload(
            False,
            action,
            error="caldav/icalendar not installed",
            exception=type(exc).__name__,
        )

    username = _setting("ICLOUD_USERNAME")
    password = _setting("ICLOUD_APP_SPECIFIC_PASSWORD")
    if not username or not password:
        return None, [], _event_payload(
            False,
            action,
            error="iCloud credentials not configured",
        )

    url = _setting("ICLOUD_CALDAV_URL", DEFAULT_ICLOUD_CALDAV_URL) or DEFAULT_ICLOUD_CALDAV_URL
    requested_name = _setting("ICLOUD_CALENDAR_NAME")
    try:
        client = caldav.DAVClient(url=url, username=username, password=password)
        principal = client.principal()
        calendars = list(principal.calendars())
        calendar, names = _select_calendar(calendars, requested_name)
    except Exception as exc:
        return None, [], _event_payload(
            False,
            action,
            error="无法连接 iCloud CalDAV，请检查 Apple ID、App 专用密码和网络",
            exception=type(exc).__name__,
        )

    if calendar is None:
        error = (
            f"找不到名为 {requested_name!r} 的 iCloud 日历"
            if requested_name
            else "iCloud 账号下没有可用日历"
        )
        return None, names, _event_payload(
            False,
            action,
            error=error,
            available_calendars=names,
        )
    return calendar, names, None


def _connect_and_select_calendar():
    calendar, _, setup_error = _connect_and_select_calendar_with_names()
    return calendar, setup_error


def _schedule_event(
    *,
    action: str,
    title: str,
    start_iso: str,
    duration_min: int,
    description: str = "",
    idempotency_key: str = "",
    user_id: str = "",
    category: str = "event",
) -> str:
    user_id = _target_user_id(user_id)
    title = (title or "").strip()
    description = (description or "").strip()
    idempotency_key = (idempotency_key or "").strip()
    if not title:
        event = _event_payload(False, action, user_id=user_id, error="title 必填")
        return _response(event, "")
    try:
        duration = int(duration_min or 0)
    except Exception:
        duration = 0
    if duration <= 0:
        event = _event_payload(False, action, user_id=user_id, error="duration_min 必须大于 0")
        return _response(event, "")
    try:
        start_dt = _parse_start(start_iso)
    except Exception:
        event = _event_payload(False, action, user_id=user_id, error="start_iso 必须是 ISO 时间")
        return _response(event, "")

    end_dt = start_dt + timedelta(minutes=duration)
    uid = _stable_uid(
        action=action,
        user_id=user_id,
        title=title,
        start_iso=start_dt.isoformat(),
        duration_min=duration,
        description=description,
        idempotency_key=idempotency_key,
    )
    calendar, setup_error = _connect_and_select_calendar()
    if setup_error:
        setup_error["action"] = action
        setup_error.update(
            {
                "uid": uid,
                "user_id": user_id,
                "start_iso": start_dt.isoformat(),
                "end_iso": end_dt.isoformat(),
                "idempotency_key": idempotency_key or uid,
            }
        )
        return _response(setup_error, "")

    calendar_name = _calendar_name(calendar)
    try:
        existing = _find_existing_event(calendar, uid)
        if existing is not None:
            event = _event_payload(
                True,
                action,
                uid=uid,
                duplicate=True,
                event_url=_calendar_url(existing),
                calendar_name=calendar_name,
                user_id=user_id,
                start_iso=start_dt.isoformat(),
                end_iso=end_dt.isoformat(),
                idempotency_key=idempotency_key or uid,
            )
            return _response(event, "")

        payload = _icalendar_payload(
            uid=uid,
            title=title,
            start_dt=start_dt,
            end_dt=end_dt,
            description=description,
            category=category,
        )
        saved = calendar.save_event(payload)
    except ImportError as exc:
        event = _event_payload(
            False,
            action,
            uid=uid,
            calendar_name=calendar_name,
            user_id=user_id,
            start_iso=start_dt.isoformat(),
            end_iso=end_dt.isoformat(),
            idempotency_key=idempotency_key or uid,
            error="caldav/icalendar not installed",
            exception=type(exc).__name__,
        )
        return _response(event, "")
    except Exception as exc:
        event = _event_payload(
            False,
            action,
            uid=uid,
            calendar_name=calendar_name,
            user_id=user_id,
            start_iso=start_dt.isoformat(),
            end_iso=end_dt.isoformat(),
            idempotency_key=idempotency_key or uid,
            error="CalDAV 事件创建失败",
            exception=type(exc).__name__,
        )
        return _response(event, "")

    event = _event_payload(
        True,
        action,
        uid=uid,
        duplicate=False,
        event_url=_calendar_url(saved),
        calendar_name=calendar_name,
        user_id=user_id,
        start_iso=start_dt.isoformat(),
        end_iso=end_dt.isoformat(),
        idempotency_key=idempotency_key or uid,
    )
    return _response(event, "Apple Calendar 日程已创建。")


def list_apple_calendars() -> dict:
    """Return visible iCloud calendars without exposing credentials."""
    calendar, names, setup_error = _connect_and_select_calendar_with_names("list_apple_calendars")
    if setup_error:
        return setup_error
    selected = _calendar_name(calendar)
    return {
        "ok": True,
        "action": "list_apple_calendars",
        "table": "apple_calendar",
        "row_id": None,
        "ts": _now_epoch(),
        "calendar_name": selected,
        "available_calendars": names,
    }


@tool
def schedule_calendar_event(
    title: str,
    start_iso: str,
    duration_min: int = 30,
    description: str = "",
    idempotency_key: str = "",
    user_id: str = "",
) -> str:
    """把健康相关事项写入 Apple Calendar。start_iso 必须是 ISO 时间，建议带时区。"""
    return _schedule_event(
        action="schedule_calendar_event",
        title=title,
        start_iso=start_iso,
        duration_min=duration_min,
        description=description,
        idempotency_key=idempotency_key,
        user_id=user_id,
        category="health",
    )


@tool
def schedule_workout(
    title: str,
    start_iso: str,
    duration_min: int = 60,
    description: str = "",
    idempotency_key: str = "",
    user_id: str = "",
) -> str:
    """把训练、跑步或恢复安排写入 Apple Calendar。start_iso 必须是 ISO 时间，建议带时区。"""
    return _schedule_event(
        action="schedule_workout",
        title=title,
        start_iso=start_iso,
        duration_min=duration_min,
        description=description,
        idempotency_key=idempotency_key,
        user_id=user_id,
        category="workout",
    )
