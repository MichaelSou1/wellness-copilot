"""Local SQLite health logs and LangChain tools.

The tables are intentionally small and append-friendly. Every mutating tool
accepts an idempotency_key and uses a UNIQUE constraint so checkpoint replay or
worker retries do not duplicate real-world side effects.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from ..config import HEALTH_LOGS_DB_PATH


ACTUATION_PREFIX = "[ACTUATION]"
_CONN: sqlite3.Connection | None = None


def _db_path() -> str:
    return os.environ.get("HEALTH_LOGS_DB_PATH") or HEALTH_LOGS_DB_PATH


def _connect() -> sqlite3.Connection:
    global _CONN
    if _CONN is not None:
        return _CONN
    path = Path(_db_path())
    if path.parent and str(path.parent) not in {"", "."}:
        path.parent.mkdir(parents=True, exist_ok=True)
    _CONN = sqlite3.connect(str(path), check_same_thread=False)
    _CONN.row_factory = sqlite3.Row
    _CONN.execute("PRAGMA journal_mode=WAL")
    _CONN.execute("PRAGMA busy_timeout=5000")
    _ensure_init()
    return _CONN


def _ensure_init() -> None:
    path = Path(_db_path())
    if path.parent and str(path.parent) not in {"", "."}:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = _CONN or sqlite3.connect(str(path), check_same_thread=False)
    try:
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                date_iso TEXT NOT NULL,
                items_json TEXT,
                kcal INTEGER,
                protein_g INTEGER,
                carbs_g INTEGER,
                fat_g INTEGER,
                source TEXT,
                idempotency_key TEXT UNIQUE,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS workouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                date_iso TEXT NOT NULL,
                plan_json TEXT,
                status TEXT,
                idempotency_key TEXT UNIQUE,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS wellness (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                date_iso TEXT NOT NULL,
                sleep_h REAL,
                mood TEXT,
                notes TEXT,
                idempotency_key TEXT UNIQUE,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                target_wxid TEXT,
                context_token TEXT,
                remind_at_iso TEXT NOT NULL,
                remind_at_epoch INTEGER NOT NULL,
                text TEXT NOT NULL,
                priority TEXT DEFAULT 'normal',
                delivered INTEGER DEFAULT 0,
                delivered_at INTEGER,
                idempotency_key TEXT UNIQUE,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_meals_user_date ON meals(user_id, date_iso);
            CREATE INDEX IF NOT EXISTS ix_workouts_user_date ON workouts(user_id, date_iso);
            CREATE INDEX IF NOT EXISTS ix_wellness_user_date ON wellness(user_id, date_iso);
            CREATE INDEX IF NOT EXISTS ix_reminders_due ON reminders(delivered, remind_at_epoch);
            """
        )
        conn.commit()
    finally:
        if _CONN is None:
            conn.close()


def init_db() -> None:
    _connect()


def _target_user_id(user_id: str = "") -> str:
    return user_id or os.environ.get("HEALTH_GUIDE_USER_ID", "default_user")


def _now_epoch() -> int:
    return int(time.time())


def _today_iso() -> str:
    return datetime.now().date().isoformat()


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _stable_key(kind: str, user_id: str, *parts: Any) -> str:
    raw = json.dumps([kind, user_id, *parts], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _actuation_event(ok: bool, action: str, table: str, row_id: int | None, **extra) -> dict:
    event = {
        "ok": bool(ok),
        "action": action,
        "table": table,
        "row_id": row_id,
        "ts": _now_epoch(),
    }
    event.update({k: v for k, v in extra.items() if v is not None})
    return event


def _actuation_response(event: dict, human_text: str = "") -> str:
    return f"{ACTUATION_PREFIX}{json.dumps(event, ensure_ascii=False, sort_keys=True)}" + (
        f"\n{human_text}" if human_text else ""
    )


def extract_actuation_events(messages: list[Any]) -> list[dict]:
    events: list[dict] = []
    for msg in messages or []:
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            texts = [str(x.get("text", "")) for x in content if isinstance(x, dict)]
            content = "\n".join(texts)
        text = str(content or "")
        start = 0
        while True:
            idx = text.find(ACTUATION_PREFIX, start)
            if idx < 0:
                break
            line = text[idx + len(ACTUATION_PREFIX):].splitlines()[0].strip()
            try:
                event = json.loads(line)
            except Exception:
                start = idx + len(ACTUATION_PREFIX)
                continue
            if isinstance(event, dict):
                events.append(event)
            start = idx + len(ACTUATION_PREFIX) + len(line)
    return events


def _insert_or_get(table: str, values: dict) -> tuple[int | None, bool]:
    conn = _connect()
    columns = list(values)
    placeholders = ", ".join("?" for _ in columns)
    cur = conn.cursor()
    cur.execute(
        f"INSERT OR IGNORE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
        [values[col] for col in columns],
    )
    conn.commit()
    inserted = cur.rowcount > 0
    if inserted:
        return int(cur.lastrowid), False
    idem = values.get("idempotency_key")
    if not idem:
        return None, False
    row = conn.execute(f"SELECT id FROM {table} WHERE idempotency_key = ?", (idem,)).fetchone()
    return (int(row["id"]) if row else None), bool(row)


@tool
def log_meal(
    date_iso: str = "",
    items_json: str = "",
    kcal: int = 0,
    protein_g: int = 0,
    carbs_g: int = 0,
    fat_g: int = 0,
    source: str = "manual",
    idempotency_key: str = "",
    user_id: str = "",
) -> str:
    """记录一餐到本地 SQLite。items_json 是食物列表 JSON 字符串。"""
    user_id = _target_user_id(user_id)
    date_iso = (date_iso or _today_iso()).strip()
    items_json = _json_text(items_json)
    idempotency_key = idempotency_key or _stable_key(
        "meal",
        user_id,
        date_iso,
        items_json,
        kcal,
        protein_g,
        carbs_g,
        fat_g,
    )
    row_id, duplicate = _insert_or_get(
        "meals",
        {
            "user_id": user_id,
            "date_iso": date_iso,
            "items_json": items_json,
            "kcal": int(kcal or 0),
            "protein_g": int(protein_g or 0),
            "carbs_g": int(carbs_g or 0),
            "fat_g": int(fat_g or 0),
            "source": source or "manual",
            "idempotency_key": idempotency_key,
            "created_at": _now_epoch(),
        },
    )
    event = _actuation_event(
        True,
        "log_meal",
        "meals",
        row_id,
        duplicate=duplicate,
        idempotency_key=idempotency_key,
        user_id=user_id,
        date_iso=date_iso,
    )
    return _actuation_response(event, "餐食日志已记录。" if not duplicate else "餐食日志已存在，未重复写入。")


@tool
def log_workout(
    date_iso: str = "",
    plan_json: str = "",
    status: str = "planned",
    idempotency_key: str = "",
    user_id: str = "",
) -> str:
    """记录训练计划或训练完成状态到本地 SQLite。plan_json 是训练计划 JSON 字符串。"""
    user_id = _target_user_id(user_id)
    date_iso = (date_iso or _today_iso()).strip()
    plan_json = _json_text(plan_json)
    status = (status or "planned").strip()
    idempotency_key = idempotency_key or _stable_key("workout", user_id, date_iso, plan_json, status)
    row_id, duplicate = _insert_or_get(
        "workouts",
        {
            "user_id": user_id,
            "date_iso": date_iso,
            "plan_json": plan_json,
            "status": status,
            "idempotency_key": idempotency_key,
            "created_at": _now_epoch(),
        },
    )
    event = _actuation_event(
        True,
        "log_workout",
        "workouts",
        row_id,
        duplicate=duplicate,
        idempotency_key=idempotency_key,
        user_id=user_id,
        date_iso=date_iso,
    )
    return _actuation_response(event, "训练日志已记录。" if not duplicate else "训练日志已存在，未重复写入。")


@tool
def log_wellness_checkin(
    date_iso: str = "",
    sleep_h: float = 0.0,
    mood: str = "",
    notes: str = "",
    idempotency_key: str = "",
    user_id: str = "",
) -> str:
    """记录睡眠、情绪或恢复 check-in 到本地 SQLite。"""
    user_id = _target_user_id(user_id)
    date_iso = (date_iso or _today_iso()).strip()
    mood = (mood or "").strip()
    notes = (notes or "").strip()
    idempotency_key = idempotency_key or _stable_key("wellness", user_id, date_iso, sleep_h, mood, notes)
    row_id, duplicate = _insert_or_get(
        "wellness",
        {
            "user_id": user_id,
            "date_iso": date_iso,
            "sleep_h": float(sleep_h or 0),
            "mood": mood,
            "notes": notes,
            "idempotency_key": idempotency_key,
            "created_at": _now_epoch(),
        },
    )
    event = _actuation_event(
        True,
        "log_wellness_checkin",
        "wellness",
        row_id,
        duplicate=duplicate,
        idempotency_key=idempotency_key,
        user_id=user_id,
        date_iso=date_iso,
    )
    return _actuation_response(event, "恢复/情绪日志已记录。" if not duplicate else "恢复/情绪日志已存在，未重复写入。")


def _table_for_kind(kind: str) -> str:
    normalized = (kind or "").strip().lower()
    aliases = {
        "meal": "meals",
        "meals": "meals",
        "workout": "workouts",
        "workouts": "workouts",
        "training": "workouts",
        "wellness": "wellness",
        "sleep": "wellness",
        "mood": "wellness",
        "reminder": "reminders",
        "reminders": "reminders",
    }
    if normalized not in aliases:
        raise ValueError("kind 必须是 meal/workout/wellness/reminder/all")
    return aliases[normalized]


def _query_table(table: str, user_id: str, days_back: int) -> list[dict]:
    since = (datetime.now().date() - timedelta(days=max(0, int(days_back or 0)))).isoformat()
    conn = _connect()
    date_col = "remind_at_iso" if table == "reminders" else "date_iso"
    rows = conn.execute(
        f"SELECT * FROM {table} WHERE user_id = ? AND {date_col} >= ? ORDER BY {date_col} DESC, id DESC LIMIT 200",
        (user_id, since),
    ).fetchall()
    return [dict(row) for row in rows]


@tool
def query_logs(kind: str = "all", days_back: int = 7, user_id: str = "") -> str:
    """读取本地健康日志。kind 可选 meal/workout/wellness/reminder/all。"""
    user_id = _target_user_id(user_id)
    if (kind or "all").strip().lower() == "all":
        payload = {
            table: _query_table(table, user_id, days_back)
            for table in ("meals", "workouts", "wellness", "reminders")
        }
    else:
        table = _table_for_kind(kind)
        payload = {table: _query_table(table, user_id, days_back)}
    return json.dumps(
        {
            "ok": True,
            "user_id": user_id,
            "days_back": int(days_back or 7),
            "data": payload,
        },
        ensure_ascii=False,
        default=str,
    )


def summarize_recent_logs(user_id: str = "", days_back: int = 7) -> str:
    user_id = _target_user_id(user_id)
    try:
        meals = _query_table("meals", user_id, days_back)
        workouts = _query_table("workouts", user_id, days_back)
        wellness = _query_table("wellness", user_id, days_back)
    except Exception:
        return ""
    if not meals and not workouts and not wellness:
        return ""

    lines = []
    if meals:
        kcal = sum(int(row.get("kcal") or 0) for row in meals)
        protein = sum(int(row.get("protein_g") or 0) for row in meals)
        count = len(meals)
        lines.append(
            f"近{days_back}日记录{count}餐，均值约 {round(kcal / count)} kcal、蛋白 {round(protein / count)}g/餐。"
        )
    if workouts:
        planned = sum(1 for row in workouts if str(row.get("status") or "").lower() == "planned")
        done = sum(1 for row in workouts if str(row.get("status") or "").lower() in {"done", "completed", "finished"})
        lines.append(f"训练日志{len(workouts)}条，其中已完成{done}条、计划中{planned}条。")
    if wellness:
        sleep_values = [float(row.get("sleep_h") or 0) for row in wellness if float(row.get("sleep_h") or 0) > 0]
        sleep_text = f"，睡眠均值 {round(sum(sleep_values) / len(sleep_values), 1)}h" if sleep_values else ""
        moods = [str(row.get("mood") or "").strip() for row in wellness if str(row.get("mood") or "").strip()]
        mood_text = f"，近期情绪：{'、'.join(moods[:3])}" if moods else ""
        lines.append(f"恢复/情绪日志{len(wellness)}条{sleep_text}{mood_text}。")
    return " ".join(lines)[:300]


def create_reminder(
    *,
    user_id: str,
    remind_at_iso: str,
    text: str,
    idempotency_key: str = "",
    target_wxid: str = "",
    context_token: str = "",
    priority: str = "normal",
) -> dict:
    remind_at_iso = (remind_at_iso or "").strip()
    text = (text or "").strip()
    if not remind_at_iso or not text:
        return _actuation_event(False, "push_reminder", "reminders", None, error="remind_at_iso 和 text 必填")
    try:
        remind_dt = datetime.fromisoformat(remind_at_iso.replace("Z", "+00:00"))
        remind_epoch = int(remind_dt.timestamp())
    except Exception:
        return _actuation_event(False, "push_reminder", "reminders", None, error="remind_at_iso 必须是 ISO 时间")
    idempotency_key = idempotency_key or _stable_key("reminder", user_id, remind_at_iso, text)
    row_id, duplicate = _insert_or_get(
        "reminders",
        {
            "user_id": user_id,
            "target_wxid": target_wxid,
            "context_token": context_token,
            "remind_at_iso": remind_at_iso,
            "remind_at_epoch": remind_epoch,
            "text": text,
            "priority": priority or "normal",
            "delivered": 0,
            "delivered_at": None,
            "idempotency_key": idempotency_key,
            "created_at": _now_epoch(),
        },
    )
    return _actuation_event(
        True,
        "push_reminder",
        "reminders",
        row_id,
        duplicate=duplicate,
        idempotency_key=idempotency_key,
        user_id=user_id,
        remind_at_iso=remind_at_iso,
    )


def due_reminders(now_epoch: int | None = None, limit: int = 50) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        """
        SELECT * FROM reminders
        WHERE delivered = 0 AND remind_at_epoch <= ?
        ORDER BY remind_at_epoch ASC, id ASC
        LIMIT ?
        """,
        (int(now_epoch or _now_epoch()), int(limit or 50)),
    ).fetchall()
    return [dict(row) for row in rows]


def mark_reminder_delivered(reminder_id: int) -> None:
    _connect().execute(
        "UPDATE reminders SET delivered = 1, delivered_at = ? WHERE id = ?",
        (_now_epoch(), int(reminder_id)),
    )
    _connect().commit()


def get_kv(key: str, default: str = "") -> str:
    row = _connect().execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row else default


def set_kv(key: str, value: str) -> None:
    _connect().execute(
        """
        INSERT INTO kv(key, value, updated_at) VALUES(?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """,
        (key, value, _now_epoch()),
    )
    _connect().commit()
