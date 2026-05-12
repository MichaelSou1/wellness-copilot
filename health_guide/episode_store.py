"""Episode store — per-user episodic memory across conversation threads.

Each completed turn is recorded as one episode:
  {ts, query, experts, gist}

This fills the gap that SQLite checkpoints cannot: when a user starts a new
thread, the previous thread's checkpoint is inaccessible, but episodes are
keyed by user_id and survive across sessions.

Write path: critic_node (end of every turn).
Read path:  turn_start_node (beginning of every turn) → episode_context state field.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .config import EPISODE_STORE_PATH

MAX_EPISODES_PER_USER = 10


def _store_path() -> Path:
    return Path(EPISODE_STORE_PATH)


def _read_store() -> Dict[str, List]:
    p = _store_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_store(data: Dict[str, List]) -> None:
    _store_path().write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def append_episode(
    user_id: str,
    query: str,
    experts: List[str],
    gist: str,
) -> None:
    """Record one completed turn. Silently no-ops on any I/O error."""
    if not query:
        return
    data = _read_store()
    episodes: List[Dict[str, Any]] = data.get(user_id, [])
    episodes.append(
        {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "query": query[:120],
            "experts": list(experts or []),
            "gist": (gist or "").strip()[:150],
        }
    )
    data[user_id] = episodes[-MAX_EPISODES_PER_USER:]
    try:
        _write_store(data)
    except Exception:
        pass


def get_recent_episodes(user_id: str, n: int = 5) -> List[Dict[str, Any]]:
    """Return the n most recent episodes for a user, oldest first."""
    episodes = _read_store().get(user_id, [])
    return episodes[-n:]


def format_episodes_for_prompt(episodes: List[Dict[str, Any]]) -> str:
    if not episodes:
        return ""
    lines = []
    for ep in reversed(episodes):  # most recent first
        experts_str = "、".join(ep.get("experts") or []) or "General"
        gist = (ep.get("gist") or "").strip()
        gist_part = f"：{gist}" if gist else ""
        lines.append(
            f"• [{ep.get('ts', '')}] {ep.get('query', '')}"
            f"（专家：{experts_str}）{gist_part}"
        )
    return "\n".join(lines)
