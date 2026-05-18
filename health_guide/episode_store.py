"""Episode store — per-user episodic memory across conversation threads.

Each completed turn is recorded as one episode:
  {ts, query, experts, gist}

This fills the gap that SQLite checkpoints cannot: when a user starts a new
thread, the previous thread's checkpoint is inaccessible, but episodes are
keyed by user_id and survive across sessions.

Write path: critic_node (end of every turn).
Read path:  turn_start_node (beginning of every turn) → episode_context state field.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import EPISODE_STORE_PATH

MAX_EPISODES_PER_USER = 20


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


def episode_id(ep: Dict[str, Any]) -> str:
    seed = f"{ep.get('ts', '')}|{ep.get('query', '')}|{ep.get('gist', '')[:80]}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
    return f"{ep.get('ts', 'episode')}-{digest}"


def _facts_as_string(facts: Optional[Dict[str, Any]]) -> str:
    if not facts:
        return ""
    parts = []
    for key, value in facts.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            rendered = "、".join(str(x) for x in value if str(x))
        else:
            rendered = str(value)
        if rendered:
            parts.append(f"{key}={rendered}")
    return "、".join(parts)


def episode_index_text(ep: Dict[str, Any]) -> str:
    return " | ".join(
        x
        for x in [
            str(ep.get("query") or "").strip(),
            str(ep.get("gist") or "").strip(),
            _facts_as_string(ep.get("facts") or {}),
        ]
        if x
    )


def append_episode(
    user_id: str,
    query: str,
    experts: List[str],
    gist: str,
    facts: Optional[Dict[str, Any]] = None,
) -> None:
    """Record one completed turn. Silently no-ops on any I/O error."""
    if not query:
        return
    data = _read_store()
    episodes: List[Dict[str, Any]] = data.get(user_id, [])
    episode = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "query": query[:120],
        "experts": list(experts or []),
        "gist": (gist or "").strip()[:400],
    }
    if facts:
        episode["facts"] = facts
    episode["id"] = episode_id(episode)
    episodes.append(episode)
    data[user_id] = episodes[-MAX_EPISODES_PER_USER:]
    try:
        _write_store(data)
    except Exception:
        pass
    try:
        from .episode_memory import EpisodeMemory

        EpisodeMemory(user_id).index_episode(
            episode["id"],
            episode_index_text(episode),
            episode=episode,
        )
    except Exception:
        # Embed-on-write is a warm cache only; failed indexing must not block
        # the user-visible answer.
        pass


def get_recent_episodes(user_id: str, n: int = 5) -> List[Dict[str, Any]]:
    """Return the n most recent episodes for a user, oldest first."""
    episodes = _read_store().get(user_id, [])
    return episodes[-n:]


def get_all_episodes(user_id: str) -> List[Dict[str, Any]]:
    return list(_read_store().get(user_id, []) or [])


def total_episode_count(user_id: str) -> int:
    return len(_read_store().get(user_id, []) or [])


def format_episodes_for_prompt(episodes: List[Dict[str, Any]], mark_source: bool = False) -> str:
    if not episodes:
        return ""
    lines = []
    iter_eps = episodes if mark_source else reversed(episodes)
    for ep in iter_eps:  # default path keeps legacy most-recent-first rendering
        experts_str = "、".join(ep.get("experts") or []) or "General"
        gist = (ep.get("gist") or "").strip()
        gist_part = f"：{gist}" if gist else ""
        facts = _facts_as_string(ep.get("facts") or {})
        facts_part = f"（已记录：{facts}）" if facts else ""
        source = ""
        if mark_source:
            source = f"[{ep.get('_memory_source') or '最近'}] "
        lines.append(
            f"• {source}[{ep.get('ts', '')}] {ep.get('query', '')}"
            f"（专家：{experts_str}）{facts_part}{gist_part}"
        )
    return "\n".join(lines)
