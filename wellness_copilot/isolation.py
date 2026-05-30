"""Runtime-switchable subagent context-isolation controls.

The dispatcher hands each specialist child agent an *isolated* context
(see ``agents/dispatcher.py``). Isolation is enforced by three independent
mechanisms:

  profile  ① role-cropped 画像 — each expert sees only its role's user card
            (``personalization.build_personalization_ctx``).
  peer     ② same-batch peer notes are hidden — experts in one plan batch run
            in parallel and don't see each other's scratchpad
            (``dispatcher._run_plan`` + ``_scratchpad.format_peer_notes``).
  history  ③ no full transcript — each expert receives only the rewritten
            question, never the parent's accumulated message history.

All three default to ON, so production behavior is unchanged. They exist so the
A/B isolation evaluation (``scripts/evaluate_isolation_ab.py``) can flip them —
per-dimension, in-process — to measure isolation's causal effect.

Why a mutable module instead of import-time env constants: the A/B runner must
toggle isolation *within one process* (run arm A, then arm B), so the flags are
read at call time via ``current()`` rather than frozen at import.

NOTE: the ``profile`` switch flips only the dominant crop — the per-role
``role_user_cards``. Decision points (``build_personalization_decision_points``)
are also role-cropped but remain isolated in both arms; treat the profile arm as
"full user card to every expert", not "every personalization surface un-cropped".
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Iterator, List

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _env_tristate(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = raw.strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    return default


@dataclass(frozen=True)
class IsolationConfig:
    """Which isolation mechanisms are active. All True == current production."""

    profile: bool = True  # ① role-cropped 画像
    peer: bool = True     # ② 同批 peer notes 过滤
    history: bool = True  # ③ 不注入完整 history


def _config_from_env() -> IsolationConfig:
    # Master switch sets the baseline; per-dimension vars override it.
    master = _env_tristate("WELLNESS_ISOLATION", True)
    return IsolationConfig(
        profile=_env_tristate("WELLNESS_ISOLATION_PROFILE", master),
        peer=_env_tristate("WELLNESS_ISOLATION_PEER", master),
        history=_env_tristate("WELLNESS_ISOLATION_HISTORY", master),
    )


_LOCK = threading.RLock()
_STATE: IsolationConfig = _config_from_env()


def current() -> IsolationConfig:
    """The isolation config in effect right now (read at call time)."""
    return _STATE


def set_isolation(*, profile: bool | None = None, peer: bool | None = None,
                  history: bool | None = None) -> IsolationConfig:
    """Mutate the active isolation config. Returns the new config.

    Only the keyword args you pass are changed; the rest are preserved.
    """
    global _STATE
    with _LOCK:
        changes = {}
        if profile is not None:
            changes["profile"] = profile
        if peer is not None:
            changes["peer"] = peer
        if history is not None:
            changes["history"] = history
        _STATE = replace(_STATE, **changes)
        return _STATE


@contextmanager
def isolation_override(*, profile: bool | None = None, peer: bool | None = None,
                       history: bool | None = None) -> Iterator[IsolationConfig]:
    """Temporarily flip isolation within a ``with`` block, then restore.

    Used by the A/B runner: ``with isolation_override(profile=False, peer=False,
    history=False):`` runs the non-isolated arm. With no args it pins the current
    state (handy as the explicit isolated arm).
    """
    global _STATE
    with _LOCK:
        previous = _STATE
    try:
        yield set_isolation(profile=profile, peer=peer, history=history)
    finally:
        with _LOCK:
            _STATE = previous


# ---------------------------------------------------------------------------
# History injection helpers (used when isolation.history is OFF)
# ---------------------------------------------------------------------------

# Key under which the dispatcher stashes the rendered transcript into pctx so it
# reaches each expert without changing the run_* / _build_*_agent signatures.
PCTX_HISTORY_KEY = "_noniso_history"

_MAX_HISTORY_CHARS = 6000


def render_transcript(messages: List) -> str:
    """Render LangChain messages into a plain 用户/助手 transcript.

    Skips system messages and empty/tool messages. Truncates from the front so
    the most recent turns survive the char cap.
    """
    lines: List[str] = []
    for msg in messages or []:
        msg_type = getattr(msg, "type", "")
        if msg_type in ("system", "tool"):
            continue
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            # multimodal content blocks → keep only text parts
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        content = str(content or "").strip()
        if not content:
            continue
        speaker = "用户" if msg_type == "human" else "助手"
        lines.append(f"{speaker}：{content}")
    if not lines:
        return ""
    text = "\n".join(lines)
    if len(text) > _MAX_HISTORY_CHARS:
        text = "…（已截断较早内容）\n" + text[-_MAX_HISTORY_CHARS:]
    return text


def noniso_history_section(pctx: dict | None) -> str:
    """System-prompt section carrying the full transcript — only when history
    isolation is OFF and the dispatcher provided a transcript. Empty otherwise,
    so the isolated arm (and production) is byte-identical to before."""
    if current().history:
        return ""
    transcript = (pctx or {}).get(PCTX_HISTORY_KEY) or ""
    if not transcript.strip():
        return ""
    return (
        "\n【完整对话历史（未隔离模式注入）】\n"
        f"{transcript}\n"
        "说明：以上为本会话的完整往来记录，供参考。\n"
    )
