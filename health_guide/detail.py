"""Detail / traceroute mode.

Toggled by ``python main.py --detail``. When on, prints fine-grained
behavior of the agent team to stdout:

- Each MCP tool invocation (args + result preview, retries, errors).
- Each expert sub-agent: start, every ReAct step (tool_call / tool_result),
  end (final answer preview).

All public helpers are no-ops when the flag is off, so importing this
module is free for non-detail callers.
"""
from __future__ import annotations

import os
from typing import Any, Iterable


_DETAIL: bool = False


def set_detail(enabled: bool) -> None:
    """Turn detail mode on/off process-wide.

    Also mirrors to ``HEALTH_GUIDE_DETAIL`` so any module imported lazily
    (after argparse runs) still sees the same value via ``is_detail()``.
    """
    global _DETAIL
    _DETAIL = bool(enabled)
    os.environ["HEALTH_GUIDE_DETAIL"] = "1" if _DETAIL else "0"


def is_detail() -> bool:
    if _DETAIL:
        return True
    return os.environ.get("HEALTH_GUIDE_DETAIL") == "1"


def _truncate(value: Any, n: int) -> str:
    s = str(value).replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


def print_mcp_call(tool_name: str, kwargs: dict) -> None:
    if not is_detail():
        return
    try:
        args_preview = _truncate(kwargs, 200)
    except Exception:
        args_preview = "<unprintable>"
    print(f"    [MCP→{tool_name}] args={args_preview}")


def print_mcp_result(tool_name: str, result: Any, attempt: int = 1) -> None:
    if not is_detail():
        return
    suffix = f" (attempt {attempt})" if attempt > 1 else ""
    print(f"    [MCP←{tool_name}]{suffix} {_truncate(result, 240)}")


def print_mcp_error(tool_name: str, err: BaseException, attempt: int = 1) -> None:
    if not is_detail():
        return
    print(
        f"    [MCP✗{tool_name}] attempt {attempt} failed: "
        f"{type(err).__name__}: {_truncate(err, 200)}"
    )


def print_expert_start(role: str, question: str) -> None:
    if not is_detail():
        return
    print(f"  [Expert→{role}] question={_truncate(question, 160)}")


def print_expert_trace(role: str, messages: Iterable) -> None:
    """Walk the ReAct message list, emit one line per tool_call/tool_result."""
    if not is_detail():
        return
    try:
        for msg in messages:
            for call in getattr(msg, "tool_calls", None) or []:
                name = call.get("name", "?")
                args = call.get("args", {})
                print(f"    [{role}·tool_call] {name} args={_truncate(args, 140)}")
            if getattr(msg, "type", "") == "tool":
                name = getattr(msg, "name", "?")
                content = getattr(msg, "content", "")
                print(f"    [{role}·tool_result] {name} → {_truncate(content, 200)}")
    except Exception:
        pass


def print_expert_end(role: str, used_tools: list, answer: str) -> None:
    if not is_detail():
        return
    tools_str = ", ".join(used_tools) if used_tools else "(none)"
    print(f"  [Expert←{role}] tools=[{tools_str}] answer={_truncate(answer, 160)}")
