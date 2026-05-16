"""Smoke test for the community MCP registry.

Runs ``MCP_REGISTRY.startup()`` with whichever ``MCP_*_ENABLED`` flags the
current environment has set, lists the tools each server contributed, and
optionally exercises one tool per server end-to-end (sync invoke that crosses
the dedicated event-loop thread back to the npx subprocess).

This isn't a unit test — it's a CI/local sanity check. The runtime that owns
the MCP subprocesses lives in ``health_guide/mcp_client.py``; the smoke just
verifies the wiring.

Run: python scripts/smoke_mcp_tools.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from health_guide import config  # noqa: E402
from health_guide.mcp_client import MCP_REGISTRY  # noqa: E402


_SAMPLE_INVOCATIONS = {
    "wger": ("search_exercises", {"query": "squat", "limit": 1}),
    "usda": ("search-foods", {"query": "chicken breast", "pageSize": 1}),
    "medical": ("search-medical-literature", {"query": "ibuprofen safety", "max_results": 1}),
}


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    _print_header("config flags")
    print(f"MCP_TRAINER_ENABLED      = {config.MCP_TRAINER_ENABLED}")
    print(f"MCP_NUTRITIONIST_ENABLED = {config.MCP_NUTRITIONIST_ENABLED}")
    print(f"MCP_CRITIC_ENABLED       = {config.MCP_CRITIC_ENABLED}")
    print(f"USDA_API_KEY set         = {bool(config.USDA_API_KEY)}")
    print(f"MCP_USDA_SCRIPT_PATH     = {config.MCP_USDA_SCRIPT_PATH or '(unset)'}")

    if not any(
        [
            config.MCP_TRAINER_ENABLED,
            config.MCP_NUTRITIONIST_ENABLED,
            config.MCP_CRITIC_ENABLED,
        ]
    ):
        print(
            "\nAll MCP_*_ENABLED flags are false — nothing to smoke. "
            "Export at least one before re-running."
        )
        return 0

    _print_header("startup")
    MCP_REGISTRY.startup()
    available = MCP_REGISTRY.available_servers()
    print(f"available servers: {available or '(none — check logs above)'}")

    rc = 0
    for server in available:
        _print_header(f"server: {server}")
        tools = MCP_REGISTRY.get_tools(server)
        for t in tools:
            print(f"  - {t.name}")
        if server in _SAMPLE_INVOCATIONS:
            tool_name, payload = _SAMPLE_INVOCATIONS[server]
            target = next((t for t in tools if t.name == tool_name), None)
            if target is None:
                print(f"  [skip] tool {tool_name!r} not present, can't sample-invoke")
                continue
            print(f"  [invoke] {tool_name}({payload})")
            try:
                result = target.invoke(payload)
            except Exception as e:
                print(f"  [error] {type(e).__name__}: {e}")
                rc = 1
                continue
            preview = str(result)
            if len(preview) > 400:
                preview = preview[:400] + "..."
            print(f"  [result] {preview}")
            if isinstance(result, str) and result.startswith("[MCP Error]"):
                rc = 1

    _print_header("shutdown")
    MCP_REGISTRY.shutdown()
    print("ok")
    return rc


if __name__ == "__main__":
    sys.exit(main())
