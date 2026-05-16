"""Community MCP server registry (Trainer / Nutritionist / Critic).

Spawns the configured MCP subprocesses once at process start and exposes their
tools as synchronous LangChain `StructuredTool` objects to the rest of the
codebase (the experts' ReAct loops and the Critic's pre-injection helper are
all synchronous).

Three servers are wired:

- ``wger``     → Trainer        exercise encyclopedia (5 public tools)
- ``usda``     → Nutritionist   USDA FoodData Central (1 tool: ``search-foods``)
- ``medical``  → Critic         FDA / PubMed / WHO / RxNorm (16 tools)

Failure model: if any server fails to come up (missing ``npx``, missing
``USDA_API_KEY``, handshake timeout, etc.), its bucket stays empty and
``get_tools(name)`` returns ``[]``. The experts then run RAG-only with no
visible regression — same shape as the existing ``[RAG Error] ...`` graceful
degradation in ``tools.py``.

Async/sync bridge: ``langchain-mcp-adapters`` tools are async-only. We run a
dedicated asyncio loop on a background daemon thread so the MCP subprocesses
stay alive for the lifetime of the process, and each synchronous tool call
dispatches its coroutine to that loop via ``run_coroutine_threadsafe``. This
plays nicely with the Dispatcher's ThreadPoolExecutor fan-out (each worker
thread submits to the shared loop instead of spinning up its own).
"""
from __future__ import annotations

import asyncio
import os
import threading
from typing import Optional

from langchain_core.tools import StructuredTool

from . import config
from .detail import print_mcp_call, print_mcp_error, print_mcp_result


# Only the read-only wger tools — the remaining 9 mutate user workouts and
# require wger account credentials. Surfacing them would let the LLM try
# write operations that will always fail. Out of scope for this project.
_WGER_PUBLIC_TOOLS = {
    "search_exercises",
    "get_exercise_details",
    "list_categories",
    "list_muscles",
    "list_equipment",
}


class _BackgroundLoop:
    """A dedicated asyncio loop running on a daemon thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="mcp-asyncio-loop", daemon=True
        )
        self._thread.start()

    def run(self, coro, timeout: Optional[float] = None):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    def stop(self) -> None:
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=3)
        except Exception:
            pass


# Some upstream APIs (notably USDA FoodData Central) intermittently return
# HTML 404 portal pages instead of JSON — about 1 in 3 requests during the
# observed window. Retry once on transient-looking failures.
_TRANSIENT_HINTS = ("404", "5", "ToolException", "AxiosError", "ECONNRESET", "ETIMEDOUT")


def _looks_transient(value) -> bool:
    s = str(value)
    return any(h in s for h in _TRANSIENT_HINTS)


def _sync_wrap(async_tool, loop: _BackgroundLoop) -> StructuredTool:
    name = async_tool.name
    description = (async_tool.description or "").strip() or name
    args_schema = getattr(async_tool, "args_schema", None)

    def _invoke_once():
        return loop.run(async_tool.ainvoke(kwargs_holder["v"]), timeout=60)

    kwargs_holder = {"v": None}

    def _run(**kwargs):
        kwargs_holder["v"] = kwargs
        print_mcp_call(name, kwargs)
        # Try up to 3 times. Empirically USDA's /foods/search returns 404+HTML
        # on ~1/3 of requests (their api-umbrella load balancer intercepts
        # the path nondeterministically). 3 attempts brings success rate from
        # 67% → ~96%. Permanent errors (auth / schema) don't match the
        # transient hints and bail on the first failure.
        last_err = None
        for attempt in (1, 2, 3):
            try:
                result = _invoke_once()
            except Exception as e:
                last_err = e
                print_mcp_error(name, e, attempt=attempt)
                if attempt < 3 and _looks_transient(e):
                    continue
                msg = str(e).replace("\n", " ").strip()[:300]
                return (
                    f"[MCP Error] {name} 调用失败：{type(e).__name__}: {msg}。"
                    "请使用其他工具或基于通用知识保守回答。"
                )
            if attempt < 3 and isinstance(result, str) and _looks_transient(result):
                print_mcp_result(name, result, attempt=attempt)
                continue
            print_mcp_result(name, result, attempt=attempt)
            return result
        return (
            f"[MCP Error] {name} 重试 3 次后仍失败：{type(last_err).__name__ if last_err else 'unknown'}。"
            "请使用其他工具或基于通用知识保守回答。"
        )

    kwargs = {
        "func": _run,
        "name": name,
        "description": description,
    }
    if args_schema is not None:
        kwargs["args_schema"] = args_schema
    return StructuredTool.from_function(**kwargs)


class MCPRegistry:
    def __init__(self) -> None:
        self._tools_by_server: dict[str, list] = {}
        self._loop: Optional[_BackgroundLoop] = None
        self._client = None
        self._started = False

    def _server_configs(self) -> dict:
        servers: dict[str, dict] = {}
        if config.MCP_TRAINER_ENABLED:
            if not config.WGER_API_KEY:
                print(
                    "[MCP] 'wger' skipped: WGER_API_KEY unset. "
                    "Register a free account at https://wger.de/en/user/registration "
                    "and put the API key into WGER_API_KEY. (The wger MCP server "
                    "refuses to start without it, even for read-only tools.)"
                )
            elif not config.MCP_WGER_SCRIPT_PATH:
                print(
                    "[MCP] 'wger' skipped: MCP_WGER_SCRIPT_PATH unset. "
                    "Run `bash scripts/setup_mcp_servers.sh` — it installs "
                    "wger-mcp to a fixed path and patches the `variations` "
                    "schema (which is out of sync with wger.de's current API)."
                )
            else:
                # wger-mcp's `info`-level logs go to console.info → stdout,
                # which corrupts the MCP JSON-RPC stream (pydantic blows up
                # parsing "[2026-...] INFO Registered tool: ..." as a JSON-RPC
                # message). Setting LOG_LEVEL=warn suppresses info/debug;
                # warn+error still fire but use stderr.
                wger_env = {
                    **os.environ,
                    "WGER_API_KEY": config.WGER_API_KEY,
                    "LOG_LEVEL": os.environ.get("WGER_LOG_LEVEL", "warn"),
                }
                servers["wger"] = {
                    "command": "node",
                    "args": [config.MCP_WGER_SCRIPT_PATH],
                    "env": wger_env,
                    "transport": "stdio",
                }
        if (
            config.MCP_NUTRITIONIST_ENABLED
            and config.USDA_API_KEY
            and config.MCP_USDA_SCRIPT_PATH
        ):
            # USDA's api.nal.usda.gov sometimes serves HTML through HTTP
            # proxies (observed via Clash / V2Ray on 127.0.0.1:7890 — the
            # proxy mangles /foods/search responses). Force the USDA
            # subprocess to bypass the proxy for the USDA host while still
            # honoring it for other domains (e.g. npm registry).
            usda_env = {**os.environ, "USDA_API_KEY": config.USDA_API_KEY}
            existing_no = usda_env.get("no_proxy") or usda_env.get("NO_PROXY") or ""
            usda_hosts = "api.nal.usda.gov,fdc.nal.usda.gov"
            merged = (
                f"{existing_no},{usda_hosts}" if existing_no else usda_hosts
            )
            usda_env["no_proxy"] = merged
            usda_env["NO_PROXY"] = merged
            servers["usda"] = {
                "command": "npx",
                "args": ["tsx", config.MCP_USDA_SCRIPT_PATH],
                "env": usda_env,
                "transport": "stdio",
            }
        if config.MCP_CRITIC_ENABLED:
            if not config.MCP_MEDICAL_SCRIPT_PATH:
                print(
                    "[MCP] 'medical' skipped: MCP_MEDICAL_SCRIPT_PATH unset. "
                    "Run `bash scripts/setup_mcp_servers.sh` to install medical-mcp "
                    "to a fixed location (its npm `bin` link is missing a shebang so "
                    "`npx medical-mcp` fails — we invoke node on the script directly)."
                )
            else:
                # medical-mcp's puppeteer-backed tools (search-google-scholar,
                # search-medical-databases) shell out to ~/.cache/puppeteer/
                # chrome/.../chrome which dynamically loads libnspr4 / libnss3
                # / etc. Those system libs are installed via conda into
                # $CONDA_PREFIX/lib (not /usr/lib), so we prepend the env's
                # lib dir to LD_LIBRARY_PATH for the medical subprocess only.
                medical_env = dict(os.environ)
                conda_prefix = os.environ.get("CONDA_PREFIX", "")
                if conda_prefix:
                    extra = os.path.join(conda_prefix, "lib")
                    existing = medical_env.get("LD_LIBRARY_PATH", "")
                    medical_env["LD_LIBRARY_PATH"] = (
                        f"{extra}:{existing}" if existing else extra
                    )
                servers["medical"] = {
                    "command": "node",
                    "args": [config.MCP_MEDICAL_SCRIPT_PATH],
                    "env": medical_env,
                    "transport": "stdio",
                }
        return servers

    def startup(self) -> None:
        if self._started:
            return
        self._started = True
        servers = self._server_configs()
        if not servers:
            return

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError as e:
            print(
                f"[MCP] langchain-mcp-adapters not installed ({e}); "
                "experts will run RAG-only."
            )
            return

        try:
            self._loop = _BackgroundLoop()
            self._client = MultiServerMCPClient(servers)
        except Exception as e:
            print(f"[MCP] init failed: {type(e).__name__}: {e}")
            return

        timeout = float(config.MCP_STARTUP_TIMEOUT_SEC)
        for name in servers.keys():
            try:
                tools = self._fetch_tools(name, timeout=timeout)
            except Exception as e:
                print(f"[MCP] '{name}' startup failed: {type(e).__name__}: {e}")
                continue

            wrapped = []
            for t in tools:
                if name == "wger" and t.name not in _WGER_PUBLIC_TOOLS:
                    continue
                try:
                    wrapped.append(_sync_wrap(t, self._loop))
                except Exception as e:
                    print(
                        f"[MCP] '{name}' wrap tool '{getattr(t, 'name', '?')}' "
                        f"failed: {type(e).__name__}: {e}"
                    )
            if wrapped:
                self._tools_by_server[name] = wrapped
                print(f"[MCP] '{name}' ready: {len(wrapped)} tools")
            else:
                print(f"[MCP] '{name}' produced 0 usable tools")

    def _fetch_tools(self, server_name: str, timeout: float):
        """Fetch tools for one server. Tries server_name kwarg first, falls
        back to filtering the flat list if the installed adapter version
        doesn't accept it."""
        try:
            return self._loop.run(
                self._client.get_tools(server_name=server_name), timeout=timeout
            )
        except TypeError:
            all_tools = self._loop.run(self._client.get_tools(), timeout=timeout)
            # Without per-server tagging we cannot reliably bucket — return the
            # whole list and let the caller's name-based filter (e.g. wger
            # public set) reject mismatches. For usda/medical, the names are
            # distinct enough that this still works in practice.
            return all_tools

    def get_tools(self, server: str) -> list:
        return self._tools_by_server.get(server, [])

    def available_servers(self) -> list[str]:
        return sorted(self._tools_by_server.keys())

    def shutdown(self) -> None:
        if self._loop is not None:
            self._loop.stop()
            self._loop = None
        self._client = None
        self._tools_by_server.clear()
        self._started = False


MCP_REGISTRY = MCPRegistry()
