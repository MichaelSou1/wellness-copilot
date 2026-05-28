"""Shared Agent runtime for CLI/API/job worker entrypoints."""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from langchain_core.messages import HumanMessage

from . import config
from .backend_metrics import AGENT_NODE_LATENCY, AGENT_TURN_LATENCY, RAG_HITS, TOOL_CALLS
from .backend_telemetry import json_log, new_trace_id, span
from .llm import extract_text_content
from .observability import ObservabilityTracker, TurnRecord
from .state import RESET_SENTINEL


@dataclass
class AgentStreamEvent:
    event: str
    trace_id: str
    node: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTurnResult:
    trace_id: str
    thread_id: str
    user_id: str
    answer: str
    route: str
    tools_used: list[str]
    retrieval_hits: int
    citations_count: int
    latency_ms: float
    actuation_count: int
    vision_calls: int
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fake_mode() -> bool:
    raw = os.environ.get("FAKE_AGENT_MODE")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes"}
    return bool(config.FAKE_AGENT_MODE)


def _fake_delay_sec() -> float:
    raw = os.environ.get("FAKE_AGENT_DELAY_MS")
    try:
        return int(raw or config.FAKE_AGENT_DELAY_MS) / 1000
    except Exception:
        return 0.15


def _safe_int(value: Any) -> int:
    if isinstance(value, tuple):
        value = value[-1] if value else 0
    try:
        return int(value or 0)
    except Exception:
        return 0


def _real_actuation_events(value: Any) -> list[dict]:
    return [item for item in (value or []) if isinstance(item, dict)]


def _has_real_dict(value: Any) -> bool:
    return isinstance(value, dict) and any(key != RESET_SENTINEL for key in value.keys())


def _event_summary(node: str, value: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"node": node}
    if value.get("messages"):
        summary["answer_preview"] = extract_text_content(value["messages"][-1])[:240]
    for key in (
        "input_accumulator_status",
        "input_accumulator_reason",
        "orchestrator_decision",
        "critic_verdict",
        "replan_context",
    ):
        if value.get(key):
            summary[key] = value[key]
    if value.get("executed"):
        summary["executed"] = value["executed"]
    if value.get("last_tools"):
        summary["tools"] = [t for t in value["last_tools"] if t != "__RESET__"]
    if value.get("retrieval_hits") is not None:
        summary["retrieval_hits"] = _safe_int(value.get("retrieval_hits"))
    if value.get("draft_answer"):
        summary["draft_preview"] = str(value["draft_answer"])[:240]
    actuation = _real_actuation_events(value.get("actuation_log"))
    if actuation:
        summary["actuation_count"] = len(actuation)
    if _has_real_dict(value.get("vision_extractions")):
        summary["vision_calls"] = 1
    return summary


def _coerce_node_name(raw: str) -> str:
    text = (raw or "").strip()
    return text or "unknown"


def _collect_result_from_events(
    *,
    trace_id: str,
    thread_id: str,
    user_id: str,
    user_query: str,
    source: str,
    start_ts: float,
    events: list[dict[str, Any]],
    final_answer: str,
    routes: list[str],
    tools_used: list[str],
    retrieval_hits: int,
    actuation_events: list[dict],
    vision_calls: int,
) -> AgentTurnResult:
    latency_ms = (time.perf_counter() - start_ts) * 1000
    route = ",".join(r for r in routes if r != "FINISH") or "FINISH"
    citations_count = final_answer.count("[source:")
    result = AgentTurnResult(
        trace_id=trace_id,
        thread_id=thread_id,
        user_id=user_id,
        answer=final_answer,
        route=route,
        tools_used=tools_used,
        retrieval_hits=retrieval_hits,
        citations_count=citations_count,
        latency_ms=latency_ms,
        actuation_count=len(actuation_events),
        vision_calls=vision_calls,
        events=events,
    )
    AGENT_TURN_LATENCY.labels(source or "api").observe(latency_ms / 1000)
    if retrieval_hits:
        RAG_HITS.inc(retrieval_hits)
    for tool_name in tools_used:
        TOOL_CALLS.labels(str(tool_name)).inc()
    try:
        ObservabilityTracker().log_turn(
            TurnRecord(
                thread_id=thread_id,
                turn_index=int(time.time()),
                route=route,
                user_query=user_query,
                final_answer=final_answer,
                tools_used=tools_used,
                retrieval_hits=retrieval_hits,
                citations_count=citations_count,
                latency_ms=latency_ms,
                actuation_count=len(actuation_events),
                vision_calls=vision_calls,
                wechat_msgs_in=1 if source == "wechat" else 0,
                wechat_msgs_out=0,
            )
        )
    except Exception as exc:
        json_log("observability_write_failed", trace_id=trace_id, error=type(exc).__name__)
    return result


def stream_agent_turn(
    *,
    user_id: str,
    thread_id: str,
    message: str,
    content: Any = None,
    source: str = "api",
    trace_id: str = "",
    wechat_context: dict | None = None,
) -> Iterator[AgentStreamEvent]:
    trace_id = trace_id or new_trace_id()
    content = content if content is not None else message
    start_ts = time.perf_counter()
    events: list[dict[str, Any]] = []
    routes: list[str] = []
    tools_used: list[str] = []
    retrieval_hits = 0
    actuation_events: list[dict] = []
    vision_calls = 0
    final_answer = ""

    if _fake_mode():
        yield AgentStreamEvent("node_start", trace_id, "FakeAgent", {"source": source})
        time.sleep(_fake_delay_sec())
        final_answer = f"[FAKE_AGENT] 已收到：{message or '[non-text input]'}"
        payload = {"answer_preview": final_answer, "route": "FAKE"}
        events.append({"event": "node_output", "node": "FakeAgent", "data": payload})
        yield AgentStreamEvent("node_output", trace_id, "FakeAgent", payload)
        result = _collect_result_from_events(
            trace_id=trace_id,
            thread_id=thread_id,
            user_id=user_id,
            user_query=message,
            source=source,
            start_ts=start_ts,
            events=events,
            final_answer=final_answer,
            routes=["FAKE"],
            tools_used=[],
            retrieval_hits=0,
            actuation_events=[],
            vision_calls=0,
        )
        yield AgentStreamEvent("final", trace_id, "FakeAgent", result.to_dict())
        return

    from .graph import graph

    graph_input = {
        "messages": [HumanMessage(content=content)],
        "profile_user_id": user_id,
    }
    if wechat_context:
        graph_input["wechat_context"] = wechat_context
    graph_config = {"configurable": {"thread_id": thread_id}}

    json_log("agent_turn_start", trace_id=trace_id, user_id=user_id, thread_id=thread_id, source=source)
    try:
        with span("agent.turn", trace_id=trace_id, user_id=user_id, thread_id=thread_id, source=source):
            previous_emit = time.perf_counter()
            for event in graph.stream(graph_input, graph_config):
                for node, value in event.items():
                    now = time.perf_counter()
                    node = _coerce_node_name(node)
                    node_latency = max(0.0, now - previous_emit)
                    previous_emit = now
                    AGENT_NODE_LATENCY.labels(node, source or "api").observe(node_latency)
                    json_log(
                        "agent_node_done",
                        trace_id=trace_id,
                        node=node,
                        latency_ms=round(node_latency * 1000, 2),
                    )
                    value = value or {}
                    yield AgentStreamEvent("node_start", trace_id, node, {"node": node})
                    summary = _event_summary(node, value)
                    events.append({"event": "node_output", "node": node, "data": summary})
                    if value.get("messages"):
                        final_answer = extract_text_content(value["messages"][-1])
                    if value.get("last_tools"):
                        tools_used.extend(t for t in value["last_tools"] if t != "__RESET__")
                    if value.get("retrieval_hits") is not None:
                        retrieval_hits += _safe_int(value.get("retrieval_hits"))
                    if value.get("executed"):
                        routes = value["executed"]
                    actuation = _real_actuation_events(value.get("actuation_log"))
                    if actuation:
                        actuation_events.extend(actuation)
                    if _has_real_dict(value.get("vision_extractions")):
                        vision_calls += 1
                    yield AgentStreamEvent("node_output", trace_id, node, summary)
    except Exception as exc:
        json_log("agent_turn_error", trace_id=trace_id, error=type(exc).__name__, detail=str(exc)[:300])
        raise

    if not final_answer:
        final_answer = "抱歉，我这轮没有生成有效回复，请再发一次。"
    result = _collect_result_from_events(
        trace_id=trace_id,
        thread_id=thread_id,
        user_id=user_id,
        user_query=message,
        source=source,
        start_ts=start_ts,
        events=events,
        final_answer=final_answer,
        routes=routes,
        tools_used=tools_used,
        retrieval_hits=retrieval_hits,
        actuation_events=actuation_events,
        vision_calls=vision_calls,
    )
    json_log("agent_turn_done", trace_id=trace_id, latency_ms=round(result.latency_ms, 2), route=result.route)
    yield AgentStreamEvent("final", trace_id, "", result.to_dict())


def run_agent_turn(**kwargs) -> AgentTurnResult:
    final: AgentTurnResult | None = None
    for item in stream_agent_turn(**kwargs):
        if item.event == "final":
            final = AgentTurnResult(**{k: v for k, v in item.data.items() if k in AgentTurnResult.__dataclass_fields__})
    if final is None:
        raise RuntimeError("Agent turn finished without final event")
    return final
