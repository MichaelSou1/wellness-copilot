import argparse
from contextlib import contextmanager
import json
import os
import sys
import time
import uuid
from pathlib import Path
from langchain_core.messages import HumanMessage
from health_guide.config import (
    MCP_CRITIC_ENABLED,
    MCP_NUTRITIONIST_ENABLED,
    MCP_TRAINER_ENABLED,
)
from health_guide.detail import set_detail
from health_guide.graph import graph
from health_guide.llm import extract_text_content
from health_guide.mcp_client import MCP_REGISTRY
from health_guide.observability import ObservabilityTracker, TurnRecord

SESSION_STORE_PATH = Path(os.environ.get("SESSION_STORE_PATH", "session_store.json"))


@contextmanager
def _suppress_process_output(enabled: bool):
    """Silence Python and child-process stdout/stderr while preserving stdin."""
    if not enabled:
        yield
        return

    try:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
    except Exception:
        yield
        return

    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
        finally:
            os.close(stdout_fd)
            os.close(stderr_fd)


def _load_session_store() -> dict:
    try:
        return json.loads(SESSION_STORE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_session_store(store: dict) -> None:
    try:
        SESSION_STORE_PATH.write_text(
            json.dumps(store, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def _resolve_thread_id(user_id: str, interactive: bool = True) -> str:
    """Resume the user's last thread or start a new one.

    The checkpointer (SqliteSaver -> checkpoints.db) persists the full
    conversation across runs, but only if we hand it the same thread_id.
    `session_store.json` maps user_id -> last thread_id so the user can
    pick "continue" instead of pasting a UUID.
    """
    store = _load_session_store()
    last = store.get(user_id)
    if last and interactive:
        choice = input(
            f"检测到上次会话 thread_id={last[:8]}…，输入 [Enter]=继续 / n=新建 / 粘贴一个 thread_id 切换： "
        ).strip()
        if not choice:
            return last
        if choice.lower() in ("n", "new"):
            return str(uuid.uuid4())
        return choice
    if last:
        return last
    return str(uuid.uuid4())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="health-guide",
        description="Health Guide multi-agent CLI",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="详细输出 MCP 工具调用与各子 agent（专家）ReAct 步骤，用于行为 traceroute。",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    set_detail(args.detail)
    detail = args.detail
    if args.detail:
        print("[detail] traceroute 模式开启：将打印 MCP 工具调用与专家 ReAct 步骤")

    tracker = ObservabilityTracker()

    if any([MCP_TRAINER_ENABLED, MCP_NUTRITIONIST_ENABLED, MCP_CRITIC_ENABLED]):
        try:
            with _suppress_process_output(not detail):
                MCP_REGISTRY.startup()
            available = MCP_REGISTRY.available_servers()
            if detail and available:
                print(f"[MCP] available: {available}")
            elif detail:
                print("[MCP] no servers came up, experts will run RAG-only")
        except Exception as e:
            if detail:
                print(
                    "[MCP] startup failed, falling back to RAG-only: "
                    f"{type(e).__name__}: {e}"
                )

    user_id = input("User ID (默认 default_user): ").strip().lstrip("﻿") or "default_user"
    os.environ["HEALTH_GUIDE_USER_ID"] = user_id

    thread_id = _resolve_thread_id(user_id)
    config = {"configurable": {"thread_id": thread_id}}
    if detail:
        print(f"[session] user={user_id} thread={thread_id}")

    if detail:
        print("=== 开始运行健康管理 Agent 团队 ===")
        print("输入 'q' 或 'exit' 退出对话。\n")

    turn_index = 0

    while True:
        try:
            user_input = input("User: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() in ["q", "quit", "exit"]:
            if detail:
                print("Goodbye!")
            break

        turn_index += 1
        start_ts = time.perf_counter()
        routes = []
        tools_used = []
        final_answer = ""
        retrieval_hits = 0

        try:
            with _suppress_process_output(not detail):
                stream_iter = graph.stream(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "profile_user_id": user_id,
                    },
                    config,
                )
                for event in stream_iter:
                    for key, value in event.items():
                        if detail:
                            print(f"\n[当前节点]: {key}")
                        if value is None:
                            value = {}

                        if "messages" in value:
                            last_msg = value["messages"][-1]
                            text = extract_text_content(last_msg)
                            if detail:
                                print(f"[回复内容]: {text}")
                            final_answer = text

                        if "expert_responses" in value and value["expert_responses"]:
                            for expert, resp in value["expert_responses"].items():
                                if not isinstance(resp, str):
                                    continue
                                if detail:
                                    print(
                                        f"[{expert} 回答]: "
                                        f"{resp[:120]}{'...' if len(resp) > 120 else ''}"
                                    )

                        if "agent_notes" in value and value["agent_notes"]:
                            for expert, note in value["agent_notes"].items():
                                if not isinstance(note, str):
                                    continue
                                if detail:
                                    print(
                                        f"[scratchpad/{expert}]: "
                                        f"{note[:100]}{'...' if len(note) > 100 else ''}"
                                    )

                        if "draft_answer" in value and value["draft_answer"]:
                            draft = value["draft_answer"]
                            if detail:
                                print(
                                    f"[Aggregator 草稿]: "
                                    f"{draft[:160]}{'...' if len(draft) > 160 else ''}"
                                )

                        if "critic_verdict" in value and value["critic_verdict"]:
                            if detail:
                                print(f"[Critic 审核]: {value['critic_verdict']}")

                        if "last_tools" in value and value["last_tools"]:
                            real_tools = [t for t in value["last_tools"] if t != "__RESET__"]
                            tools_used.extend(real_tools)
                            if detail:
                                for tool_name in real_tools:
                                    print(f"[调用工具]: {tool_name}")

                        if "retrieval_hits" in value:
                            rh = value.get("retrieval_hits", 0)
                            if isinstance(rh, (int, float, str)):
                                retrieval_hits += int(rh)

                        if "plan" in value and value["plan"] is not None:
                            plan_list = value["plan"]
                            if detail and plan_list:
                                print(f"[Plan 待执行]: {' -> '.join(plan_list)}")

                        if "executed" in value and value["executed"]:
                            routes = value["executed"]
                            if detail:
                                print(f"[Plan 已执行]: {', '.join(routes)}")

                        if "history_summary" in value and value["history_summary"]:
                            if detail:
                                print("[TurnStart] 已压缩较早历史为摘要（保留最近 8 条原文）")

                        if "next" in value and detail:
                            next_val = value["next"]
                            if isinstance(next_val, list):
                                if next_val:
                                    print(f"[Dispatch -> {next_val[0]}]")
                            else:
                                print(f"[Dispatch -> {next_val}]")
        except Exception as e:
            if detail:
                print(
                    "[ERROR] 本轮 Agent 链路执行失败，已保留会话进程："
                    f"{type(e).__name__}: {e}"
                )
            final_answer = final_answer or "抱歉，本轮处理时发生系统错误，请稍后重试。"

        if not detail:
            print(f"Health-Guide-Agent: {final_answer}\n")

        # Persist thread for resume on next launch (after at least one turn).
        store = _load_session_store()
        store[user_id] = thread_id
        _save_session_store(store)

        route = ",".join(r for r in routes if r != "FINISH") or "FINISH"

        rag_tools = sorted({t for t in tools_used if "retrieve" in t and "knowledge" in t})
        if detail and rag_tools:
            print(f"[RAG调用]: YES (hits={retrieval_hits}, tools={', '.join(rag_tools)})")
        elif detail:
            print("[RAG调用]: NO (本轮回复未调用本地检索工具，可能为纯 LLM 输出)")

        latency_ms = (time.perf_counter() - start_ts) * 1000
        citations_count = final_answer.count("[source:")
        try:
            tracker.log_turn(
                TurnRecord(
                    thread_id=thread_id,
                    turn_index=turn_index,
                    route=route,
                    user_query=user_input,
                    final_answer=final_answer,
                    tools_used=tools_used,
                    retrieval_hits=retrieval_hits,
                    citations_count=citations_count,
                    latency_ms=latency_ms,
                )
            )
        except Exception as e:
            if detail:
                print(f"[WARN] 写入可观测指标失败：{type(e).__name__}: {e}")

    if detail:
        try:
            summary = tracker.get_thread_summary(thread_id)
            report_path = tracker.export_thread_report(thread_id)
            print("\n=== 会话评估摘要 ===")
            print(f"- turn_count: {summary['turn_count']}")
            print(f"- avg_latency_ms: {summary['avg_latency_ms']}")
            print(f"- retrieval_hit_rate: {summary['retrieval_hit_rate']}")
            print(f"- citation_rate: {summary['citation_rate']}")
            print(f"- routes: {summary['routes']}")
            print(f"- tool_counts: {summary['tool_counts']}")
            print(f"- report: {report_path}")
        except Exception as e:
            print(f"[WARN] 导出会话评估摘要失败：{type(e).__name__}: {e}")

    try:
        with _suppress_process_output(not detail):
            MCP_REGISTRY.shutdown()
    except Exception:
        pass

if __name__ == "__main__":
    main()
