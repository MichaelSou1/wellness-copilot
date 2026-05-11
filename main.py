import uuid
import os
import time
from langchain_core.messages import HumanMessage
from health_guide.graph import graph
from health_guide.llm import extract_text_content
from health_guide.observability import ObservabilityTracker, TurnRecord

def main():
    # 配置 Checkpoint 的 thread_id
    # 使用 UUID 生成唯一的会话 ID，这样每次运行都是新的会话，
    # 或者固定一个 ID 以测试持久化记忆 (由于目前是 :memory:，重启后记忆会丢失)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    tracker = ObservabilityTracker()

    user_id = input("User ID (默认 default_user): ").strip().lstrip("﻿") or "default_user"
    os.environ["HEALTH_GUIDE_USER_ID"] = user_id

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
            print("Goodbye!")
            break

        turn_index += 1
        start_ts = time.perf_counter()
        routes = []
        tools_used = []
        final_answer = ""
        retrieval_hits = 0

        for event in graph.stream(
            {
                "messages": [HumanMessage(content=user_input)],
                "profile_user_id": user_id,
            },
            config,
        ):
            for key, value in event.items():
                print(f"\n[当前节点]: {key}")
                if value is None:
                    value = {}

                if "messages" in value:
                    last_msg = value["messages"][-1]
                    text = extract_text_content(last_msg)
                    print(f"[回复内容]: {text}")
                    final_answer = text

                if "expert_responses" in value and value["expert_responses"]:
                    for expert, resp in value["expert_responses"].items():
                        print(f"[{expert} 回答]: {resp[:120]}{'...' if len(resp) > 120 else ''}")

                if "agent_notes" in value and value["agent_notes"]:
                    for expert, note in value["agent_notes"].items():
                        print(f"[scratchpad/{expert}]: {note[:100]}{'...' if len(note) > 100 else ''}")

                if "draft_answer" in value and value["draft_answer"]:
                    draft = value["draft_answer"]
                    print(f"[Aggregator 草稿]: {draft[:160]}{'...' if len(draft) > 160 else ''}")

                if "critic_verdict" in value and value["critic_verdict"]:
                    print(f"[Critic 审核]: {value['critic_verdict']}")

                if "last_tools" in value and value["last_tools"]:
                    tools_used.extend(value["last_tools"])
                    for tool_name in value["last_tools"]:
                        print(f"[调用工具]: {tool_name}")

                if "retrieval_hits" in value:
                    retrieval_hits += int(value.get("retrieval_hits", 0))

                if "plan" in value and value["plan"] is not None:
                    plan_list = value["plan"]
                    if plan_list:
                        print(f"[Plan 待执行]: {' -> '.join(plan_list)}")

                if "executed" in value and value["executed"]:
                    routes = value["executed"]
                    print(f"[Plan 已执行]: {', '.join(routes)}")

                if "next" in value:
                    next_val = value["next"]
                    if isinstance(next_val, list):
                        if next_val:
                            print(f"[Dispatch -> {next_val[0]}]")
                    else:
                        print(f"[Dispatch -> {next_val}]")

        route = ",".join(r for r in routes if r != "FINISH") or "FINISH"

        rag_tools = sorted({t for t in tools_used if "retrieve" in t and "knowledge" in t})
        if rag_tools:
            print(f"[RAG调用]: YES (hits={retrieval_hits}, tools={', '.join(rag_tools)})")
        else:
            print("[RAG调用]: NO (本轮回复未调用本地检索工具，可能为纯 LLM 输出)")

        latency_ms = (time.perf_counter() - start_ts) * 1000
        citations_count = final_answer.count("[source:")
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

if __name__ == "__main__":
    main()
