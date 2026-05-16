"""Smoke tests for long-chain error fallbacks.

Run: python scripts/smoke_error_fallbacks.py
"""
import os
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402


class RaisingLLM:
    def invoke(self, messages):
        raise RuntimeError("forced failure")


class PassLLM:
    def invoke(self, messages):
        return AIMessage(content="VERDICT: PASS")


def test_planner_fresh_fallback():
    import health_guide.agents.planner as planner

    old_llm = planner.llm
    planner.llm = RaisingLLM()
    try:
        out = planner.planner_node({"messages": [HumanMessage(content="你好")]})
    finally:
        planner.llm = old_llm

    assert out.get("plan") == ["General"], out
    assert out.get("next") == [], out
    print("  ✓ Planner fresh failure falls back to General")


def test_rag_tool_error_fallback():
    import health_guide.tools as tools

    class BadKB:
        def retrieve(self, query, top_k):
            raise RuntimeError("forced rag failure")

    old_get = tools._get_agent_kb
    tools._get_agent_kb = lambda agent: BadKB()
    try:
        out = tools._retrieve_by_agent("膝盖疼怎么练", 4, "trainer")
    finally:
        tools._get_agent_kb = old_get

    assert out.startswith("[RAG Error]"), out
    assert "RuntimeError" in out, out
    print("  ✓ RAG retrieval failure returns [RAG Error] text")


def test_expert_failure_graph_continues():
    """Trainer raises → Dispatcher catches via expert_error_update,
    Aggregator + Critic still run on the fallback answer."""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph

    import health_guide.agents.critic as critic
    import health_guide.agents.dispatcher as dispatcher_mod
    from health_guide.agents.aggregator import aggregator_node
    from health_guide.agents.dispatcher import dispatcher_node
    from health_guide.graph import _route_after_dispatch, _route_after_judge
    from health_guide.agents.replan_judge import replan_judge_node
    from health_guide.state import AgentState

    def boom(*args, **kwargs):
        raise RuntimeError("forced expert failure")

    old_trainer_runner = dispatcher_mod.EXPERT_RUNNERS["Trainer"]
    old_critic_llm = critic.llm
    dispatcher_mod.EXPERT_RUNNERS["Trainer"] = boom
    critic.llm = PassLLM()
    try:
        workflow = StateGraph(AgentState)
        workflow.add_node("Dispatcher", dispatcher_node)
        workflow.add_node("ReplanJudge", replan_judge_node)
        workflow.add_node("Aggregator", aggregator_node)
        workflow.add_node("Critic", critic.critic_node)
        workflow.set_entry_point("Dispatcher")
        workflow.add_conditional_edges(
            "Dispatcher", _route_after_dispatch,
            ["ReplanJudge", END],
        )
        workflow.add_conditional_edges(
            "ReplanJudge", _route_after_judge,
            ["Dispatcher", "Aggregator"],
        )
        workflow.add_edge("Aggregator", "Critic")
        workflow.add_edge("Critic", END)
        graph = workflow.compile(checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        final_answer = ""
        saw_aggregator = False
        saw_critic = False
        saw_error_tool = False
        for event in graph.stream(
            {
                "messages": [HumanMessage(content="膝盖疼还能练腿吗？")],
                "profile_user_id": "smoke_error_user",
                "plan": ["Trainer"],
                # Bypass ReplanJudge for this fault-injection test (it would
                # otherwise issue an extra LLM call we don't want here).
                "replan_count": 2,
            },
            config,
        ):
            for node, value in event.items():
                value = value or {}
                if node == "Aggregator" and value.get("draft_answer"):
                    saw_aggregator = True
                if node == "Critic":
                    saw_critic = True
                    if value.get("messages"):
                        final_answer = value["messages"][-1].content
                if value.get("last_tools"):
                    saw_error_tool = any(
                        str(t).startswith("ERROR:Trainer:") for t in value["last_tools"]
                    )
    finally:
        dispatcher_mod.EXPERT_RUNNERS["Trainer"] = old_trainer_runner
        critic.llm = old_critic_llm

    assert saw_error_tool, "Trainer error marker missing"
    assert saw_aggregator, "Aggregator did not run after expert failure"
    assert saw_critic, "Critic did not run after expert failure"
    assert final_answer, "Final answer is empty"
    print("  ✓ Expert failure degrades and graph continues to Aggregator/Critic")


def test_aggregator_llm_fallback():
    import health_guide.agents.aggregator as aggregator

    old_llm = aggregator.llm
    aggregator.llm = RaisingLLM()
    try:
        out = aggregator.aggregator_node(
            {
                "executed": ["Trainer", "Nutritionist"],
                "expert_responses": {
                    "Trainer": "先暂停深蹲，改低冲击训练。",
                    "Nutritionist": "晚餐补充蛋白质和主食。",
                },
                "messages": [HumanMessage(content="练后怎么恢复？")],
            }
        )
    finally:
        aggregator.llm = old_llm

    draft = out.get("draft_answer", "")
    assert "训练教练建议" in draft, draft
    assert "营养师建议" in draft, draft
    print("  ✓ Aggregator LLM failure preserves completed expert answers")


def test_critic_error_guard_adds_warning():
    import health_guide.agents.critic as critic

    old_llm = critic.llm
    critic.llm = RaisingLLM()
    draft = "可以先休息一下，观察身体反应。"
    try:
        out = critic.critic_node(
            {
                "draft_answer": draft,
                "messages": [HumanMessage(content="我运动后胸闷而且心率异常怎么办？")],
                "executed": ["Wellness"],
                "profile_user_id": "smoke_error_user",
            }
        )
    finally:
        critic.llm = old_llm

    final = out["messages"][-1].content
    assert out.get("critic_verdict", "").startswith("ERROR_GUARDED:"), out
    assert final.startswith("安全提示："), final
    assert draft in final, final
    print("  ✓ Critic failure adds safety warning and preserves draft")


def main():
    print("\n[unit] Planner fallback")
    test_planner_fresh_fallback()
    print("\n[unit] RAG tool fallback")
    test_rag_tool_error_fallback()
    print("\n[integration] Expert failure graph continuation")
    test_expert_failure_graph_continues()
    print("\n[unit] Aggregator fallback")
    test_aggregator_llm_fallback()
    print("\n[unit] Critic guarded fallback")
    test_critic_error_guard_adds_warning()
    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()
