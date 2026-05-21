"""Smoke test for dynamic replan compatibility and current Critic replan path.

Layers:

1. **Unit**: ReplanJudge._parse_verdict accepts CONTINUE / REPLAN forms.
2. **Unit**: dispatcher_node honors `replan_request` and clears it.
3. **Unit**: dispatcher_node respects REPLAN_CAP.
4. **Integration**: build a graph matching the current topology where Critic
   sets ``replan_context`` once, routes back to Orchestrator, then finishes.
"""
import os
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from langchain_core.messages import HumanMessage  # noqa: E402

from wellness_copilot.agents.dispatcher import dispatcher_node  # noqa: E402
from wellness_copilot.agents.replan_judge import _parse_verdict  # noqa: E402


class PassLLM:
    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        return AIMessage(content="VERDICT: PASS")


def test_judge_parse():
    assert _parse_verdict("VERDICT: CONTINUE") == ""
    assert _parse_verdict("verdict: continue\n") == ""
    r = _parse_verdict("VERDICT: REPLAN\nREASON: 用户睡眠问题需要 Psychologist 补充")
    assert "睡眠" in r and "Psychologist" in r, r
    # REPLAN without REASON falls back to a generic synthesized reason
    r2 = _parse_verdict("VERDICT: REPLAN")
    assert r2, "REPLAN without REASON should still produce a non-empty reason"
    # Garbage → no replan
    assert _parse_verdict("hello world") == ""
    assert _parse_verdict("") == ""
    print("  ✓ ReplanJudge._parse_verdict handles CONTINUE / REPLAN / garbage")


def test_dispatcher_replan_trigger():
    state = {
        "replan_request": "需要 Psychologist 补充睡眠建议",
        "plan": ["Nutritionist"],
        "executed": ["Trainer"],
        "replan_count": 0,
    }
    out = dispatcher_node(state)
    assert out.get("next") == ["__REPLAN__"], f"expected __REPLAN__, got {out.get('next')}"
    assert out.get("replan_request") == ""
    assert out.get("replan_context") == "需要 Psychologist 补充睡眠建议"
    assert out.get("replan_count") == 1
    print("  ✓ dispatcher_node routes to __REPLAN__ and clears the request")


def test_dispatcher_replan_cap():
    import wellness_copilot.agents.dispatcher as dispatcher_mod

    def psychologist_stub(*args, **kwargs):
        return {
            "expert_responses": {"Psychologist": "cap reached, executing remaining plan"},
            "agent_notes": {"Psychologist": "cap reached"},
            "last_tools": [],
            "retrieval_hits": 0,
        }

    state = {
        "replan_request": "再叫一个",
        "plan": ["Psychologist"],
        "executed": ["Trainer", "Nutritionist"],
        "replan_count": 2,
    }
    old_psychologist = dispatcher_mod.EXPERT_RUNNERS["Psychologist"]
    dispatcher_mod.EXPERT_RUNNERS["Psychologist"] = psychologist_stub
    try:
        out = dispatcher_node(state)
    finally:
        dispatcher_mod.EXPERT_RUNNERS["Psychologist"] = old_psychologist
    assert out.get("next") == []
    assert out.get("replan_request") == ""
    assert "replan_context" not in out
    assert out.get("executed") == ["Trainer", "Nutritionist", "Psychologist"]
    print("  ✓ dispatcher_node respects REPLAN_CAP and falls through")


def test_integration_replan_loop():
    """End-to-end with current Critic -> Orchestrator replan path."""
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.messages import AIMessage

    from wellness_copilot.state import AgentState
    from wellness_copilot.graph import _route_after_critic, _route_after_orchestrator

    def orchestrator_stub(state):
        if state.get("replan_context"):
            executed = [role for role in (state.get("executed") or []) if role != "Orchestrator"]
            return {
                "expert_responses": {"Psychologist": "睡前 30 分钟降低刺激，做 5 分钟呼吸放松。"},
                "agent_notes": {"Psychologist": "睡眠：睡前 30 分钟降刺激 + 5 分钟呼吸"},
                "last_tools": [],
                "retrieval_hits": 0,
                "executed": executed + ["Psychologist"],
                "plan": [],
                "replan_context": "",
                "next": [],
                "orchestrator_decision": "CALLED_CHILD",
            }
        return {
            "expert_responses": {"Trainer": "今晚做 20 分钟轻度有氧。"},
            "agent_notes": {"Trainer": "训练：20 分钟轻度有氧"},
            "last_tools": [],
            "retrieval_hits": 0,
            "executed": ["Trainer"],
            "plan": [],
            "replan_count": 0,
            "replan_context": "",
            "next": [],
            "orchestrator_decision": "CALLED_CHILD",
        }

    def aggregator_stub(state):
        executed = [role for role in (state.get("executed") or []) if role != "Orchestrator"]
        responses = state.get("expert_responses") or {}
        draft = "\n".join(responses.get(role, "") for role in executed if responses.get(role))
        return {"draft_answer": draft}

    fired = {"v": False}

    def critic_stub(state):
        executed = state.get("executed") or []
        if "Trainer" in executed and "Psychologist" not in executed and not fired["v"]:
            fired["v"] = True
            return {
                "critic_verdict": "REPLAN",
                "replan_context": "安全审核员要求补叫 Psychologist：用户睡眠问题没解决",
                "replan_count": int(state.get("replan_count") or 0) + 1,
                "draft_answer": "",
            }
        return {
            "messages": [AIMessage(content=state.get("draft_answer") or "done")],
            "critic_verdict": "PASS",
        }

    workflow = StateGraph(AgentState)
    workflow.add_node("Orchestrator", orchestrator_stub)
    workflow.add_node("Aggregator", aggregator_stub)
    workflow.add_node("Critic", critic_stub)

    workflow.set_entry_point("Orchestrator")
    workflow.add_conditional_edges(
        "Orchestrator",
        _route_after_orchestrator,
        ["Aggregator", "Critic", END],
    )
    workflow.add_edge("Aggregator", "Critic")
    workflow.add_conditional_edges(
        "Critic",
        _route_after_critic,
        ["Orchestrator", END],
    )

    test_graph = workflow.compile(checkpointer=MemorySaver())

    thread_id = str(uuid.uuid4())
    user_id = "smoke_judge_user"
    os.environ["WELLNESS_COPILOT_USER_ID"] = user_id
    query = "我深蹲后膝盖酸，今晚训练应该怎么调？"
    print(f"\n  USER: {query}")

    config = {"configurable": {"thread_id": thread_id}}
    orchestrator_visits = 0
    saw_replan_route = False
    executed_final = []
    final_answer = ""
    critic_verdict = ""
    for event in test_graph.stream(
        {"messages": [HumanMessage(content=query)], "profile_user_id": user_id},
        config,
    ):
        for node, value in event.items():
            if value is None:
                value = {}
            if node == "Orchestrator":
                orchestrator_visits += 1
                if value.get("executed"):
                    executed_final = value["executed"]
                print(f"  [Orchestrator #{orchestrator_visits}] executed={value.get('executed', [])}")
            elif node == "Critic":
                critic_verdict = value.get("critic_verdict", "")
                if value.get("replan_context"):
                    saw_replan_route = True
                    print(f"  [Critic] requests replan: {value['replan_context']}")
                if value.get("messages"):
                    last_msg = value["messages"][-1]
                    final_answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                print(f"  [Critic] verdict={critic_verdict}")

    assert fired["v"], "Judge stub never fired (state plumbing broken?)"
    assert orchestrator_visits >= 2, f"expected Orchestrator to run >=2x, got {orchestrator_visits}"
    assert saw_replan_route, "Critic never routed back to Orchestrator"
    assert "Trainer" in executed_final, f"Trainer must run, got {executed_final}"
    assert "Psychologist" in executed_final, f"Psychologist must join after replan, got {executed_final}"
    assert len(executed_final) >= 2, (
        f"expected ≥2 experts after replan, got {executed_final}"
    )
    assert critic_verdict, "Critic did not produce a verdict"
    assert final_answer, "no final answer"
    print("  ✓ integration: Critic requested replan, Orchestrator re-ran, Psychologist joined, graph finished")
    print(f"     orchestrator_visits={orchestrator_visits}, executed={executed_final}, verdict={critic_verdict}")


def main():
    print("\n[unit] ReplanJudge._parse_verdict")
    test_judge_parse()
    print("\n[unit] dispatcher replan trigger")
    test_dispatcher_replan_trigger()
    print("\n[unit] dispatcher replan cap")
    test_dispatcher_replan_cap()
    print("\n[integration] current graph with Critic stub forcing one replan")
    test_integration_replan_loop()
    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()
