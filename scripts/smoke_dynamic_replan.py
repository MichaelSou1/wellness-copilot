"""Smoke test for dynamic replan via meta-LLM ReplanJudge.

Layers:

1. **Unit**: ReplanJudge._parse_verdict accepts CONTINUE / REPLAN forms.
2. **Unit**: dispatcher_node honors `replan_request` and clears it.
3. **Unit**: dispatcher_node respects REPLAN_CAP.
4. **Integration**: build a graph that swaps in a deterministic Trainer
   stub *and* a deterministic ReplanJudge stub, then verifies the replan
   loop runs end-to-end and the graph finishes via Aggregator → Critic.
"""
import os
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from langchain_core.messages import HumanMessage  # noqa: E402

from health_guide.agents.dispatcher import dispatcher_node  # noqa: E402
from health_guide.agents.replan_judge import _parse_verdict  # noqa: E402


class PassLLM:
    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        return AIMessage(content="VERDICT: PASS")


def test_judge_parse():
    assert _parse_verdict("VERDICT: CONTINUE") == ""
    assert _parse_verdict("verdict: continue\n") == ""
    r = _parse_verdict("VERDICT: REPLAN\nREASON: 用户睡眠问题需要 Wellness 补充")
    assert "睡眠" in r and "Wellness" in r, r
    # REPLAN without REASON falls back to a generic synthesized reason
    r2 = _parse_verdict("VERDICT: REPLAN")
    assert r2, "REPLAN without REASON should still produce a non-empty reason"
    # Garbage → no replan
    assert _parse_verdict("hello world") == ""
    assert _parse_verdict("") == ""
    print("  ✓ ReplanJudge._parse_verdict handles CONTINUE / REPLAN / garbage")


def test_dispatcher_replan_trigger():
    state = {
        "replan_request": "需要 Wellness 补充睡眠建议",
        "plan": ["Nutritionist"],
        "executed": ["Trainer"],
        "replan_count": 0,
    }
    out = dispatcher_node(state)
    assert out.get("next") == ["__REPLAN__"], f"expected __REPLAN__, got {out.get('next')}"
    assert out.get("replan_request") == ""
    assert out.get("replan_context") == "需要 Wellness 补充睡眠建议"
    assert out.get("replan_count") == 1
    print("  ✓ dispatcher_node routes to __REPLAN__ and clears the request")


def test_dispatcher_replan_cap():
    import health_guide.agents.dispatcher as dispatcher_mod

    def wellness_stub(*args, **kwargs):
        return {
            "expert_responses": {"Wellness": "cap reached, executing remaining plan"},
            "agent_notes": {"Wellness": "cap reached"},
            "last_tools": [],
            "retrieval_hits": 0,
        }

    state = {
        "replan_request": "再叫一个",
        "plan": ["Wellness"],
        "executed": ["Trainer", "Nutritionist"],
        "replan_count": 2,
    }
    old_wellness = dispatcher_mod.EXPERT_RUNNERS["Wellness"]
    dispatcher_mod.EXPERT_RUNNERS["Wellness"] = wellness_stub
    try:
        out = dispatcher_node(state)
    finally:
        dispatcher_mod.EXPERT_RUNNERS["Wellness"] = old_wellness
    assert out.get("next") == []
    assert out.get("replan_request") == ""
    assert "replan_context" not in out
    assert out.get("executed") == ["Trainer", "Nutritionist", "Wellness"]
    print("  ✓ dispatcher_node respects REPLAN_CAP and falls through")


def test_integration_replan_loop():
    """End-to-end with stubbed Trainer + stubbed ReplanJudge.

    Trainer stub: returns a fixed answer (no marker — markers are gone).
    Judge stub: forces REPLAN once for Trainer, CONTINUE afterwards.
    """
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver

    from health_guide.state import AgentState
    import health_guide.agents.critic as critic
    import health_guide.agents.dispatcher as dispatcher_mod
    from health_guide.agents.dispatcher import dispatcher_node as real_dispatcher
    from health_guide.agents.aggregator import aggregator_node
    from health_guide.agents.critic import critic_node
    from health_guide.graph import _route_after_dispatch, _route_after_judge, _route_after_orchestrator

    def orchestrator_stub(state):
        if state.get("replan_context"):
            return {
                "plan": ["Wellness"],
                "replan_context": "",
                "next": [],
                "orchestrator_decision": "PLAN",
            }
        return {
            "plan": ["Trainer"],
            "executed": ["Orchestrator"],
            "replan_count": 0,
            "replan_context": "",
            "next": [],
            "orchestrator_decision": "PLAN",
        }

    def trainer_stub(*args, **kwargs):
        return {
            "expert_responses": {"Trainer": "今晚做 20 分钟轻度有氧。"},
            "agent_notes": {"Trainer": "训练：20 分钟轻度有氧"},
            "last_tools": [],
            "retrieval_hits": 0,
        }

    def wellness_stub(*args, **kwargs):
        return {
            "expert_responses": {"Wellness": "睡前 30 分钟降低刺激，做 5 分钟呼吸放松。"},
            "agent_notes": {"Wellness": "睡眠：睡前 30 分钟降刺激 + 5 分钟呼吸"},
            "last_tools": [],
            "retrieval_hits": 0,
        }

    # Stateful stub via closure — fire REPLAN once on Trainer, then always CONTINUE.
    fired = {"v": False}

    def judge_stub(state):
        executed = state.get("executed") or []
        if not executed:
            return {}
        last = executed[-1]
        if last == "Trainer" and not fired["v"]:
            fired["v"] = True
            return {"replan_request": "用户睡眠问题没解决，需要 Wellness 补充建议"}
        return {}

    old_runners = dict(dispatcher_mod.EXPERT_RUNNERS)
    old_critic_llm = critic.llm
    dispatcher_mod.EXPERT_RUNNERS = {
        "Trainer": trainer_stub,
        "Wellness": wellness_stub,
    }
    critic.llm = PassLLM()

    try:
        workflow = StateGraph(AgentState)
        workflow.add_node("Orchestrator", orchestrator_stub)
        workflow.add_node("Dispatcher", real_dispatcher)
        workflow.add_node("ReplanJudge", judge_stub)
        workflow.add_node("Aggregator", aggregator_node)
        workflow.add_node("Critic", critic_node)

        workflow.set_entry_point("Orchestrator")
        workflow.add_conditional_edges(
            "Orchestrator",
            _route_after_orchestrator,
            ["Dispatcher", "Aggregator", END],
        )
        workflow.add_conditional_edges(
            "Dispatcher",
            _route_after_dispatch,
            ["Orchestrator", "ReplanJudge", END],
        )
        workflow.add_conditional_edges(
            "ReplanJudge",
            _route_after_judge,
            ["Dispatcher", "Aggregator"],
        )
        workflow.add_edge("Aggregator", "Critic")
        workflow.add_edge("Critic", END)

        test_graph = workflow.compile(checkpointer=MemorySaver())
    finally:
        dispatcher_mod.EXPERT_RUNNERS = old_runners
        critic.llm = old_critic_llm

    thread_id = str(uuid.uuid4())
    user_id = "smoke_judge_user"
    os.environ["HEALTH_GUIDE_USER_ID"] = user_id
    query = "我深蹲后膝盖酸，今晚训练应该怎么调？"
    print(f"\n  USER: {query}")

    config = {"configurable": {"thread_id": thread_id}}
    orchestrator_visits = 0
    saw_replan_route = False
    executed_final = []
    final_answer = ""
    critic_verdict = ""
    old_runners = dict(dispatcher_mod.EXPERT_RUNNERS)
    old_critic_llm = critic.llm
    dispatcher_mod.EXPERT_RUNNERS = {
        "Trainer": trainer_stub,
        "Wellness": wellness_stub,
    }
    critic.llm = PassLLM()
    try:
        for event in test_graph.stream(
            {"messages": [HumanMessage(content=query)], "profile_user_id": user_id},
            config,
        ):
            for node, value in event.items():
                if value is None:
                    value = {}
                if node == "Orchestrator":
                    orchestrator_visits += 1
                    print(f"  [Orchestrator #{orchestrator_visits}] plan={value.get('plan', [])}")
                elif node == "Dispatcher":
                    nxt = value.get("next")
                    if nxt == ["__REPLAN__"]:
                        saw_replan_route = True
                        print(f"  [Dispatcher] -> __REPLAN__ (ctx={value.get('replan_context')!r})")
                    elif nxt:
                        print(f"  [Dispatcher] -> {nxt[0]}")
                    if value.get("executed"):
                        executed_final = value["executed"]
                elif node == "ReplanJudge":
                    if value.get("replan_request"):
                        print(f"  [ReplanJudge] requests replan: {value['replan_request']}")
                    else:
                        print("  [ReplanJudge] CONTINUE")
                elif node == "Critic":
                    critic_verdict = value.get("critic_verdict", "")
                    if value.get("messages"):
                        last_msg = value["messages"][-1]
                        final_answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                    print(f"  [Critic] verdict={critic_verdict}")
    finally:
        dispatcher_mod.EXPERT_RUNNERS = old_runners
        critic.llm = old_critic_llm

    assert fired["v"], "Judge stub never fired (state plumbing broken?)"
    assert orchestrator_visits >= 2, f"expected Orchestrator to run >=2x, got {orchestrator_visits}"
    assert saw_replan_route, "Dispatcher never routed to __REPLAN__"
    assert "Trainer" in executed_final, f"Trainer must run, got {executed_final}"
    assert "Wellness" in executed_final, f"Wellness must join after replan, got {executed_final}"
    assert len(executed_final) >= 2, (
        f"expected ≥2 experts after replan, got {executed_final}"
    )
    assert critic_verdict, "Critic did not produce a verdict"
    assert final_answer, "no final answer"
    print("  ✓ integration: Judge requested replan, Orchestrator re-ran, Wellness joined, graph finished")
    print(f"     orchestrator_visits={orchestrator_visits}, executed={executed_final}, verdict={critic_verdict}")


def main():
    print("\n[unit] ReplanJudge._parse_verdict")
    test_judge_parse()
    print("\n[unit] dispatcher replan trigger")
    test_dispatcher_replan_trigger()
    print("\n[unit] dispatcher replan cap")
    test_dispatcher_replan_cap()
    print("\n[integration] full graph with Judge stub forcing one replan")
    test_integration_replan_loop()
    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()
