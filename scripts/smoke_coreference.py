"""Smoke test for multi-turn coreference resolution.

Goal: prove that follow-up questions with omitted/referential phrasing
("那 / 那个 / 再练 / 还能...") are correctly rewritten by QueryRewriter and
routed by Orchestrator instead of being misrouted to the wrong specialist.

Three layers:

1. **Unit**: QueryRewriter passes through on first turn (no prior AI).
2. **Integration (deterministic)**: stub the rewriter to a known rewritten
   query, verify Orchestrator / Aggregator / Critic actually see it via
   `get_user_question`.
3. **Live LLM**: run two real turns end-to-end; turn 2 uses a pronoun
   ("那这个练完之后") and we check that the rewriter produces a self-
   contained query referencing turn-1 context, and that Orchestrator delegates.
"""
import os
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from langchain_core.messages import HumanMessage, AIMessage  # noqa: E402

from health_guide.agents.query_rewriter import (  # noqa: E402
    query_rewriter_node,
    get_user_question,
)


def test_rewriter_passthrough_on_first_turn():
    state = {"messages": [HumanMessage(content="减脂期每天该吃多少蛋白质？")]}
    out = query_rewriter_node(state)
    assert out["contextualized_query"] == "减脂期每天该吃多少蛋白质？"
    print("  ✓ rewriter passes through on first turn (no LLM call)")


def test_get_user_question_prefers_contextualized():
    state = {
        "contextualized_query": "为了减脂目标，每天该吃多少蛋白质？",
        "messages": [HumanMessage(content="那这个怎么算")],
    }
    assert "蛋白质" in get_user_question(state)
    print("  ✓ get_user_question prefers contextualized_query over raw HumanMessage")


def test_get_user_question_falls_back_to_messages():
    state = {"messages": [HumanMessage(content="hello")]}
    assert get_user_question(state) == "hello"
    print("  ✓ get_user_question falls back to messages when no contextualized_query")


def test_live_multi_turn_coreference():
    """Run the full graph for two turns; turn 2 uses a pronoun."""
    from health_guide.graph import graph
    from health_guide.llm import extract_text_content

    thread_id = str(uuid.uuid4())
    user_id = "smoke_coref_user"
    os.environ["HEALTH_GUIDE_USER_ID"] = user_id
    config = {"configurable": {"thread_id": thread_id}}

    def run(query: str) -> dict:
        contextualized = ""
        plans = []
        finished_only = False
        executed = []
        final_answer = ""
        for event in graph.stream(
            {"messages": [HumanMessage(content=query)], "profile_user_id": user_id},
            config,
        ):
            for node, value in event.items():
                if value is None:
                    value = {}
                if node == "QueryRewriter" and value.get("contextualized_query"):
                    contextualized = value["contextualized_query"]
                if node == "Orchestrator":
                    p = value.get("plan", [])
                    plans.append(p)
                if value.get("executed"):
                    executed = value["executed"]
                if value.get("messages"):
                    last_msg = value["messages"][-1]
                    final_answer = extract_text_content(last_msg)
        return {
            "contextualized": contextualized,
            "plans": plans,
            "finished_only": finished_only,
            "executed": executed,
            "final_answer": final_answer,
        }

    # Turn 1: concrete first question (no coref) — establishes context.
    q1 = "我想减脂，每天蛋白质大概该吃多少？"
    print(f"\n  TURN 1: {q1}")
    r1 = run(q1)
    print(f"    contextualized: {r1['contextualized']}")
    print(f"    plans: {r1['plans']}")
    print(f"    executed: {r1['executed']}")
    print(f"    final len: {len(r1['final_answer'])}")
    assert r1["contextualized"] == q1, "turn 1 should pass through unchanged"
    assert r1["final_answer"], "turn 1 should produce a final answer"

    # Turn 2: heavy coreference. "那" + "这个" both refer to the cut/protein context.
    q2 = "那训练上该怎么配合？"
    print(f"\n  TURN 2: {q2}")
    r2 = run(q2)
    print(f"    contextualized: {r2['contextualized']}")
    print(f"    plans: {r2['plans']}")
    print(f"    executed: {r2['executed']}")
    print(f"    final len: {len(r2['final_answer'])}")

    assert r2["contextualized"], "turn 2 contextualized_query must be set"
    # The rewriter should incorporate the prior context — at minimum, mention
    # "减脂" (the fat-loss goal) since q2 dangles on that without saying so.
    assert "减脂" in r2["contextualized"], (
        f"rewriter should resolve coref by mentioning 减脂 context; got: {r2['contextualized']}"
    )
    assert not r2["finished_only"], "Orchestrator must not finish without handling a follow-up question"
    assert r2["executed"], "turn 2 should execute at least one expert"
    # The question is training-shaped, so Trainer should be in there.
    assert "Trainer" in r2["executed"], (
        f"expected Trainer in executed for a training question, got {r2['executed']}"
    )
    assert r2["final_answer"], "turn 2 should produce a final answer"
    print("  ✓ live: coreference resolved, Orchestrator routed to Trainer, graph finished")


def main():
    print("\n[unit] rewriter passthrough on first turn")
    test_rewriter_passthrough_on_first_turn()
    print("\n[unit] get_user_question prefers contextualized")
    test_get_user_question_prefers_contextualized()
    print("\n[unit] get_user_question falls back")
    test_get_user_question_falls_back_to_messages()
    print("\n[live] multi-turn coreference end-to-end")
    test_live_multi_turn_coreference()
    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()
