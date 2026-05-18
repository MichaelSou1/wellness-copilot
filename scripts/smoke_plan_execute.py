"""Smoke test for plan-and-execute (Planner + Dispatcher + expert batch).

Goal: prove that a multi-expert turn routes through Planner/Dispatcher,
produces scratchpad notes for both experts, reaches Critic, and carries the
per-turn personalization context built by TurnStart.

Run: python scripts/smoke_plan_execute.py
"""
import os
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from langchain_core.messages import HumanMessage  # noqa: E402

from health_guide.graph import graph  # noqa: E402
from health_guide.llm import extract_text_content  # noqa: E402


def run_turn(thread_id: str, user_id: str, query: str):
    print("\n" + "=" * 70)
    print(f"USER ({user_id}): {query}")
    print("=" * 70)
    config = {"configurable": {"thread_id": thread_id}}
    events = []
    final_answer = ""
    for event in graph.stream(
        {
            "messages": [HumanMessage(content=query)],
            "profile_user_id": user_id,
        },
        config,
    ):
        for node, value in event.items():
            print(f"\n-- node: {node}")
            if value is None:
                value = {}
            events.append((node, value))
            if value.get("plan"):
                print(f"   plan_remaining: {value['plan']}")
            if value.get("executed"):
                print(f"   executed: {value['executed']}")
            if value.get("next"):
                print(f"   next: {value['next']}")
            if value.get("expert_responses"):
                for k, v in value["expert_responses"].items():
                    if not isinstance(v, str):
                        continue
                    print(f"   expert[{k}]: {v[:100]}{'...' if len(v) > 100 else ''}")
            if value.get("agent_notes"):
                for k, v in value["agent_notes"].items():
                    if not isinstance(v, str):
                        continue
                    print(f"   note[{k}]: {v[:100]}{'...' if len(v) > 100 else ''}")
            if value.get("draft_answer"):
                d = value["draft_answer"]
                print(f"   draft: {d[:140]}{'...' if len(d) > 140 else ''}")
            if value.get("critic_verdict"):
                print(f"   critic_verdict: {value['critic_verdict']}")
            if value.get("messages"):
                text = extract_text_content(value["messages"][-1])
                final_answer = text
                print(f"   final_msg: {text[:200]}{'...' if len(text) > 200 else ''}")
    snapshot = graph.get_state(config)
    final_state = getattr(snapshot, "values", {}) or {}
    return events, final_answer, final_state


def main():
    thread_id = str(uuid.uuid4())
    user_id = "smoke_plan_user"
    os.environ["HEALTH_GUIDE_USER_ID"] = user_id

    # Multi-expert query — keyword router will pick Trainer + Nutritionist
    # ('练腿' + '吃什么') and Planner sorts them as Trainer -> Nutritionist.
    events, final, final_state = run_turn(
        thread_id, user_id,
        "我刚练完腿，今晚该吃什么帮助恢复？训练动作上有没有要注意的？",
    )

    # ---- Assertions ----
    executed = final_state.get("executed") or []
    assert executed[:2] == ["Trainer", "Nutritionist"], (
        f"expected Trainer -> Nutritionist, got {executed}"
    )
    notes = final_state.get("agent_notes") or {}
    assert notes.get("Trainer"), "missing Trainer scratchpad note"
    assert notes.get("Nutritionist"), "missing Nutritionist scratchpad note"
    assert final_state.get("personalization_ctx"), "TurnStart did not build personalization_ctx"

    critic_verdicts = [v.get("critic_verdict") for _, v in events if v.get("critic_verdict")]
    assert critic_verdicts, "Critic did not run"

    assert final, "no final answer"
    print("\n--- assertions passed ---")
    print(f"   executed: {executed}")
    print(f"   personalization_ctx: {bool(final_state.get('personalization_ctx'))}")
    print(f"   critic verdict: {critic_verdicts[-1]}")
    print(f"   final length: {len(final)}")
    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()
