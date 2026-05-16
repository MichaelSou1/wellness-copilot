"""Smoke test for Critic + shared scratchpad.

Run: python scripts/smoke_critic_scratchpad.py
"""
import os
import sys
import uuid

# Ensure repo root on path when invoked from anywhere
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from langchain_core.messages import HumanMessage  # noqa: E402

from health_guide.graph import graph  # noqa: E402
from health_guide.llm import extract_text_content  # noqa: E402


def run_turn(thread_id: str, user_id: str, query: str) -> None:
    print("\n" + "=" * 70)
    print(f"USER ({user_id}): {query}")
    print("=" * 70)
    config = {"configurable": {"thread_id": thread_id}}
    saw_critic = False
    saw_notes = False
    saw_draft = False
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
            if "next" in value:
                print(f"   next: {value['next']}")
            if "expert_responses" in value and value["expert_responses"]:
                for k, v in value["expert_responses"].items():
                    if not isinstance(v, str):
                        continue
                    print(f"   expert[{k}]: {v[:120]}{'...' if len(v) > 120 else ''}")
            if "agent_notes" in value and value["agent_notes"]:
                saw_notes = True
                for k, v in value["agent_notes"].items():
                    if not isinstance(v, str):
                        continue
                    print(f"   note[{k}]: {v[:100]}{'...' if len(v) > 100 else ''}")
            if "draft_answer" in value and value["draft_answer"]:
                saw_draft = True
                d = value["draft_answer"]
                print(f"   draft: {d[:140]}{'...' if len(d) > 140 else ''}")
            if "critic_verdict" in value and value["critic_verdict"]:
                saw_critic = True
                print(f"   critic_verdict: {value['critic_verdict']}")
            if "messages" in value:
                last_msg = value["messages"][-1]
                text = extract_text_content(last_msg)
                final_answer = text
                print(f"   final_msg: {text[:200]}{'...' if len(text) > 200 else ''}")

    print("\n--- assertions ---")
    print(f"   saw scratchpad note: {saw_notes}")
    print(f"   saw aggregator draft: {saw_draft}")
    print(f"   saw critic verdict:  {saw_critic}")
    print(f"   final answer len:    {len(final_answer)}")
    assert final_answer, "final answer is empty"
    assert saw_critic, "Critic did not run"
    assert saw_draft, "Aggregator did not produce draft"


def main():
    thread_id = str(uuid.uuid4())
    user_id = "smoke_test_user"
    os.environ["HEALTH_GUIDE_USER_ID"] = user_id

    # Turn 1: cross-domain question that should fan out to multiple experts.
    run_turn(thread_id, user_id, "我刚练完腿，今晚应该吃什么帮助恢复？")

    # Turn 2: follow-up that should see prior-turn scratchpad notes.
    run_turn(thread_id, user_id, "那训练后多久睡觉比较好？")

    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()
