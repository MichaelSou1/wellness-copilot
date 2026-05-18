"""Smoke test for profile-driven personalization.

Run:
    python scripts/smoke_personalization.py
"""
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_TMP = tempfile.TemporaryDirectory()
tmp_path = Path(_TMP.name)
os.environ.setdefault("PROFILE_STORE_PATH", str(tmp_path / "profile_store.json"))
os.environ.setdefault("EPISODE_STORE_PATH", str(tmp_path / "episode_store.json"))
os.environ.setdefault("EPISODE_INDEX_DIR", str(tmp_path / "episode_indices"))
os.environ.setdefault("EPISODE_SEMANTIC_RETRIEVAL_ENABLED", "false")

from langchain_core.messages import HumanMessage  # noqa: E402

from health_guide.graph import graph  # noqa: E402
from health_guide.llm import extract_text_content  # noqa: E402
from health_guide.profile_store import get_user_profile, update_user_profile  # noqa: E402


def run_turn(thread_id: str, user_id: str, query: str):
    print("\n" + "=" * 70)
    print(f"USER ({user_id}): {query}")
    print("=" * 70)
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "profile_user_id": user_id,
        },
        config,
    )
    answer = extract_text_content(state["messages"][-1]) if state.get("messages") else ""
    print(answer[:900] + ("..." if len(answer) > 900 else ""))
    print(f"executed={state.get('executed')} critic={state.get('critic_verdict')}")
    return state, answer


def main():
    user_id = f"smoke_personalization_{uuid.uuid4().hex[:8]}"
    thread_id = str(uuid.uuid4())
    os.environ["HEALTH_GUIDE_USER_ID"] = user_id

    update_user_profile(
        user_id,
        {
            "name": "Michael",
            "identity": "CS研究生",
            "physical_stats": {
                "age": 24,
                "height": 178,
                "weight": 75,
                "injuries": ["ACL 撕裂术后早期"],
            },
            "dietary_context": {
                "goal": "增肌",
                "preferences": ["不吃香菜"],
            },
            "mental_state": {"stress_sources": ["论文 deadline"]},
        },
    )

    state, answer = run_turn(thread_id, user_id, "我想增肌，该怎么练和吃？")
    assert state.get("personalization_ctx"), "missing personalization_ctx"
    assert "24" in answer or "75" in answer or "178" in answer, "answer did not cite profile numbers"
    assert "ACL" in answer or "韧带" in answer, "answer did not cite ACL injury"
    assert "理疗师" in answer or "医生" in answer, "answer did not preserve professional-supervision constraint"

    run_turn(thread_id, user_id, "我喜欢你回答简洁一点，以后请用 concise tone。请记住。")
    profile = get_user_profile(user_id)
    print("\nprofile after style turn:")
    print(json.dumps(profile.get("response_style", {}), ensure_ascii=False))
    tone = (profile.get("response_style") or {}).get("tone", "")
    assert tone in {"concise", "简洁", "简洁一点"}, f"response_style.tone not updated: {tone!r}"

    _, concise_answer = run_turn(thread_id, user_id, "那明天训练和晚餐怎么安排？")
    assert len(concise_answer) < 600, f"expected concise answer, got {len(concise_answer)} chars"

    print("\nSMOKE PERSONALIZATION OK")


if __name__ == "__main__":
    main()
