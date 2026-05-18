"""Smoke test for hybrid episodic memory retrieval.

This exercises TurnStart directly: recent 2 episodes must always be present,
and an older ACL-related episode should be recalled semantically.

Run:
    python scripts/smoke_semantic_episode.py
"""
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
os.environ["PROFILE_STORE_PATH"] = str(tmp_path / "profile_store.json")
os.environ["EPISODE_STORE_PATH"] = str(tmp_path / "episode_store.json")
os.environ["EPISODE_INDEX_DIR"] = str(tmp_path / "episode_indices")
os.environ["EPISODE_SEMANTIC_RETRIEVAL_ENABLED"] = "true"
os.environ["EPISODE_SEMANTIC_MIN_COUNT"] = "8"
os.environ["EPISODE_SEMANTIC_TOP_K"] = "3"

from langchain_core.messages import HumanMessage  # noqa: E402

from health_guide.agents import turn_start as turn_start_mod  # noqa: E402
from health_guide.episode_store import append_episode  # noqa: E402


def main():
    user_id = f"smoke_semantic_{uuid.uuid4().hex[:8]}"

    seed = [
        ("旧伤记录", ["Trainer"], "用户提到膝盖 ACL 前交叉韧带术后康复，不能跑步、跳跃、急停变向；须经理疗师许可。"),
        ("早餐吃什么", ["Nutritionist"], "讨论了早餐蛋白质和燕麦搭配。"),
        ("工作压力", ["Wellness"], "讨论了 deadline 压力和睡前放松。"),
        ("肩部活动", ["Trainer"], "讨论了肩部热身。"),
        ("喝水", ["Nutritionist"], "讨论了饮水量。"),
        ("通勤久坐", ["Wellness"], "讨论了久坐拉伸。"),
        ("训练后酸痛", ["Trainer"], "讨论了 DOMS 主动恢复。"),
        ("补剂", ["Nutritionist"], "讨论了肌酸 3-5g。"),
        ("晚餐", ["Nutritionist"], "讨论了晚餐蔬菜和蛋白。"),
        ("睡眠", ["Wellness"], "讨论了固定起床时间。"),
        ("最近一：胸背训练", ["Trainer"], "最近计划是胸背训练，未涉及膝盖。"),
        ("最近二：加班吃饭", ["Nutritionist"], "最近讨论加班时如何点外卖。"),
    ]
    for query, experts, gist in seed:
        append_episode(user_id=user_id, query=query, experts=experts, gist=gist)

    state = {
        "profile_user_id": user_id,
        "messages": [HumanMessage(content="我膝盖最近有点疼，能跑步吗？")],
    }
    update = turn_start_mod.turn_start_node(state)
    ctx = update.get("episode_context", "")
    print("\n--- hybrid episode_context ---")
    print(ctx)
    assert "[最近]" in ctx, "recent episodes missing"
    assert "最近一" in ctx and "最近二" in ctx, "recent 2 episodes not preserved"
    assert "[相关]" in ctx and ("ACL" in ctx or "前交叉" in ctx), "ACL episode not semantically recalled"

    old_min = turn_start_mod.EPISODE_SEMANTIC_MIN_COUNT
    turn_start_mod.EPISODE_SEMANTIC_MIN_COUNT = 999
    try:
        degraded = turn_start_mod.turn_start_node(state).get("episode_context", "")
    finally:
        turn_start_mod.EPISODE_SEMANTIC_MIN_COUNT = old_min
    print("\n--- degraded episode_context ---")
    print(degraded)
    assert "最近一" in degraded and "最近二" in degraded, "degraded mode lost recent episodes"
    assert "ACL" not in degraded and "前交叉" not in degraded, "degraded mode should not semantic-recall ACL"

    print("\nSMOKE SEMANTIC EPISODE OK")


if __name__ == "__main__":
    main()
