"""Critic / Safety Reviewer.

Reads the Aggregator's draft answer along with the shared scratchpad notes
and the user's profile, and decides whether the draft is safe to ship.

Output protocol (the LLM must return one of):
  VERDICT: PASS
  VERDICT: REVISE
  ---
  <revised answer text, only present when REVISE>

If PASS, the draft is forwarded as-is. If REVISE, the revised version is
used as the final answer.
"""
import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..llm import extract_text_content, llm
from ..profile_store import (
    get_user_profile as get_profile_from_store,
    profile_to_prompt_text,
)
from .query_rewriter import get_user_question


_EXPERT_LABELS = {
    "Trainer": "训练教练",
    "Nutritionist": "营养师",
    "Wellness": "身心康复师",
    "General": "助理",
}


_CRITIC_SYSTEM = """\
你是健康团队的安全审核员。你的职责是对即将发给用户的回答做一次"安全 & 一致性"复核，**只关注严重问题**，不要为了挑刺而挑刺。

需要重点关注的风险：
1. **健康安全**：是否给出了与用户画像冲突的建议？例如忽略已记录的伤病/慢病、给出极端热量赤字（女性 <1200 kcal/d、男性 <1500 kcal/d）、推荐对未经训练者高风险的动作等。
2. **跨专家矛盾**：综合回答与各专家的核心要点是否存在事实性冲突？
3. **越界**：是否在没有医疗背景信息时给出诊断/用药/剂量类建议？

判断标准要克制：
- 如果只是措辞不够漂亮、缺少边角细节，请直接 PASS。
- 只有当真的存在上述风险时才 REVISE。

输出格式（严格遵守）：
- 如果通过，仅输出一行：`VERDICT: PASS`
- 如果需要修订，输出：
```
VERDICT: REVISE
REASON: <一句话写明触发修订的具体风险>
---
<修订后的完整回答，以用户视角直接说话，不要提及"审核""修订"等元话语>
```
"""


def _format_notes(agent_notes):
    if not agent_notes:
        return "（无）"
    lines = []
    for role, note in agent_notes.items():
        if not note:
            continue
        label = _EXPERT_LABELS.get(role, role)
        lines.append(f"- [{label}] {note}")
    return "\n".join(lines) if lines else "（无）"


def _parse_verdict(text: str):
    """Return (verdict, revised_or_none). verdict ∈ {'PASS','REVISE','UNKNOWN'}."""
    if not text:
        return "UNKNOWN", None
    first_line = text.strip().splitlines()[0].strip().upper()
    if first_line.startswith("VERDICT: PASS"):
        return "PASS", None
    if first_line.startswith("VERDICT: REVISE"):
        # split on first '---' line
        parts = re.split(r"\n\s*---\s*\n", text, maxsplit=1)
        revised = parts[1].strip() if len(parts) == 2 else ""
        return "REVISE", revised or None
    return "UNKNOWN", None


def critic_node(state):
    draft = state.get("draft_answer", "") or ""
    if not draft:
        # Nothing to review (e.g., all experts produced empty). Fail safe.
        return {
            "messages": [AIMessage(content="抱歉，未能生成有效回答，请重试。")],
            "critic_verdict": "EMPTY",
        }

    user_id = state.get("profile_user_id", "default_user")
    try:
        profile_text = profile_to_prompt_text(get_profile_from_store(user_id))
    except Exception:
        profile_text = "（画像不可用）"

    notes_text = _format_notes(state.get("agent_notes") or {})
    user_question = get_user_question(state) or "（未获取到原始问题）"

    review_prompt = (
        f"用户画像：\n{profile_text}\n\n"
        f"用户本轮问题：\n{user_question}\n\n"
        f"各专家共享 scratchpad 要点：\n{notes_text}\n\n"
        f"即将发送给用户的草稿回答：\n{draft}\n\n"
        "请按指定格式给出审核结论。"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=_CRITIC_SYSTEM),
            HumanMessage(content=review_prompt),
        ])
        raw = extract_text_content(response)
    except Exception as e:
        # Critic failure must not block the user — fall back to draft.
        return {
            "messages": [AIMessage(content=draft)],
            "critic_verdict": f"ERROR:{type(e).__name__}",
        }

    verdict, revised = _parse_verdict(raw)

    if verdict == "REVISE" and revised:
        final_text = revised
    else:
        # PASS or UNKNOWN → ship the draft unchanged.
        final_text = draft
        if verdict == "UNKNOWN":
            verdict = "PASS_FALLBACK"

    return {
        "messages": [AIMessage(content=final_text)],
        "critic_verdict": verdict,
    }
