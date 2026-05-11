"""TurnStart — per-turn boundary node.

Runs once at the entry of every user turn (before QueryRewriter):

1. Resets *turn-scoped* state fields so the previous turn's scratchpad,
   tool log, plan queue, and replan counter don't leak into the new turn.
   This is required because LangGraph reducers like ``operator.add`` /
   merge-dict otherwise accumulate forever across the persisted thread.

2. If the conversation has grown past ``MAX_MESSAGES_BEFORE_SUMMARY``,
   compresses the older half into a running Chinese summary and removes
   the original messages from the thread (via ``RemoveMessage``), keeping
   only the most recent tail verbatim. The summary is re-injected as a
   ``SystemMessage`` with a stable id so subsequent compactions just
   replace it, never duplicate.
"""

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)

from ..llm import extract_text_content, llm
from ..state import RESET_SENTINEL


MAX_MESSAGES_BEFORE_SUMMARY = 20
KEEP_RECENT_MESSAGES = 8
HISTORY_SUMMARY_ID = "__history_summary__"


_SUMMARIZER_SYSTEM = """\
你是对话摘要助手。把给定的旧对话历史压缩成简洁的中文要点摘要，供后续 Agent 参考。

要求：
- 用 5-10 条要点，覆盖：用户的身体状况/目标/约束、已讨论过的训练/营养/恢复方案、用户的偏好或反馈
- 如果给出了"已有摘要"，把新对话内容并入它，去重并保持简洁
- 不要逐字复述对话，提炼为陈述句
- 不要回答任何问题，不要给建议，只做摘要
- 直接输出摘要正文，不要前缀、引号、解释
"""


def _render_messages(msgs):
    lines = []
    for m in msgs:
        role = getattr(m, "type", "")
        if role == "human":
            tag = "用户"
        elif role == "ai":
            tag = "助手"
        else:
            continue
        text = extract_text_content(m).strip()
        if not text:
            continue
        if len(text) > 600:
            text = text[:600] + "…"
        lines.append(f"{tag}：{text}")
    return "\n".join(lines)


def turn_start_node(state):
    update = {
        # Reset accumulator fields via sentinels honored by state.py reducers
        "agent_notes": {RESET_SENTINEL: True},
        "expert_responses": {RESET_SENTINEL: True},
        "last_tools": [RESET_SENTINEL],
        "retrieval_hits": (RESET_SENTINEL, 0),
        # Plain-overwrite fields (no reducer)
        "plan": [],
        "executed": [],
        "next": [],
        "replan_count": 0,
        # take-last reducers: explicit clear
        "replan_request": "",
        "replan_context": "",
        "contextualized_query": "",
        "draft_answer": "",
        "critic_verdict": "",
    }

    messages = state.get("messages", []) or []
    # Don't count the synthetic summary marker against the window budget.
    non_summary = [m for m in messages if getattr(m, "id", None) != HISTORY_SUMMARY_ID]
    if len(non_summary) <= MAX_MESSAGES_BEFORE_SUMMARY:
        return update

    head = non_summary[:-KEEP_RECENT_MESSAGES]
    if not head:
        return update

    prior_summary = (state.get("history_summary") or "").strip()
    head_text = _render_messages(head)
    if not head_text:
        return update

    user_prompt = (
        (f"已有摘要：\n{prior_summary}\n\n" if prior_summary else "")
        + f"需要并入摘要的更早对话：\n{head_text}\n\n"
        "请输出新的合并摘要。"
    )
    try:
        response = llm.invoke([
            SystemMessage(content=_SUMMARIZER_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
        new_summary = extract_text_content(response).strip()
    except Exception:
        # Summarization is best-effort; on failure, leave history intact.
        return update

    if not new_summary:
        return update

    removes = [RemoveMessage(id=m.id) for m in head if getattr(m, "id", None)]
    summary_msg = SystemMessage(
        content=f"[此前对话摘要]\n{new_summary}",
        id=HISTORY_SUMMARY_ID,
    )
    update["messages"] = removes + [summary_msg]
    update["history_summary"] = new_summary
    return update
