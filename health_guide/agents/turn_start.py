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

from ..config import (
    EPISODE_SEMANTIC_MIN_COUNT,
    EPISODE_SEMANTIC_RETRIEVAL_ENABLED,
    EPISODE_SEMANTIC_TOP_K,
)
from ..episode_store import (
    episode_id,
    format_episodes_for_prompt,
    get_recent_episodes,
    total_episode_count,
)
from ..episode_memory import EpisodeMemory
from ..llm import extract_text_content, llm
from ..personalization import build_personalization_ctx
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


def _latest_human_text(messages) -> str:
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            return extract_text_content(msg).strip()
    return ""


def _load_episode_context(user_id: str, current_query: str) -> str:
    recent_eps = get_recent_episodes(user_id, n=2)
    for ep in recent_eps:
        ep["_memory_source"] = "最近"

    semantic_eps = []
    if (
        EPISODE_SEMANTIC_RETRIEVAL_ENABLED
        and current_query
        and total_episode_count(user_id) >= EPISODE_SEMANTIC_MIN_COUNT
    ):
        recent_ids = {ep.get("id") or episode_id(ep) for ep in recent_eps}
        try:
            semantic_eps = EpisodeMemory(user_id).retrieve_similar(
                current_query,
                top_k=EPISODE_SEMANTIC_TOP_K,
                exclude_ids=recent_ids,
            )
        except Exception:
            semantic_eps = []

    # Keep recency as the first anchor in prompt order, then add older semantic
    # recalls. get_recent_episodes returns oldest-first, so reverse the recent
    # slice for newest-first display.
    merged = list(reversed(recent_eps)) + semantic_eps
    return format_episodes_for_prompt(merged, mark_source=True)


def turn_start_node(state):
    user_id = state.get("profile_user_id", "default_user")
    messages = state.get("messages", []) or []
    current_query = _latest_human_text(messages)
    try:
        episode_context = _load_episode_context(user_id, current_query)
    except Exception:
        episode_context = ""
    try:
        personalization_ctx = build_personalization_ctx(user_id)
    except Exception:
        personalization_ctx = {}

    update = {
        "episode_context": episode_context,
        "personalization_ctx": personalization_ctx,
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
        "orchestrator_decision": "",
    }

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
