"""QueryRewriter — coreference resolution for multi-turn questions.

Problem: in multi-turn dialogue users send messages like "那个怎么吃" /
"再练一次行吗" — these are uninterpretable without the prior context.
We previously made the Planner look only at the latest user message to avoid
spurious FINISH outputs on follow-ups. That fixes FINISH but loses
coreference resolution.

Solution: a dedicated rewriter LLM call that produces a *self-contained*
restatement of the latest user message, drawing on prior turns. Downstream
"meta" nodes (Planner / ReplanJudge / Aggregator / Critic) read this
restatement via `get_user_question()`. Experts still see the full
conversation, so they continue to handle context naturally.

The rewriter only runs at turn boundaries (when there is at least one prior
AIMessage in history). It is *not* re-invoked during replan — replan
shares the same `contextualized_query` because it's still the same user
turn.
"""
from typing import Optional

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)

from ..llm import extract_text_content, llm


_REWRITER_SYSTEM = """\
你是一个对话改写器。给定多轮对话历史和用户的最新一条消息，你的任务是把最新消息改写为一个**独立、自包含的中文问题**——即不依赖对话历史也能被理解。

规则：
- 如果最新消息已经自包含（没有"那个/这个/它/上面/之前说的/再/还/也"等指代或省略），原样输出
- 如果存在指代或省略，从历史中找到指代对象，把必要的上下文补全到最新消息里
- 只输出改写后的消息文本，不要任何前缀、解释、标点装饰
- 改写后的长度应接近"原消息 + 少量补充上下文"，不要把整段历史塞进去
- 不要回答用户的问题，只做改写
"""


def _last_human_index(messages) -> Optional[int]:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage) or getattr(messages[i], "type", None) == "human":
            return i
    return None


def _has_prior_ai(messages, latest_human_idx: int) -> bool:
    for i in range(latest_human_idx - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage) or getattr(msg, "type", None) == "ai":
            text = extract_text_content(msg).strip()
            if text:
                return True
    return False


def _format_history(messages, latest_human_idx: int) -> str:
    """Render the conversation prior to the latest user message."""
    lines = []
    for msg in messages[:latest_human_idx]:
        role = getattr(msg, "type", None)
        if role == "human" or isinstance(msg, HumanMessage):
            tag = "用户"
        elif role == "ai" or isinstance(msg, AIMessage):
            tag = "助手"
        else:
            continue
        text = extract_text_content(msg).strip()
        if not text:
            continue
        # Trim very long assistant responses to keep the rewriter prompt focused.
        if len(text) > 400:
            text = text[:400] + "…"
        lines.append(f"{tag}：{text}")
    return "\n".join(lines)


def query_rewriter_node(state):
    messages = state.get("messages", [])
    if not messages:
        return {"contextualized_query": ""}

    idx = _last_human_index(messages)
    if idx is None:
        return {"contextualized_query": ""}

    latest_text = extract_text_content(messages[idx]).strip()
    if not latest_text:
        return {"contextualized_query": ""}

    # First turn (no prior assistant message) → no coreference possible.
    if not _has_prior_ai(messages, idx):
        return {"contextualized_query": latest_text}

    history = _format_history(messages, idx)
    user_prompt = (
        f"对话历史：\n{history}\n\n"
        f"用户最新消息：\n{latest_text}\n\n"
        "请按规则改写。"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=_REWRITER_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
        rewritten = extract_text_content(response).strip().strip("`").strip()
    except Exception:
        # Rewriter is best-effort. On failure, fall back to the raw message.
        return {"contextualized_query": latest_text}

    if not rewritten:
        return {"contextualized_query": latest_text}

    # Light sanity guards: avoid the model leaking "改写：" / quote wrappers.
    for prefix in ("改写：", "改写:", "结果：", "结果:"):
        if rewritten.startswith(prefix):
            rewritten = rewritten[len(prefix):].strip()
    if rewritten.startswith("「") and rewritten.endswith("」"):
        rewritten = rewritten[1:-1].strip()
    if rewritten.startswith('"') and rewritten.endswith('"'):
        rewritten = rewritten[1:-1].strip()

    return {"contextualized_query": rewritten or latest_text}


# ---- Shared helper used by Planner / Aggregator / Critic / ReplanJudge ----

def get_user_question(state, messages: Optional[list] = None) -> str:
    """Return the question to reason about.

    Prefers `contextualized_query` (set by QueryRewriter) over the raw
    latest HumanMessage so meta-nodes see a self-contained phrasing.
    """
    cq = (state.get("contextualized_query") or "").strip()
    if cq:
        return cq
    msgs = messages if messages is not None else state.get("messages", [])
    for msg in reversed(msgs or []):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            return extract_text_content(msg)
    return ""


def get_user_message_for_planner(state) -> AnyMessage:
    """Build a HumanMessage carrying the contextualized query for Planner."""
    return HumanMessage(content=get_user_question(state))
