from langchain_core.messages import SystemMessage, HumanMessage
from ..llm import extract_text_content, llm
from .fallbacks import aggregate_fallback
from .query_rewriter import get_user_question

_EXPERT_DOMAIN_LABELS = {
    "Trainer":      "训练与运动恢复",
    "Nutritionist": "饮食与营养",
    "Wellness":     "睡眠、压力与身心恢复",
    "General":      "综合健康常识",
}

_SYSTEM_PROMPT = """\
你是一位综合健康顾问，擅长将来自不同专业领域的建议整合为一份自然流畅的回答。

写作原则：
- 以统一的第一人称视角写作，读起来像一个人说话，而不是"汇报各专家的意见"
- 不要在正文中出现任何专家名称、角色头衔或来源标注
- 不同专业维度之间用过渡语句自然衔接，而非分块罗列
- 内容有层次：先给整体方向或核心判断，再展开具体建议，最后如有必要点出注意事项
- 如果各维度建议有重叠，只保留一次，取最完整的表达
- 如果建议有矛盾，选择更保守、更安全的一方并简要说明原因
"""

_SYNTHESIS_TEMPLATE = """\
用户的问题：
{user_question}

以下是从不同专业维度整理的参考建议（仅供你整合用，不要原文照抄或提及维度标签）：

{expert_sections}

请根据上述参考内容，直接回答用户的问题。整合时遵循以下结构：
1. 首句直接给出核心判断或整体方向（不要用"根据…""综合来看…"之类的引言）
2. 按逻辑顺序展开各维度的具体建议，用"在此基础上""配合""同样值得注意的是"等过渡语句衔接
3. 结尾用 1～2 句话给出最关键的行动要点
"""


def aggregator_node(state):
    """Produce draft_answer for Critic to review (no messages append here)."""
    # plan-and-execute: 用 executed (本轮已执行的专家顺序列表) 而非 next
    current_experts = state.get("executed") or state.get("next", [])
    all_responses = state.get("expert_responses", {})
    relevant = {k: v for k, v in all_responses.items() if k in current_experts}

    if not relevant:
        return {"draft_answer": "抱歉，未能获取专家建议，请重试。"}

    if len(relevant) == 1:
        return {"draft_answer": next(iter(relevant.values()))}

    user_question = get_user_question(state)

    expert_sections = "\n\n".join(
        f"[{_EXPERT_DOMAIN_LABELS.get(k, k)}]\n{v}"
        for k, v in relevant.items()
    )

    synthesis_prompt = _SYNTHESIS_TEMPLATE.format(
        user_question=user_question or "（未获取到原始问题）",
        expert_sections=expert_sections,
    )

    try:
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=synthesis_prompt),
        ])
        return {"draft_answer": extract_text_content(response)}
    except Exception:
        return {"draft_answer": aggregate_fallback(current_experts, relevant)}
