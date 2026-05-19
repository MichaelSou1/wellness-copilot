from langchain_core.messages import SystemMessage, HumanMessage
import re

from ..personalization import (
    apply_personalization_boost,
    build_personalization_decision_points,
    format_decision_points_for_prompt,
)
from ..llm import extract_text_content, llm
from .doctor import ensure_doctor_disclaimer
from .fallbacks import aggregate_fallback
from .query_rewriter import get_user_question

_EXPERT_DOMAIN_LABELS = {
    "Trainer":      "训练与运动恢复",
    "Nutritionist": "饮食与营养",
    "Psychologist":     "心理支持与情绪调节",
    "Doctor":       "医学建议与就医边界",
    "Orchestrator": "综合健康常识",
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
- 必须保留各专家已引用的画像数值，不得为了顺口改写成「根据你的情况」
- 若某条建议由年龄、体重、身高、BMI、目标或压力源推导而来，必须同时保留原始画像值和推导结果；
  例如不要只写「心率114-133次/分钟」，要写「按你30岁估算，心率约114-133次/分钟」
- 必须保留所有伤病、过敏、饮食禁忌等硬约束，不得软化或省略
- 如果参考建议中包含 Doctor/医学建议，最终回答必须保留「仅供参考，如有不适请就医」
"""

_SYNTHESIS_TEMPLATE = """\
{user_card}

{decision_section}

用户的问题：
{user_question}

以下是从不同专业维度整理的参考建议（仅供你整合用，不要原文照抄或提及维度标签）：

{expert_sections}

请根据上述参考内容，直接回答用户的问题。整合时遵循以下结构：
1. 首句直接给出核心判断或整体方向（不要用"根据…""综合来看…"之类的引言）
2. 按逻辑顺序展开各维度的具体建议，用"在此基础上""配合""同样值得注意的是"等过渡语句衔接
3. 结尾用 1～2 句话给出最关键的行动要点
"""


def _num(value):
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _fmt(value, suffix=""):
    n = _num(value)
    if n is None:
        return ""
    return f"{int(n) if n.is_integer() else round(n, 1)}{suffix}"


def _ensure_final_anchors(answer: str, pctx: dict, user_question: str) -> str:
    text = apply_personalization_boost(answer, pctx, user_question, max_notes=3)
    q = user_question or ""
    if re.search(r"(?<!\d)10\s*(?:K|公里)(?!g)", q, re.IGNORECASE) and "10公里" not in (text or ""):
        if re.match(r"^\s*赛前一周", text or ""):
            return re.sub(r"^\s*赛前一周", "这场10公里跑的赛前一周", text, count=1)
        return f"这场10公里跑，{text}"
    return text


def _recent_user_context(state) -> str:
    texts = []
    for msg in state.get("messages") or []:
        if getattr(msg, "type", "") != "human":
            continue
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            texts.append(content.strip())
    return "\n".join(texts[-3:])


def _decision_points_for_roles(pctx: dict, user_question: str, roles: list[str]) -> list:
    points = []
    seen = set()
    for role in roles:
        for point in build_personalization_decision_points(pctx, user_question, role=role):
            if point.id in seen:
                continue
            seen.add(point.id)
            points.append(point)
    return sorted(points, key=lambda p: (p.priority, p.domain, p.id))


def aggregator_node(state):
    """Produce draft_answer for Critic to review (no messages append here)."""
    # plan-and-execute: 用 executed (本轮已执行的专家顺序列表) 而非 next
    current_experts = [
        role for role in (state.get("executed") or state.get("next", []))
        if role != "Orchestrator"
    ]
    all_responses = state.get("expert_responses", {})
    relevant = {k: v for k, v in all_responses.items() if k in current_experts}

    if not relevant:
        return {"draft_answer": "抱歉，未能获取专家建议，请重试。"}

    has_doctor = "Doctor" in relevant
    user_question = get_user_question(state) or ""
    question_context = "\n".join(
        item for item in (user_question, _recent_user_context(state)) if item
    )

    if len(relevant) == 1:
        pctx = state.get("personalization_ctx") or {}
        draft = _ensure_final_anchors(next(iter(relevant.values())), pctx, question_context)
        if has_doctor:
            draft = ensure_doctor_disclaimer(draft)
        return {"draft_answer": draft}

    pctx = state.get("personalization_ctx") or {}
    user_card = pctx.get("user_card") or "【关于该用户】\n用户画像暂不可用。"
    decision_section = format_decision_points_for_prompt(
        _decision_points_for_roles(pctx, user_question or "", current_experts)
    )

    expert_sections = "\n\n".join(
        f"[{_EXPERT_DOMAIN_LABELS.get(k, k)}]\n{v}"
        for k, v in relevant.items()
    )

    synthesis_prompt = _SYNTHESIS_TEMPLATE.format(
        user_card=user_card,
        decision_section=decision_section,
        user_question=user_question or "（未获取到原始问题）",
        expert_sections=expert_sections,
    )

    try:
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=synthesis_prompt),
        ])
        draft = _ensure_final_anchors(extract_text_content(response), pctx, question_context)
    except Exception:
        draft = _ensure_final_anchors(aggregate_fallback(current_experts, relevant), pctx, question_context)
    if has_doctor:
        draft = ensure_doctor_disclaimer(draft)
    return {"draft_answer": draft}
