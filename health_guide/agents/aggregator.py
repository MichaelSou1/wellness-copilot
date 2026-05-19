from langchain_core.messages import SystemMessage, HumanMessage
import re

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
- 必须保留所有伤病、过敏、饮食禁忌等硬约束，不得软化或省略
- 如果参考建议中包含 Doctor/医学建议，最终回答必须保留「仅供参考，如有不适请就医」
"""

_SYNTHESIS_TEMPLATE = """\
{user_card}

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
    profile = pctx.get("raw_profile") or {}
    stats = profile.get("physical_stats") or {}
    dietary = profile.get("dietary_context") or {}
    mental = profile.get("mental_state") or {}
    text = answer or ""
    prefix = []

    anchors = [
        _fmt(stats.get("age"), "岁"),
        _fmt(stats.get("weight"), "kg"),
        _fmt(stats.get("height"), "cm"),
    ]
    anchors = [a for a in anchors if a]
    injuries = [str(x).strip() for x in (stats.get("injuries") or []) if str(x).strip()]
    stress = [str(x).strip() for x in (mental.get("stress_sources") or []) if str(x).strip()]
    goal = str(dietary.get("goal") or "").strip()

    if anchors and not any(a.rstrip("岁kgcm") in text for a in anchors):
        prefix.append(f"按你目前 {', '.join(anchors)}" + (f"、目标{goal}" if goal and goal != "健康" else "") + "，下面的建议按可执行数字落地。")
    if injuries and not any(i in text for i in injuries):
        prefix.append(f"同时要保留你的伤病限制：{', '.join(injuries)}，任何诱发疼痛或不稳定的动作都先暂停。")
    if stress and not any(s in text for s in stress):
        prefix.append(f"你的压力源包括 {', '.join(stress)}，恢复建议需要贴合这些具体场景。")

    q = user_question or ""
    endurance_race = re.search(
        r"10\s*(?:k|公里)|5\s*(?:k|公里)|半马|马拉松|越野赛|备赛|跑量|补给|碳水加载|跑步.{0,6}比赛|比赛.{0,6}跑",
        q,
        re.IGNORECASE,
    )
    if endurance_race:
        if "补给" not in text:
            prefix.append("比赛补给也要提前练习：赛前 1.5-2 小时吃易消化碳水，10K 中后段按天气和用时安排水、电解质或能量胶。")
        if "跑量" not in text:
            prefix.append("跑量上不要临近比赛硬堆量，赛前一周应明显减量，把状态留到比赛日。")

    if re.search(r"熬夜|睡不好|睡眠|入睡|比赛.*紧张|紧张.*比赛", q):
        if "睡眠" not in text and "入睡" not in text and "作息" not in text:
            prefix.append("睡眠是这轮恢复的优先项：固定起床时间，睡前 30-60 分钟降刺激。")
        if re.search(r"比赛|紧张", q) and "焦虑" not in text:
            prefix.append("赛前焦虑很常见，目标是通过减量、呼吸放松和固定作息把兴奋度压到可控范围。")

    if not prefix:
        return answer
    return "\n\n".join(prefix + [answer])


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

    if len(relevant) == 1:
        pctx = state.get("personalization_ctx") or {}
        draft = _ensure_final_anchors(next(iter(relevant.values())), pctx, get_user_question(state) or "")
        if has_doctor:
            draft = ensure_doctor_disclaimer(draft)
        return {"draft_answer": draft}

    user_question = get_user_question(state)
    pctx = state.get("personalization_ctx") or {}
    user_card = pctx.get("user_card") or "【关于该用户】\n用户画像暂不可用。"

    expert_sections = "\n\n".join(
        f"[{_EXPERT_DOMAIN_LABELS.get(k, k)}]\n{v}"
        for k, v in relevant.items()
    )

    synthesis_prompt = _SYNTHESIS_TEMPLATE.format(
        user_card=user_card,
        user_question=user_question or "（未获取到原始问题）",
        expert_sections=expert_sections,
    )

    try:
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=synthesis_prompt),
        ])
        draft = _ensure_final_anchors(extract_text_content(response), pctx, user_question or "")
    except Exception:
        draft = _ensure_final_anchors(aggregate_fallback(current_experts, relevant), pctx, user_question or "")
    if has_doctor:
        draft = ensure_doctor_disclaimer(draft)
    return {"draft_answer": draft}
