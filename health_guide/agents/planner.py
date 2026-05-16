"""Planner — pure LLM-based, supports dynamic replan.

Two modes (auto-detected from state):

1. **Fresh turn** (state.replan_context is empty): brand-new user message;
   produce an ordered plan, reset `executed` and `replan_count`.

2. **Replan** (state.replan_context is non-empty): one of the experts asked
   to bring in additional specialists. Read the reason, the already-executed
   list, and current expert responses, then propose an *append-only* delta
   plan (must not repeat anyone in `executed`).
"""
from typing import List, Dict

from langchain_core.messages import SystemMessage, HumanMessage

from ..llm import extract_text_content, llm
from ..profile_store import get_user_profile as _get_profile
from .fallbacks import default_fresh_plan, empty_replan
from .query_rewriter import get_user_question, get_user_message_for_planner


_VALID_EXPERTS = {"Trainer", "Nutritionist", "Wellness", "General"}

# 推荐顺序：训练量决策 → 营养匹配训练量 → 恢复匹配两者 → 通用兜底
_PRIORITY_ORDER = {"Trainer": 0, "Nutritionist": 1, "Wellness": 2, "General": 3}


def _sort_by_priority(roles: List[str]) -> List[str]:
    seen = []
    for r in roles:
        if r in _PRIORITY_ORDER and r not in seen:
            seen.append(r)
    seen.sort(key=lambda r: _PRIORITY_ORDER[r])
    return seen


_FRESH_PLAN_SYSTEM = (
    "你是健康管理团队的任务规划器。判断**当前用户消息**需要哪些专家协作，并按以下推荐顺序排列：\n"
    "Trainer → Nutritionist → Wellness → General（顺序代表先后执行，后执行的专家可以看到前面的要点）。\n"
    "可选角色: Trainer, Nutritionist, Wellness, General。\n"
    "【默认规则】优先只选 1 个最相关的专家。\n"
    "【多专家条件】仅当问题明确、同时涉及两个不同领域的核心诉求时，才选多个。\n"
    "  例如：'练完腿该吃什么' -> Trainer,Nutritionist\n"
    "  例如：'训练后睡不好' -> Trainer,Wellness\n"
    "  例如：'训练后多久睡觉比较好' -> Trainer,Wellness\n"
    "  例如：'你好/谢谢/再见' -> General\n"
    "【画像联动规则】若消息中含有「用户画像」，需结合其中的伤病和目标做路由决策：\n"
    "  - 用户有伤病记录，且问题涉及训练、运动量或体能提升 → 必须包含 Trainer\n"
    "  - 用户目标为增肌/减脂，且问题涉及饮食或营养 → 同时派 Nutritionist\n"
    "  例如：画像含「ACL 撕裂」，问「增肌吃什么」→ Trainer,Nutritionist\n"
    "  例如：画像含「减脂目标」，问「睡眠质量差」→ Wellness（无跨域需求不强制扩展）\n"
    "【历史记录联动】若消息中含有「近期对话记录」，可辅助路由：\n"
    "  - 近期记录中出现的伤病/身体问题，即使本次未明说，也应纳入路由考量\n"
    "  例如：近期问过「膝盖 ACL 术后恢复」，本次问「增肌饮食」→ Trainer,Nutritionist\n"
    "【FINISH 规则】仅当用户消息**不需要任何回答**时才输出 FINISH（比如用户只是道谢/告别且无后续问题）。\n"
    "只要用户提了任何健康相关问题，就必须选择至少一个专家而不是 FINISH。\n"
    "直接输出按执行顺序排列的角色名称，多个时用英文逗号分隔（不加空格），不要输出其他内容。"
)


_REPLAN_SYSTEM = (
    "你是健康团队的任务规划器。前一位专家已经回答，但提出了"
    "「需要其他专家补充协作」的请求。你的任务是：决定接下来要追加哪些专家。\n"
    "可选角色: Trainer, Nutritionist, Wellness, General。\n"
    "规则：\n"
    "1. 不能重复已经派出过的角色（即使他们没参与本轮也不重复派）。\n"
    "2. 只追加确实有必要的角色（默认 1 个；最多 2 个）。\n"
    "3. 如果你认为不需要再追加任何人，直接输出: NONE\n"
    "4. 否则按推荐顺序 Trainer → Nutritionist → Wellness → General 输出角色名，\n"
    "   多个时用英文逗号分隔（不加空格），不要输出其他内容。"
)


def _profile_summary(profile: dict) -> str:
    """Return a concise routing-relevant summary; empty string if nothing meaningful."""
    parts = []
    stats = profile.get("physical_stats") or {}
    if stats.get("age"):
        parts.append(f"{stats['age']}岁")
    if stats.get("weight"):
        parts.append(f"{stats['weight']}kg")
    injuries = stats.get("injuries") or []
    if injuries:
        parts.append(f"伤病：{'、'.join(str(x) for x in injuries)}")
    dietary = profile.get("dietary_context") or {}
    goal = (dietary.get("goal") or "").strip()
    if goal and goal != "健康":
        parts.append(f"目标：{goal}")
    prefs = dietary.get("preferences") or []
    if prefs:
        parts.append(f"饮食偏好：{'、'.join(str(x) for x in prefs)}")
    mental = profile.get("mental_state") or {}
    stress = mental.get("stress_sources") or []
    if stress:
        parts.append(f"压力源：{'、'.join(str(x) for x in stress)}")
    return "，".join(parts)


def _format_responses(responses: Dict[str, str]) -> str:
    if not responses:
        return "（无）"
    parts = []
    for k, v in responses.items():
        snippet = (v or "").strip()
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"
        parts.append(f"[{k}]\n{snippet}")
    return "\n\n".join(parts)


def _parse_role_list(content: str) -> List[str]:
    content = (content or "").strip().replace("'", "").replace('"', "")
    if not content:
        return []
    roles = [r.strip() for r in content.split(",")]
    return [r for r in roles if r in _VALID_EXPERTS]


def _fresh_plan(state) -> dict:
    user_id = state.get("profile_user_id", "default_user")
    profile = _get_profile(user_id)
    summary = _profile_summary(profile)

    user_question = get_user_question(state)
    episode_ctx = (state.get("episode_context") or "").strip()

    parts = []
    if summary:
        parts.append(f"用户画像：{summary}")
    if episode_ctx:
        parts.append(f"近期对话记录：\n{episode_ctx}")
    parts.append(f"用户问题：{user_question}")
    routing_content = "\n".join(parts)

    routing_msg = HumanMessage(content=routing_content)

    response = llm.invoke([SystemMessage(content=_FRESH_PLAN_SYSTEM), routing_msg])
    content = extract_text_content(response).strip().replace("'", "").replace('"', "")

    if content.upper() == "FINISH":
        return {
            "plan": [],
            "executed": [],
            "replan_count": 0,
            "replan_context": "",
            "next": ["FINISH"],
        }

    roles = _parse_role_list(content)
    plan = _sort_by_priority(roles)
    if not plan:
        plan = ["General"]
    return {
        "plan": plan,
        "executed": [],
        "replan_count": 0,
        "replan_context": "",
        "next": [],
    }


def _replan(state) -> dict:
    replan_ctx = state.get("replan_context", "")
    executed = state.get("executed") or []
    responses = state.get("expert_responses") or {}

    user_text = get_user_question(state)

    user_prompt = (
        f"用户最初的问题：\n{user_text or '（未获取到）'}\n\n"
        f"已经派出过的专家（按顺序）：{executed or '（无）'}\n\n"
        f"各专家已给出的回答摘要：\n{_format_responses(responses)}\n\n"
        f"上一位专家提出的补叫请求理由：\n{replan_ctx}\n\n"
        "请决定接下来要追加哪些专家。"
    )

    response = llm.invoke([
        SystemMessage(content=_REPLAN_SYSTEM),
        HumanMessage(content=user_prompt),
    ])
    content = extract_text_content(response).strip().replace("'", "").replace('"', "")

    if content.upper() == "NONE":
        return {
            "plan": [],
            "replan_context": "",
            "next": [],
        }

    candidates = _parse_role_list(content)
    # 去重：不能重复已执行的角色
    filtered = [r for r in _sort_by_priority(candidates) if r not in executed]
    return {
        "plan": filtered,
        "replan_context": "",
        "next": [],
    }


def planner_node(state):
    try:
        if state.get("replan_context"):
            return _replan(state)
        return _fresh_plan(state)
    except Exception:
        if state.get("replan_context"):
            return empty_replan()
        return default_fresh_plan()
