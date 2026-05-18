"""Planner — pure LLM-based, supports dynamic replan.

Two modes (auto-detected from state):

1. **Fresh turn** (state.replan_context is empty): brand-new user message;
   produce an ordered plan, reset `executed` and `replan_count`.

2. **Replan** (state.replan_context is non-empty): one of the experts asked
   to bring in additional specialists. Read the reason, the already-executed
   list, and current expert responses, then propose an *append-only* delta
   plan (must not repeat anyone in `executed`).
"""
import re
from typing import List, Dict

from langchain_core.messages import SystemMessage, HumanMessage

from ..llm import extract_text_content, llm
from ..profile_store import get_user_profile as _get_profile
from .fallbacks import default_fresh_plan, empty_replan
from .query_rewriter import get_user_question, get_user_message_for_planner


# ---- Deterministic routing rules ------------------------------------------
# When the LLM-based planner misses a domain that profile-context obviously
# implies (e.g., user has an active ACL injury and asks about diet — Trainer
# still has training-safety implications), patch the plan post hoc.

_TRAINING_INTENT = re.compile(
    r"训练|锻炼|健身|动作|计划|频次|频率|强度|"
    r"减脂|增肌|塑形|燃脂|瘦身|减重|增重|健身房|"
    r"有氧|力量|跑步|跑量|游泳|骑行|跳绳|HIIT|拉伸|"
    r"运动量|活动量|腿|腰|肩|背|核心|腹|胸|臂",
    re.IGNORECASE,
)

_FOOD_INTENT = re.compile(
    r"吃|食|餐|喝|饮食|营养|蛋白|碳水|脂肪|热量|kcal|大卡|"
    r"零食|早餐|午餐|晚餐|补剂|蛋白粉|奶|蔬菜|水果|食谱|菜谱",
    re.IGNORECASE,
)

_WELLNESS_INTENT = re.compile(
    r"睡|失眠|入睡|压力|焦虑|抑郁|情绪|疲劳|倦怠|心情|放松|休息|恢复",
    re.IGNORECASE,
)

# Calculator-style training topics that should always involve Trainer even
# when the LLM is tempted to route to Nutritionist on the word "热量"/"calorie".
_TRAINER_ONLY_TOPICS = re.compile(
    r"TDEE|BMR|基础代谢|总能量消耗|maintenance.?calories|代谢率",
    re.IGNORECASE,
)

# Cardiac / acute-symptom-during-exercise signals — needs Trainer (for the
# "stop training" guidance) AND General/Wellness (escalate-to-doctor framing).
_CARDIAC_EXERCISE_SIGNAL = re.compile(
    r"(?:运动|训练|跑步|健身|锻炼).{0,8}(?:胸闷|胸痛|胸口闷|胸口痛|心悸|心慌|头晕|晕|喘不上)"
    r"|(?:胸闷|胸痛|胸口闷|胸口痛|心悸|心慌|头晕|晕|喘不上).{0,8}(?:运动|训练|跑步|健身|锻炼)",
    re.IGNORECASE,
)


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
    "【画像联动规则·强制】若消息中含有「用户画像」，必须严格按下列规则扩展专家集合：\n"
    "  R1 [伤病联动]：画像中 injuries 不为空（任何伤病/术后/康复状态）时，"
    "若用户问题涉及训练、运动、动作、计划、强度、减脂/增肌/塑形等任何会改变身体负荷的诉求 → **必须包含 Trainer**\n"
    "    例如：画像含「ACL 术后」，问「减脂每天吃多少」→ Trainer,Nutritionist（Trainer 必须出现，"
    "    因为减脂目标会影响训练计划，且术后训练有禁忌）\n"
    "    例如：画像含「半月板损伤」，问「想增肌该怎么吃」→ Trainer,Nutritionist\n"
    "  R2 [饮食禁忌联动]：画像中 dietary_context.preferences 含过敏/禁忌/特殊饮食（如花生过敏、纯素、乳糖不耐）时，"
    "若用户问题涉及吃喝、零食、餐食、营养补充 → **必须包含 Nutritionist**\n"
    "  R3 [慢病联动]：画像中含心脏病/冠心病/糖尿病/高血压等慢病提及时，"
    "若问题涉及高强度运动 → **必须包含 Trainer**\n"
    "【历史记录联动】若消息中含有「近期对话记录」，可辅助路由：\n"
    "  - 近期记录中出现的伤病/身体问题，即使本次未明说，也应纳入路由考量\n"
    "  例如：近期问过「膝盖 ACL 术后恢复」，本次问「增肌饮食」→ Trainer,Nutritionist\n"
    "【FINISH 规则】FINISH 极少使用。只有用户消息是纯粹的告别（如'再见''拜拜''bye'）且无任何其他内容时才输出 FINISH。\n"
    "  - '谢谢' / '辛苦了' / '挺有帮助' / '我懂了' 等表达感谢或反馈的消息 → General（回应即可，不要 FINISH）\n"
    "  - 任何健康相关问题 → 必须选择至少一个专家\n"
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


def _profile_has_injury(profile: dict) -> bool:
    stats = profile.get("physical_stats") or {}
    return bool(stats.get("injuries"))


def _profile_has_dietary_restriction(profile: dict) -> bool:
    dietary = profile.get("dietary_context") or {}
    prefs = dietary.get("preferences") or []
    if not prefs:
        return False
    restriction_hits = re.compile(r"过敏|不耐|纯素|素食|忌|不吃|不喝|无糖|低糖|无麸质")
    for p in prefs:
        if restriction_hits.search(str(p)):
            return True
    return False


def _profile_has_chronic_condition(profile: dict, summary: str) -> bool:
    return bool(re.search(r"心脏|冠心|糖尿|高血压|哮喘|肾病", summary))


def _enforce_profile_rules(plan: List[str], profile: dict, profile_summary: str, query: str) -> List[str]:
    """Apply R1/R2/R3 deterministically to recover from LLM misses.

    Looks at the merged ``query + profile_summary`` for domain-intent keywords
    so injury context referenced only in the profile still triggers Trainer.
    """
    text = f"{query or ''}\n{profile_summary or ''}"
    plan = list(plan)

    if _profile_has_injury(profile):
        if _TRAINING_INTENT.search(text) and "Trainer" not in plan:
            plan.append("Trainer")
        if _profile_has_dietary_restriction(profile) and _FOOD_INTENT.search(text) and "Nutritionist" not in plan:
            plan.append("Nutritionist")

    if _profile_has_dietary_restriction(profile):
        if _FOOD_INTENT.search(text) and "Nutritionist" not in plan:
            plan.append("Nutritionist")

    if _profile_has_chronic_condition(profile, profile_summary):
        if _TRAINING_INTENT.search(text) and "Trainer" not in plan:
            plan.append("Trainer")

    # R4 [Topic override]: TDEE/BMR/基础代谢 questions belong to Trainer (the
    # node that holds the calculate_tdee tool), even if the wording mentions
    # "热量" and the LLM gravitates to Nutritionist.
    if _TRAINER_ONLY_TOPICS.search(text) and "Trainer" not in plan:
        plan.append("Trainer")

    # R5 [Cardiac-during-exercise]: needs both Trainer (stop training) and
    # General (escalate to doctor framing).
    cardiac_signal = bool(_CARDIAC_EXERCISE_SIGNAL.search(text))
    if cardiac_signal:
        if "Trainer" not in plan:
            plan.append("Trainer")
        if "General" not in plan:
            plan.append("General")

    # Drop the catch-all General if a specialist now covers the request.
    # Exception: when R5 (cardiac+exercise) explicitly demanded General, keep it.
    specialists = [r for r in plan if r in {"Trainer", "Nutritionist", "Wellness"}]
    if specialists and "General" in plan and not cardiac_signal:
        plan = [r for r in plan if r != "General"]

    return _sort_by_priority(plan)


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
    # Deterministic post-process: enforce profile-driven multi-expert rules.
    plan = _enforce_profile_rules(plan, profile, summary, user_question)
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
