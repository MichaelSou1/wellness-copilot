"""Personalization context rendering.

This module is the single prompt-facing source of truth for the user's long
term profile. TurnStart builds the context once per turn, then downstream
nodes read it from state instead of reloading profile JSON independently.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .profile_store import get_user_profile, profile_subset_for, profile_to_prompt_text


_DEFAULT_NAMES = {"", "user", "用户", "default_user"}
_DEFAULT_IDENTITIES = {"", "用户"}
_DEFAULT_GOALS = {"", "健康"}


def _as_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _num(value: Any) -> float | None:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _fmt_num(value: float | int, suffix: str = "") -> str:
    if float(value).is_integer():
        return f"{int(value)}{suffix}"
    return f"{value:.1f}{suffix}"


def _bmi(height_cm: Any, weight_kg: Any) -> float | None:
    h = _num(height_cm)
    w = _num(weight_kg)
    if not h or not w:
        return None
    meters = h / 100
    if meters <= 0:
        return None
    return round(w / (meters * meters), 1)


def _meaningful_name(profile: Dict[str, Any]) -> str:
    name = str(profile.get("name") or "").strip()
    return "" if name.lower() in _DEFAULT_NAMES else name


def _meaningful_identity(profile: Dict[str, Any]) -> str:
    identity = str(profile.get("identity") or "").strip()
    return "" if identity.lower() in _DEFAULT_IDENTITIES else identity


def _goal(profile: Dict[str, Any]) -> str:
    dietary = profile.get("dietary_context") or {}
    return str(dietary.get("goal") or "").strip()


def is_meaningful(profile: Dict[str, Any]) -> bool:
    """Return True when the profile contains user-provided signal."""
    if not isinstance(profile, dict):
        return False
    stats = profile.get("physical_stats") or {}
    dietary = profile.get("dietary_context") or {}
    mental = profile.get("mental_state") or {}
    style = profile.get("response_style") or {}

    if _meaningful_name(profile) or _meaningful_identity(profile):
        return True
    if any(_num(stats.get(k)) for k in ("age", "height", "weight")):
        return True
    if _as_list(stats.get("injuries")):
        return True
    if _goal(profile) not in _DEFAULT_GOALS:
        return True
    if _as_list(dietary.get("preferences")):
        return True
    provider = str(dietary.get("provider") or "").strip()
    if provider and provider.lower() != "self":
        return True
    if _as_list(mental.get("stress_sources")):
        return True
    if str(mental.get("relaxation_preference") or "").strip():
        return True
    if any(str(v or "").strip() for v in style.values()) if isinstance(style, dict) else False:
        return True
    return False


def profile_routing_digest(profile: Dict[str, Any]) -> str:
    """Concise routing-relevant summary for Orchestrator rules."""
    parts: List[str] = []
    name = _meaningful_name(profile)
    identity = _meaningful_identity(profile)
    if name:
        parts.append(f"姓名：{name}")
    if identity:
        parts.append(f"身份：{identity}")

    stats = profile.get("physical_stats") or {}
    if _num(stats.get("age")):
        parts.append(f"{_fmt_num(_num(stats.get('age')))}岁")
    if _num(stats.get("height")):
        parts.append(f"{_fmt_num(_num(stats.get('height')), 'cm')}")
    if _num(stats.get("weight")):
        parts.append(f"{_fmt_num(_num(stats.get('weight')), 'kg')}")
    bmi = _bmi(stats.get("height"), stats.get("weight"))
    if bmi:
        parts.append(f"BMI {bmi}")
    injuries = _as_list(stats.get("injuries"))
    if injuries:
        parts.append(f"伤病：{'、'.join(injuries)}")

    dietary = profile.get("dietary_context") or {}
    goal = _goal(profile)
    if goal and goal != "健康":
        parts.append(f"目标：{goal}")
    prefs = _as_list(dietary.get("preferences"))
    if prefs:
        parts.append(f"饮食偏好/限制：{'、'.join(prefs)}")

    mental = profile.get("mental_state") or {}
    stress = _as_list(mental.get("stress_sources"))
    if stress:
        parts.append(f"压力源：{'、'.join(stress)}")
    relaxation = str(mental.get("relaxation_preference") or "").strip()
    if relaxation:
        parts.append(f"放松偏好：{relaxation}")

    style = profile.get("response_style") or {}
    if isinstance(style, dict):
        style_bits = [str(v).strip() for v in style.values() if str(v or "").strip()]
        if style_bits:
            parts.append(f"回答风格：{'、'.join(style_bits)}")
    return "，".join(parts)


def _describe_injury_constraint(injury: str) -> str:
    text = str(injury or "").strip()
    low = text.lower()
    if not text:
        return ""

    if re.search(r"acl|前交叉|十字韧带|韧带", low, re.IGNORECASE):
        return (
            f"{text}：禁止跑步、跳跃、急停变向、深蹲/弓步等膝关节高剪切或高冲击动作；"
            "下肢训练须在医生或理疗师评估许可下分阶段推进。"
        )
    if "半月板" in text:
        return (
            f"{text}：避免深蹲、跳跃、跑步、扭转和大重量屈膝负荷；"
            "低冲击替代方案也须在理疗师许可下渐进。"
        )
    if re.search(r"冠心|心脏|心血管|心肌|心梗", text):
        return (
            f"{text}：禁止自行开始 HIIT、冲刺或极限强度训练；训练前需医生运动风险评估，"
            "出现胸闷、胸痛、心悸、头晕应停止并就医。"
        )
    if re.search(r"腰椎|腰间盘|椎间盘|下背|腰痛", text):
        return (
            f"{text}：避免大重量硬拉、深蹲、负重扭转和疼痛诱发动作；"
            "核心/髋主导训练需低负荷、无痛并经专业评估。"
        )
    if re.search(r"肩袖|肩峰|肩关节|肩", text):
        return (
            f"{text}：避免过顶推举、爆发推拉和疼痛范围训练；"
            "上肢动作需降低负荷并在无痛范围内推进。"
        )
    if re.search(r"踝|脚踝", text):
        return (
            f"{text}：避免跳跃、冲刺、急停变向和不稳定地面高负荷；"
            "从低冲击有氧和本体感觉训练开始。"
        )
    if "膝" in text:
        return (
            f"{text}：避免跑跳、深蹲、弓步和大重量膝主导动作；"
            "低冲击替代动作须在理疗师许可下进行。"
        )
    if re.search(r"术后|康复|恢复中", text):
        return (
            f"{text}：禁止自行增加强度或做疼痛诱发动作；"
            "训练需遵守医生/理疗师给出的阶段性限制。"
        )
    return (
        f"{text}：训练和饮食建议需以不加重症状为前提；涉及疼痛部位负荷的动作须先获医生或理疗师许可。"
    )


def _dietary_constraint(preferences: Iterable[str]) -> List[str]:
    prefs = [p for p in preferences if p]
    if not prefs:
        return []
    allergies = []
    avoid = []
    soft = []
    allergy_pat = re.compile(r"过敏|不耐|乳糖|无麸质")
    avoid_pat = re.compile(r"忌|禁忌|不吃|不喝|避免|纯素|素食")
    for pref in prefs:
        if allergy_pat.search(pref):
            allergies.append(pref)
        elif avoid_pat.search(pref):
            avoid.append(pref)
        else:
            soft.append(pref)
    lines = []
    if allergies:
        lines.append(
            f"饮食过敏/不耐：{'、'.join(allergies)}；推荐食材和补剂不得包含这些项目，并提醒交叉污染风险。"
        )
    if avoid:
        lines.append(f"饮食禁忌/限制：{'、'.join(avoid)}；推荐餐食时避开，但不要把它称为过敏。")
    if soft:
        lines.append(f"饮食偏好：{'、'.join(soft)}；安排餐食时优先适配。")
    return lines


def render_active_constraints(profile: Dict[str, Any]) -> List[str]:
    constraints: List[str] = []
    stats = profile.get("physical_stats") or {}
    for injury in _as_list(stats.get("injuries")):
        desc = _describe_injury_constraint(injury)
        if desc:
            constraints.append(desc)

    dietary = profile.get("dietary_context") or {}
    constraints.extend(_dietary_constraint(_as_list(dietary.get("preferences"))))

    goal = _goal(profile)
    weight = _num(stats.get("weight"))
    if goal == "增肌":
        if weight:
            low = int(round(weight * 1.6))
            high = int(round(weight * 2.2))
            constraints.append(
                f"增肌目标：蛋白质可先按 1.6-2.2g/kg/天估算，当前体重 {_fmt_num(weight, 'kg')} 对应约 {low}-{high}g/天；热量通常从 TDEE + 200-300 kcal 起。"
            )
        else:
            constraints.append("增肌目标：需优先补齐体重/活动量，再估算蛋白质和热量盈余。")
    elif goal == "减脂":
        if weight:
            low = int(round(weight * 1.6))
            high = int(round(weight * 2.2))
            constraints.append(
                f"减脂目标：蛋白质可先按 1.6-2.2g/kg/天估算，当前体重 {_fmt_num(weight, 'kg')} 对应约 {low}-{high}g/天；热量赤字优先控制在 300-500 kcal/天。"
            )
        else:
            constraints.append("减脂目标：热量赤字优先控制在 300-500 kcal/天，避免极端节食。")
    return constraints


def _render_style(profile: Dict[str, Any]) -> str:
    style = profile.get("response_style") or {}
    if not isinstance(style, dict):
        return ""
    labels = {
        "tone": "语气",
        "humor": "幽默程度",
        "formality": "正式程度",
        "language": "语言",
    }
    parts = []
    for key in ("tone", "humor", "formality", "language"):
        value = str(style.get(key) or "").strip()
        if value:
            parts.append(f"{labels[key]}：{value}")
    return "；".join(parts)


def render_user_card(profile: Dict[str, Any]) -> str:
    if not is_meaningful(profile):
        return "【关于该用户】\n该用户的个人信息暂未填写；请在恰当处主动询问关键数据，不要臆测。"

    stats = profile.get("physical_stats") or {}
    dietary = profile.get("dietary_context") or {}
    mental = profile.get("mental_state") or {}

    name = _meaningful_name(profile) or "该用户"
    identity = _meaningful_identity(profile)
    descriptors: List[str] = []
    age = _num(stats.get("age"))
    height = _num(stats.get("height"))
    weight = _num(stats.get("weight"))
    if identity:
        descriptors.append(identity)
    if age:
        descriptors.append(f"{_fmt_num(age)} 岁")
    if height:
        descriptors.append(f"身高 {_fmt_num(height, 'cm')}")
    if weight:
        descriptors.append(f"体重 {_fmt_num(weight, 'kg')}")
    bmi = _bmi(height, weight)
    if bmi:
        descriptors.append(f"BMI {bmi}")
    goal = str(dietary.get("goal") or "").strip()
    if goal:
        descriptors.append(f"目标定位为「{goal}」")

    about = f"你目前对话的对象是 {name}"
    if descriptors:
        about += "，" + "，".join(descriptors)
    about += "。"

    stress = _as_list(mental.get("stress_sources"))
    relaxation = str(mental.get("relaxation_preference") or "").strip()
    if stress or relaxation:
        mental_bits = []
        if stress:
            mental_bits.append(f"压力源：{'、'.join(stress)}")
        if relaxation:
            mental_bits.append(f"放松偏好：{relaxation}")
        about += "\n身心背景：" + "；".join(mental_bits) + "。"

    constraints = render_active_constraints(profile)
    if constraints:
        constraint_text = "\n".join(f"- {line}" for line in constraints)
    else:
        constraint_text = "- 暂无已知伤病、过敏或饮食硬性限制；未提供的信息不要臆测。"

    card = f"【关于该用户】\n{about}\n\n【必须遵守的个性化约束】\n{constraint_text}"
    style_text = _render_style(profile)
    if style_text:
        card += f"\n\n【风格偏好】\n{style_text}"
    return card


def build_personalization_ctx(user_id: str) -> Dict[str, Any]:
    """Build the per-turn personalization context for state."""
    profile = get_user_profile(user_id)
    raw_json = profile_to_prompt_text(profile)
    constraints = render_active_constraints(profile)
    role_user_cards = {
        role: render_user_card(profile_subset_for(role, profile))
        for role in ("Trainer", "Nutritionist", "Psychologist", "Doctor", "Orchestrator")
    }
    return {
        "user_card": render_user_card(profile),
        "role_user_cards": role_user_cards,
        "active_constraints": constraints,
        "routing_digest": profile_routing_digest(profile),
        "raw_profile": profile,
        "raw_profile_json": raw_json,
        "has_meaningful_data": is_meaningful(profile),
    }


def profile_anchor_terms(profile: Dict[str, Any]) -> List[str]:
    """Terms Critic can use to judge whether personalization surfaced."""
    terms: List[str] = []
    stats = profile.get("physical_stats") or {}
    for key in ("age", "weight", "height"):
        value = _num(stats.get(key))
        if value:
            terms.append(_fmt_num(value))
            if key == "weight":
                terms.append(_fmt_num(value, "kg"))
            elif key == "height":
                terms.append(_fmt_num(value, "cm"))
    for injury in _as_list(stats.get("injuries")):
        terms.append(injury)
        if re.search(r"acl|前交叉|韧带", injury, re.IGNORECASE):
            terms.extend(["ACL", "韧带"])
        if "半月板" in injury:
            terms.append("半月板")
    dietary = profile.get("dietary_context") or {}
    terms.extend(_as_list(dietary.get("preferences")))
    goal = _goal(profile)
    if goal and goal not in _DEFAULT_GOALS:
        terms.append(goal)
    mental = profile.get("mental_state") or {}
    terms.extend(_as_list(mental.get("stress_sources")))
    return [t for t in terms if t]


_ADVICE_SIGNAL = re.compile(
    r"建议|方案|计划|怎么|如何|能不能|可以吗|应该|多少|安排|推荐|吃|练|训练|运动|"
    r"恢复|减脂|增肌|睡|压力|焦虑|疼|痛|不适|用药|补剂|比赛|备赛|[？?]",
    re.IGNORECASE,
)

_PURE_CHITCHAT_SIGNAL = re.compile(r"^(你好|您好|再见|拜拜|谢谢|感谢|今天天气|天气真好)[。！!.\s]*$")
_ENDURANCE_RACE_SIGNAL = re.compile(
    r"(?<!\d)(?:10|5)\s*(?:K|公里)(?!g).{0,20}(?:比赛|备赛|补给|赛前|赛中|赛后)|"
    r"(?:比赛|备赛|补给|赛前|赛中|赛后).{0,20}(?<!\d)(?:10|5)\s*(?:K|公里)(?!g)|"
    r"半马|马拉松|比赛|备赛|补给|赛前|赛中|赛后|碳水加载",
    re.IGNORECASE,
)
_RUNNING_BEGINNER_SIGNAL = re.compile(r"从零开始跑|跑5公里|5\s*公里|4周入门|跑步入门", re.IGNORECASE)
_CRISIS_SIGNAL = re.compile(r"活着没意思|不想活|轻生|自杀|自伤|结束生命|想死|消失算了")
_MEDICAL_REDFLAG_SIGNAL = re.compile(
    r"浓茶|茶色尿|酱油色尿|血压|血糖|胸痛|胸闷|胸口压迫|压迫感|呼吸困难|晕厥|静息心率|心率异常|"
    r"布洛芬|对乙酰氨基酚|阿司匹林|抗生素|处方|剂量|继续用药|是什么病|诊断|急诊",
    re.IGNORECASE,
)
_TDEE_ESTIMATE_SIGNAL = re.compile(r"TDEE|BMR|基础代谢|每日.*消耗|消耗.*热量|总消耗", re.IGNORECASE)
_DIABETES_SIGNAL = re.compile(r"糖尿病|血糖", re.IGNORECASE)
_HIIT_SIGNAL = re.compile(r"HIIT|高强度|间歇", re.IGNORECASE)


@dataclass(frozen=True)
class PersonalizationDecisionPoint:
    id: str
    domain: str
    priority: int
    instruction: str
    evidence_terms: List[str]
    source_profile_keys: List[str]


def _as_answer_text(answer: str | None) -> str:
    return (answer or "").strip()


def _contains_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _fmt_liters(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _extract_target_weight(text: str) -> float | None:
    match = re.search(r"(?:减到|降到|瘦到|到)\s*(\d+(?:\.\d+)?)\s*kg", text, re.IGNORECASE)
    if not match:
        match = re.search(r"(\d+(?:\.\d+)?)\s*kg", text, re.IGNORECASE)
    if not match:
        return None
    return _num(match.group(1))


def _underweight_weight_goal_answer(profile: Dict[str, Any], user_question: str) -> str:
    stats = profile.get("physical_stats") or {}
    dietary = profile.get("dietary_context") or {}
    mental = profile.get("mental_state") or {}
    weight = _num(stats.get("weight"))
    height = _num(stats.get("height"))
    age = _num(stats.get("age"))
    if not weight or not height:
        return ""
    bmi = _bmi(height, weight)
    target = _extract_target_weight(user_question or "")
    target_bmi = round(target / ((height / 100) ** 2), 1) if target else None
    if not bmi or bmi >= 18.5 and (not target_bmi or target_bmi >= 18.5):
        return ""
    if not re.search(r"减到|降到|瘦到|减脂|减重|体重|40\s*kg", user_question or "", re.IGNORECASE):
        return ""

    protein_low = int(round(weight * 1.2))
    protein_high = int(round(weight * 1.6))
    min_healthy_weight = round(18.5 * ((height / 100) ** 2), 1)
    target_text = f"；如果把目标设为 {_fmt_num(target, 'kg')}，BMI 会到 {target_bmi}" if target_bmi else ""
    age_text = f"{_fmt_num(age, '岁')}、" if age else ""
    goal = str(dietary.get("goal") or "").strip()
    stress = "、".join(_as_list(mental.get("stress_sources")))
    stress_text = f"你记录的压力源是 {stress}，这会让体重数字更容易变成焦虑焦点。" if stress else ""
    goal_text = f"虽然画像目标写着「{goal}」，但这个目标需要先让位于安全。" if goal and goal != "健康" else ""

    return (
        f"按你目前 {age_text}{_fmt_num(height, 'cm')}、{_fmt_num(weight, 'kg')} 计算，BMI 约 {bmi}，已经低于 18.5 的健康下限"
        f"{target_text}。我不建议再降体重，也不能支持把目标设为更低体重；当前更合理的方向是体重维持、营养恢复和体象/体重焦虑评估。\n\n"
        f"{goal_text}{stress_text}\n\n"
        "饮食上不要做热量赤字，也不要减少主食或跳餐。先保证每天 3 餐 + 1-2 次加餐：每餐有主食、蛋白质和脂肪；"
        f"蛋白质按当前 {_fmt_num(weight, 'kg')} 可先放在 1.2-1.6g/kg/天，约 {protein_low}-{protein_high}g/天；"
        f"健康 BMI 下限对应体重大约 {min_healthy_weight}kg，所以 40kg 不是安全目标。\n\n"
        "心理层面需要把“体重焦虑/体象压力/进食障碍风险”放到优先级前面。建议尽快联系临床营养师、心理咨询师，"
        "必要时到精神心理科或进食障碍相关门诊做专业评估；如果已经有明显限制进食、进食后强烈内疚、月经紊乱、头晕乏力或脱发，更应尽快就医。\n\n"
        "今天能做的第一步：先正常吃下一餐，不做补偿性少吃；把体重秤收起来 1 周，记录精力、睡眠、情绪和饥饿感，而不是继续追低数字。"
    )


def _query_allergy_terms(question: str, profile: Dict[str, Any]) -> List[str]:
    terms: List[str] = []
    dietary = profile.get("dietary_context") or {}
    for pref in _as_list(dietary.get("preferences")):
        if re.search(r"过敏|不耐", pref):
            if "乳糖" in pref:
                terms.append("乳糖不耐")
            else:
                terms.append(pref)
    q = question or ""
    if re.search(r"花生.*过敏|过敏.*花生", q):
        terms.append("花生过敏")
    if re.search(r"乳糖.*不耐|不耐.*乳糖", q):
        terms.append("乳糖不耐")
    if re.search(r"海鲜.*过敏|过敏.*海鲜", q):
        terms.append("海鲜过敏")

    seen: List[str] = []
    for term in terms:
        cleaned = str(term).strip().replace("过敏：", "")
        if "乳糖" in cleaned:
            cleaned = "乳糖不耐"
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


def _is_vegetarian(question: str, profile: Dict[str, Any]) -> bool:
    dietary = profile.get("dietary_context") or {}
    prefs = " ".join(_as_list(dietary.get("preferences")))
    return bool(re.search(r"素食|纯素|不吃肉|vegetarian|vegan", f"{question or ''} {prefs}", re.IGNORECASE))


def _is_advice_query(question: str) -> bool:
    q = (question or "").strip()
    return bool(q and _ADVICE_SIGNAL.search(q) and not _PURE_CHITCHAT_SIGNAL.search(q))


def _profile_from_ctx(pctx: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(pctx, dict):
        return {}
    profile = pctx.get("raw_profile")
    if isinstance(profile, dict):
        return profile
    return pctx if isinstance(pctx, dict) else {}


def _role_allows(domain: str, role: str | None) -> bool:
    if not role:
        return True
    if role == domain:
        return True
    if role in {"Aggregator", "Critic", "Orchestrator"}:
        return True
    return False


def _domain_query_match(domain: str, question: str) -> bool:
    q = question or ""
    if domain == "Trainer":
        return bool(
            re.search(
                r"练|训练|运动|健身|跑|步数|深蹲|卧推|热身|冷身|强度|RPE|心率|恢复|康复|"
                r"HIIT|增肌|5\s*公里|10\s*(?:K|公里)|半马|马拉松|TDEE|BMR|基础代谢|热量消耗",
                q,
                re.IGNORECASE,
            )
        )
    if domain == "Nutritionist":
        return bool(
            re.search(
                r"吃|饮食|营养|蛋白|热量|kcal|减脂餐|食谱|补剂|肌酸|乳清|餐|水|补水|电解质|"
                r"碳水|低碳|米饭|加餐|零食|过敏|咖啡|咖啡因|饮品|饮料|茶|牛奶|能量饮料",
                q,
                re.IGNORECASE,
            )
        ) or bool(_ENDURANCE_RACE_SIGNAL.search(q) and not _RUNNING_BEGINNER_SIGNAL.search(q))
    if domain == "Psychologist":
        return bool(
            re.search(
                r"睡|失眠|入睡|压力|焦虑|紧张|倦怠|动力|精力|疲惫|疲劳|恢复不过来|躺|情绪|内疚|暴食|手机|屏幕|汇报|"
                r"轻生|自伤|不想活|活着没意思",
                q,
                re.IGNORECASE,
            )
        )
    if domain == "Doctor":
        return bool(
            re.search(
                r"病|诊断|处方|用药|剂量|疼|痛|头晕|恶心|胸痛|胸闷|胸口压迫|压迫感|心率|血压|血糖|糖尿病|HIIT|"
                r"尿色|浓茶|晕厥|呼吸困难|药",
                q,
                re.IGNORECASE,
            )
        )
    return True


def _injury_terms(injuries: Iterable[str], question: str = "") -> str:
    text = " ".join(str(x) for x in injuries if str(x).strip())
    return f"{text} {question or ''}".strip()


def _add_point(
    points: List[PersonalizationDecisionPoint],
    *,
    domain: str,
    point_id: str,
    priority: int,
    instruction: str,
    evidence_terms: Iterable[str],
    source_profile_keys: Iterable[str],
    role: str | None,
) -> None:
    if not _role_allows(domain, role):
        return
    if point_id in {p.id for p in points}:
        return
    terms = [str(t).strip() for t in evidence_terms if str(t).strip()]
    points.append(
        PersonalizationDecisionPoint(
            id=point_id,
            domain=domain,
            priority=priority,
            instruction=instruction.strip(),
            evidence_terms=terms,
            source_profile_keys=[str(k).strip() for k in source_profile_keys if str(k).strip()],
        )
    )


def build_personalization_decision_points(
    pctx: Dict[str, Any] | None,
    user_question: str,
    role: str | None = None,
) -> List[PersonalizationDecisionPoint]:
    """Build profile-driven decisions that must be integrated into advice text.

    Unlike the old answer-prefix guardrail, these points are question and domain
    scoped. They intentionally avoid generic profile echo; each point must
    change a concrete recommendation, number, limit, or scenario action.
    """
    profile = _profile_from_ctx(pctx)
    if not is_meaningful(profile) or not _is_advice_query(user_question):
        return []

    q = user_question or ""
    stats = profile.get("physical_stats") or {}
    dietary = profile.get("dietary_context") or {}
    mental = profile.get("mental_state") or {}
    age = _num(stats.get("age"))
    height = _num(stats.get("height"))
    weight = _num(stats.get("weight"))
    bmi = _bmi(height, weight)
    goal = str(dietary.get("goal") or "").strip()
    injuries = _as_list(stats.get("injuries"))
    injury_text = _injury_terms(injuries, q)
    stress = _as_list(mental.get("stress_sources"))
    sleep_quality = str(mental.get("sleep_quality") or "").strip()
    prefs = _as_list(dietary.get("preferences"))

    points: List[PersonalizationDecisionPoint] = []

    if _CRISIS_SIGNAL.search(q):
        _add_point(
            points,
            domain="Psychologist",
            point_id="psych_crisis_safety",
            priority=1,
            instruction="把轻生/自伤表述作为即时心理危机处理：先联系身边可信任的人，若有伤害冲动则急救/危机热线/急诊优先；不要把重点放在普通睡眠或运动建议。",
            evidence_terms=["危机", "可信任", "急救", "急诊", "不要一个人"],
            source_profile_keys=["mental_state.stress_sources"],
            role=role,
        )
        return sorted(points, key=lambda p: (p.priority, p.domain, p.id))

    if injuries and _domain_query_match("Trainer", q):
        if re.search(r"ACL|前交叉|韧带", injury_text, re.IGNORECASE) and re.search(r"深蹲|蹲|下肢|练腿|负重", q):
            _add_point(
                points,
                domain="Trainer",
                point_id="trainer_acl_squat_gate",
                priority=1,
                instruction="把 ACL/韧带术后或康复状态作为训练许可门槛：说明不能只看时间，深蹲需医生/理疗师评估；写出伸直0度/屈曲约120度且无痛、单腿30度控制、股四头肌约80%、训练后24小时无明显肿胀疼痛等通过标准，并先用低冲击/等长替代。",
                evidence_terms=["ACL", "医生", "理疗师", "评估", "120", "30度", "80%", "24小时", "肿胀", "疼痛"],
                source_profile_keys=["physical_stats.injuries"],
                role=role,
            )
        elif re.search(r"肩袖|肩关节|肩", injury_text) and re.search(r"练胸|卧推|胸|推", q):
            _add_point(
                points,
                domain="Trainer",
                point_id="trainer_rotator_cuff_chest",
                priority=1,
                instruction="把肩袖恢复期转成练胸动作选择：暂避杠铃卧推、双杠臂屈伸、过头推举和疼痛范围推举；优先弹力带外旋、Y-T-W、墙壁俯卧撑，并以理疗评估为前提。",
                evidence_terms=["肩袖", "杠铃卧推", "弹力带外旋", "Y-T-W", "理疗"],
                source_profile_keys=["physical_stats.injuries"],
                role=role,
            )
        elif re.search(r"半月板|膝", injury_text) and re.search(r"步|走|跑|深蹲|练腿|训练|运动", q):
            _add_point(
                points,
                domain="Trainer",
                point_id="trainer_knee_low_impact",
                priority=1,
                instruction="把膝部伤病转成低冲击训练限制：避免跑跳、深屈膝、扭转和高冲击；用直腿抬高、股四头肌等长、低阻力固定单车/游泳等无痛替代，并用24小时反应决定是否加量。",
                evidence_terms=["膝", "低阻力固定单车", "24小时", "无痛", "理疗师"],
                source_profile_keys=["physical_stats.injuries"],
                role=role,
            )
        else:
            _add_point(
                points,
                domain="Trainer",
                point_id="trainer_injury_constraint",
                priority=1,
                instruction=f"把已知伤病/康复限制（{'、'.join(injuries)}）转成训练动作禁忌和替代动作；涉及相关部位负荷必须以无痛和医生/理疗师许可为前提。",
                evidence_terms=[*injuries, "无痛", "医生", "理疗师"],
                source_profile_keys=["physical_stats.injuries"],
                role=role,
            )

    if _RUNNING_BEGINNER_SIGNAL.search(q) and _domain_query_match("Trainer", q):
        _add_point(
            points,
            domain="Trainer",
            point_id="trainer_5k_beginner_run_walk",
            priority=1,
            instruction="从零开始 5 公里必须按跑走结合落地：第1周慢跑1分钟+快走2分钟循环，第2周2跑2走，第3周3-4跑1-2走，第4周再延长连续慢跑；每周3次，不要求立刻跑完5km。",
            evidence_terms=["跑走", "快走", "慢跑", "第1周", "每周3次"],
            source_profile_keys=["question"],
            role=role,
        )

    if (
        _ENDURANCE_RACE_SIGNAL.search(q)
        and not _RUNNING_BEGINNER_SIGNAL.search(q)
        and _domain_query_match("Trainer", q)
        and re.search(r"练|训练|跑量|轻松跑|配速|减量|怎么练|安排", q)
    ):
        _add_point(
            points,
            domain="Trainer",
            point_id="trainer_race_week_taper",
            priority=1,
            instruction="赛前一周训练必须按减量落地：总跑量降到平时约50%，保留2-3次20-30分钟轻松跑和少量比赛配速/加速唤醒；不要赛前加倍训练或新增高强度腿部训练。",
            evidence_terms=["减量", "50%", "轻松跑", "20-30", "配速"],
            source_profile_keys=["question"],
            role=role,
        )

    if re.search(r"新手|健身房几次|每周.*几次", q) and _domain_query_match("Trainer", q):
        _add_point(
            points,
            domain="Trainer",
            point_id="trainer_beginner_frequency",
            priority=1,
            instruction="健身新手方案要按低容量起步：每周2-3次、每次45-60分钟、间隔至少1天，前4-6周优先深蹲模式/髋铰链/推/拉/核心等动作模式和稳定节奏。",
            evidence_terms=["2-3", "45-60", "间隔", "4-6周", "动作模式"],
            source_profile_keys=["physical_stats.age", "physical_stats.weight", "physical_stats.height"],
            role=role,
        )

    if goal == "增肌" and _domain_query_match("Trainer", q) and re.search(r"增肌|怎么练|训练|力量|健身|练", q):
        _add_point(
            points,
            domain="Trainer",
            point_id="trainer_muscle_gain_plan",
            priority=1,
            instruction="把增肌目标转成训练决策：每周3-5次力量训练，复合动作优先，主训练动作落在8-12次区间，并用渐进超负荷记录重量/次数；不要只给饮食方案。",
            evidence_terms=["3-5", "力量", "8-12", "渐进", "复合动作"],
            source_profile_keys=["dietary_context.goal", "physical_stats.age", "physical_stats.weight"],
            role=role,
        )

    if age and height and weight and _TDEE_ESTIMATE_SIGNAL.search(q):
        bmr = int(round(10 * weight + 6.25 * height - 5 * age + 5))
        sedentary = int(round(bmr * 1.2))
        active = int(round(bmr * 1.55))
        deficit_low = max(1500, active - 500)
        deficit_high = max(1500, active - 300)
        _add_point(
            points,
            domain="Trainer",
            point_id="trainer_tdee_bmr_estimate",
            priority=1,
            instruction=f"把 TDEE/BMR 问题交给训练侧估算：按 {_fmt_num(age, '岁')}、{_fmt_num(weight, 'kg')}、{_fmt_num(height, 'cm')} 用 Mifflin-St Jeor 估 BMR 约 {bmr} kcal/天，并给久坐约 {sedentary}、中等活动约 {active} kcal/天；减脂起点从 TDEE 下调 300-500 kcal。",
            evidence_terms=["TDEE", "BMR", str(bmr), str(sedentary), str(active), "300-500"],
            source_profile_keys=["physical_stats.age", "physical_stats.weight", "physical_stats.height"],
            role=role,
        )

    if age and re.search(r"热身|冷身|有氧|跑|5公里|10公里|心率|强度|熬夜|睡眠不足", q):
        max_hr = int(round(220 - age))
        low_hr = int(round(max_hr * 0.6))
        high_hr = int(round(max_hr * 0.7))
        _add_point(
            points,
            domain="Trainer",
            point_id="trainer_age_intensity",
            priority=2,
            instruction=f"用年龄把强度具体化：按 {_fmt_num(age, '岁')} 粗估最大心率约 {max_hr}，轻中等强度先落在 {low_hr}-{high_hr} 次/分钟，或用 RPE 4-6/能完整说话校准。",
            evidence_terms=[str(max_hr), f"{low_hr}-{high_hr}", "RPE", "完整说话"],
            source_profile_keys=["physical_stats.age"],
            role=role,
        )

    training_load_context = bool(re.search(r"训练|运动|健身|跑|强度|练|HIIT|力量|有氧", q, re.IGNORECASE))
    if (sleep_quality or stress) and training_load_context and re.search(r"睡|熬夜|压力|恢复|强度|训练|运动|练", q) and _domain_query_match("Trainer", q):
        source = "、".join(stress) if stress else sleep_quality
        _add_point(
            points,
            domain="Trainer",
            point_id="trainer_sleep_load",
            priority=1,
            instruction=f"把睡眠/压力状态（{source}）转成训练负荷调整：睡眠差或压力高的几天降到 RPE 5-6、时长约20-45分钟，优先技术练习/低强度有氧，避免高强度硬扛。",
            evidence_terms=["RPE 5-6", "睡眠", "低强度", "避免高强度"],
            source_profile_keys=["mental_state.sleep_quality", "mental_state.stress_sources"],
            role=role,
        )

    allergies = _query_allergy_terms(q, profile)
    if allergies and _domain_query_match("Nutritionist", q):
        _add_point(
            points,
            domain="Nutritionist",
            point_id="nutrition_allergy_guard",
            priority=1,
            instruction=f"把饮食限制（{'、'.join(allergies)}）作为食物/补剂筛选硬约束：避开相关成分，过敏/不耐场景提醒配料表、过敏原标识和交叉污染。",
            evidence_terms=[*allergies, "配料表", "过敏原", "交叉污染"],
            source_profile_keys=["dietary_context.preferences"],
            role=role,
        )

    nutrition_query = _domain_query_match("Nutritionist", q)
    if nutrition_query and re.search(r"咖啡|咖啡因|饮品|饮料|茶|能量饮料", q, re.IGNORECASE):
        _add_point(
            points,
            domain="Nutritionist",
            point_id="nutrition_caffeine_beverage_timing",
            priority=1,
            instruction="把下午咖啡因与焦虑/睡眠问题连接起来：咖啡因窗口前移到上午，下午换低咖啡因或无咖啡因饮品，并提醒能量饮料、浓茶也可能含咖啡因。",
            evidence_terms=["咖啡因", "上午", "下午", "无咖啡因", "能量饮料"],
            source_profile_keys=["mental_state.sleep_quality", "mental_state.stress_sources"],
            role=role,
        )

    if weight and nutrition_query and re.search(r"蛋白|增肌|减脂|保肌|每日饮食|减脂餐|食谱|吃多少|摄入", q):
        low = int(round(weight * 1.6))
        high = int(round(weight * 2.2))
        _add_point(
            points,
            domain="Nutritionist",
            point_id="nutrition_weight_protein",
            priority=1,
            instruction=f"按体重把蛋白质落到克数：{_fmt_num(weight, 'kg')} 对应 1.6-2.2g/kg/天，约 {low}-{high}g/天，并分到餐次/食物选择中。",
            evidence_terms=[f"{low}", f"{high}", "1.6", "2.2", "每餐"],
            source_profile_keys=["physical_stats.weight", "dietary_context.goal"],
            role=role,
        )

    if injuries and weight and nutrition_query and re.search(r"饮食|营养|恢复|蛋白|吃", q):
        low = int(round(weight * 1.2))
        high = int(round(weight * 1.6))
        _add_point(
            points,
            domain="Nutritionist",
            point_id="nutrition_injury_recovery_support",
            priority=1,
            instruction=f"当前问题明确问到伤病恢复与饮食时，才把营养用于恢复环境：按 {_fmt_num(weight, 'kg')} 给蛋白质约 {low}-{high}g/天，保证总能量、蔬果/维生素C和 Omega-3 来源；不要替代医生/理疗师的训练许可。",
            evidence_terms=[str(low), str(high), "蛋白质", "维生素C", "Omega-3"],
            source_profile_keys=["physical_stats.weight", "physical_stats.injuries"],
            role=role,
        )

    if (
        weight
        and goal == "减脂"
        and nutrition_query
        and not _TDEE_ESTIMATE_SIGNAL.search(q)
        and re.search(r"减脂|热量|每日饮食|减脂餐|平台|吃", q)
    ):
        _add_point(
            points,
            domain="Nutritionist",
            point_id="nutrition_fat_loss_deficit",
            priority=2,
            instruction="把减脂目标转成温和热量策略：优先300-500 kcal/天赤字，避免极端低热量，并用围度、训练表现和睡眠一起判断。",
            evidence_terms=["300-500", "热量", "训练表现", "睡眠"],
            source_profile_keys=["dietary_context.goal"],
            role=role,
        )

    if weight and nutrition_query and re.search(r"水|饮水|补水|出汗|电解质", q):
        low_l = _fmt_liters(weight * 30 / 1000)
        high_l = _fmt_liters(weight * 40 / 1000)
        _add_point(
            points,
            domain="Nutritionist",
            point_id="nutrition_weight_hydration",
            priority=1,
            instruction=f"按体重把补水量具体化：{_fmt_num(weight, 'kg')} 的基础饮水约 {low_l}-{high_l}L/天；大量出汗时按运动前中后和体重变化补水/电解质。",
            evidence_terms=[f"{low_l}-{high_l}", "1.2-1.5L", "电解质", "体重变化"],
            source_profile_keys=["physical_stats.weight"],
            role=role,
        )

    if weight and _ENDURANCE_RACE_SIGNAL.search(q) and not _RUNNING_BEGINNER_SIGNAL.search(q):
        carb_low = int(round(weight * 1.0))
        carb_high = int(round(weight * 2.0))
        _add_point(
            points,
            domain="Nutritionist",
            point_id="nutrition_race_carbs",
            priority=1,
            instruction=f"真正比赛/备赛补给才给碳水策略：按 {_fmt_num(weight, 'kg')}，赛前餐可先放 {carb_low}-{carb_high}g 易消化碳水；赛前2-3天熟悉食物，炎热或>60分钟再安排水/电解质/能量胶。",
            evidence_terms=[f"{carb_low}-{carb_high}", "赛前2-3天", "电解质", "能量胶"],
            source_profile_keys=["physical_stats.weight"],
            role=role,
        )

    if _domain_query_match("Psychologist", q):
        if stress and re.search(r"睡|压力|焦虑|倦怠|动力|暴食|内疚|恢复|精力|放松|紧张|汇报|手机", q):
            source = "、".join(stress)
            if re.search(r"手机|屏幕|刷手机", f"{q} {source}"):
                instruction = f"把压力源/触发源（{source}）映射到睡前手机流程：睡前60分钟手机离床外并开勿扰，用纸质书/洗澡/拉伸/呼吸替代；先连续3天做到一个动作，降低门槛。"
                terms = ["手机", "60分钟", "勿扰", "3天", "睡前"]
                point_id = "psych_phone_sleep_flow"
            elif re.search(r"项目|工作|deadline", source, re.IGNORECASE):
                instruction = f"把压力源（{source}）映射到睡前降速流程：5分钟担忧/待办清单，约定明天处理窗口，再做10分钟呼吸或渐进放松，避免把工作带上床。"
                terms = ["项目", "担忧清单", "10分钟", "明天", "睡前"]
                point_id = "psych_project_sleep_flow"
            elif re.search(r"汇报|公开|演讲|presentation", q, re.IGNORECASE):
                instruction = f"把压力源（{source}）映射到公开汇报场景：20分钟拆开场/要点/结尾，每天10-15分钟演练，临场2分钟慢呼吸。"
                terms = ["汇报", "20分钟", "10-15分钟", "2分钟", "呼吸"]
                point_id = "psych_presentation_flow"
            else:
                instruction = f"把压力源（{source}）转成可执行最小行动：5-10分钟启动动作、固定恢复窗或下一步任务拆解，避免只说放松。"
                terms = [*stress, "5-10分钟", "恢复窗", "下一步"]
                point_id = "psych_stress_min_action"
            _add_point(
                points,
                domain="Psychologist",
                point_id=point_id,
                priority=1,
                instruction=instruction,
                evidence_terms=terms,
                source_profile_keys=["mental_state.stress_sources"],
                role=role,
            )
        if sleep_quality and re.search(r"睡|失眠|入睡|咖啡|精力|恢复|手机", q):
            _add_point(
                points,
                domain="Psychologist",
                point_id="psych_sleep_quality_flow",
                priority=1,
                instruction=f"把睡眠质量“{sleep_quality}”转成睡眠卫生动作：固定起床时间，睡前30-60分钟降刺激，必要时用担忧清单、呼吸或渐进式肌肉放松。",
                evidence_terms=["睡眠", "30-60", "固定起床", "担忧清单", "呼吸"],
                source_profile_keys=["mental_state.sleep_quality"],
                role=role,
            )

    if _domain_query_match("Doctor", q):
        if _DIABETES_SIGNAL.search(q) and _HIIT_SIGNAL.search(q):
            _add_point(
                points,
                domain="Doctor",
                point_id="doctor_diabetes_hiit_guard",
                priority=1,
                instruction="把糖尿病画像与 HIIT 意图相连：不能直接开始30分钟HIIT，需医生评估运动许可；运动前后监测血糖，考虑用药、低血糖、并发症、足部保护和心血管风险。",
                evidence_terms=["糖尿病", "HIIT", "医生", "血糖", "低血糖"],
                source_profile_keys=["question", "physical_stats.age", "physical_stats.weight", "dietary_context.goal"],
                role=role,
            )
        elif _MEDICAL_REDFLAG_SIGNAL.search(q):
            _add_point(
                points,
                domain="Doctor",
                point_id="doctor_redflag_triage",
                priority=1,
                instruction="把症状/用药红旗转成医学边界：先建议医生/急诊/药师评估，再给就诊前记录项和暂时避免事项；不要先做诊断或训练/用药处方。",
                evidence_terms=["医生", "评估", "记录", "不要自行", "仅供参考"],
                source_profile_keys=["question", "physical_stats.age", "physical_stats.weight"],
                role=role,
            )
        elif age and re.search(r"HIIT|高强度|心率|胸痛|胸闷|血压|血糖", q, re.IGNORECASE):
            _add_point(
                points,
                domain="Doctor",
                point_id="doctor_age_high_intensity",
                priority=2,
                instruction=f"把年龄和高强度意图相连：{_fmt_num(age, '岁')} 用户开始 HIIT/高强度训练前应先做风险评估，尤其关注心血管、血压/血糖和不适信号。",
                evidence_terms=[_fmt_num(age, "岁"), "风险评估", "心血管", "血压", "血糖"],
                source_profile_keys=["physical_stats.age"],
                role=role,
            )

    if bmi and re.search(r"减到|降到|瘦到|减脂|减重|体重", q):
        target = _extract_target_weight(q)
        target_bmi = round(target / ((height / 100) ** 2), 1) if target and height else None
        if bmi < 18.5 or (target_bmi and target_bmi < 18.5):
            _add_point(
                points,
                domain="Nutritionist",
                point_id="nutrition_underweight_safety",
                priority=1,
                instruction=f"把身高体重转成安全边界：当前 BMI 约 {bmi}，若目标会低于18.5则不能支持继续减重；目标应改为体重维持、规律进食和体象/焦虑支持。",
                evidence_terms=[str(bmi), "18.5", "不建议", "维持", "体象"],
                source_profile_keys=["physical_stats.height", "physical_stats.weight"],
                role=role,
            )
            if stress or re.search(r"心理|体象|焦虑|进食", q):
                source = "、".join(stress) if stress else "体重/体象压力"
                _add_point(
                    points,
                    domain="Psychologist",
                    point_id="psych_underweight_body_image",
                    priority=1,
                    instruction=f"把低体重目标与心理风险相连：点出 {source} 可能让体重数字变成焦虑焦点，建议体象/进食障碍风险评估；今天先正常吃下一餐、暂停频繁称重，而不是继续追低数字。",
                    evidence_terms=["体重焦虑", "体象", "进食障碍", "正常吃下一餐", "称重"],
                    source_profile_keys=["mental_state.stress_sources", "physical_stats.height", "physical_stats.weight"],
                    role=role,
                )

    return sorted(points, key=lambda p: (p.priority, p.domain, p.id))


def format_decision_points_for_prompt(points: Iterable[PersonalizationDecisionPoint]) -> str:
    pts = list(points or [])
    if not pts:
        return ""
    lines = [
        "【必须融入正文的个性化决策点】",
        "下面每一点都必须变成正文里的具体建议、数字、限制或场景动作；不要把它们作为开头清单照搬，也不要只复述画像。",
    ]
    for i, point in enumerate(pts, start=1):
        evidence = f"（落地证据应包含：{'、'.join(point.evidence_terms)}）" if point.evidence_terms else ""
        lines.append(f"{i}. [{point.domain}/P{point.priority}] {point.instruction}{evidence}")
    return "\n".join(lines) + "\n"


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def _term_present(answer: str, term: str) -> bool:
    if not term:
        return False
    if term.startswith("re:"):
        try:
            return bool(re.search(term[3:], answer or "", re.IGNORECASE))
        except re.error:
            return False
    return _normalized_text(term) in _normalized_text(answer)


def _point_landed(answer: str, point: PersonalizationDecisionPoint) -> bool:
    terms = point.evidence_terms or []
    if not terms:
        return False
    hits = sum(1 for term in terms if _term_present(answer, term))
    needed = 1 if len(terms) == 1 else 2
    return hits >= needed


def check_decision_points_landed(
    answer: str,
    points: Iterable[PersonalizationDecisionPoint],
) -> Dict[str, Any]:
    pts = list(points or [])
    landed = [point for point in pts if _point_landed(answer or "", point)]
    required_total = len(pts)
    required_to_land = min(2, required_total) if required_total else 0
    missing = [point for point in pts if point not in landed]
    return {
        "required_total": required_total,
        "required_to_land": required_to_land,
        "landed_count": len(landed),
        "satisfied": required_total == 0 or len(landed) >= required_to_land,
        "landed_ids": [p.id for p in landed],
        "missing_ids": [p.id for p in missing],
        "missing_instructions": [p.instruction for p in missing],
    }


def personalized_action_notes(profile: Dict[str, Any], user_question: str, answer: str = "") -> List[str]:
    """Render short, computed personalization notes for advice answers.

    Backward-compatible wrapper around the structured decision-point API.
    New generation paths should inject formatted decision points before writing
    the answer instead of prepending these strings after the fact.
    """
    points = build_personalization_decision_points({"raw_profile": profile}, user_question)
    if points:
        return [point.instruction for point in points[:3]]
    if not isinstance(profile, dict):
        return []
    q = user_question or ""
    text = answer or ""
    if not _ADVICE_SIGNAL.search(q) or _PURE_CHITCHAT_SIGNAL.search(q):
        return []

    stats = profile.get("physical_stats") or {}
    dietary = profile.get("dietary_context") or {}
    mental = profile.get("mental_state") or {}
    age = _num(stats.get("age"))
    height = _num(stats.get("height"))
    weight = _num(stats.get("weight"))
    bmi = _bmi(height, weight)
    goal = str(dietary.get("goal") or "").strip()
    injuries = _as_list(stats.get("injuries"))
    stress = _as_list(mental.get("stress_sources"))
    sleep_quality = str(mental.get("sleep_quality") or "").strip()
    medical_red_flag = bool(
        re.search(
            r"浓茶|茶色尿|酱油色尿|血压|血糖|胸痛|胸闷|呼吸困难|晕厥|静息心率|心率异常|"
            r"布洛芬|对乙酰氨基酚|阿司匹林|抗生素|处方|剂量|继续用药|是什么病|诊断|急诊",
            q,
            re.IGNORECASE,
        )
    )

    notes: List[str] = []

    def add(line: str, *, unless: Iterable[str] = ()) -> None:
        line = line.strip()
        if not line:
            return
        if any(line == existing for existing in notes):
            return
        if unless and _contains_any(text, unless):
            return
        notes.append(line)

    if re.search(r"活着没意思|不想活|轻生|自杀|自伤|结束生命|不想跟任何人说|没意思", q):
        add(
            "这属于心理危机信号：现在先不要一个人扛，立刻联系身边可信任的人；若有伤害自己的冲动，请马上拨打当地急救/危机热线或去急诊。",
            unless=[r"危机|热线|急救|急诊|可信任的人|不要一个人|988"],
        )
        return notes[:2]

    if medical_red_flag:
        return []

    profile_anchors = [
        _fmt_num(age, "岁") if age else "",
        _fmt_num(weight, "kg") if weight else "",
        _fmt_num(height, "cm") if height else "",
    ]
    profile_anchors = [x for x in profile_anchors if x]
    if profile_anchors and not any(anchor.rstrip("岁kgcm") in text for anchor in profile_anchors):
        add(
            f"你的基础画像是 {'、'.join(profile_anchors)}；下面的克数、步数或训练量要围绕这些数值调整。",
        )

    if bmi and re.search(r"减到|降到|瘦到|减脂|减重|体重", q):
        target = _extract_target_weight(q)
        if bmi < 18.5 or (target and height and target / ((height / 100) ** 2) < 18.5):
            target_text = f"；若减到 {_fmt_num(target, 'kg')}，BMI 约 {round(target / ((height / 100) ** 2), 1)}" if target and height else ""
            add(
                f"你当前 BMI 约 {bmi}，已在偏低范围{target_text}，不建议再降体重，也不能支持把目标设为更低体重；本轮目标应改成体重维持、规律进食和体象/焦虑支持。",
                unless=[r"不能支持|不建议.*继续减|不建议.*减重|不支持"],
            )

    if injuries:
        injury_text = "、".join(injuries)
        if re.search(r"步|走|跑|深蹲|训练|运动|恢复|练", q):
            if any("膝关节炎" in item or "膝" in item for item in injuries) and re.search(r"步|走", q):
                add(
                    f"考虑到 {injury_text}，步数要循序渐进：先从低痛感步数开始，若 24 小时无肿胀/夜间痛，再每 1-2 周小幅增加 500-1000 步。",
                    unless=[r"循序渐进|每\s*1-2\s*周|小幅增加|逐步"],
                )
            else:
                add(
                    f"已知伤病/康复限制是 {injury_text}；涉及该部位负荷的动作必须以无痛为前提，并先获医生或理疗师许可。",
                    unless=[re.escape(item) for item in injuries],
                )

    allergies = _query_allergy_terms(q, profile)
    if allergies and re.search(r"吃|零食|蛋白|食谱|补剂|餐|饮食|推荐", q):
        allergy_like = [item for item in allergies if "过敏" in item or item in {"花生", "海鲜"}]
        if allergy_like:
            add(
                f"饮食限制先锁定：你前面明确提到 {'、'.join(allergies)}。买包装食品要看配料表和过敏原提示，并避开可能交叉污染的产品。",
                unless=[r"配料表|过敏原"],
            )
        else:
            add(
                f"饮食限制先锁定：{'、'.join(allergies)}。买包装食品要看配料表/成分表，优先选择明确无相关成分的产品。",
                unless=[r"配料表|成分表"],
            )

    if _is_vegetarian(q, profile) and re.search(r"增肌|蛋白|餐|晚餐|训练", q):
        add(
            "素食增肌除豆制品/豆类蛋白外，还要补齐 B12、铁、锌和 Omega-3 来源，避免只把蛋白质克数算够但微量营养素掉线。",
            unless=[r"B12|Omega|铁|锌"],
        )

    if weight and re.search(r"蛋白|增肌|减脂|保肌|恢复|平台|训练后|ACL|术后|康复", q, re.IGNORECASE):
        low = int(round(weight * 1.6))
        high = int(round(weight * 2.2))
        per3_low = int(round(low / 3))
        per4_high = int(round(high / 4))
        add(
            f"按 {_fmt_num(weight, 'kg')} 换算，蛋白质可先放在 1.6-2.2g/kg/天，也就是约 {low}-{high}g/天；分 3-4 餐时每餐大约 {per3_low}-{per4_high}g。",
            unless=[rf"{low}\s*[-~到至]\s*{high}|{low}g|{high}g|{low}克|{high}克"],
        )

    if weight and re.search(r"水|饮水|补水|出汗|电解质|尿色", q):
        low_l = weight * 30 / 1000
        high_l = weight * 40 / 1000
        add(
            f"按 {_fmt_num(weight, 'kg')} 估算，基础饮水约 {_fmt_liters(low_l)}-{_fmt_liters(high_l)}L/天；训练大量出汗时再按体重变化补，运动后每少 1kg 补 1.2-1.5L。",
            unless=[r"30\s*[-~到至]\s*40\s*ml|1\.2\s*[-~到至]\s*1\.5L|2\.7\s*[-~到至]\s*3\.6"],
        )

    if weight and _ENDURANCE_RACE_SIGNAL.search(q):
        carb_low = int(round(weight * 1.0))
        carb_high = int(round(weight * 2.0))
        add(
            f"按 {_fmt_num(weight, 'kg')} 估算，比赛日赛前餐可先放在约 {carb_low}-{carb_high}g 易消化碳水；赛前 2-3 天仍以熟悉食物为主，不临时尝试新品。",
            unless=[rf"{carb_low}\s*[-~到至]\s*{carb_high}|赛前\s*2-3\s*天.*碳水"],
        )

    if age and re.search(r"热身|冷身|有氧|跑|5公里|10公里|心率|强度|熬夜|睡眠不足", q):
        max_hr = int(round(220 - age))
        low_hr = int(round(max_hr * 0.6))
        high_hr = int(round(max_hr * 0.7))
        add(
            f"按 {_fmt_num(age, '岁')} 粗略估算最大心率约 {max_hr}，轻中等强度可先落在约 {low_hr}-{high_hr} 次/分钟，或用 RPE 4-6/能完整说话来校准。",
            unless=[rf"{max_hr}|RPE\s*4-6|最大心率"],
        )

    if re.search(r"碳水|米饭|低碳|戒", q):
        add(
            "减脂期更适合调份量而不是戒碳水；长期极低碳水容易影响训练表现、情绪和坚持度，训练日前后尤其要保留主食。",
            unless=[r"情绪|坚持度|训练表现"],
        )

    if stress and re.search(r"睡|压力|焦虑|倦怠|动力|暴食|内疚|恢复|精力|放松|紧张|汇报", q):
        source = "、".join(stress)
        if re.search(r"育儿|工作", source):
            add(
                f"你的压力源是 {source}，恢复动作要切到真实场景：每天固定 10-15 分钟无打扰恢复窗，把育儿/工作待办只写下一步，不在睡前继续拆大任务。",
                unless=[r"育儿|恢复窗|无打扰"],
            )
        elif re.search(r"工作|deadline|项目", source, re.IGNORECASE):
            add(
                f"你的压力源是 {source}，睡前先做 5 分钟担忧清单并约定明天处理窗口，再做 10 分钟呼吸/渐进放松，避免把工作带上床。",
                unless=[r"担忧清单|deadline|项目压力"],
            )
        else:
            add(
                f"你的压力源是 {source}，先用 5-10 分钟的最小行动降低启动成本，再逐步恢复节奏。",
                unless=[r"5-10\s*分钟|压力源"],
            )
    elif sleep_quality and re.search(r"睡|恢复|精力|咖啡|焦虑", q):
        add(
            f"你记录的睡眠质量是「{sleep_quality}」，先把咖啡因截止时间、睡前 30-60 分钟降刺激和固定起床时间作为优先级。",
            unless=[r"睡眠质量|30-60\s*分钟|固定起床"],
        )

    if goal and goal not in _DEFAULT_GOALS and weight and re.search(r"计划|方案|怎么练|怎么吃|平台|安排", q):
        if goal == "增肌":
            add(
                f"你的目标是增肌，体重增长速度先控制在每周约 {_fmt_num(weight * 0.0025, 'kg')}-{_fmt_num(weight * 0.005, 'kg')}，超过太多多半是脂肪增长过快。",
                unless=[r"每周.*0\.25|每周.*0\.5|体重.*每周"],
            )
        elif goal == "减脂":
            add(
                "你的目标是减脂，热量赤字优先放在 300-500 kcal/天，并用围度、训练表现和睡眠共同判断，不只盯体重。",
                unless=[r"300\s*[-~到至]\s*500|训练表现.*睡眠"],
            )

    return notes[:3]


def apply_personalization_boost(
    answer: str,
    pctx: Dict[str, Any] | None,
    user_question: str,
    *,
    max_notes: int = 3,
) -> str:
    """Compatibility hook for legacy callers.

    Structured decision points are now injected before generation and checked by
    Critic. This hook keeps only the underweight safety override, because that
    path must replace unsafe weight-loss advice rather than decorate it.
    """
    text = _as_answer_text(answer)
    if not text:
        return answer
    ctx = pctx or {}
    profile = ctx.get("raw_profile") if isinstance(ctx, dict) else None
    if not isinstance(profile, dict):
        profile = {}
    underweight_answer = _underweight_weight_goal_answer(profile, user_question)
    if underweight_answer:
        return underweight_answer
    return answer


def personalization_debug_json(ctx: Dict[str, Any]) -> str:
    """Compact JSON for debugging/eval reports."""
    safe = {
        "routing_digest": ctx.get("routing_digest", ""),
        "has_meaningful_data": bool(ctx.get("has_meaningful_data")),
        "active_constraints": ctx.get("active_constraints") or [],
    }
    return json.dumps(safe, ensure_ascii=False)
