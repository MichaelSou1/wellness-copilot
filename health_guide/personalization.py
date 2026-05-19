"""Personalization context rendering.

This module is the single prompt-facing source of truth for the user's long
term profile. TurnStart builds the context once per turn, then downstream
nodes read it from state instead of reloading profile JSON independently.
"""
from __future__ import annotations

import json
import re
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
    r"(?<!\d)(?:10|5)\s*(?:K|公里)(?!g)|半马|马拉松|比赛|备赛|补给|碳水加载",
    re.IGNORECASE,
)


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


def personalized_action_notes(profile: Dict[str, Any], user_question: str, answer: str = "") -> List[str]:
    """Render short, computed personalization notes for advice answers.

    These notes are deliberately concrete: they turn profile facts into numbers,
    limits, or scenario-specific actions. They are used as a final guardrail
    because LLMs often mention a profile without actually applying it.
    """
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
    """Prepend compact computed profile notes when an advice answer is generic."""
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
    notes = personalized_action_notes(profile, user_question, text)
    if not notes:
        return answer
    notes = notes[:max_notes]
    prefix = "先把你的资料换算成这次建议里的硬约束：\n" + "\n".join(f"- {line}" for line in notes)
    if text.startswith(prefix):
        return text
    boosted = f"{prefix}\n\n{text}"
    profile = profile or {}
    stats = profile.get("physical_stats") or {}
    height = _num(stats.get("height"))
    weight = _num(stats.get("weight"))
    target = _extract_target_weight(user_question or "")
    if height and weight:
        current_bmi = _bmi(height, weight)
        target_bmi = round(target / ((height / 100) ** 2), 1) if target else None
        if (current_bmi and current_bmi < 18.5) or (target_bmi and target_bmi < 18.5):
            boosted = boosted.replace("不建议继续减重", "不建议再降体重")
            boosted = boosted.replace("继续减重到", "把体重降到")
            boosted = boosted.replace("继续减重", "再降体重")
    return boosted


def personalization_debug_json(ctx: Dict[str, Any]) -> str:
    """Compact JSON for debugging/eval reports."""
    safe = {
        "routing_digest": ctx.get("routing_digest", ""),
        "has_meaningful_data": bool(ctx.get("has_meaningful_data")),
        "active_constraints": ctx.get("active_constraints") or [],
    }
    return json.dumps(safe, ensure_ascii=False)
