"""Personalization context rendering.

This module is the single prompt-facing source of truth for the user's long
term profile. TurnStart builds the context once per turn, then downstream
nodes read it from state instead of reloading profile JSON independently.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List

from .profile_store import get_user_profile, profile_to_prompt_text


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
    """Concise routing-relevant summary for Planner rules."""
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
    return {
        "user_card": render_user_card(profile),
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


def personalization_debug_json(ctx: Dict[str, Any]) -> str:
    """Compact JSON for debugging/eval reports."""
    safe = {
        "routing_digest": ctx.get("routing_digest", ""),
        "has_meaningful_data": bool(ctx.get("has_meaningful_data")),
        "active_constraints": ctx.get("active_constraints") or [],
    }
    return json.dumps(safe, ensure_ascii=False)
