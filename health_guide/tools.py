import os
import re
from pathlib import Path
from langchain_core.tools import tool

import json

from .rag import LocalKnowledgeBase
from .config import KNOWLEDGE_BASE_DIR, KNOWLEDGE_BASE_AGENT_SUBDIRS
from .profile_store import get_user_profile as get_profile_from_store
from .profile_store import update_user_profile as update_profile_in_store
from .integrations.local_logs import (
    log_meal,
    log_wellness_checkin,
    log_workout,
    query_logs,
)
from .integrations.push_reminder import push_reminder

# Per-agent KB singletons — created lazily on first call.
_AGENT_KBS: dict = {}


def _get_agent_kb(agent: str) -> LocalKnowledgeBase:
    if agent not in _AGENT_KBS:
        subdir = KNOWLEDGE_BASE_AGENT_SUBDIRS.get(agent, agent)
        kb_dir = str(Path(KNOWLEDGE_BASE_DIR) / subdir)
        _AGENT_KBS[agent] = LocalKnowledgeBase(kb_dir=kb_dir)
    return _AGENT_KBS[agent]


def _retrieve_by_agent(query: str, top_k: int, agent: str) -> str:
    try:
        kb = _get_agent_kb(agent)
        results = kb.retrieve(query=query, top_k=top_k)
    except Exception as e:
        return (
            "[RAG Error] 本地知识库暂不可用，请基于通用安全知识保守回答。"
            f"原因: {type(e).__name__}"
        )
    if not results:
        return "[RAG] 未命中本地知识库，请尝试改写查询或补充 knowledge_base 文档。"

    lines = ["[RAG] 命中以下知识片段："]
    for i, r in enumerate(results, start=1):
        snippet = re.sub(r"\s+", " ", r["content"]).strip()
        if len(snippet) > 220:
            snippet = snippet[:220] + "..."
        lines.append(
            f"{i}. [source: {r['source']} | chunk: {r['chunk_id']} | score: {r['score']}] {snippet}"
        )
    return "\n".join(lines)


@tool
def retrieve_trainer_knowledge(query: str, top_k: int = 4):
    """训练教练专用：从 trainer 知识库检索训练/运动/康复知识。"""
    return _retrieve_by_agent(query=query, top_k=top_k, agent="trainer")


@tool
def retrieve_nutritionist_knowledge(query: str, top_k: int = 4):
    """营养师专用：从 nutritionist 知识库检索饮食/营养/热量知识。"""
    return _retrieve_by_agent(query=query, top_k=top_k, agent="nutritionist")


@tool
def retrieve_psychologist_knowledge(query: str, top_k: int = 4):
    """心理疗愈师专用：从 psychologist 知识库检索心理健康、睡眠与压力管理知识。"""
    return _retrieve_by_agent(query=query, top_k=top_k, agent="psychologist")


@tool
def retrieve_doctor_knowledge(query: str, top_k: int = 4):
    """医学顾问专用：从 doctor 知识库检索症状分诊、慢病指标、用药边界等一般医学知识。"""
    return _retrieve_by_agent(query=query, top_k=top_k, agent="doctor")


@tool
def retrieve_safety_guidelines(query: str, top_k: int = 3):
    """安全审核员专用：从 safety 知识库检索运动伤病/症状就医/饮食极端等安全条目。"""
    return _retrieve_by_agent(query=query, top_k=top_k, agent="safety")


@tool
def get_user_profile(user_id: str = ""):
    """获取用户画像。user_id 为空时，将使用环境变量 HEALTH_GUIDE_USER_ID。"""
    target_user_id = user_id or os.environ.get("HEALTH_GUIDE_USER_ID", "default_user")
    profile = get_profile_from_store(target_user_id)
    return json.dumps(profile, ensure_ascii=False)


@tool
def update_user_profile(patch_json: str, user_id: str = ""):
    """更新用户画像。patch_json 需是 JSON 字符串，将做深度合并。"""
    target_user_id = user_id or os.environ.get("HEALTH_GUIDE_USER_ID", "default_user")
    try:
        patch = json.loads(patch_json)
        if not isinstance(patch, dict):
            return "[Profile Update Error] patch_json 必须是 JSON 对象。"
    except Exception as e:
        return f"[Profile Update Error] 无法解析 JSON: {e}"

    updated = update_profile_in_store(target_user_id, patch)
    return f"用户画像已更新：{json.dumps(updated, ensure_ascii=False)}"


def _target_user_id(user_id: str = "") -> str:
    return user_id or os.environ.get("HEALTH_GUIDE_USER_ID", "default_user")


def _append_unique(items: list, value: str) -> list:
    value = (value or "").strip()
    if not value:
        return items
    normalized = {str(x).strip() for x in items}
    if value not in normalized:
        items.append(value)
    return items


@tool
def set_physical_stats(
    age: int = 0,
    weight_kg: float = 0,
    height_cm: float = 0,
    user_id: str = "",
):
    """结构化记录年龄、体重(kg)、身高(cm)。未知字段传 0 即可跳过。"""
    patch = {"physical_stats": {}}
    if age and age > 0:
        patch["physical_stats"]["age"] = int(age)
    if weight_kg and weight_kg > 0:
        patch["physical_stats"]["weight"] = float(weight_kg)
    if height_cm and height_cm > 0:
        patch["physical_stats"]["height"] = float(height_cm)
    if not patch["physical_stats"]:
        return "[Profile Update Skip] 未提供有效年龄/体重/身高。"
    updated = update_profile_in_store(_target_user_id(user_id), patch)
    return f"身体数据已更新：{json.dumps(updated['physical_stats'], ensure_ascii=False)}"


@tool
def add_injury(injury: str, user_id: str = ""):
    """结构化记录伤病/术后/康复状态，例如 ACL 术后、半月板损伤。"""
    injury = (injury or "").strip()
    if not injury:
        return "[Profile Update Skip] injury 为空。"
    target = _target_user_id(user_id)
    profile = get_profile_from_store(target)
    stats = profile.get("physical_stats") or {}
    injuries = list(stats.get("injuries") or [])
    _append_unique(injuries, injury)
    updated = update_profile_in_store(target, {"physical_stats": {"injuries": injuries}})
    return f"伤病记录已更新：{json.dumps(updated['physical_stats'].get('injuries', []), ensure_ascii=False)}"


@tool
def set_dietary_goal(goal: str, user_id: str = ""):
    """结构化记录饮食/体型目标，例如 增肌、减脂、健康。"""
    goal = (goal or "").strip()
    if not goal:
        return "[Profile Update Skip] goal 为空。"
    updated = update_profile_in_store(
        _target_user_id(user_id),
        {"dietary_context": {"goal": goal}},
    )
    return f"饮食目标已更新：{updated.get('dietary_context', {}).get('goal', '')}"


@tool
def add_dietary_preference(preference: str, kind: str = "dislike", user_id: str = ""):
    """结构化记录饮食偏好。kind 可选 like, dislike, allergy。"""
    preference = (preference or "").strip()
    kind = (kind or "dislike").strip().lower()
    if not preference:
        return "[Profile Update Skip] preference 为空。"
    if kind not in {"like", "dislike", "allergy"}:
        return "[Profile Update Error] kind 必须是 like/dislike/allergy。"
    prefix = {
        "like": "喜欢",
        "dislike": "不喜欢/不吃",
        "allergy": "过敏",
    }[kind]
    value = preference if any(x in preference for x in ("喜欢", "不吃", "过敏", "不耐")) else f"{prefix}：{preference}"
    target = _target_user_id(user_id)
    profile = get_profile_from_store(target)
    dietary = profile.get("dietary_context") or {}
    prefs = list(dietary.get("preferences") or [])
    _append_unique(prefs, value)
    updated = update_profile_in_store(target, {"dietary_context": {"preferences": prefs}})
    return f"饮食偏好已更新：{json.dumps(updated['dietary_context'].get('preferences', []), ensure_ascii=False)}"


@tool
def add_stress_source(source: str, user_id: str = ""):
    """结构化记录压力来源，例如 工作加班、论文 deadline、比赛压力。"""
    source = (source or "").strip()
    if not source:
        return "[Profile Update Skip] source 为空。"
    target = _target_user_id(user_id)
    profile = get_profile_from_store(target)
    mental = profile.get("mental_state") or {}
    sources = list(mental.get("stress_sources") or [])
    _append_unique(sources, source)
    updated = update_profile_in_store(target, {"mental_state": {"stress_sources": sources}})
    return f"压力源已更新：{json.dumps(updated['mental_state'].get('stress_sources', []), ensure_ascii=False)}"


@tool
def set_response_style(
    tone: str = "",
    humor: str = "",
    formality: str = "",
    language: str = "",
    user_id: str = "",
):
    """结构化记录回答风格偏好，例如 tone=concise, humor=light, formality=casual, language=zh。"""
    style_patch = {}
    for key, value in {
        "tone": tone,
        "humor": humor,
        "formality": formality,
        "language": language,
    }.items():
        value = (value or "").strip()
        if value:
            style_patch[key] = value
    if not style_patch:
        return "[Profile Update Skip] 未提供有效风格字段。"
    updated = update_profile_in_store(
        _target_user_id(user_id),
        {"response_style": style_patch},
    )
    return f"回答风格已更新：{json.dumps(updated.get('response_style', {}), ensure_ascii=False)}"

# 2. 定义 TDEE 计算工具
@tool
def calculate_tdee(weight_kg: float, height_cm: float, age: int, activity_level: str = "sedentary"):
    """根据体重、身高、年龄计算每日热量消耗(TDEE)。activity_level 可选: sedentary, active, very_active"""
    # Mifflin-St Jeor 公式
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5

    multipliers = {
        "sedentary": 1.2,
        "active": 1.55,
        "very_active": 1.725
    }
    tdee = bmr * multipliers.get(activity_level, 1.2)
    return f"根据公式计算，基础代谢(BMR)为 {bmr} kcal，每日总消耗(TDEE)约为 {int(tdee)} kcal。"

# 工具列表
tools = [
    retrieve_trainer_knowledge,
    retrieve_nutritionist_knowledge,
    retrieve_psychologist_knowledge,
    retrieve_doctor_knowledge,
    retrieve_safety_guidelines,
    calculate_tdee,
    get_user_profile,
    update_user_profile,
    set_physical_stats,
    add_injury,
    set_dietary_goal,
    add_dietary_preference,
    add_stress_source,
    set_response_style,
    log_meal,
    log_workout,
    log_wellness_checkin,
    query_logs,
    push_reminder,
]
