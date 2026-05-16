"""Fallback helpers for long agent chains.

These helpers keep node failure behavior consistent without changing the
normal successful path.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List


EXPERT_LABELS = {
    "Trainer": "训练教练",
    "Nutritionist": "营养师",
    "Wellness": "身心康复师",
    "General": "通用助理",
}

_EXPERT_FALLBACKS = {
    "Trainer": (
        "抱歉，训练建议节点暂时不可用。我先给你一个保守原则：如果有疼痛、肿胀、术后恢复或旧伤，"
        "先暂停相关部位负重训练，优先做低强度活动，并在医生或理疗师确认后再恢复进阶训练。"
    ),
    "Nutritionist": (
        "抱歉，营养建议节点暂时不可用。我先给你一个保守原则：保持规律三餐，优先选择足量蛋白质、"
        "蔬果和主食，避免极端低热量或不明补剂；如有慢病、过敏或特殊饮食限制，请以医生/营养师建议为准。"
    ),
    "Wellness": (
        "抱歉，身心恢复建议节点暂时不可用。我先给你一个保守原则：先降低训练和工作负荷，保证睡眠，"
        "若出现持续疼痛、胸闷、头晕、心率异常或失眠超过两周，请优先就医评估。"
    ),
    "General": (
        "抱歉，通用助理节点暂时不可用。你可以稍后重试；如果问题涉及明显身体不适或急性症状，"
        "请优先联系医生或当地急救服务。"
    ),
}

_SAFETY_PATTERNS = [
    r"胸痛|胸闷|胸口痛|呼吸困难|喘不上气",
    r"心率异常|心跳异常|心悸|心律不齐|心跳很快|心跳过快",
    r"头晕|眩晕|晕厥|昏厥|快晕倒",
    r"持续疼痛|疼痛持续|剧痛|明显疼痛",
    r"肿胀不消|肿胀不退|明显肿胀",
    r"术后|手术后|ACL|前交叉韧带|半月板|韧带|骨折|撕裂",
    r"用药剂量|药物剂量|处方|吃多少药|服药量",
    r"极端低卡|极低热量|低于\s*1200|低于\s*1500|每天\s*[0-9]{2,3}\s*(?:kcal|千卡|大卡)",
]

_SAFETY_RE = re.compile("|".join(f"(?:{p})" for p in _SAFETY_PATTERNS), re.IGNORECASE)

SAFETY_WARNING = (
    "安全提示：你的问题或草稿中包含可能需要专业评估的健康风险信号。"
    "在获得医生、运动医学医生或理疗师确认前，请先暂停可能加重症状的训练/饮食调整；"
    "如果有胸痛、胸闷、明显心率异常、头晕晕厥、持续疼痛或肿胀，请尽快就医。"
)


def default_fresh_plan() -> dict:
    return {
        "plan": ["General"],
        "executed": [],
        "replan_count": 0,
        "replan_context": "",
        "next": [],
    }


def empty_replan() -> dict:
    return {
        "plan": [],
        "replan_context": "",
        "next": [],
    }


def expert_fallback_answer(role: str, exc: Exception | None = None) -> str:
    answer = _EXPERT_FALLBACKS.get(role, _EXPERT_FALLBACKS["General"])
    if exc is None:
        return answer
    return f"{answer}\n\n（系统提示：{role} 节点临时失败，已启用保守兜底。）"


def expert_error_update(role: str, exc: Exception) -> dict:
    answer = expert_fallback_answer(role, exc)
    return {
        "expert_responses": {role: answer},
        "agent_notes": {role: f"{EXPERT_LABELS.get(role, role)}节点失败，已给出保守兜底建议。"},
        "last_tools": [f"ERROR:{role}:{type(exc).__name__}"],
        "retrieval_hits": 0,
    }


def ordered_expert_responses(executed: Iterable[str], responses: Dict[str, str]) -> List[str]:
    sections = []
    seen = set()
    for role in executed or []:
        text = (responses.get(role) or "").strip()
        if not text:
            continue
        seen.add(role)
        label = EXPERT_LABELS.get(role, role)
        sections.append(f"{label}建议：\n{text}")
    for role, text in (responses or {}).items():
        if role in seen or not (text or "").strip():
            continue
        label = EXPERT_LABELS.get(role, role)
        sections.append(f"{label}建议：\n{text.strip()}")
    return sections


def aggregate_fallback(executed: Iterable[str], responses: Dict[str, str]) -> str:
    sections = ordered_expert_responses(executed, responses)
    if not sections:
        return "抱歉，未能获取专家建议，请重试。"
    return "我先保留已完成专家的建议，供你参考：\n\n" + "\n\n".join(sections)


def has_safety_risk(*texts: str) -> bool:
    combined = "\n".join(t for t in texts if t)
    return bool(_SAFETY_RE.search(combined))


def add_safety_warning(draft: str) -> str:
    if draft.startswith("安全提示："):
        return draft
    return f"{SAFETY_WARNING}\n\n{draft}"
