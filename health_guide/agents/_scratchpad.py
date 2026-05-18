"""Shared scratchpad helpers.

Each expert reads prior notes from teammates (injected into the prompt) and
writes a brief note for downstream Critic + future turns.

Replan decisions are made by the dedicated ReplanJudge node (see
`replan_judge.py`), not by the experts themselves — so this module no longer
exposes a replan marker.
"""
import re
from typing import Dict


_NOTE_MAX_CHARS = 280

_EXPERT_LABELS = {
    "Trainer": "训练教练",
    "Nutritionist": "营养师",
    "Wellness": "身心康复师",
    "General": "助理",
}


def format_peer_notes(agent_notes: Dict[str, str], self_role: str) -> str:
    if not agent_notes:
        return ""
    lines = []
    for role, note in agent_notes.items():
        if role == self_role or not note:
            continue
        label = _EXPERT_LABELS.get(role, role)
        lines.append(f"- [{label}] {note}")
    if not lines:
        return ""
    return (
        "【协作伙伴最近的要点（来自共享 scratchpad，可作为参考但以你的专业判断为准）】\n"
        + "\n".join(lines)
        + "\n"
    )


def build_scratchpad_note(role: str, answer: str) -> str:
    if not answer:
        return ""
    text = answer.strip().replace("\n\n", "\n")
    if len(text) <= _NOTE_MAX_CHARS:
        return text
    return text[:_NOTE_MAX_CHARS].rstrip() + "…"


def extract_facts_from_notes(agent_notes: Dict[str, str], user_question: str = "") -> Dict[str, list]:
    """Cheap fact hints for episodic memory.

    This is intentionally heuristic: it adds retrieval-friendly labels to the
    episode without making another LLM call.
    """
    text = f"{user_question or ''}\n" + "\n".join(str(v) for v in (agent_notes or {}).values())
    facts: Dict[str, list] = {}

    injury_hits = []
    for pat in (
        r"ACL|前交叉韧带|韧带",
        r"半月板",
        r"冠心病|心脏病|心血管",
        r"腰椎|椎间盘|腰痛",
        r"肩袖|肩关节|肩痛",
        r"膝盖|膝关节",
    ):
        for hit in re.findall(pat, text, flags=re.IGNORECASE):
            injury_hits.append(hit.upper() if hit.lower() == "acl" else hit)
    if injury_hits:
        facts["injuries_or_risks"] = sorted(set(injury_hits))

    goals = []
    for goal in ("增肌", "减脂", "康复", "睡眠", "压力管理"):
        if goal in text:
            goals.append(goal)
    if goals:
        facts["goals"] = sorted(set(goals))

    numbers = re.findall(r"\b\d+(?:\.\d+)?\s*(?:kg|cm|kcal|g|分钟|次|组|天|岁)\b", text, flags=re.IGNORECASE)
    if numbers:
        facts["numbers"] = numbers[:8]
    return facts
