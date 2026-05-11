"""Shared scratchpad helpers.

Each expert reads prior notes from teammates (injected into the prompt) and
writes a brief note for downstream Critic + future turns.

Replan decisions are made by the dedicated ReplanJudge node (see
`replan_judge.py`), not by the experts themselves — so this module no longer
exposes a replan marker.
"""
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
