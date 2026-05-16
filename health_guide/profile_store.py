import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .config import DEFAULT_USER_PROFILE, PROFILE_STORE_PATH


# Top-level identity keys all experts get (cheap, useful for personalization tone).
_COMMON_TOP_FIELDS = ("name", "identity")

# Per-expert whitelist of (section_key, [field_keys] | None).
# `None` for field_keys = include the whole section.
PROFILE_FIELDS_BY_ROLE: Dict[str, Dict[str, Optional[list]]] = {
    "Trainer": {
        "physical_stats": ["age", "weight", "height", "injuries"],
        # Trainer wants to know the goal (e.g. 增肌/减脂 affects programming).
        "dietary_context": ["goal"],
    },
    "Nutritionist": {
        "physical_stats": ["age", "weight", "height"],
        "dietary_context": None,  # preferences/goal/provider all matter
    },
    "Wellness": {
        "physical_stats": ["age", "injuries"],
        "mental_state": None,
    },
    "General": {
        "physical_stats": ["age"],
        "dietary_context": ["goal"],
        "mental_state": ["stress_sources"],
    },
}


def _store_path() -> Path:
    return Path(PROFILE_STORE_PATH)


def _ensure_store_exists() -> None:
    p = _store_path()
    if not p.exists():
        p.write_text("{}", encoding="utf-8")


def _read_store() -> Dict[str, Any]:
    _ensure_store_exists()
    p = _store_path()
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_store(data: Dict[str, Any]) -> None:
    p = _store_path()
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def get_user_profile(user_id: str) -> Dict[str, Any]:
    data = _read_store()
    if user_id not in data:
        data[user_id] = copy.deepcopy(DEFAULT_USER_PROFILE)
        _write_store(data)
    return data[user_id]


def update_user_profile(user_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    data = _read_store()
    current = data.get(user_id, copy.deepcopy(DEFAULT_USER_PROFILE))
    merged = _deep_merge(current, patch)
    data[user_id] = merged
    _write_store(data)
    return merged


def profile_to_prompt_text(profile: Dict[str, Any]) -> str:
    return json.dumps(profile, ensure_ascii=False)


def profile_subset_for(role: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Return a role-specific subset of profile fields.

    Reduces token cost and prompt noise — Trainer doesn't need dietary
    preferences, Nutritionist doesn't need workout history, etc.
    Falls back to the full profile for unknown roles.
    """
    spec = PROFILE_FIELDS_BY_ROLE.get(role)
    if spec is None:
        return profile

    subset: Dict[str, Any] = {}
    for key in _COMMON_TOP_FIELDS:
        if key in profile:
            subset[key] = profile[key]

    for section_key, field_keys in spec.items():
        section = profile.get(section_key)
        if not isinstance(section, dict):
            continue
        if field_keys is None:
            subset[section_key] = section
            continue
        picked = {k: section[k] for k in field_keys if k in section}
        if picked:
            subset[section_key] = picked
    return subset


def profile_to_prompt_text_for(role: str, profile: Dict[str, Any]) -> str:
    return json.dumps(profile_subset_for(role, profile), ensure_ascii=False)
