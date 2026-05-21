"""Smoke checks for structured personalization decision points.

Usage:
    python3 scripts/smoke_personalization_decision_points.py
"""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wellness_copilot.personalization import (  # noqa: E402
    build_personalization_decision_points,
    check_decision_points_landed,
)


def _ctx(profile: dict) -> dict:
    return {"raw_profile": profile}


def _ids(points) -> set[str]:
    return {p.id for p in points}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    phone_profile = {
        "physical_stats": {"age": 35, "weight": 70, "height": 170, "injuries": []},
        "dietary_context": {"goal": "健康"},
        "mental_state": {"sleep_quality": "差", "stress_sources": ["手机使用"]},
    }
    phone_q = "我睡前总是刷手机停不下来，越刷越清醒，怎么建立睡前习惯？"
    phone_points = build_personalization_decision_points(_ctx(phone_profile), phone_q, role="Psychologist")
    phone_ids = _ids(phone_points)
    _assert("psych_phone_sleep_flow" in phone_ids, "missing phone sleep decision point")
    _assert("psych_sleep_quality_flow" in phone_ids, "missing sleep quality decision point")
    phone_answer = "睡前60分钟把手机放到床外并开启勿扰，固定起床时间来修复睡眠；先连续3天只完成这个流程。"
    _assert(
        check_decision_points_landed(phone_answer, phone_points)["satisfied"],
        "phone sleep answer should satisfy P3",
    )

    run_profile = {
        "physical_stats": {"age": 29, "weight": 66, "height": 169, "injuries": []},
        "dietary_context": {"goal": "健康"},
        "mental_state": {},
    }
    run_q = "我想从零开始跑5公里，给我一个保守的4周入门计划。"
    run_points = build_personalization_decision_points(_ctx(run_profile), run_q, role="Orchestrator")
    run_ids = _ids(run_points)
    _assert("trainer_5k_beginner_run_walk" in run_ids, "missing 5K run-walk point")
    _assert("trainer_age_intensity" in run_ids, "missing age intensity point")
    _assert("nutrition_race_carbs" not in run_ids, "5K beginner plan must not trigger race-carb point")

    acl_profile = {
        "physical_stats": {"age": 30, "weight": 78, "height": 180, "injuries": ["ACL术后6个月"]},
        "dietary_context": {"goal": "健康"},
        "mental_state": {},
    }
    acl_q = "我能开始做深蹲了吗？"
    acl_points = build_personalization_decision_points(_ctx(acl_profile), acl_q, role="Trainer")
    acl_ids = _ids(acl_points)
    _assert("trainer_acl_squat_gate" in acl_ids, "missing ACL squat gate point")
    acl_nutrition_ids = _ids(build_personalization_decision_points(_ctx(acl_profile), acl_q, role="Nutritionist"))
    _assert("nutrition_weight_protein" not in acl_nutrition_ids, "ACL squat permission must not trigger protein point")

    diabetes_profile = {
        "physical_stats": {"age": 45, "weight": 82, "height": 170, "injuries": []},
        "dietary_context": {"goal": "减脂"},
        "mental_state": {},
    }
    diabetes_q = "我有糖尿病，想直接开始HIIT 30分钟减脂，可以吗？"
    diabetes_points = build_personalization_decision_points(_ctx(diabetes_profile), diabetes_q, role="Doctor")
    diabetes_ids = _ids(diabetes_points)
    _assert("doctor_diabetes_hiit_guard" in diabetes_ids, "missing diabetes HIIT doctor point")
    all_diabetes_ids = _ids(build_personalization_decision_points(_ctx(diabetes_profile), diabetes_q, role="Orchestrator"))
    _assert("nutrition_weight_protein" not in all_diabetes_ids, "diabetes HIIT must not trigger unrelated protein point")

    generic = "根据你的情况，建议循序渐进，注意安全。"
    _assert(
        not check_decision_points_landed(generic, diabetes_points)["satisfied"],
        "generic profile echo must not satisfy P3",
    )

    print("personalization decision point smoke checks passed")


if __name__ == "__main__":
    main()
