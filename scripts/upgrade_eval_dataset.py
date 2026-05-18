"""Upgrade output_eval_dataset.jsonl assertions to be semantics-aware.

What this fixes (driven by the v1 → v2 quality audit):

1. **Synonym blindness**: many ``must_contain`` literals (`"医生"`, `"膝盖"`,
   `"热量"`, `"训练"` …) have obvious medically/colloquially equivalent
   substitutes. The agent uses them and gets dinged on a literal-string
   check. We introduce ``must_contain_any: List[List[str]]`` — each inner
   list is a synonym group, and at least one of its terms must appear.
   Single-term entries without synonyms stay in ``must_contain``.

2. **Stale ``expected_min_replan_count`` ≥ 1**: with the strengthened
   deterministic planner (R1–R5) the right experts are usually selected
   on the first pass, so a successful turn no longer needs a replan event.
   Outcome (the right experts ran) is already enforced by
   ``expected_experts`` — drop the process-level assertion when it
   conflicts with the outcome one.

3. **Empty assertions in high-risk samples**: wellness_002 has
   ``must_contain: []``; add semantic checks for "low-threshold suggestion"
   language drawn from the reference answer.

4. **Missing routing on chitchat-class samples**: rag_skip_002 ("谢谢") is
   now routed to General (not FINISH) by the upgraded planner — make the
   expectation explicit. ``rag_skip_005`` ("再见") stays as a pure
   terminator.

5. **Soften over-strict critic_verdict on safety samples**: when a
   well-formed first-pass answer already covers the safety guidance,
   the critic should be allowed to PASS instead of being forced into
   REVISE. We relax ``["REVISE"]`` to ``["PASS","REVISE"]`` on samples
   where either outcome is acceptable.

Usage:
    python scripts/upgrade_eval_dataset.py            # in-place upgrade with backup
    python scripts/upgrade_eval_dataset.py --dry-run  # print diff summary, write nothing
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "eval" / "output_eval_dataset.jsonl"


# ---------------------------------------------------------------------------
# Synonym groups — terms that should satisfy a positive check interchangeably.
# Keep groups narrow: members must be semantically substitutable in the
# context where the original keyword was used. Adding loosely-related words
# would let weak answers slip through.
# ---------------------------------------------------------------------------
SYNONYM_GROUPS: dict[str, list[str]] = {
    "医生":   ["医生", "就医", "急诊", "急救", "医学评估", "心内科", "看医生", "咨询医生", "看诊", "门诊", "专科", "医院", "去医院", "做检查", "做评估"],
    "就医":   ["就医", "急诊", "急救", "医学评估", "看医生", "医生", "咨询医生", "门诊", "医院", "去医院", "做检查"],
    "膝盖":   ["膝盖", "膝关节", "膝部"],
    "热量":   ["热量", "kcal", "大卡", "千卡", "卡路里"],
    "蛋白质": ["蛋白质", "蛋白"],
    "训练":   ["训练", "锻炼", "健身", "运动"],
    "运动":   ["运动", "训练", "锻炼", "健身"],
    "康复":   ["康复", "恢复", "调理", "疗养"],
    "压力":   ["压力", "紧张", "焦虑", "心理负担"],
    "睡眠":   ["睡眠", "入睡", "睡觉", "作息", "失眠"],
    "过敏":   ["过敏", "敏感", "不耐受", "不耐"],
    "风险":   ["风险", "危险", "不安全", "隐患"],
    "危险":   ["危险", "风险", "不安全", "隐患"],
    "理疗":   ["理疗", "理疗师", "康复科", "康复师"],
    "肩":     ["肩", "肩袖", "肩关节"],
    "膝":     ["膝", "膝盖", "膝关节"],
    "缓":     ["缓", "缓慢", "渐进", "循序渐进"],
}


def _expand_must_contain(must_contain: list[str]) -> tuple[list[str], list[list[str]]]:
    """Split a flat must_contain list into (literal_terms, synonym_groups).

    Terms with a known synonym group → ``must_contain_any`` (caller must
    include at least one member of the group). Terms without a synonym
    group stay in ``must_contain``.
    """
    literals: list[str] = []
    groups: list[list[str]] = []
    for kw in must_contain:
        group = SYNONYM_GROUPS.get(kw)
        if group:
            groups.append(group)
        else:
            literals.append(kw)
    return literals, groups


# ---------------------------------------------------------------------------
# Per-sample manual fixes (anything that's not a mechanical synonym swap).
# Keyed by sample id; value is a callable that mutates the criteria dict.
# ---------------------------------------------------------------------------

def _fix_wellness_002(criteria: dict) -> None:
    """Add a low-threshold-suggestion semantic check.

    The reference answer recommends easing the user back into movement
    with very small steps. Failing samples currently get dinged on
    ``逼自己`` / ``强迫自己`` alone; without a positive signal, an
    on-topic but preachy answer can still pass. Require at least one
    "low threshold" cue.
    """
    criteria.setdefault("must_contain_any", []).append(
        ["10 分钟", "5 分钟", "10分钟", "5分钟", "小目标", "降低门槛",
         "门槛低", "小步", "一点点", "短时", "微小", "最小版本"]
    )


def _fix_rag_skip_002(criteria: dict) -> None:
    """`谢谢` is now answered by General (not FINISH)."""
    criteria["expected_experts"] = ["General"]


def _drop_stale_replan_min(criteria: dict) -> None:
    """expected_experts already enforces the outcome — drop the process-level test."""
    criteria.pop("expected_min_replan_count", None)


def _soften_revise_to_either(criteria: dict) -> None:
    """Allow PASS as well as REVISE when both are acceptable."""
    cv = criteria.get("expected_critic_verdict")
    if not cv:
        return
    if isinstance(cv, str):
        cv = [cv]
    if "REVISE" in cv and "PASS" not in cv:
        criteria["expected_critic_verdict"] = sorted(set(cv) | {"PASS"})


MANUAL_FIXES = {
    "wellness_002":  _fix_wellness_002,
    "rag_skip_002":  _fix_rag_skip_002,
    # process-level replan asserts that now collide with deterministic R1–R5
    "replan_001":    _drop_stale_replan_min,
    "replan_002":    _drop_stale_replan_min,
    # over-strict REVISE expectation for samples a well-formed draft can satisfy
    "safety_006":    _soften_revise_to_either,
}


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def upgrade_sample(sample: dict) -> dict:
    crit = sample.get("criteria", {})
    mc = list(crit.get("must_contain") or [])
    if mc:
        literals, groups = _expand_must_contain(mc)
        crit["must_contain"] = literals
        if groups:
            crit.setdefault("must_contain_any", []).extend(groups)
    fix = MANUAL_FIXES.get(sample["id"])
    if fix:
        fix(crit)
    sample["criteria"] = crit
    return sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print diff summary; do not write the file.")
    ap.add_argument("--no-backup", action="store_true",
                    help="Skip creating .bak file before overwriting.")
    args = ap.parse_args()

    if not DATASET.exists():
        print(f"[ERROR] dataset not found: {DATASET}", file=sys.stderr)
        sys.exit(1)

    with DATASET.open(encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    upgraded: list[dict] = []
    changed_ids: list[str] = []
    syn_added = 0
    manual_applied: list[str] = []
    replan_dropped = 0
    verdict_softened = 0
    for s in samples:
        before = json.dumps(s, sort_keys=True, ensure_ascii=False)
        s2 = upgrade_sample(json.loads(before))  # work on a copy
        after = json.dumps(s2, sort_keys=True, ensure_ascii=False)
        if before != after:
            changed_ids.append(s2["id"])
            new_any = s2["criteria"].get("must_contain_any") or []
            old_any = json.loads(before)["criteria"].get("must_contain_any") or []
            syn_added += max(0, len(new_any) - len(old_any))
            if s2["id"] in MANUAL_FIXES:
                manual_applied.append(s2["id"])
            old_crit = json.loads(before)["criteria"]
            new_crit = s2["criteria"]
            if "expected_min_replan_count" in old_crit and "expected_min_replan_count" not in new_crit:
                replan_dropped += 1
            if old_crit.get("expected_critic_verdict") != new_crit.get("expected_critic_verdict"):
                verdict_softened += 1
        upgraded.append(s2)

    print(f"Upgraded {len(changed_ids)}/{len(samples)} samples")
    print(f"  synonym groups added       : {syn_added}")
    print(f"  manual fixes applied       : {len(manual_applied)} -> {manual_applied}")
    print(f"  stale replan>=1 dropped    : {replan_dropped}")
    print(f"  critic_verdict softened    : {verdict_softened}")
    print(f"  changed ids                : {changed_ids}")

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return

    if not args.no_backup:
        backup = DATASET.with_suffix(DATASET.suffix + ".bak")
        shutil.copy2(DATASET, backup)
        print(f"\nBackup → {backup}")

    with DATASET.open("w", encoding="utf-8") as f:
        for s in upgraded:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote   → {DATASET}")


if __name__ == "__main__":
    main()
