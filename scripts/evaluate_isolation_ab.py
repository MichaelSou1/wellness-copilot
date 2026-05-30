"""A/B evaluation of subagent context isolation (隔离 ON vs OFF).

This is an *intervention* experiment, complementary to:
  - L4 evaluate_architecture.py  — asserts isolation *happens* (structural).
  - L5 evaluate_output.py        — absolute E2E quality (ceiling-bound).

Here we run the SAME input twice — once with isolation ON (production default)
and once with it OFF (full profile to every expert + same-batch peer notes +
full transcript) — and measure the *effect* of isolation at two layers:

  1. subagent layer (primary diagnostic) — pairwise A/B judge of each expert's
     own output, plus a deterministic cross-domain *leakage* rate driven by the
     dataset's `leak_traps`. This is where isolation acts most directly.
  2. E2E layer (acceptance) — pairwise A/B judge of the final merged answer.

Scoring is pairwise (not absolute) because the existing L5 absolute scores are
ceiling-bound and can't resolve isolation's small deltas. A/B order is randomized
per comparison to cancel position bias.

Usage:
    python scripts/evaluate_isolation_ab.py                       # full dataset, all 3 dims OFF
    python scripts/evaluate_isolation_ab.py --samples iso_hard_001,iso_hard_002
    python scripts/evaluate_isolation_ab.py --ablate profile      # flip only the profile dim
    python scripts/evaluate_isolation_ab.py --no-judge            # leakage + cost only (fast)

Output: reports/isolation_ab_report_YYYYMMDD-HHMMSS.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Reuse the L5 harness: graph runner (+profiler), judge factory, profile seeding.
import evaluate_output as eo  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from wellness_copilot import isolation  # noqa: E402
from wellness_copilot.llm import extract_text_content  # noqa: E402
from wellness_copilot.profile_store import update_user_profile  # noqa: E402


# ---------------------------------------------------------------------------
# Pairwise judge prompts
# ---------------------------------------------------------------------------

_E2E_JUDGE_SYSTEM = """\
你是健康类 AI 助手的对比评审员。下面给出同一问题的两个回答（回答X / 回答Y），
请逐维度判断哪个更好。每个维度只能选 "X" / "Y" / "tie"。

维度：
- relevance 切题性
- completeness 完整性（对照参考答案的知识点覆盖）
- safety 安全性（红线场景是否建议就医；是否给出与已知伤病冲突的危险建议）
- personalization 个性化（是否基于画像数值做推导，而非泛泛而谈；是否出现与该问题无关的跨域信息堆砌）
- coherence 连贯性（多专家观点是否融合、无内部矛盾）
- overall 综合

注意：更长、堆砌更多无关信息不等于更好。若两者实质相当，请判 "tie"。

严格输出一行 JSON，不要 markdown：
{"overall":"X|Y|tie","relevance":"...","completeness":"...","safety":"...","personalization":"...","coherence":"...","reason":"<一句话>"}
"""

_SUBAGENT_JUDGE_SYSTEM = """\
你是评审单个专家子回答质量的对比评审员。这是「{role}」专家针对用户问题给出的子回答。
给出两个版本（回答X / 回答Y），逐维度判断哪个更适合作为「{role}」的专业输出。
每个维度只能选 "X" / "Y" / "tie"。

维度：
- role_relevance 角色切题（是否聚焦本专家职责领域）
- focus 聚焦/不串域（是否避免谈论本专家职责之外、应由其他专家负责的内容）
- safety 安全性（是否守住专业边界、必要时建议就医、不与已知风险冲突）
- usefulness 可执行性（建议是否具体、含数值/步骤）
- overall 综合

注意：把不属于本专家职责的跨域内容塞进回答，应在 focus 上判为更差。

严格输出一行 JSON，不要 markdown：
{{"overall":"X|Y|tie","role_relevance":"...","focus":"...","safety":"...","usefulness":"...","reason":"<一句话>"}}
"""

_E2E_DIMS = ("overall", "relevance", "completeness", "safety", "personalization", "coherence")
_SUBAGENT_DIMS = ("overall", "role_relevance", "focus", "safety", "usefulness")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ablate_kwargs(ablate: str) -> dict:
    """Return the isolation_override kwargs for the OFF arm."""
    if ablate == "all":
        return {"profile": False, "peer": False, "history": False}
    if ablate in ("profile", "peer", "history"):
        return {ablate: False}
    raise ValueError(f"unknown --ablate {ablate!r}")


def _run_arm(sample: dict, profile: dict, override_kwargs: dict) -> tuple[str, dict, dict]:
    """Seed a fresh user and run the full graph for one arm.

    A fresh user_id per arm keeps episode_store / profile state from leaking
    across arms. Both arms get the identical profile patch.
    """
    user_id = f"isoab_{sample['id']}_{uuid.uuid4().hex[:8]}"
    if profile:
        update_user_profile(user_id, profile)
    with isolation.isolation_override(**override_kwargs):
        answer, state, perf = eo._run_sample(sample, user_id, verbose=False)
    expert_outputs = dict(state.get("expert_responses") or {})
    return answer, {
        "answer": answer,
        "executed": state.get("executed", []),
        "expert_outputs": expert_outputs,
        "input_tokens": int((perf or {}).get("input_tokens", 0) or 0),
        "total_tokens": int((perf or {}).get("total_tokens", 0) or 0),
        "wall_ms": float((perf or {}).get("total_wall_ms", 0) or 0),
    }, perf


def _count_leaks(expert_outputs: dict, leak_traps: dict) -> dict:
    """Deterministic cross-domain leakage check.

    A leak = a trapped (cross-domain) term appears verbatim in the output of an
    expert whose role crop should have removed it. Plain substring presence —
    any mention by the wrong expert counts (negation context is irrelevant here:
    the expert shouldn't know the field at all).
    """
    per_expert: dict[str, dict] = {}
    checked = 0
    leaked_outputs = 0
    total_hits = 0
    for role, terms in (leak_traps or {}).items():
        out = expert_outputs.get(role)
        if not out:
            continue
        checked += 1
        hits = [t for t in terms if t and t in out]
        if hits:
            leaked_outputs += 1
            total_hits += len(hits)
        per_expert[role] = {"trap_terms": terms, "leaked_terms": hits}
    return {
        "checked_outputs": checked,
        "leaked_outputs": leaked_outputs,
        "total_term_hits": total_hits,
        "per_expert": per_expert,
    }


def _pairwise_judge(judge_llm, system: str, iso_text: str, noniso_text: str,
                    context: str, dims: tuple, rng: random.Random) -> dict:
    iso_is_x = rng.random() < 0.5
    x_text, y_text = (iso_text, noniso_text) if iso_is_x else (noniso_text, iso_text)
    prompt = (
        f"{context}\n\n"
        f"[回答X]\n{x_text}\n\n"
        f"[回答Y]\n{y_text}"
    )
    try:
        resp = judge_llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ])
        raw = extract_text_content(resp).strip()
        if raw.startswith("```"):
            inner = raw.split("```", 2)
            raw = inner[1].lstrip("json").strip() if len(inner) >= 2 else raw
        verdict = json.loads(raw)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    def _decode(v: str) -> str:
        v = str(v or "").strip().upper()
        if v == "TIE":
            return "tie"
        if v == "X":
            return "iso" if iso_is_x else "noniso"
        if v == "Y":
            return "noniso" if iso_is_x else "iso"
        return "tie"

    out = {d: _decode(verdict.get(d, "tie")) for d in dims}
    out["reason"] = verdict.get("reason", "")
    out["_iso_was_X"] = iso_is_x
    return out


def _build_e2e_context(sample: dict) -> str:
    profile = sample.get("profile", {})
    turns = sample.get("turns", [])
    last_user = next((t["content"] for t in reversed(turns) if t["role"] == "user"), "")
    reference = sample.get("reference_answer", "（无参考答案）")
    return (
        f"[用户画像]\n{json.dumps(profile, ensure_ascii=False)}\n\n"
        f"[用户问题]\n{last_user}\n\n"
        f"[参考答案]\n{reference}"
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _tally(records: list[dict], dims: tuple) -> dict:
    """Count iso/noniso/tie winners per dimension across a list of verdicts."""
    out = {}
    for d in dims:
        c = {"iso": 0, "noniso": 0, "tie": 0}
        for r in records:
            v = r.get(d)
            if v in c:
                c[v] += 1
        total = sum(c.values())
        c["n"] = total
        c["iso_win_rate"] = round(c["iso"] / total, 3) if total else None
        c["noniso_win_rate"] = round(c["noniso"] / total, 3) if total else None
        out[d] = c
    return out


def _aggregate(results: list[dict], dataset: str, ablate: str, judge_enabled: bool) -> dict:
    e2e_verdicts = [r["e2e_judge"] for r in results if r.get("e2e_judge") and not r["e2e_judge"].get("error")]

    sub_verdicts_all = []
    sub_by_expert: dict[str, list] = {}
    for r in results:
        for role, v in (r.get("subagent_judge") or {}).items():
            if v and not v.get("error"):
                sub_verdicts_all.append(v)
                sub_by_expert.setdefault(role, []).append(v)

    # Leakage per arm
    def _leak_summary(arm_key: str) -> dict:
        checked = sum(r["leaks"][arm_key]["checked_outputs"] for r in results if r.get("leaks"))
        leaked = sum(r["leaks"][arm_key]["leaked_outputs"] for r in results if r.get("leaks"))
        hits = sum(r["leaks"][arm_key]["total_term_hits"] for r in results if r.get("leaks"))
        return {
            "checked_outputs": checked,
            "leaked_outputs": leaked,
            "leak_rate": round(leaked / checked, 3) if checked else None,
            "total_term_hits": hits,
        }

    # Cost per arm
    def _avg(arm_key: str, field: str) -> float | None:
        vals = [r[arm_key][field] for r in results if r.get(arm_key)]
        return round(sum(vals) / len(vals), 1) if vals else None

    by_category_e2e: dict[str, list] = {}
    for r in results:
        v = r.get("e2e_judge")
        if v and not v.get("error"):
            by_category_e2e.setdefault(r["category"], []).append(v)

    return {
        "run_id": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        "dataset": dataset,
        "ablate": ablate,
        "off_arm_override": _ablate_kwargs(ablate),
        "judge_enabled": judge_enabled,
        "total_samples": len(results),
        "leakage": {
            "isolated_arm": _leak_summary("iso"),
            "non_isolated_arm": _leak_summary("noniso"),
        },
        "cost": {
            "isolated_arm": {
                "avg_input_tokens": _avg("iso", "input_tokens"),
                "avg_total_tokens": _avg("iso", "total_tokens"),
                "avg_wall_ms": _avg("iso", "wall_ms"),
            },
            "non_isolated_arm": {
                "avg_input_tokens": _avg("noniso", "input_tokens"),
                "avg_total_tokens": _avg("noniso", "total_tokens"),
                "avg_wall_ms": _avg("noniso", "wall_ms"),
            },
        },
        "subagent_pairwise": {
            "overall": _tally(sub_verdicts_all, _SUBAGENT_DIMS),
            "by_expert": {
                role: _tally(vs, _SUBAGENT_DIMS) for role, vs in sorted(sub_by_expert.items())
            },
        },
        "e2e_pairwise": {
            "overall": _tally(e2e_verdicts, _E2E_DIMS),
            "by_category": {
                cat: _tally(vs, _E2E_DIMS) for cat, vs in sorted(by_category_e2e.items())
            },
        },
        "details": results,
    }


def _arm_view(arm: dict) -> dict:
    """Trim per-arm payload for the report (drop full expert outputs duplication)."""
    return {
        "answer": arm["answer"],
        "executed": arm["executed"],
        "input_tokens": arm["input_tokens"],
        "total_tokens": arm["total_tokens"],
        "wall_ms": arm["wall_ms"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="A/B evaluation of subagent context isolation")
    parser.add_argument("--dataset", default="eval/isolation_hard_dataset.jsonl")
    parser.add_argument("--out", default="reports/isolation_ab_report.json")
    parser.add_argument("--samples", default="", help="comma-separated sample IDs")
    parser.add_argument("--ablate", default="all", choices=["all", "profile", "peer", "history"],
                        help="which isolation dimension(s) the OFF arm flips")
    parser.add_argument("--no-judge", action="store_true", help="leakage + cost only")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed for A/B order")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    off_kwargs = _ablate_kwargs(args.ablate)

    ds_path = PROJECT_ROOT / args.dataset
    if not ds_path.exists():
        print(f"[ERROR] dataset not found: {ds_path}", file=sys.stderr)
        sys.exit(1)

    samples: list[dict] = []
    with ds_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] skip line {lineno}: {exc}", file=sys.stderr)

    if args.samples:
        wanted = set(args.samples.split(","))
        samples = [s for s in samples if s["id"] in wanted]
        if not samples:
            print(f"[ERROR] no samples matched {wanted}", file=sys.stderr)
            sys.exit(1)

    print("=" * 64)
    print("Isolation A/B Evaluation — Wellness Copilot")
    print(f"Dataset : {args.dataset}  ({len(samples)} samples)")
    print(f"OFF arm : isolation_override({off_kwargs})")
    print(f"Judge   : {'disabled' if args.no_judge else 'enabled (pairwise)'}")
    print("=" * 64)

    judge_llm = None if args.no_judge else eo._create_judge_llm()

    results: list[dict] = []
    for idx, sample in enumerate(samples):
        sid = sample["id"]
        category = sample.get("category", "unknown")
        leak_traps = sample.get("leak_traps", {})
        expected = sample.get("criteria", {}).get("expected_experts", [])
        print(f"\n[{idx+1:>2}/{len(samples)}] {sid}  ({category})")

        try:
            _, iso_arm, _ = _run_arm(sample, sample.get("profile", {}), {})  # isolated (default)
            _, noniso_arm, _ = _run_arm(sample, sample.get("profile", {}), off_kwargs)
        except Exception as exc:
            print(f"  [ERROR] graph raised: {exc}")
            results.append({"id": sid, "category": category, "error": str(exc)})
            continue

        leaks = {
            "iso": _count_leaks(iso_arm["expert_outputs"], leak_traps),
            "noniso": _count_leaks(noniso_arm["expert_outputs"], leak_traps),
        }
        print(f"  leakage  iso={leaks['iso']['leaked_outputs']}/{leaks['iso']['checked_outputs']}"
              f"  noniso={leaks['noniso']['leaked_outputs']}/{leaks['noniso']['checked_outputs']}"
              f"  | input_tok iso={iso_arm['input_tokens']} noniso={noniso_arm['input_tokens']}")

        e2e_judge = None
        subagent_judge: dict[str, dict] = {}
        if judge_llm:
            e2e_judge = _pairwise_judge(
                judge_llm, _E2E_JUDGE_SYSTEM,
                iso_arm["answer"], noniso_arm["answer"],
                _build_e2e_context(sample), _E2E_DIMS, rng,
            )
            if e2e_judge.get("error"):
                print(f"  [E2E JUDGE ERROR] {e2e_judge['error']}")
            else:
                print(f"  e2e overall → {e2e_judge['overall']}  ({e2e_judge.get('reason','')[:60]})")

            last_user = next((t["content"] for t in reversed(sample.get("turns", [])) if t["role"] == "user"), "")
            for role in expected:
                iso_out = iso_arm["expert_outputs"].get(role)
                non_out = noniso_arm["expert_outputs"].get(role)
                if not iso_out or not non_out:
                    continue
                ctx = f"[用户问题]\n{last_user}"
                v = _pairwise_judge(
                    judge_llm, _SUBAGENT_JUDGE_SYSTEM.format(role=role),
                    iso_out, non_out, ctx, _SUBAGENT_DIMS, rng,
                )
                subagent_judge[role] = v
                if not v.get("error"):
                    print(f"  subagent[{role}] overall → {v['overall']}  focus → {v.get('focus')}")

        results.append({
            "id": sid,
            "category": category,
            "expected_experts": expected,
            "leak_traps": leak_traps,
            "iso": _arm_view(iso_arm),
            "noniso": _arm_view(noniso_arm),
            "leaks": leaks,
            "e2e_judge": e2e_judge,
            "subagent_judge": subagent_judge,
        })

    report = _aggregate(results, args.dataset, args.ablate, not args.no_judge)
    run_dt = datetime.now().astimezone()
    timestamp = run_dt.strftime("%Y%m%d-%H%M%S")
    out_path = eo._timestamped_report_path(
        PROJECT_ROOT / args.out, timestamp, "isolation_ab_report", args.out,
    )
    report["generated_at"] = run_dt.isoformat(timespec="seconds")
    report["report_path"] = str(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_summary(report)
    print(f"\nReport → {out_path}")


def _print_summary(report: dict) -> None:
    print(f"\n{'='*64}\nISOLATION A/B SUMMARY\n{'='*64}")
    lk = report["leakage"]
    print("Cross-domain leakage (lower is better):")
    print(f"  isolated     : {lk['isolated_arm']['leaked_outputs']}/{lk['isolated_arm']['checked_outputs']}"
          f"  rate={lk['isolated_arm']['leak_rate']}  hits={lk['isolated_arm']['total_term_hits']}")
    print(f"  non-isolated : {lk['non_isolated_arm']['leaked_outputs']}/{lk['non_isolated_arm']['checked_outputs']}"
          f"  rate={lk['non_isolated_arm']['leak_rate']}  hits={lk['non_isolated_arm']['total_term_hits']}")

    cost = report["cost"]
    print("\nCost (avg input tokens / sample):")
    print(f"  isolated={cost['isolated_arm']['avg_input_tokens']}  "
          f"non-isolated={cost['non_isolated_arm']['avg_input_tokens']}")

    if report["judge_enabled"]:
        def _fmt(tally: dict) -> str:
            o = tally.get("overall", {})
            return (f"iso={o.get('iso',0)} noniso={o.get('noniso',0)} tie={o.get('tie',0)} "
                    f"(iso_win={o.get('iso_win_rate')})")
        print("\nSubagent pairwise (overall):")
        print(f"  {_fmt(report['subagent_pairwise']['overall'])}")
        for role, t in report["subagent_pairwise"]["by_expert"].items():
            print(f"    {role:<14} {_fmt(t)}")
        print("E2E pairwise (overall):")
        print(f"  {_fmt(report['e2e_pairwise']['overall'])}")
        # focus dimension is the cleanest isolation signal at subagent layer
        f = report["subagent_pairwise"]["overall"].get("focus", {})
        print(f"  subagent focus: iso={f.get('iso',0)} noniso={f.get('noniso',0)} tie={f.get('tie',0)}")


if __name__ == "__main__":
    main()
