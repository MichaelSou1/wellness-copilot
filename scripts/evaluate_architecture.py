"""L4 architecture regression evaluation for Health Guide Agent.

This runner targets six architecture properties that the L5 output-quality
evaluation does not surface directly:

  1. context_isolation
     Each expert callable receives an isolated (SystemMessage + HumanMessage)
     pair. The SystemMessage contains a role-cropped profile and (optionally)
     peer scratchpad notes — NEVER fields outside the per-role whitelist
     and NEVER the parent's accumulated messages.
     We assert this by monkey-patching `health_guide.utils.create_agent`
     to capture every system_prompt string at construction time.

  2. rag_on_demand
     Chitchat / pure-profile-update questions must not call any
     retrieve_*_knowledge tool. We read `state.last_tools` after the run.

  3. parallel_fanout
     When the plan contains ≥2 experts, Dispatcher should run them
     concurrently. We monkey-patch each EXPERT_RUNNERS entry to record
     (start_ts, end_ts) and assert wall-clock < sum(per-expert) × threshold.

  4. replan_triggers / replan_cap / replan_skip_on_full_plan
     We read `state.replan_count` and `state.executed`. The cap test asserts
     replan_count <= REPLAN_CAP (no infinite loop).

  5. episodic_memory_cross_thread
     We pre-seed episode_store with a synthetic prior turn for a unique
     user_id, then run a brand-new thread and assert the answer reflects
     the earlier episode (e.g. mentions the injury / allergy).

  6. history_summary_compression
     Run many turns on a single thread, then assert
     state.history_summary is non-empty AND total messages stayed bounded
     (i.e. TurnStart actually emitted RemoveMessage).

Each test is data-driven from `eval/architecture_eval_dataset.jsonl`. The
runner picks an executor per `arch_check` field.

Usage:
    python scripts/evaluate_architecture.py
    python scripts/evaluate_architecture.py --samples arch_isolation_001
    python scripts/evaluate_architecture.py --dataset eval/architecture_eval_dataset.jsonl

Output: reports/architecture_eval_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage  # noqa: E402

from health_guide import config as _cfg  # noqa: E402
from health_guide import utils as _utils_mod  # noqa: E402
from health_guide.agents import dispatcher as _disp_mod  # noqa: E402
from health_guide.episode_store import append_episode  # noqa: E402
from health_guide.graph import graph  # noqa: E402
from health_guide.llm import extract_text_content  # noqa: E402
from health_guide.profile_store import update_user_profile  # noqa: E402


# ---------------------------------------------------------------------------
# Capture instrumentation
# ---------------------------------------------------------------------------

# Per-run capture state. Reset before each sample.
_CAPTURE: Dict[str, Any] = {
    "system_prompts": [],     # list of dicts: {"prompt": str, "role_guess": str|None}
    "expert_timings": [],     # list of dicts: {"role": str, "start": float, "end": float}
}


def _reset_capture() -> None:
    _CAPTURE["system_prompts"] = []
    _CAPTURE["expert_timings"] = []


def _install_capture() -> None:
    """Wrap create_agent and EXPERT_RUNNERS to record what was passed in.

    Idempotent: re-installing replaces existing wrappers with fresh ones
    pointing at the SAME underlying originals (stored once on first install).
    """
    # 1) Wrap create_agent to capture every system_prompt.
    if not hasattr(_utils_mod, "_orig_create_agent"):
        _utils_mod._orig_create_agent = _utils_mod.create_agent  # type: ignore[attr-defined]

    def _capturing_create_agent(llm, tools, system_prompt):
        _CAPTURE["system_prompts"].append({
            "prompt": system_prompt,
            # Best-effort role detection from the leading sentence.
            "role_guess": _guess_role_from_prompt(system_prompt),
        })
        return _utils_mod._orig_create_agent(llm, tools, system_prompt)  # type: ignore[attr-defined]

    _utils_mod.create_agent = _capturing_create_agent  # type: ignore[assignment]
    # Also patch every expert module's already-imported reference to create_agent.
    from health_guide.agents import trainer as _t, nutritionist as _n, wellness as _w, orchestrator as _o
    for mod in (_t, _n, _w, _o):
        if hasattr(mod, "create_agent"):
            mod.create_agent = _capturing_create_agent  # type: ignore[assignment]

    # 2) Wrap each EXPERT_RUNNERS entry to record wall-clock timing.
    if not hasattr(_disp_mod, "_orig_expert_runners"):
        _disp_mod._orig_expert_runners = dict(_disp_mod.EXPERT_RUNNERS)  # type: ignore[attr-defined]

    def _make_timed_runner(role: str, fn: Callable) -> Callable:
        def _runner(*args, **kwargs) -> dict:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                t1 = time.perf_counter()
                _CAPTURE["expert_timings"].append({
                    "role": role, "start": t0, "end": t1, "wall_ms": (t1 - t0) * 1000.0,
                })
        return _runner

    _disp_mod.EXPERT_RUNNERS = {
        role: _make_timed_runner(role, fn)
        for role, fn in _disp_mod._orig_expert_runners.items()  # type: ignore[attr-defined]
    }


def _guess_role_from_prompt(prompt: str) -> str | None:
    head = (prompt or "")[:30]
    for role, tag in (
        ("Trainer", "训练教练"),
        ("Nutritionist", "营养师"),
        ("Wellness", "身心"),
        ("Orchestrator", "主 agent"),
    ):
        if tag in head:
            return role
    return None


# ---------------------------------------------------------------------------
# Graph runner
# ---------------------------------------------------------------------------

def _run_graph(sample: dict, user_id: str, thread_id: str | None = None) -> dict:
    """Run all user turns of a sample against the graph. Returns final state."""
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    final_state: dict = {}
    for turn in sample.get("turns", []):
        if turn.get("role") != "user":
            continue
        final_state = graph.invoke(
            {
                "messages": [HumanMessage(content=turn["content"])],
                "profile_user_id": user_id,
            },
            config,
        )
    return final_state


# ---------------------------------------------------------------------------
# Per-check evaluators. Each returns (passed: bool, detail: dict).
# ---------------------------------------------------------------------------

def _eval_context_isolation(sample: dict, user_id: str) -> tuple[bool, dict]:
    """Capture system_prompt for the targeted expert and check whitelist."""
    a = sample["asserts"]
    target_role = a["expert_role"]
    must_not = a.get("system_prompt_must_not_contain", [])
    must_any = a.get("system_prompt_must_contain_at_least_one_of", [])
    max_chars = a.get("system_prompt_max_chars", 99999)

    _reset_capture()
    _ = _run_graph(sample, user_id)

    # Find the system_prompt for the targeted role.
    candidates = [c for c in _CAPTURE["system_prompts"] if c["role_guess"] == target_role]
    if not candidates:
        return False, {"error": f"no captured system_prompt for role={target_role}", "captured_roles": [c["role_guess"] for c in _CAPTURE["system_prompts"]]}

    sp = candidates[-1]["prompt"]
    leaked = [kw for kw in must_not if kw in sp]
    any_anchor_present = (not must_any) or any(kw in sp for kw in must_any)
    too_long = len(sp) > max_chars

    passed = (not leaked) and any_anchor_present and (not too_long)
    return passed, {
        "system_prompt_len": len(sp),
        "leaked_keywords": leaked,
        "any_anchor_present": any_anchor_present,
        "too_long": too_long,
        "system_prompt_excerpt": sp[:300],
    }


def _eval_rag_on_demand(sample: dict, user_id: str) -> tuple[bool, dict]:
    a = sample["asserts"]
    state = _run_graph(sample, user_id)
    tools = list(state.get("last_tools") or [])
    rag_calls = [t for t in tools if isinstance(t, str) and t.startswith("retrieve_") and t.endswith("_knowledge")]

    max_rag = a.get("max_rag_calls", 0)
    max_replan = a.get("max_replan_count", 99)
    actual_replan = int(state.get("replan_count", 0) or 0)

    rag_ok = len(rag_calls) <= max_rag
    replan_ok = actual_replan <= max_replan
    return (rag_ok and replan_ok), {
        "rag_calls": rag_calls,
        "rag_call_count": len(rag_calls),
        "tools_used": tools,
        "replan_count": actual_replan,
        "limit_rag": max_rag,
        "limit_replan": max_replan,
    }


def _eval_parallel_fanout(sample: dict, user_id: str) -> tuple[bool, dict]:
    a = sample["asserts"]
    _reset_capture()
    state = _run_graph(sample, user_id)

    executed = state.get("executed", []) or []
    timings = list(_CAPTURE["expert_timings"])

    min_experts = a.get("min_experts_executed", 2)
    if len(executed) < min_experts:
        return False, {"error": f"need ≥{min_experts} experts, got {executed}"}

    # Only consider the first batch (before any replan) — that's where the
    # parallel fan-out is supposed to happen. Replan rounds bring in a single
    # expert each and don't have peer parallelism to measure.
    first_batch = timings[:len(executed)] if timings else []
    if len(first_batch) < 2:
        return False, {"error": "fewer than 2 timed expert calls", "timings": timings}

    sum_ms = sum(t["wall_ms"] for t in first_batch)
    wall_ms = (max(t["end"] for t in first_batch) - min(t["start"] for t in first_batch)) * 1000.0
    ratio = wall_ms / sum_ms if sum_ms else 1.0

    threshold = a.get("max_wall_clock_ratio", 0.85)
    parallel_ok = ratio <= threshold

    # Optional: both notes present in agent_notes for Aggregator/Critic to use.
    notes = state.get("agent_notes") or {}
    both_notes_present = all(notes.get(role) for role in executed[:min_experts])
    notes_ok = (not a.get("both_notes_present_in_aggregator", False)) or both_notes_present

    return (parallel_ok and notes_ok), {
        "executed": executed,
        "first_batch_timings": first_batch,
        "sum_ms": round(sum_ms, 1),
        "wall_ms": round(wall_ms, 1),
        "ratio": round(ratio, 3),
        "threshold": threshold,
        "agent_notes_present": {k: bool(v) for k, v in notes.items()},
    }


def _eval_replan(sample: dict, user_id: str) -> tuple[bool, dict]:
    a = sample["asserts"]
    state = _run_graph(sample, user_id)
    executed = state.get("executed", []) or []
    replan = int(state.get("replan_count", 0) or 0)

    must_any = a.get("executed_must_contain_any_of") or []
    min_experts = a.get("min_experts_executed", 0)
    rc_min = a.get("replan_count_min", 0)
    rc_max = a.get("replan_count_max", 99)

    any_ok = (not must_any) or any(role in executed for role in must_any)
    count_ok = (len(executed) >= min_experts)
    rc_ok = (rc_min <= replan <= rc_max)

    return (any_ok and count_ok and rc_ok), {
        "executed": executed,
        "replan_count": replan,
        "any_required": must_any,
        "limits": {"min_experts": min_experts, "rc_min": rc_min, "rc_max": rc_max},
        "checks": {"any_ok": any_ok, "count_ok": count_ok, "rc_ok": rc_ok},
    }


def _eval_episodic_memory(sample: dict, user_id: str) -> tuple[bool, dict]:
    """Pre-seed episode_store with prior episodes, then run a fresh thread."""
    a = sample["asserts"]
    prime = a.get("prime_episodes") or []

    # Seed
    for ep in prime:
        append_episode(
            user_id=user_id,
            query=ep.get("query", ""),
            experts=ep.get("experts") or [],
            gist=ep.get("gist", ""),
        )

    # Fresh thread — verifies cross-thread episodic injection.
    state = _run_graph(sample, user_id, thread_id=str(uuid.uuid4()))
    answer = extract_text_content(state["messages"][-1]) if state.get("messages") else ""

    mention_any = a.get("session_b_answer_should_mention_any_of") or []
    must_not = a.get("session_b_answer_must_not_contain") or []

    any_ok = (not mention_any) or any(kw in answer for kw in mention_any)
    not_ok = all(kw not in answer for kw in must_not)
    episode_ctx_seen = bool((state.get("episode_context") or "").strip())

    return (any_ok and not_ok and episode_ctx_seen), {
        "answer_excerpt": answer[:300],
        "executed": state.get("executed", []),
        "episode_context_injected": episode_ctx_seen,
        "checks": {"any_ok": any_ok, "not_ok": not_ok, "episode_ctx_seen": episode_ctx_seen},
        "primed_episodes": prime,
    }


def _eval_history_summary(sample: dict, user_id: str) -> tuple[bool, dict]:
    """Run many turns on a single thread; require summary fired."""
    a = sample["asserts"]
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    final_state: dict = {}
    for turn in sample.get("turns", []):
        if turn.get("role") != "user":
            continue
        final_state = graph.invoke(
            {
                "messages": [HumanMessage(content=turn["content"])],
                "profile_user_id": user_id,
            },
            config,
        )

    summary = (final_state.get("history_summary") or "").strip()
    total_msgs = len(final_state.get("messages", []) or [])
    summary_ok = (not a.get("history_summary_should_be_nonempty")) or bool(summary)
    bounded_ok = total_msgs <= a.get("max_total_messages_after_compression", 99999)

    return (summary_ok and bounded_ok), {
        "history_summary_len": len(summary),
        "history_summary_excerpt": summary[:200],
        "total_messages": total_msgs,
        "checks": {"summary_ok": summary_ok, "bounded_ok": bounded_ok},
    }


_EVALUATORS: Dict[str, Callable[[dict, str], tuple[bool, dict]]] = {
    "context_isolation": _eval_context_isolation,
    "rag_on_demand": _eval_rag_on_demand,
    "parallel_fanout": _eval_parallel_fanout,
    "replan_triggers": _eval_replan,
    "replan_cap": _eval_replan,
    "replan_skip_on_full_plan": _eval_replan,
    "episodic_memory_cross_thread": _eval_episodic_memory,
    "history_summary_compression": _eval_history_summary,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _seed_profile(user_id: str, profile: dict) -> None:
    if profile:
        update_user_profile(user_id, profile)


def main() -> None:
    parser = argparse.ArgumentParser(description="L4 architecture regression evaluation")
    parser.add_argument("--dataset", default="eval/architecture_eval_dataset.jsonl")
    parser.add_argument("--out", default="reports/architecture_eval_report.json")
    parser.add_argument("--samples", default="", help="comma-separated sample IDs to run")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    ds_path = PROJECT_ROOT / args.dataset
    if not ds_path.exists():
        print(f"[ERROR] dataset not found: {ds_path}", file=sys.stderr)
        sys.exit(1)

    samples: List[dict] = []
    with ds_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    if args.samples:
        wanted = set(args.samples.split(","))
        samples = [s for s in samples if s["id"] in wanted]
        if not samples:
            print(f"[ERROR] no samples matched: {wanted}", file=sys.stderr)
            sys.exit(1)

    print(f"{'='*64}")
    print("L4 Architecture Regression — Health Guide Agent")
    print(f"Dataset: {args.dataset}  ({len(samples)} samples)")
    print(f"{'='*64}")

    _install_capture()

    results: List[dict] = []
    pass_count = 0
    pass_by_check: Dict[str, list] = {}

    for idx, sample in enumerate(samples):
        sid = sample["id"]
        check = sample["arch_check"]
        desc = sample.get("description", "")
        print(f"\n[{idx+1:>2}/{len(samples)}] {sid}  ({check})")
        if desc:
            print(f"    {desc}")

        evaluator = _EVALUATORS.get(check)
        if not evaluator:
            print(f"    [ERROR] no evaluator for arch_check={check}")
            results.append({"id": sid, "arch_check": check, "passed": False, "error": "no evaluator"})
            continue

        user_id = f"arch_{sid}_{uuid.uuid4().hex[:8]}"
        _seed_profile(user_id, sample.get("profile") or {})

        try:
            passed, detail = evaluator(sample, user_id)
        except Exception as exc:
            print(f"    [ERROR] {type(exc).__name__}: {exc}")
            results.append({
                "id": sid, "arch_check": check, "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
            pass_by_check.setdefault(check, []).append(False)
            continue

        results.append({
            "id": sid, "arch_check": check, "passed": passed,
            "description": desc,
            "detail": detail,
        })
        pass_by_check.setdefault(check, []).append(passed)
        if passed:
            pass_count += 1
            print("    PASS")
        else:
            print("    FAIL")
            if args.verbose:
                print(f"    detail: {json.dumps(detail, ensure_ascii=False)[:400]}")

    # ---- Aggregate ------------------------------------------------------
    by_check_summary = {
        k: {"total": len(v), "passed": sum(1 for x in v if x),
            "pass_rate": round(sum(1 for x in v if x) / len(v), 3) if v else None}
        for k, v in pass_by_check.items()
    }

    report = {
        "run_id": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        "dataset": args.dataset,
        "total_samples": len(results),
        "passed": pass_count,
        "failed": len(results) - pass_count,
        "pass_rate": round(pass_count / len(results), 3) if results else None,
        "by_check": by_check_summary,
        "details": results,
    }

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Print summary -------------------------------------------------
    print(f"\n{'='*64}")
    print("L4 SUMMARY")
    print(f"{'='*64}")
    print(f"Total          : {len(results)}")
    print(f"Passed         : {pass_count}  ({report['pass_rate']:.1%})" if report["pass_rate"] is not None else "")
    print("By arch_check:")
    for k, summ in sorted(by_check_summary.items()):
        pr = summ["pass_rate"]
        bar = "█" * int((pr or 0) * 20)
        print(f"  {k:<36} {summ['passed']}/{summ['total']}   {bar}")
    print(f"\nReport → {out_path}")


if __name__ == "__main__":
    main()
