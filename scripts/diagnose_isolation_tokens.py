"""Diagnose WHY the isolated arm can cost more tokens than the non-isolated arm.

The A/B report showed isolated input tokens > non-isolated, which is counter to
the naive "isolation injects less context → fewer tokens" expectation. Hypothesis:
isolated experts, lacking context, call RAG (retrieve_*_knowledge) more often, and
each retrieval injects a large chunk of KB text — outweighing the context the
non-isolated arm adds.

This runs a few samples under both arms and reports, per arm:
  - total tokens / input tokens / LLM call count (from the graph profiler)
  - RAG retrieve_*_knowledge call count and full tool-call count (from state.last_tools)
  - per-component token breakdown (which expert/node burned the tokens)

Usage:
    python scripts/diagnose_isolation_tokens.py --samples iso_hard_001,iso_hard_010
    python scripts/diagnose_isolation_tokens.py --n 6 --ablate all
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evaluate_output as eo  # noqa: E402
from wellness_copilot import isolation  # noqa: E402
from wellness_copilot.profile_store import update_user_profile  # noqa: E402


def _ablate_kwargs(ablate: str) -> dict:
    if ablate == "all":
        return {"profile": False, "peer": False, "history": False}
    return {ablate: False}


def _rag_count(tools: list) -> int:
    return sum(1 for t in tools if isinstance(t, str) and t.startswith("retrieve_") and t.endswith("_knowledge"))


def _run(sample: dict, override: dict) -> dict:
    user_id = f"diag_{sample['id']}_{uuid.uuid4().hex[:6]}"
    if sample.get("profile"):
        update_user_profile(user_id, sample["profile"])
    with isolation.isolation_override(**override):
        _, state, perf = eo._run_sample(sample, user_id, verbose=False)
    tools = list(state.get("last_tools") or [])
    comp = perf.get("component_totals") or {}
    return {
        "total_tokens": int(perf.get("total_tokens", 0) or 0),
        "input_tokens": int(perf.get("input_tokens", 0) or 0),
        "llm_calls": int(perf.get("llm_calls", 0) or 0),
        "rag_calls": _rag_count(tools),
        "tool_calls": len(tools),
        "tools": tools,
        "components": {k: int(v.get("total_tokens", 0) or 0) for k, v in comp.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose isolated-vs-non-isolated token cost")
    ap.add_argument("--dataset", default="eval/isolation_hard_dataset.jsonl")
    ap.add_argument("--samples", default="")
    ap.add_argument("--n", type=int, default=6, help="first N samples if --samples not given")
    ap.add_argument("--ablate", default="all", choices=["all", "profile", "peer", "history"])
    args = ap.parse_args()

    override = _ablate_kwargs(args.ablate)
    ds = PROJECT_ROOT / args.dataset
    samples = [json.loads(l) for l in ds.open(encoding="utf-8") if l.strip()]
    if args.samples:
        wanted = set(args.samples.split(","))
        samples = [s for s in samples if s["id"] in wanted]
    else:
        samples = samples[: args.n]

    print("=" * 70)
    print(f"TOKEN DIAGNOSIS  dataset={args.dataset}  off-arm={override}  n={len(samples)}")
    print("=" * 70)

    agg = {"iso": {"total": 0, "rag": 0, "llm": 0, "tool": 0},
           "non": {"total": 0, "rag": 0, "llm": 0, "tool": 0}}

    for s in samples:
        iso = _run(s, {})            # isolated (default)
        non = _run(s, override)      # non-isolated
        for key, arm in (("iso", iso), ("non", non)):
            agg[key]["total"] += arm["total_tokens"]
            agg[key]["rag"] += arm["rag_calls"]
            agg[key]["llm"] += arm["llm_calls"]
            agg[key]["tool"] += arm["tool_calls"]
        print(f"\n[{s['id']}]")
        print(f"  ISO : tokens={iso['total_tokens']:>7}  llm_calls={iso['llm_calls']:>2}  "
              f"rag_calls={iso['rag_calls']}  tool_calls={iso['tool_calls']}  tools={iso['tools']}")
        print(f"  NON : tokens={non['total_tokens']:>7}  llm_calls={non['llm_calls']:>2}  "
              f"rag_calls={non['rag_calls']}  tool_calls={non['tool_calls']}  tools={non['tools']}")
        # which component drove the difference
        comps = sorted(set(iso["components"]) | set(non["components"]))
        diffs = [(c, iso["components"].get(c, 0) - non["components"].get(c, 0)) for c in comps]
        diffs = [d for d in diffs if abs(d[1]) > 50]
        diffs.sort(key=lambda x: -abs(x[1]))
        if diffs:
            print("  token Δ(iso−non) by component: " +
                  "  ".join(f"{c}={d:+d}" for c, d in diffs[:6]))

    print("\n" + "=" * 70)
    print("AGGREGATE")
    print(f"  ISO : total_tokens={agg['iso']['total']:>8}  rag_calls={agg['iso']['rag']}  "
          f"llm_calls={agg['iso']['llm']}  tool_calls={agg['iso']['tool']}")
    print(f"  NON : total_tokens={agg['non']['total']:>8}  rag_calls={agg['non']['rag']}  "
          f"llm_calls={agg['non']['llm']}  tool_calls={agg['non']['tool']}")
    dt = agg["iso"]["total"] - agg["non"]["total"]
    dr = agg["iso"]["rag"] - agg["non"]["rag"]
    print(f"  Δ(iso−non): tokens={dt:+d}  rag_calls={dr:+d}")
    if dr > 0 and dt > 0:
        print("  → isolated arm makes MORE RAG calls; consistent with RAG-driven token inflation.")
    elif dt > 0 and dr <= 0:
        print("  → isolated arm costs more tokens but NOT via extra RAG; look elsewhere (replan/tool loops).")


if __name__ == "__main__":
    main()
