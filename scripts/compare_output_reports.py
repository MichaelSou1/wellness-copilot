#!/usr/bin/env python3
"""Compare two output-eval reports (paired by sample id).

Used for Phase 1 of todos/rag-diagnosis-plan.md: control (no episodes) vs
treatment (seeded episodes). Prints per-dimension deltas and the paired
per-sample personalization diff so we can attribute the 3.70 personalization
score to b1 (episodic RAG empty) vs b2 (model weaving weak).

Usage:
    python scripts/compare_output_reports.py CONTROL.json TREATMENT.json
"""
import json
import sys
from statistics import mean

DIMS = ("relevance", "completeness", "safety", "personalization", "coherence")


def load(path):
    rep = json.loads(open(path, encoding="utf-8").read())
    by_id = {}
    for d in rep.get("details", []):
        sc = d.get("scores") or {}
        if isinstance(sc, dict) and sc.get("personalization") is not None:
            by_id[d["id"]] = {
                "scores": {k: sc.get(k) for k in DIMS},
                "category": d.get("category"),
                "episode_context_len": d.get("episode_context_len", 0),
            }
    arch = rep.get("architecture_summary", {})
    return rep, by_id, arch


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    ctrl_path, treat_path = sys.argv[1], sys.argv[2]
    _, ctrl, ctrl_arch = load(ctrl_path)
    _, treat, treat_arch = load(treat_path)

    paired = sorted(set(ctrl) & set(treat))
    print(f"paired samples: {len(paired)}  (control={len(ctrl)}, treatment={len(treat)})")
    print(f"\nepisode_context present rate:  control={ctrl_arch.get('episode_context_present_rate')}"
          f"  treatment={treat_arch.get('episode_context_present_rate')}")
    print(f"avg episode_context len     :  control={ctrl_arch.get('avg_episode_context_len')}"
          f"  treatment={treat_arch.get('avg_episode_context_len')}")

    print("\n=== per-dimension means (paired) ===")
    print(f"{'dim':<16}{'control':>10}{'treatment':>12}{'delta':>10}")
    for dim in DIMS:
        c = mean(ctrl[i]["scores"][dim] for i in paired)
        t = mean(treat[i]["scores"][dim] for i in paired)
        star = "  <==" if dim == "personalization" else ""
        print(f"{dim:<16}{c:>10.3f}{t:>12.3f}{t-c:>+10.3f}{star}")

    # Paired personalization detail.
    print("\n=== personalization paired diff (per sample) ===")
    deltas = []
    for i in paired:
        c = ctrl[i]["scores"]["personalization"]
        t = treat[i]["scores"]["personalization"]
        deltas.append(t - c)
        flag = "" if t == c else (" +" if t > c else " -")
        print(f"  {i:<24} {ctrl[i]['category']:<22} ctrl={c} treat={t} "
              f"ctx_len={treat[i]['episode_context_len']}{flag}")
    up = sum(1 for d in deltas if d > 0)
    down = sum(1 for d in deltas if d < 0)
    same = sum(1 for d in deltas if d == 0)
    print(f"\npersonalization: mean Δ={mean(deltas):+.3f}  | up={up} down={down} same={same}")

    # Per-category personalization delta.
    cats = {}
    for i in paired:
        cats.setdefault(ctrl[i]["category"], []).append(
            treat[i]["scores"]["personalization"] - ctrl[i]["scores"]["personalization"]
        )
    print("\nper-category personalization Δ:")
    for c, ds in sorted(cats.items()):
        print(f"  {c:<22} n={len(ds):<3} Δ={mean(ds):+.3f}")


if __name__ == "__main__":
    main()
