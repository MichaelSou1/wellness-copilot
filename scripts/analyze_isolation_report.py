"""Causal-chain analysis over an isolation A/B report (from evaluate_isolation_ab.py).

Tests the claim "isolation reduces rotten context, which improves the subagent's
answer" by joining, per (sample, expert):
  - whether the NON-isolated arm leaked a cross-domain / stale trap term, and
  - which arm the judge preferred on the `focus` dimension (and `overall`).

If isolation's focus/quality advantage concentrates in exactly the cases where
the non-isolated arm leaked, that's direct evidence for: contamination → worse
focus. If the two are unrelated, isolation's effect (if any) isn't coming from
the leakage we measured.

Usage:
    python scripts/analyze_isolation_report.py reports/isolation_ab_report_XXXX.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _rate(num: int, den: int) -> str:
    return f"{num}/{den} ({num/den:.1%})" if den else f"{num}/{den} (n/a)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an isolation A/B report")
    parser.add_argument("report", help="path to isolation_ab_report_*.json")
    args = parser.parse_args()

    path = Path(args.report)
    if not path.exists():
        print(f"[ERROR] report not found: {path}", file=sys.stderr)
        sys.exit(1)
    report = json.loads(path.read_text(encoding="utf-8"))
    details = report.get("details", [])

    # Per (sample, expert) join of leakage(noniso) × focus/overall verdict.
    pairs = []  # (sample_id, role, noniso_leaked: bool, focus, overall)
    for r in details:
        if r.get("error"):
            continue
        leaks = (r.get("leaks") or {}).get("noniso", {}).get("per_expert", {})
        judges = r.get("subagent_judge") or {}
        for role, v in judges.items():
            if not v or v.get("error"):
                continue
            leaked = bool((leaks.get(role) or {}).get("leaked_terms"))
            pairs.append((r["id"], role, leaked, v.get("focus"), v.get("overall")))

    if not pairs:
        print("[WARN] no joinable (sample, expert) judged pairs in report.")
        return

    leaked = [p for p in pairs if p[2]]
    clean = [p for p in pairs if not p[2]]

    def _focus_breakdown(group):
        c = {"iso": 0, "noniso": 0, "tie": 0}
        for _, _, _, focus, _ in group:
            if focus in c:
                c[focus] += 1
        return c

    def _overall_breakdown(group):
        c = {"iso": 0, "noniso": 0, "tie": 0}
        for _, _, _, _, ov in group:
            if ov in c:
                c[ov] += 1
        return c

    print("=" * 70)
    print("ISOLATION CAUSAL-CHAIN ANALYSIS  (contamination → focus/quality)")
    print(f"report: {path.name}")
    print(f"off-arm override: {report.get('off_arm_override')}")
    print("=" * 70)
    print(f"\njoined (sample, expert) judged pairs: {len(pairs)}")
    print(f"  non-isolated LEAKED a trap : {_rate(len(leaked), len(pairs))}")
    print(f"  non-isolated stayed CLEAN  : {_rate(len(clean), len(pairs))}")

    print("\n--- focus verdict, split by whether non-isolated leaked ---")
    for label, grp in (("LEAKED", leaked), ("CLEAN", clean)):
        b = _focus_breakdown(grp)
        n = len(grp)
        iso_pref = _rate(b["iso"], n)
        print(f"  {label:<7} (n={n:>3}): iso={b['iso']} noniso={b['noniso']} tie={b['tie']}"
              f"   | judge preferred ISO on focus: {iso_pref}")

    print("\n--- overall verdict, split by whether non-isolated leaked ---")
    for label, grp in (("LEAKED", leaked), ("CLEAN", clean)):
        b = _overall_breakdown(grp)
        n = len(grp)
        print(f"  {label:<7} (n={n:>3}): iso={b['iso']} noniso={b['noniso']} tie={b['tie']}"
              f"   | judge preferred ISO overall: {_rate(b['iso'], n)}")

    # Headline: does leaking predict an iso focus-win? (2x2 lift)
    lk = _focus_breakdown(leaked)
    cl = _focus_breakdown(clean)
    iso_win_when_leaked = lk["iso"] / len(leaked) if leaked else None
    iso_win_when_clean = cl["iso"] / len(clean) if clean else None
    print("\n--- headline ---")
    if iso_win_when_leaked is not None and iso_win_when_clean is not None:
        print(f"  P(judge prefers ISO on focus | noniso leaked) = {iso_win_when_leaked:.1%}")
        print(f"  P(judge prefers ISO on focus | noniso clean ) = {iso_win_when_clean:.1%}")
        lift = iso_win_when_leaked - iso_win_when_clean
        print(f"  lift = {lift:+.1%}  "
              f"({'supports' if lift > 0.05 else 'no clear support for'} "
              f"contamination→worse-focus chain)")
    else:
        print("  insufficient data in one cell (need both leaked and clean pairs).")

    # Leakage reduction headline (arm-level, from the report aggregate)
    lkg = report.get("leakage", {})
    iso_r = (lkg.get("isolated_arm") or {}).get("leak_rate")
    non_r = (lkg.get("non_isolated_arm") or {}).get("leak_rate")
    print(f"\n  arm leak rate: isolated={iso_r}  non-isolated={non_r}")


if __name__ == "__main__":
    main()
