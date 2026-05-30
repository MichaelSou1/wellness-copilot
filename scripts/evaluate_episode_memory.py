#!/usr/bin/env python3
"""Phase 2 (todos/rag-diagnosis-plan.md): episodic-memory RAG quality eval.

The output eval measures the *system*; it cannot isolate the per-user episodic
memory RAG (FAISS ``EpisodeMemory.retrieve_similar``) because that subsystem
never fired in eval (no seeded episodes + EPISODE_SEMANTIC_MIN_COUNT=8). This
script evaluates the episodic retriever directly, mirroring the KB RAG eval:

  1. Retrieval quality — for each sample, seed its labeled prior history
     (3 relevant + 6 distractor episodes, from eval/episode_seeds.jsonl), index
     it, then query with the sample's CURRENT question and measure whether the
     "relevant" episodes are retrieved (recall@k, MRR, precision@k).
  2. TOP_K sensitivity — sweep top_k to see how recall/MRR move.
  3. Downstream adoption (optional, --output-report) — given a SEEDED output-eval
     report, measure whether episode-only content actually surfaced in answers.

No LLM is required for (1)/(2): only the embedding model. Runs on GPU if available.

Usage:
    python scripts/evaluate_episode_memory.py
    python scripts/evaluate_episode_memory.py --topk 1,2,3,5,8
    python scripts/evaluate_episode_memory.py --output-report reports/output_eval_report_SEEDED.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Isolate episode storage/index from the production stores. Must be set BEFORE
# importing wellness_copilot modules (config captures these at import time).
_TMP = Path(tempfile.gettempdir()) / f"episode_eval_{uuid.uuid4().hex[:8]}"
os.environ.setdefault("EPISODE_STORE_PATH", str(_TMP / "store.json"))
os.environ.setdefault("EPISODE_INDEX_DIR", str(_TMP / "index"))
_TMP.mkdir(parents=True, exist_ok=True)

from wellness_copilot.episode_store import (  # noqa: E402
    append_episode,
    get_all_episodes,
    total_episode_count,
)
from wellness_copilot.episode_memory import EpisodeMemory  # noqa: E402

SEED_PATH = PROJECT_ROOT / "eval" / "episode_seeds.jsonl"
DATASET = PROJECT_ROOT / "eval" / "output_eval_dataset.jsonl"


def _load_seeds() -> dict:
    seeds = {}
    for line in SEED_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rec = json.loads(line)
            seeds[rec["id"]] = rec.get("episodes", [])
    return seeds


def _load_dataset() -> dict:
    out = {}
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def _current_query(sample: dict) -> str:
    users = [t["content"] for t in sample.get("turns", []) if t.get("role") == "user"]
    return users[-1] if users else ""


def _seed_and_index(user_id: str, episodes: list[dict]) -> dict:
    """Seed episodes, build the index, return {stored_id: relevant_bool}.

    We match seeded episodes back to their stored ids by truncated-query text
    (append_episode stores query[:120] and a derived id).
    """
    rel_by_query = {ep.get("query", "")[:120]: bool(ep.get("relevant")) for ep in episodes}
    for ep in episodes:
        append_episode(
            user_id,
            query=ep.get("query", ""),
            experts=ep.get("experts", []) or [],
            gist=ep.get("gist", ""),
            facts=ep.get("facts") or None,
        )
    EpisodeMemory(user_id).rebuild_from_store()
    stored = get_all_episodes(user_id)
    relevance = {}
    for ep in stored:
        relevance[ep["id"]] = rel_by_query.get(ep.get("query", "")[:120], False)
    return relevance


def _metrics_at_k(ranked_ids: list[str], relevance: dict, k: int) -> dict:
    relevant_ids = {eid for eid, r in relevance.items() if r}
    n_rel = len(relevant_ids)
    topk = ranked_ids[:k]
    hits = [eid for eid in topk if eid in relevant_ids]
    recall = len(hits) / n_rel if n_rel else 0.0
    precision = len(hits) / k if k else 0.0
    # MRR over the full ranking (rank of first relevant).
    rr = 0.0
    for rank, eid in enumerate(ranked_ids, start=1):
        if eid in relevant_ids:
            rr = 1.0 / rank
            break
    # hit@k: at least one relevant in top-k
    hit = 1.0 if hits else 0.0
    return {"recall": recall, "precision": precision, "mrr": rr, "hit": hit, "n_rel": n_rel}


def run_retrieval_eval(topks: list[int], max_k: int, verbose: bool) -> dict:
    seeds = _load_seeds()
    dataset = _load_dataset()
    ids = [sid for sid in dataset if sid in seeds]
    print(f"[INFO] episodic retrieval eval over {len(ids)} sample(s)")

    per_sample = []
    for idx, sid in enumerate(ids):
        sample = dataset[sid]
        episodes = seeds[sid]
        query = _current_query(sample)
        user_id = f"epeval_{sid}_{uuid.uuid4().hex[:8]}"
        relevance = _seed_and_index(user_id, episodes)
        # Rank ALL episodes (top_k large), no exclude — pure retrieval quality.
        ranked = EpisodeMemory(user_id).retrieve_similar(query, top_k=max_k)
        ranked_ids = [r.get("id") for r in ranked]
        scored = ranked[0].get("_memory_score") if ranked else None
        row = {
            "id": sid,
            "category": sample.get("category", "unknown"),
            "n_episodes": total_episode_count(user_id),
            "top1_score": scored,
            "top1_query": ranked[0].get("query") if ranked else "",
            "top1_relevant": bool(relevance.get(ranked_ids[0])) if ranked_ids else False,
            "by_k": {k: _metrics_at_k(ranked_ids, relevance, k) for k in topks},
        }
        per_sample.append(row)
        if verbose:
            r3 = row["by_k"].get(3, {})
            print(
                f"[{idx+1:>2}/{len(ids)}] {sid:<22} top1={'REL' if row['top1_relevant'] else 'dis'}"
                f" score={scored:.3f}  recall@3={r3.get('recall',0):.2f} mrr={r3.get('mrr',0):.2f}"
            )

    # Aggregate.
    agg = {}
    for k in topks:
        agg[k] = {
            "recall": round(mean(r["by_k"][k]["recall"] for r in per_sample), 4),
            "precision": round(mean(r["by_k"][k]["precision"] for r in per_sample), 4),
            "mrr": round(mean(r["by_k"][k]["mrr"] for r in per_sample), 4),
            "hit_rate": round(mean(r["by_k"][k]["hit"] for r in per_sample), 4),
        }
    # Per-category recall@3 / mrr.
    by_cat: dict = {}
    for r in per_sample:
        c = r["category"]
        by_cat.setdefault(c, []).append(r)
    cat_summary = {
        c: {
            "n": len(rows),
            "recall@3": round(mean(x["by_k"].get(3, x["by_k"][topks[-1]])["recall"] for x in rows), 3),
            "mrr": round(mean(x["by_k"].get(3, x["by_k"][topks[-1]])["mrr"] for x in rows), 3),
            "top1_relevant_rate": round(mean(1.0 if x["top1_relevant"] else 0.0 for x in rows), 3),
        }
        for c, rows in by_cat.items()
    }
    top1_rel_rate = round(mean(1.0 if r["top1_relevant"] else 0.0 for r in per_sample), 4)
    return {
        "n_samples": len(per_sample),
        "topk_sweep": agg,
        "top1_relevant_rate": top1_rel_rate,
        "by_category": cat_summary,
        "per_sample": per_sample,
    }


def _select_depth(episodes: list, depth: int) -> list:
    """Keep all relevant + enough distractors to reach `depth` total."""
    relevant = [e for e in episodes if e.get("relevant")]
    distractors = [e for e in episodes if not e.get("relevant")]
    if depth <= 0:
        return relevant + distractors
    chosen = relevant[:depth]
    chosen += distractors[: max(0, depth - len(chosen))]
    return chosen


def run_depth_sweep(depths: list[int], k: int = 3) -> dict:
    """Retrieval quality as a function of how much history the user has.

    Models 'a user with N episodes': keep all relevant episodes + (N - #relevant)
    distractors, index, query, measure recall@k / mrr / hit@k. As N grows the
    index holds more unrelated history, so this shows whether shallow history
    still retrieves the relevant episodes (the MIN_COUNT trade-off, retrieval side).
    """
    seeds = _load_seeds()
    dataset = _load_dataset()
    ids = [sid for sid in dataset if sid in seeds]
    out = {}
    for depth in depths:
        rows = []
        for sid in ids:
            sample = dataset[sid]
            eps = _select_depth(seeds[sid], depth)
            n_rel = sum(1 for e in eps if e.get("relevant"))
            if n_rel == 0:
                continue
            user_id = f"epdepth_{depth}_{sid}_{uuid.uuid4().hex[:6]}"
            relevance = _seed_and_index(user_id, eps)
            ranked = EpisodeMemory(user_id).retrieve_similar(_current_query(sample), top_k=max(k, depth))
            ranked_ids = [r.get("id") for r in ranked]
            rows.append(_metrics_at_k(ranked_ids, relevance, k))
        out[depth] = {
            "n": len(rows),
            "avg_total_episodes": depth,
            f"recall@{k}": round(mean(r["recall"] for r in rows), 4) if rows else None,
            f"precision@{k}": round(mean(r["precision"] for r in rows), 4) if rows else None,
            "mrr": round(mean(r["mrr"] for r in rows), 4) if rows else None,
            f"hit@{k}": round(mean(r["hit"] for r in rows), 4) if rows else None,
            "avg_relevant_present": round(mean(r["n_rel"] for r in rows), 2) if rows else None,
        }
    return out


def run_adoption_eval(report_path: str) -> dict:
    """Heuristic downstream-adoption: did episode-only content surface in answers?

    For each sample present in BOTH the seeded output-eval report and the seed
    file, extract distinctive tokens that appear ONLY in the relevant episodes'
    gist/facts (not in the profile, current query, or reference answer), then
    check whether the produced answer references any of them.
    """
    import re

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    details = {d["id"]: d for d in report.get("details", [])}
    seeds = _load_seeds()
    dataset = _load_dataset()

    def toks(text: str) -> set:
        # crude bilingual tokenizer: 2-4 char Chinese n-grams + ascii words/numbers
        text = text or ""
        out = set(re.findall(r"[a-zA-Z]{3,}|\d+(?:\.\d+)?", text.lower()))
        zh = re.findall(r"[一-鿿]+", text)
        for run in zh:
            for n in (3, 4):
                for i in range(len(run) - n + 1):
                    out.add(run[i : i + n])
        return out

    rows = []
    for sid, seed_eps in seeds.items():
        if sid not in details or sid not in dataset:
            continue
        d = details[sid]
        answer = d.get("answer", "")
        ep_ctx_len = d.get("episode_context_len", 0)
        sample = dataset[sid]
        rel_eps = [e for e in seed_eps if e.get("relevant")]
        rel_text = " ".join((e.get("gist", "") + " " + json.dumps(e.get("facts", {}), ensure_ascii=False)) for e in rel_eps)
        baseline_text = (
            json.dumps(sample.get("profile", {}), ensure_ascii=False)
            + " " + _current_query(sample)
            + " " + sample.get("reference_answer", "")
        )
        episode_only = toks(rel_text) - toks(baseline_text)
        ans_tokens = toks(answer)
        overlap = episode_only & ans_tokens
        rows.append({
            "id": sid,
            "category": sample.get("category", "unknown"),
            "episode_context_len": ep_ctx_len,
            "episode_only_tokens": len(episode_only),
            "adopted_tokens": len(overlap),
            "adopted": bool(overlap),
            "sample_overlap": sorted(list(overlap))[:8],
        })

    n = len(rows)
    with_ctx = [r for r in rows if r["episode_context_len"] > 0]
    adopted = [r for r in with_ctx if r["adopted"]]
    return {
        "n_samples": n,
        "n_with_episode_context": len(with_ctx),
        "adoption_rate_over_with_context": round(len(adopted) / len(with_ctx), 3) if with_ctx else None,
        "avg_adopted_tokens": round(mean(r["adopted_tokens"] for r in with_ctx), 2) if with_ctx else None,
        "per_sample": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", default="1,2,3,5,8", help="comma-separated k values to sweep")
    ap.add_argument("--depth-sweep", default="", help="comma-separated history depths (e.g. 3,5,8,9) for the MIN_COUNT retrieval trade-off")
    ap.add_argument("--output-report", default="", help="seeded output-eval report for adoption analysis")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "episode_memory_eval.json"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    topks = sorted({int(x) for x in args.topk.split(",") if x.strip()})
    max_k = max(topks)

    result = {"retrieval": run_retrieval_eval(topks, max_k, args.verbose)}
    if args.depth_sweep:
        depths = sorted({int(x) for x in args.depth_sweep.split(",") if x.strip()})
        result["depth_sweep"] = run_depth_sweep(depths, k=3)
    if args.output_report:
        result["adoption"] = run_adoption_eval(args.output_report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Console summary.
    r = result["retrieval"]
    print("\n" + "=" * 64)
    print("EPISODIC-MEMORY RETRIEVAL EVAL")
    print("=" * 64)
    print(f"samples              : {r['n_samples']}")
    print(f"top-1 relevant rate  : {r['top1_relevant_rate']:.1%}")
    print("\nTOP_K sensitivity (recall / precision / mrr / hit@k):")
    for k in topks:
        a = r["topk_sweep"][k]
        print(f"  k={k:<2}  recall={a['recall']:.3f}  prec={a['precision']:.3f}  mrr={a['mrr']:.3f}  hit@k={a['hit_rate']:.3f}")
    print("\nby category (recall@3 / mrr / top1-rel):")
    for c, s in sorted(r["by_category"].items()):
        print(f"  {c:<22} n={s['n']:<3} recall@3={s['recall@3']:.2f}  mrr={s['mrr']:.2f}  top1rel={s['top1_relevant_rate']:.2f}")
    if "depth_sweep" in result:
        print("\n" + "-" * 64)
        print("HISTORY-DEPTH SWEEP (retrieval quality vs # episodes user has)")
        print(f"{'depth':>6}{'recall@3':>10}{'prec@3':>9}{'mrr':>8}{'hit@3':>8}{'rel_present':>13}")
        for d in sorted(result["depth_sweep"]):
            s = result["depth_sweep"][d]
            print(f"{d:>6}{s['recall@3']:>10.3f}{s['precision@3']:>9.3f}{s['mrr']:>8.3f}{s['hit@3']:>8.3f}{s['avg_relevant_present']:>13.2f}")
    if "adoption" in result:
        ad = result["adoption"]
        print("\n" + "-" * 64)
        print("DOWNSTREAM ADOPTION (episode-only content surfaced in answers)")
        print(f"  samples w/ episode_context : {ad['n_with_episode_context']}/{ad['n_samples']}")
        print(f"  adoption rate              : {ad['adoption_rate_over_with_context']}")
        print(f"  avg adopted tokens         : {ad['avg_adopted_tokens']}")
    print(f"\nReport → {out}")


if __name__ == "__main__":
    main()
