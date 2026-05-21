"""A/B 对比多个 embedding 模型在同一评测集上的召回质量。

为什么用子进程运行每个模型:
- `wellness_copilot.rag` 在 import 时从 `wellness_copilot.config` 读取
  `RAG_EMBED_MODEL_NAME`,属于模块级常量,运行中切换比较 tricky。
- 用 `subprocess + env var` 每次都是一个干净的 Python 进程,避免两个
  embedding 模型同时驻留显存(8GB 端侧可能 OOM),也避免 index cache 串扰。
- `wellness_copilot.rag._docs_fingerprint` 已经把模型名纳入哈希,不同模型的
  cache 自动失效,无需手动清理。

用法示例:

    python scripts/compare_embedders.py \\
        --models BAAI/bge-small-zh-v1.5,BAAI/bge-m3 \\
        --dataset eval/rag_eval_dataset.jsonl

输出:

- `reports/embedder_compare/<safe_name>.json` —— 每个模型的完整两阶段评测报告
- `reports/embedder_compare/summary.json` —— 汇总 + Δ
- 控制台打印一张 side-by-side 对比表
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _safe_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def _run_single(
    model: str,
    dataset: Path,
    stage1_ks: str,
    stage2_ks: str,
    stage1_pool: int,
    out_dir: Path,
    chunk_size: int,
    overlap: int,
    boundary_look_back: int,
    min_chunk_chars: int,
) -> Dict:
    report_path = out_dir / f"report_{_safe_name(model)}.json"
    env = os.environ.copy()
    env["RAG_EMBED_MODEL_NAME"] = model

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_rag.py"),
        "--dataset", str(dataset),
        "--stage1-ks", stage1_ks,
        "--stage2-ks", stage2_ks,
        "--stage1-pool", str(stage1_pool),
        "--chunk-size", str(chunk_size),
        "--overlap", str(overlap),
        "--boundary-look-back", str(boundary_look_back),
        "--min-chunk-chars", str(min_chunk_chars),
        "--out", str(report_path),
    ]

    print("\n" + "=" * 72)
    print(f"[Compare] running eval with RAG_EMBED_MODEL_NAME = {model}")
    print("=" * 72)
    subprocess.run(cmd, env=env, check=True)

    return json.loads(report_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Pretty-printing a side-by-side table
# --------------------------------------------------------------------------- #


def _collect_metric_row(report: Dict, stage: str, metric_path: List[str]) -> float:
    node = report.get(stage, {})
    for key in metric_path:
        node = node.get(key, {}) if isinstance(node, dict) else {}
    return float(node) if isinstance(node, (int, float)) else 0.0


def _format_table(
    models: List[str],
    reports: Dict[str, Dict],
    stage1_ks: List[int],
    stage2_ks: List[int],
) -> str:
    rows: List[List[str]] = []

    def add(metric: str, values: List[float]):
        row = [metric] + [f"{v:.4f}" for v in values]
        if len(values) == 2:
            delta = values[1] - values[0]
            marker = "↑" if delta > 0 else ("↓" if delta < 0 else "·")
            row.append(f"{marker} {delta:+.4f}")
        rows.append(row)

    # --- Embedding Stage (stage1) ---
    rows.append(["--- EMBEDDING STAGE (dense-only) ---"] + [""] * (len(models) + 1))
    add(
        "  MRR",
        [_collect_metric_row(r, "embedding_stage", ["mrr"]) for r in reports.values()],
    )
    for k in stage1_ks:
        add(
            f"  Recall@{k}",
            [
                _collect_metric_row(r, "embedding_stage", ["recall", f"recall@{k}"])
                for r in reports.values()
            ],
        )
        add(
            f"  nDCG@{k}",
            [
                _collect_metric_row(r, "embedding_stage", ["ndcg", f"ndcg@{k}"])
                for r in reports.values()
            ],
        )
        add(
            f"  MAP@{k}",
            [
                _collect_metric_row(r, "embedding_stage", ["map", f"map@{k}"])
                for r in reports.values()
            ],
        )

    # --- Rerank Stage (stage2) ---
    rows.append(["--- RERANK STAGE (cross-encoder top-N) ---"] + [""] * (len(models) + 1))
    add(
        "  MRR",
        [_collect_metric_row(r, "rerank_stage", ["mrr"]) for r in reports.values()],
    )
    for k in stage2_ks:
        add(
            f"  Recall@{k}",
            [
                _collect_metric_row(r, "rerank_stage", ["recall", f"recall@{k}"])
                for r in reports.values()
            ],
        )
        add(
            f"  nDCG@{k}",
            [
                _collect_metric_row(r, "rerank_stage", ["ndcg", f"ndcg@{k}"])
                for r in reports.values()
            ],
        )
        add(
            f"  MAP@{k}",
            [
                _collect_metric_row(r, "rerank_stage", ["map", f"map@{k}"])
                for r in reports.values()
            ],
        )

    headers = ["Metric"] + models + (["Δ (B-A)"] if len(models) == 2 else [])
    col_widths = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]

    def fmt(row):
        return "  ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers)))

    lines = [fmt(headers), "-" * (sum(col_widths) + 2 * (len(headers) - 1))]
    for r in rows:
        lines.append(fmt(r + [""] * (len(headers) - len(r))))
    return "\n".join(lines)


def _summarize(
    models: List[str],
    reports: Dict[str, Dict],
    stage1_ks: List[int],
    stage2_ks: List[int],
) -> Dict:
    summary = {"models": models, "reports": {}, "delta": {}}
    for model in models:
        r = reports[model]
        summary["reports"][model] = {
            "embedding_stage": r.get("embedding_stage"),
            "rerank_stage": r.get("rerank_stage"),
            "rerank_uplift_vs_embedding": r.get("rerank_uplift_vs_embedding"),
        }

    if len(models) == 2:
        a, b = models
        ra, rb = reports[a], reports[b]

        def d(stage, path_a, path_b):
            va = _collect_metric_row(ra, stage, path_a)
            vb = _collect_metric_row(rb, stage, path_b)
            return round(vb - va, 4)

        delta = {"embedding_stage": {}, "rerank_stage": {}}
        delta["embedding_stage"]["mrr"] = d("embedding_stage", ["mrr"], ["mrr"])
        for k in stage1_ks:
            delta["embedding_stage"][f"recall@{k}"] = d(
                "embedding_stage", ["recall", f"recall@{k}"], ["recall", f"recall@{k}"]
            )
            delta["embedding_stage"][f"ndcg@{k}"] = d(
                "embedding_stage", ["ndcg", f"ndcg@{k}"], ["ndcg", f"ndcg@{k}"]
            )
        delta["rerank_stage"]["mrr"] = d("rerank_stage", ["mrr"], ["mrr"])
        for k in stage2_ks:
            delta["rerank_stage"][f"recall@{k}"] = d(
                "rerank_stage", ["recall", f"recall@{k}"], ["recall", f"recall@{k}"]
            )
            delta["rerank_stage"][f"ndcg@{k}"] = d(
                "rerank_stage", ["ndcg", f"ndcg@{k}"], ["ndcg", f"ndcg@{k}"]
            )
        summary["delta"] = {"baseline": a, "candidate": b, "metrics": delta}

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="A/B compare multiple embedding models on the RAG eval set."
    )
    parser.add_argument(
        "--models",
        default="BAAI/bge-small-zh-v1.5,BAAI/bge-m3",
        help="Comma-separated HF model ids to compare (first = baseline).",
    )
    parser.add_argument(
        "--dataset",
        default="eval/rag_eval_dataset.jsonl",
        help="Path to JSONL eval dataset",
    )
    parser.add_argument("--stage1-ks", default="5,10,20")
    parser.add_argument("--stage2-ks", default="1,3,5")
    parser.add_argument("--stage1-pool", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=420)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--boundary-look-back", type=int, default=120)
    parser.add_argument("--min-chunk-chars", type=int, default=30)
    parser.add_argument(
        "--out-dir",
        default="reports/embedder_compare",
        help="Output directory for individual and summary reports",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(models) < 2:
        parser.error("--models must contain at least two comma-separated model ids")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1_ks = sorted({int(x) for x in args.stage1_ks.split(",")})
    stage2_ks = sorted({int(x) for x in args.stage2_ks.split(",")})

    reports: Dict[str, Dict] = {}
    for model in models:
        reports[model] = _run_single(
            model=model,
            dataset=Path(args.dataset),
            stage1_ks=args.stage1_ks,
            stage2_ks=args.stage2_ks,
            stage1_pool=args.stage1_pool,
            out_dir=out_dir,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            boundary_look_back=args.boundary_look_back,
            min_chunk_chars=args.min_chunk_chars,
        )

    summary = _summarize(models, reports, stage1_ks, stage2_ks)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("[Compare] Side-by-side comparison")
    print("=" * 72)
    print(_format_table(models, reports, stage1_ks, stage2_ks))
    print()
    print(f"[Compare] Summary written to: {summary_path}")
    print(f"[Compare] Individual reports: {out_dir}/report_<model>.json")


if __name__ == "__main__":
    main()
