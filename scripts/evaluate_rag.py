"""RAG 召回准确率评测。

本脚本将 RAG 的两个阶段拆开独立评测：

1. Embedding（Stage-1 Dense Retrieve）：只看向量召回的候选池质量
   - 目的：衡量 embedding 模型把相关片段"捞进"候选池的能力
   - 主指标：Recall@k（k 较大，例如 10/20）
   - 辅助：MRR / nDCG（排序位置的参考）

2. Rerank（Stage-2 Cross-Encoder Re-rank）：看重排后头部的质量
   - 目的：衡量重排器把最相关的片段"挤到"前排、供 LLM 作为上下文的能力
   - 主指标：MRR / nDCG@k / Recall@k（k 较小，例如 1/3/5）
   - 同时报告 vs Stage-1 的 uplift（Δ）——隔离重排器的边际贡献

指标选取理由参见 README 的"RAG 评测"章节。
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class EvalSample:
    query: str
    agent: str
    relevant_sources: Set[str]
    relevant_chunk_ids: Set[str]


# --------------------------------------------------------------------------- #
# Dataset I/O
# --------------------------------------------------------------------------- #


def load_dataset(path: Path) -> List[EvalSample]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples: List[EvalSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query = (row.get("query") or "").strip()
            if not query:
                raise ValueError(f"Line {line_no}: missing non-empty query")
            agent = (row.get("agent") or "general").strip().lower()

            sources = set(row.get("relevant_sources") or [])
            chunks = set(row.get("relevant_chunk_ids") or [])
            if not sources and not chunks:
                raise ValueError(
                    f"Line {line_no}: provide at least one of relevant_sources/relevant_chunk_ids"
                )

            samples.append(
                EvalSample(
                    query=query,
                    agent=agent,
                    relevant_sources=sources,
                    relevant_chunk_ids=chunks,
                )
            )
    return samples


# --------------------------------------------------------------------------- #
# Relevance judgement
# --------------------------------------------------------------------------- #


def _normalize_source(source: str) -> str:
    # 路由器会为 source 加 namespace 前缀 (e.g. "nutritionist/xxx.md")。
    # 为了让数据集中的裸文件名也能匹配，这里提取 basename 做一次额外比对。
    return (source or "").split("/")[-1]


def _is_relevant(item: Dict[str, object], sample: EvalSample) -> bool:
    source = str(item.get("source") or "")
    chunk_id = str(item.get("chunk_id") or "")
    basename = _normalize_source(source)

    if sample.relevant_sources:
        for s in sample.relevant_sources:
            if source == s or source.endswith("/" + s) or basename == s:
                return True

    if sample.relevant_chunk_ids:
        for c in sample.relevant_chunk_ids:
            if chunk_id == c or chunk_id.endswith(c):
                return True

    return False


def _num_relevant_total(sample: EvalSample) -> int:
    # 以更细粒度的 ground truth 作为分母：优先 chunk 级；否则 source 级。
    if sample.relevant_chunk_ids:
        return len(sample.relevant_chunk_ids)
    return max(1, len(sample.relevant_sources))


def _unique_relevance_key(item: Dict[str, object], sample: EvalSample) -> Optional[str]:
    """为同一个相关 ground truth 去重（防止同文档多 chunk 被重复计入召回率分子）。"""
    source = str(item.get("source") or "")
    chunk_id = str(item.get("chunk_id") or "")
    basename = _normalize_source(source)

    if sample.relevant_chunk_ids:
        for c in sample.relevant_chunk_ids:
            if chunk_id == c or chunk_id.endswith(c):
                return f"chunk::{c}"

    if sample.relevant_sources:
        for s in sample.relevant_sources:
            if source == s or source.endswith("/" + s) or basename == s:
                return f"source::{s}"

    return None


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _recall_at_k(results: List[Dict[str, object]], sample: EvalSample, k: int) -> float:
    top = results[:k]
    total = _num_relevant_total(sample)
    hit_keys = set()
    for r in top:
        key = _unique_relevance_key(r, sample)
        if key:
            hit_keys.add(key)
    return len(hit_keys) / total if total else 0.0


def _hit_at_k(results: List[Dict[str, object]], sample: EvalSample, k: int) -> int:
    top = results[:k]
    return 1 if any(_is_relevant(r, sample) for r in top) else 0


def _first_relevant_rank(results: List[Dict[str, object]], sample: EvalSample) -> Optional[int]:
    for idx, r in enumerate(results, start=1):
        if _is_relevant(r, sample):
            return idx
    return None


def _reciprocal_rank(results: List[Dict[str, object]], sample: EvalSample) -> float:
    rank = _first_relevant_rank(results, sample)
    return 1.0 / rank if rank else 0.0


def _dcg(rels: List[int]) -> float:
    # 使用标准形式：sum(rel_i / log2(i+1))，二元相关度
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def _ndcg_at_k(results: List[Dict[str, object]], sample: EvalSample, k: int) -> float:
    top = results[:k]
    rels = [1 if _is_relevant(r, sample) else 0 for r in top]
    ideal_hits = min(_num_relevant_total(sample), k)
    ideal = [1] * ideal_hits + [0] * (k - ideal_hits)
    idcg = _dcg(ideal)
    if idcg == 0:
        return 0.0
    return _dcg(rels) / idcg


def _average_precision(results: List[Dict[str, object]], sample: EvalSample, k: int) -> float:
    top = results[:k]
    total = _num_relevant_total(sample)
    if total == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    seen_keys: Set[str] = set()
    for idx, r in enumerate(top, start=1):
        key = _unique_relevance_key(r, sample)
        if key and key not in seen_keys:
            seen_keys.add(key)
            hits += 1
            precision_sum += hits / idx
    return precision_sum / min(total, k)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _stage_metrics(
    results: List[Dict[str, object]],
    sample: EvalSample,
    ks: List[int],
) -> Dict[str, float]:
    row: Dict[str, float] = {
        "mrr": _reciprocal_rank(results, sample),
        "first_relevant_rank": _first_relevant_rank(results, sample) or 0,
    }
    for k in ks:
        row[f"recall@{k}"] = _recall_at_k(results, sample, k)
        row[f"hit@{k}"] = _hit_at_k(results, sample, k)
        row[f"ndcg@{k}"] = _ndcg_at_k(results, sample, k)
        row[f"map@{k}"] = _average_precision(results, sample, k)
    return row


def _average(rows: List[Dict[str, float]], keys: List[str]) -> Dict[str, float]:
    n = len(rows) or 1
    out = {}
    for k in keys:
        out[k] = round(sum(r.get(k, 0.0) for r in rows) / n, 4)
    return out


def _summarize_stage(
    per_sample_rows: List[Dict[str, float]],
    ks: List[int],
) -> Dict[str, object]:
    keys = ["mrr"]
    for k in ks:
        keys.extend([f"recall@{k}", f"hit@{k}", f"ndcg@{k}", f"map@{k}"])

    aggregated = _average(per_sample_rows, keys)
    return {
        "mrr": aggregated["mrr"],
        "recall": {f"recall@{k}": aggregated[f"recall@{k}"] for k in ks},
        "hit_rate": {f"hit_rate@{k}": aggregated[f"hit@{k}"] for k in ks},
        "ndcg": {f"ndcg@{k}": aggregated[f"ndcg@{k}"] for k in ks},
        "map": {f"map@{k}": aggregated[f"map@{k}"] for k in ks},
    }


def _diff_stage(
    after: Dict[str, object],
    before: Dict[str, object],
    ks: List[int],
) -> Dict[str, object]:
    """rerank 相对 dense 的提升（Δ）。正值表示 rerank 改善。"""

    def d(a: float, b: float) -> float:
        return round(a - b, 4)

    return {
        "mrr": d(after["mrr"], before["mrr"]),
        "recall": {
            f"recall@{k}": d(after["recall"][f"recall@{k}"], before["recall"][f"recall@{k}"])
            for k in ks
        },
        "hit_rate": {
            f"hit_rate@{k}": d(
                after["hit_rate"][f"hit_rate@{k}"], before["hit_rate"][f"hit_rate@{k}"]
            )
            for k in ks
        },
        "ndcg": {
            f"ndcg@{k}": d(after["ndcg"][f"ndcg@{k}"], before["ndcg"][f"ndcg@{k}"])
            for k in ks
        },
        "map": {
            f"map@{k}": d(after["map"][f"map@{k}"], before["map"][f"map@{k}"]) for k in ks
        },
    }


# --------------------------------------------------------------------------- #
# Main evaluation loop
# --------------------------------------------------------------------------- #


def evaluate(
    samples: List[EvalSample],
    agent_kbs: Dict,
    stage1_ks: List[int],
    stage2_ks: List[int],
    stage1_pool: int,
) -> Dict[str, object]:
    # 两阶段会各自用不同的 k 列表：
    # - embedding 阶段看候选池的覆盖能力，关注较大的 k
    # - rerank 阶段看头部的精度，关注较小的 k
    max_stage1_k = max(stage1_ks + [stage1_pool])
    max_stage2_k = max(stage2_ks)

    stage1_rows_all: List[Dict[str, float]] = []
    stage2_rows_all: List[Dict[str, float]] = []

    per_agent_stage1: Dict[str, List[Dict[str, float]]] = {}
    per_agent_stage2: Dict[str, List[Dict[str, float]]] = {}

    details = []

    for sample in samples:
        kb = agent_kbs.get(sample.agent) or agent_kbs.get("general")
        if kb is None:
            raise ValueError(f"No KB found for agent '{sample.agent}'")
        staged = kb.retrieve_stages(
            query=sample.query,
            stage1_k=max_stage1_k,
            stage2_k=max_stage2_k,
        )
        stage1 = staged["stage1"]
        stage2 = staged["stage2"]

        s1_row = _stage_metrics(stage1, sample, stage1_ks)
        s2_row = _stage_metrics(stage2, sample, stage2_ks)

        stage1_rows_all.append(s1_row)
        stage2_rows_all.append(s2_row)
        per_agent_stage1.setdefault(sample.agent, []).append(s1_row)
        per_agent_stage2.setdefault(sample.agent, []).append(s2_row)

        details.append(
            {
                "query": sample.query,
                "agent": sample.agent,
                "stage1_first_rank": s1_row["first_relevant_rank"],
                "stage2_first_rank": s2_row["first_relevant_rank"],
                "stage1": {k: round(v, 4) for k, v in s1_row.items()},
                "stage2": {k: round(v, 4) for k, v in s2_row.items()},
                "stage1_top": [
                    {
                        "rank": r.get("rank"),
                        "source": r.get("source"),
                        "chunk_id": r.get("chunk_id"),
                        "dense_score": r.get("dense_score"),
                    }
                    for r in stage1[:max_stage1_k]
                ],
                "stage2_top": [
                    {
                        "rank": r.get("rank"),
                        "source": r.get("source"),
                        "chunk_id": r.get("chunk_id"),
                        "dense_score": r.get("dense_score"),
                        "rerank_score": r.get("rerank_score"),
                    }
                    for r in stage2[:max_stage2_k]
                ],
            }
        )

    stage1_summary = _summarize_stage(stage1_rows_all, stage1_ks)
    stage2_summary = _summarize_stage(stage2_rows_all, stage2_ks)

    # 仅在两阶段共享的 k 上计算 uplift（其余 k 不可比）
    shared_ks = sorted(set(stage1_ks).intersection(stage2_ks))
    uplift = (
        _diff_stage(
            _summarize_stage(stage2_rows_all, shared_ks),
            _summarize_stage(stage1_rows_all, shared_ks),
            shared_ks,
        )
        if shared_ks
        else {}
    )

    per_agent_summary = {}
    agents = sorted(set(list(per_agent_stage1.keys()) + list(per_agent_stage2.keys())))
    for agent in agents:
        per_agent_summary[agent] = {
            "sample_count": len(per_agent_stage1.get(agent, [])),
            "embedding_stage": _summarize_stage(per_agent_stage1.get(agent, []), stage1_ks),
            "rerank_stage": _summarize_stage(per_agent_stage2.get(agent, []), stage2_ks),
        }

    return {
        "config": {
            "sample_count": len(samples),
            "stage1_ks": stage1_ks,
            "stage2_ks": stage2_ks,
            "stage1_pool": stage1_pool,
        },
        "embedding_stage": stage1_summary,
        "rerank_stage": stage2_summary,
        "rerank_uplift_vs_embedding": uplift,
        "per_agent_summary": per_agent_summary,
        "details": details,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_ks(raw: str) -> List[int]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    ks = sorted({int(x) for x in items})
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers, e.g. 1,3,5")
    return ks


def _pretty_print_summary(report: Dict[str, object]) -> None:
    cfg = report["config"]
    print("=" * 72)
    print(f"[RAG Eval] samples = {cfg['sample_count']}")
    print(f"[RAG Eval] stage1 ks = {cfg['stage1_ks']} | stage1_pool = {cfg['stage1_pool']}")
    print(f"[RAG Eval] stage2 ks = {cfg['stage2_ks']}")
    print("-" * 72)
    print("[Embedding Stage  (dense retrieve only)]")
    print(json.dumps(report["embedding_stage"], ensure_ascii=False, indent=2))
    print("-" * 72)
    print("[Rerank Stage  (cross-encoder on dense top-N)]")
    print(json.dumps(report["rerank_stage"], ensure_ascii=False, indent=2))
    print("-" * 72)
    if report["rerank_uplift_vs_embedding"]:
        print("[Uplift: rerank - embedding  (positive = rerank helps)]")
        print(json.dumps(report["rerank_uplift_vs_embedding"], ensure_ascii=False, indent=2))
    print("=" * 72)


def main():
    from pathlib import Path as _Path
    from health_guide.config import KNOWLEDGE_BASE_DIR, KNOWLEDGE_BASE_AGENT_SUBDIRS
    from health_guide.rag import LocalKnowledgeBase

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RAG pipeline: Embedding recall quality and Rerank precision "
            "quality are measured separately (with uplift delta)."
        )
    )
    parser.add_argument(
        "--dataset",
        default="eval/rag_eval_dataset.jsonl",
        help="Path to JSONL eval dataset",
    )
    parser.add_argument("--kb-dir", default=KNOWLEDGE_BASE_DIR, help="Knowledge base directory")
    parser.add_argument("--chunk-size", type=int, default=420, help="Chunk size")
    parser.add_argument("--overlap", type=int, default=100, help="Chunk overlap")
    parser.add_argument(
        "--boundary-look-back",
        type=int,
        default=120,
        help="Max chars to look back for a sentence boundary when snapping chunk end (default: 120)",
    )
    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=30,
        help="Discard chunks shorter than this many characters (default: 30)",
    )
    parser.add_argument(
        "--stage1-ks",
        default="5,10,20",
        help="Comma-separated k list for embedding stage metrics (default: 5,10,20)",
    )
    parser.add_argument(
        "--stage2-ks",
        default="1,3,5",
        help="Comma-separated k list for rerank stage metrics (default: 1,3,5)",
    )
    parser.add_argument(
        "--stage1-pool",
        type=int,
        default=20,
        help="Candidate pool size for stage-1 dense retrieve (fed into rerank). Default 20.",
    )
    parser.add_argument(
        "--out",
        default="reports/rag_eval_report.json",
        help="Path to output evaluation report JSON",
    )
    args = parser.parse_args()

    stage1_ks = parse_ks(args.stage1_ks)
    stage2_ks = parse_ks(args.stage2_ks)
    dataset = load_dataset(Path(args.dataset))

    kb_root = _Path(args.kb_dir)
    agent_kbs = {}
    for agent, subdir in KNOWLEDGE_BASE_AGENT_SUBDIRS.items():
        kb = LocalKnowledgeBase(
            kb_dir=str(kb_root / subdir),
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            boundary_look_back=args.boundary_look_back,
            min_chunk_chars=args.min_chunk_chars,
        )
        kb.build(force_rebuild=False)
        agent_kbs[agent] = kb

    report = evaluate(
        samples=dataset,
        agent_kbs=agent_kbs,
        stage1_ks=stage1_ks,
        stage2_ks=stage2_ks,
        stage1_pool=args.stage1_pool,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _pretty_print_summary(report)
    print(f"[RAG Eval] Report written to: {out}")


if __name__ == "__main__":
    main()
