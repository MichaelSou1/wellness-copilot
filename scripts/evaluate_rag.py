"""RAG 召回准确率评测。

本脚本将 RAG 的两个阶段拆开独立评测：

1. Embedding（Stage-1 Dense Retrieve + PDF section expansion）：只看候选池质量
   - 目的：衡量 embedding/section parent-child 召回把相关片段"捞进"候选池的能力
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
import re
import sys
from dataclasses import dataclass
from datetime import datetime
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
    page_ranges: Set[str] = None
    line_no: int = 0
    question_type: str = ""
    difficulty: str = ""
    sample_kind: str = ""
    answer: str = ""
    supporting_span: str = ""
    quality: Optional[Dict[str, object]] = None


# --------------------------------------------------------------------------- #
# Dataset I/O
# --------------------------------------------------------------------------- #


def _normalize_path_text(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _strip_agent_namespace(value: str, agent: str) -> str:
    """Normalize legacy/new eval IDs to LocalKnowledgeBase's per-agent ID form.

    LocalKnowledgeBase is instantiated per namespace, so returned sources look like
    `bmi_body_composition.md`. Some generated/curated datasets store the namespace
    explicitly, e.g. `nutritionist/bmi_body_composition.md` or
    `nutritionist:bmi_body_composition.md#chunk-1`.
    """
    text = _normalize_path_text(value)
    agent = (agent or "").strip().lower()
    if not text or not agent:
        return text

    lowered = text.lower()
    colon_prefix = f"{agent}:"
    slash_prefix = f"{agent}/"
    if lowered.startswith(colon_prefix):
        return text[len(colon_prefix) :]
    if lowered.startswith(slash_prefix):
        return text[len(slash_prefix) :]
    return text


def _canonical_source(source: str, agent: str) -> str:
    return _strip_agent_namespace(source, agent)


def _canonical_chunk_id(chunk_id: str, agent: str) -> str:
    return _strip_agent_namespace(chunk_id, agent)


def _canonical_set(values: object, agent: str, *, chunk_ids: bool = False) -> Set[str]:
    if not values:
        return set()
    if isinstance(values, str):
        values = [values]
    out: Set[str] = set()
    for value in values:
        normalized = (
            _canonical_chunk_id(str(value), agent)
            if chunk_ids
            else _canonical_source(str(value), agent)
        )
        if normalized:
            out.add(normalized)
    return out


def _canonical_page_ranges(raw_page_range: object, chunk_ids: Set[str]) -> Set[str]:
    ranges: Set[str] = set()
    if isinstance(raw_page_range, str) and raw_page_range.strip():
        ranges.add(raw_page_range.strip())
    elif isinstance(raw_page_range, list):
        ranges.update(str(value).strip() for value in raw_page_range if str(value).strip())

    for chunk_id in chunk_ids:
        match = re.search(r"#p(\d+(?:-\d+)?)-chunk-\d+", chunk_id)
        if match:
            ranges.add(match.group(1))
    return ranges


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
            agent = (row.get("agent") or "").strip().lower()
            if not agent:
                raise ValueError(f"Line {line_no}: missing non-empty agent")

            sources = _canonical_set(row.get("relevant_sources") or [], agent)
            chunks = _canonical_set(
                row.get("relevant_chunk_ids") or [],
                agent,
                chunk_ids=True,
            )
            page_ranges = _canonical_page_ranges(row.get("page_range"), chunks)
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
                    page_ranges=page_ranges,
                    line_no=line_no,
                    question_type=str(row.get("question_type") or "").strip(),
                    difficulty=str(row.get("difficulty") or "").strip(),
                    sample_kind=str(row.get("sample_kind") or "").strip(),
                    answer=str(row.get("answer") or "").strip(),
                    supporting_span=str(row.get("supporting_span") or "").strip(),
                    quality=row.get("quality") if isinstance(row.get("quality"), dict) else None,
                )
            )
    if not samples:
        raise ValueError(f"Dataset is empty: {path}")
    return samples


# --------------------------------------------------------------------------- #
# Relevance judgement
# --------------------------------------------------------------------------- #


def _normalize_source(source: str) -> str:
    # 兼容历史数据集中的裸文件名 ground truth。
    return (source or "").split("/")[-1]


def _source_matches(item_source: str, truth_source: str, agent: str) -> bool:
    source = _canonical_source(item_source, agent)
    truth = _canonical_source(truth_source, agent)
    if source == truth:
        return True
    if source.endswith("/" + truth) or truth.endswith("/" + source):
        return True
    return _normalize_source(source) == _normalize_source(truth)


def _chunk_matches(item_chunk_id: str, truth_chunk_id: str, agent: str) -> bool:
    chunk_id = _canonical_chunk_id(item_chunk_id, agent)
    truth = _canonical_chunk_id(truth_chunk_id, agent)
    if not chunk_id or not truth:
        return False
    return chunk_id == truth or chunk_id.endswith(truth) or truth.endswith(chunk_id)


def _is_relevant(item: Dict[str, object], sample: EvalSample) -> bool:
    return _unique_relevance_key(item, sample) is not None


def _parse_page_range(page_range: object) -> Optional[tuple]:
    text = str(page_range or "").strip()
    match = re.match(r"^(\d+)(?:-(\d+))?$", text)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2) or start)
    return (min(start, end), max(start, end))


def _page_ranges_overlap(left: object, right: object) -> bool:
    l_range = _parse_page_range(left)
    r_range = _parse_page_range(right)
    if not l_range or not r_range:
        return False
    return max(l_range[0], r_range[0]) <= min(l_range[1], r_range[1])


def _chunk_sequence(chunk_id: str, agent: str) -> Optional[tuple]:
    normalized = _canonical_chunk_id(chunk_id, agent)
    match = re.match(r"^(?P<source>.+)#(?:p\d+(?:-\d+)?-)?chunk-(?P<idx>\d+)$", normalized)
    if not match:
        return None
    return (match.group("source"), int(match.group("idx")))


def _result_chunk_ids(item: Dict[str, object]) -> List[str]:
    ids = []
    primary = str(item.get("chunk_id") or "").strip()
    if primary:
        ids.append(primary)
    context_ids = item.get("context_chunk_ids") or []
    if isinstance(context_ids, str):
        context_ids = [context_ids]
    for chunk_id in context_ids:
        chunk_id = str(chunk_id or "").strip()
        if chunk_id and chunk_id not in ids:
            ids.append(chunk_id)
    return ids


def _source_hit(item: Dict[str, object], sample: EvalSample) -> bool:
    source = str(item.get("source") or "")
    truth_sources = set(sample.relevant_sources)
    if not truth_sources:
        truth_sources.update(
            _canonical_chunk_id(chunk_id, sample.agent).split("#", 1)[0]
            for chunk_id in sample.relevant_chunk_ids
            if "#" in _canonical_chunk_id(chunk_id, sample.agent)
        )
    return any(_source_matches(source, truth, sample.agent) for truth in truth_sources)


def _page_hit(item: Dict[str, object], sample: EvalSample) -> bool:
    if not sample.page_ranges or not _source_hit(item, sample):
        return False
    candidate_ranges = [
        item.get("page_range"),
        item.get("context_page_range"),
    ]
    return any(
        _page_ranges_overlap(candidate, truth)
        for candidate in candidate_ranges
        for truth in sample.page_ranges
    )


def _adjacent_chunk_hit(item: Dict[str, object], sample: EvalSample, radius: int = 1) -> bool:
    if not sample.relevant_chunk_ids:
        return False
    truth_sequences = [
        seq
        for seq in (_chunk_sequence(chunk_id, sample.agent) for chunk_id in sample.relevant_chunk_ids)
        if seq is not None
    ]
    if not truth_sequences:
        return False
    for candidate_id in _result_chunk_ids(item):
        candidate_seq = _chunk_sequence(candidate_id, sample.agent)
        if candidate_seq is None:
            continue
        candidate_source, candidate_idx = candidate_seq
        for truth_source, truth_idx in truth_sequences:
            if candidate_source == truth_source and abs(candidate_idx - truth_idx) <= radius:
                return True
    return False


def _same_page_or_adjacent_chunk_hit(item: Dict[str, object], sample: EvalSample) -> bool:
    if _is_relevant(item, sample):
        return True
    for candidate_id in _result_chunk_ids(item):
        if any(_chunk_matches(candidate_id, truth, sample.agent) for truth in sample.relevant_chunk_ids):
            return True
    return _page_hit(item, sample) or _adjacent_chunk_hit(item, sample)


def _num_relevant_total(sample: EvalSample) -> int:
    # 以更细粒度的 ground truth 作为分母：优先 chunk 级；否则 source 级。
    if sample.relevant_chunk_ids:
        return len(sample.relevant_chunk_ids)
    return max(1, len(sample.relevant_sources))


def _unique_relevance_key(item: Dict[str, object], sample: EvalSample) -> Optional[str]:
    """返回命中的 ground truth key，并用于同一相关项去重。

    如果样本提供 chunk 级 ground truth，就只按 chunk 判断相关性；source 只作为
    兼容/校验信息存在，不能让同文档的其他 chunk 被算作命中。只有在没有 chunk
    标注时，才退回 source 级评测。
    """
    source = str(item.get("source") or "")
    chunk_id = str(item.get("chunk_id") or "")

    if sample.relevant_chunk_ids:
        for c in sample.relevant_chunk_ids:
            if _chunk_matches(chunk_id, c, sample.agent):
                return f"chunk::{c}"
        return None

    if sample.relevant_sources:
        for s in sample.relevant_sources:
            if _source_matches(source, s, sample.agent):
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
    seen_keys: Set[str] = set()
    rels = []
    for r in top:
        key = _unique_relevance_key(r, sample)
        if key and key not in seen_keys:
            seen_keys.add(key)
            rels.append(1)
        else:
            rels.append(0)
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


def _first_rank_where(results: List[Dict[str, object]], predicate) -> int:
    for idx, item in enumerate(results, start=1):
        if predicate(item):
            return idx
    return 0


def _relaxed_stage_metrics(
    results: List[Dict[str, object]],
    sample: EvalSample,
    ks: List[int],
) -> Dict[str, Optional[float]]:
    strict_first = _first_relevant_rank(results, sample) or 0
    source_first = _first_rank_where(results, lambda item: _source_hit(item, sample))
    page_first = (
        _first_rank_where(results, lambda item: _page_hit(item, sample))
        if sample.page_ranges
        else 0
    )
    same_page_or_adjacent_first = _first_rank_where(
        results,
        lambda item: _same_page_or_adjacent_chunk_hit(item, sample),
    )
    row: Dict[str, Optional[float]] = {
        "first_source_rank": float(source_first),
        "first_page_rank": float(page_first),
        "first_same_page_or_adjacent_rank": float(same_page_or_adjacent_first),
        "page_eligible": 1.0 if sample.page_ranges else 0.0,
    }
    for k in ks:
        strict_hit = 1 if strict_first and strict_first <= k else 0
        source_hit = 1 if source_first and source_first <= k else 0
        page_hit = 1 if page_first and page_first <= k else 0
        same_page_or_adjacent_hit = (
            1 if same_page_or_adjacent_first and same_page_or_adjacent_first <= k else 0
        )
        row[f"source_hit@{k}"] = float(source_hit)
        row[f"page_hit@{k}"] = float(page_hit) if sample.page_ranges else None
        row[f"same_source_near_miss@{k}"] = float(1 if source_hit and not strict_hit else 0)
        row[f"same_page_or_adjacent_chunk_hit@{k}"] = float(same_page_or_adjacent_hit)
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


def _average_optional(rows: List[Dict[str, Optional[float]]], key: str) -> Optional[float]:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return round(sum(float(value) for value in values) / len(values), 4)


def _summarize_relaxed_stage(
    per_sample_rows: List[Dict[str, Optional[float]]],
    ks: List[int],
) -> Dict[str, object]:
    page_eligible_count = int(sum(float(row.get("page_eligible") or 0.0) for row in per_sample_rows))
    return {
        "sample_count": len(per_sample_rows),
        "page_eligible_count": page_eligible_count,
        "source_hit_rate": {
            f"source_hit@{k}": _average_optional(per_sample_rows, f"source_hit@{k}") or 0.0
            for k in ks
        },
        "page_hit_rate": {
            f"page_hit@{k}": _average_optional(per_sample_rows, f"page_hit@{k}")
            for k in ks
        },
        "same_source_near_miss_rate": {
            f"same_source_near_miss@{k}": (
                _average_optional(per_sample_rows, f"same_source_near_miss@{k}") or 0.0
            )
            for k in ks
        },
        "same_page_or_adjacent_chunk_hit_rate": {
            f"same_page_or_adjacent_chunk_hit@{k}": (
                _average_optional(per_sample_rows, f"same_page_or_adjacent_chunk_hit@{k}") or 0.0
            )
            for k in ks
        },
    }


def _summarize_groups(
    stage1_groups: Dict[str, List[Dict[str, float]]],
    stage2_groups: Dict[str, List[Dict[str, float]]],
    stage1_ks: List[int],
    stage2_ks: List[int],
) -> Dict[str, object]:
    out = {}
    for group in sorted(set(stage1_groups.keys()) | set(stage2_groups.keys())):
        out[group] = {
            "sample_count": len(stage1_groups.get(group, [])),
            "embedding_stage": _summarize_stage(stage1_groups.get(group, []), stage1_ks),
            "rerank_stage": _summarize_stage(stage2_groups.get(group, []), stage2_ks),
        }
    return out


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


def _kb_inventory(kb) -> Dict[str, Set[str]]:
    chunks = list(getattr(kb, "chunks", []) or [])
    if not chunks:
        docs = kb._read_documents()
        for d in docs:
            for i, (_text, page_range) in enumerate(kb._split_text(d["text"]), start=1):
                if page_range:
                    chunk_id = f"{d['source']}#p{page_range}-chunk-{i}"
                else:
                    chunk_id = f"{d['source']}#chunk-{i}"
                chunks.append(
                    {
                        "source": d["source"],
                        "chunk_id": chunk_id,
                    }
                )

    sources = set()
    chunk_ids = set()
    for chunk in chunks:
        if isinstance(chunk, dict):
            sources.add(str(chunk.get("source") or ""))
            chunk_ids.add(str(chunk.get("chunk_id") or ""))
        else:
            sources.add(str(chunk.source))
            chunk_ids.add(str(chunk.chunk_id))

    return {
        "sources": {s for s in sources if s},
        "chunk_ids": {c for c in chunk_ids if c},
    }


def validate_dataset_against_kbs(
    samples: List[EvalSample],
    agent_kbs: Dict[str, object],
) -> Dict[str, object]:
    """Fail fast when eval ground truth no longer points at the current KB.

    The RAG evaluator judges retrieval results against explicit source/chunk IDs.
    After knowledge_base files are renamed, split differently, or moved between
    experts, stale IDs otherwise look like retrieval failures and silently drag
    metrics toward zero. This validation keeps architecture drift visible.
    """
    kb_inventory: Dict[str, Dict[str, Set[str]]] = {}
    for agent, kb in agent_kbs.items():
        kb_inventory[agent] = _kb_inventory(kb)

    unknown_sources = []
    unknown_chunk_ids = []
    for sample in samples:
        inventory = kb_inventory.get(sample.agent)
        if inventory is None:
            continue
        for source in sorted(sample.relevant_sources):
            if not any(
                _source_matches(existing, source, sample.agent)
                for existing in inventory["sources"]
            ):
                unknown_sources.append(
                    {
                        "line": sample.line_no,
                        "agent": sample.agent,
                        "source": source,
                    }
                )
        for chunk_id in sorted(sample.relevant_chunk_ids):
            if not any(
                _chunk_matches(existing, chunk_id, sample.agent)
                for existing in inventory["chunk_ids"]
            ):
                unknown_chunk_ids.append(
                    {
                        "line": sample.line_no,
                        "agent": sample.agent,
                        "chunk_id": chunk_id,
                    }
                )

    return {
        "unknown_sources": unknown_sources,
        "unknown_chunk_ids": unknown_chunk_ids,
        "checked_agents": sorted(kb_inventory.keys()),
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
    # - embedding/section 扩展阶段看候选池的覆盖能力，关注较大的 k
    # - rerank 阶段看头部的精度，关注较小的 k
    max_stage1_k = max(stage1_ks + [stage1_pool])
    max_stage2_k = max(stage2_ks)

    stage1_rows_all: List[Dict[str, float]] = []
    stage2_rows_all: List[Dict[str, float]] = []
    stage1_relaxed_rows_all: List[Dict[str, Optional[float]]] = []
    stage2_relaxed_rows_all: List[Dict[str, Optional[float]]] = []

    per_agent_stage1: Dict[str, List[Dict[str, float]]] = {}
    per_agent_stage2: Dict[str, List[Dict[str, float]]] = {}
    per_agent_stage1_relaxed: Dict[str, List[Dict[str, Optional[float]]]] = {}
    per_agent_stage2_relaxed: Dict[str, List[Dict[str, Optional[float]]]] = {}
    by_question_type_stage1: Dict[str, List[Dict[str, float]]] = {}
    by_question_type_stage2: Dict[str, List[Dict[str, float]]] = {}
    by_difficulty_stage1: Dict[str, List[Dict[str, float]]] = {}
    by_difficulty_stage2: Dict[str, List[Dict[str, float]]] = {}
    by_sample_kind_stage1: Dict[str, List[Dict[str, float]]] = {}
    by_sample_kind_stage2: Dict[str, List[Dict[str, float]]] = {}

    details = []

    for sample in samples:
        kb = agent_kbs.get(sample.agent)
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
        s1_relaxed = _relaxed_stage_metrics(stage1, sample, stage1_ks)
        s2_relaxed = _relaxed_stage_metrics(stage2, sample, stage2_ks)

        stage1_rows_all.append(s1_row)
        stage2_rows_all.append(s2_row)
        stage1_relaxed_rows_all.append(s1_relaxed)
        stage2_relaxed_rows_all.append(s2_relaxed)
        per_agent_stage1.setdefault(sample.agent, []).append(s1_row)
        per_agent_stage2.setdefault(sample.agent, []).append(s2_row)
        per_agent_stage1_relaxed.setdefault(sample.agent, []).append(s1_relaxed)
        per_agent_stage2_relaxed.setdefault(sample.agent, []).append(s2_relaxed)
        if sample.question_type:
            by_question_type_stage1.setdefault(sample.question_type, []).append(s1_row)
            by_question_type_stage2.setdefault(sample.question_type, []).append(s2_row)
        if sample.difficulty:
            by_difficulty_stage1.setdefault(sample.difficulty, []).append(s1_row)
            by_difficulty_stage2.setdefault(sample.difficulty, []).append(s2_row)
        if sample.sample_kind:
            by_sample_kind_stage1.setdefault(sample.sample_kind, []).append(s1_row)
            by_sample_kind_stage2.setdefault(sample.sample_kind, []).append(s2_row)

        detail = {
            "query": sample.query,
            "agent": sample.agent,
            "stage1_first_rank": s1_row["first_relevant_rank"],
            "stage2_first_rank": s2_row["first_relevant_rank"],
            "stage1": {k: round(v, 4) for k, v in s1_row.items()},
            "stage2": {k: round(v, 4) for k, v in s2_row.items()},
            "stage1_relaxed": {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in s1_relaxed.items()
            },
            "stage2_relaxed": {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in s2_relaxed.items()
            },
            "stage1_top": [
                {
                    "rank": r.get("rank"),
                    "source": r.get("source"),
                    "chunk_id": r.get("chunk_id"),
                    "page_range": r.get("page_range"),
                    "section_path": r.get("section_path"),
                    "context_chunk_ids": r.get("context_chunk_ids"),
                    "context_page_range": r.get("context_page_range"),
                    "dense_score": r.get("dense_score"),
                    "parent_section_score": r.get("parent_section_score"),
                }
                for r in stage1[:max_stage1_k]
            ],
            "stage2_top": [
                {
                    "rank": r.get("rank"),
                    "source": r.get("source"),
                    "chunk_id": r.get("chunk_id"),
                    "page_range": r.get("page_range"),
                    "section_path": r.get("section_path"),
                    "context_chunk_ids": r.get("context_chunk_ids"),
                    "context_page_range": r.get("context_page_range"),
                    "dense_score": r.get("dense_score"),
                    "rerank_score": r.get("rerank_score"),
                    "parent_section_score": r.get("parent_section_score"),
                }
                for r in stage2[:max_stage2_k]
            ],
        }
        if sample.question_type:
            detail["question_type"] = sample.question_type
        if sample.difficulty:
            detail["difficulty"] = sample.difficulty
        if sample.sample_kind:
            detail["sample_kind"] = sample.sample_kind
        if sample.page_ranges:
            detail["page_ranges"] = sorted(sample.page_ranges)
        if sample.answer:
            detail["answer"] = sample.answer
        if sample.supporting_span:
            detail["supporting_span"] = sample.supporting_span
        if sample.quality:
            detail["quality"] = sample.quality
        details.append(detail)

    stage1_summary = _summarize_stage(stage1_rows_all, stage1_ks)
    stage2_summary = _summarize_stage(stage2_rows_all, stage2_ks)
    stage1_relaxed_summary = _summarize_relaxed_stage(stage1_relaxed_rows_all, stage1_ks)
    stage2_relaxed_summary = _summarize_relaxed_stage(stage2_relaxed_rows_all, stage2_ks)

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
            "embedding_stage_relaxed": _summarize_relaxed_stage(
                per_agent_stage1_relaxed.get(agent, []),
                stage1_ks,
            ),
            "rerank_stage_relaxed": _summarize_relaxed_stage(
                per_agent_stage2_relaxed.get(agent, []),
                stage2_ks,
            ),
        }

    metadata_summary = {}
    if by_question_type_stage1:
        metadata_summary["by_question_type"] = _summarize_groups(
            by_question_type_stage1,
            by_question_type_stage2,
            stage1_ks,
            stage2_ks,
        )
    if by_difficulty_stage1:
        metadata_summary["by_difficulty"] = _summarize_groups(
            by_difficulty_stage1,
            by_difficulty_stage2,
            stage1_ks,
            stage2_ks,
        )
    if by_sample_kind_stage1:
        metadata_summary["by_sample_kind"] = _summarize_groups(
            by_sample_kind_stage1,
            by_sample_kind_stage2,
            stage1_ks,
            stage2_ks,
        )

    return {
        "config": {
            "sample_count": len(samples),
            "stage1_ks": stage1_ks,
            "stage2_ks": stage2_ks,
            "stage1_pool": stage1_pool,
        },
        "embedding_stage": stage1_summary,
        "rerank_stage": stage2_summary,
        "embedding_stage_relaxed": stage1_relaxed_summary,
        "rerank_stage_relaxed": stage2_relaxed_summary,
        "rerank_uplift_vs_embedding": uplift,
        "per_agent_summary": per_agent_summary,
        "metadata_summary": metadata_summary,
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


def _timestamped_report_path(
    path: Path,
    timestamp: str,
    default_stem: str,
    raw_path: str = "",
) -> Path:
    raw = raw_path or str(path)
    if raw.endswith(("/", "\\")) or (path.exists() and path.is_dir()):
        candidate = path / f"{default_stem}_{timestamp}.json"
    elif path.suffix:
        candidate = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
    else:
        candidate = path.with_name(f"{path.name}_{timestamp}.json")

    if not candidate.exists():
        return candidate
    for idx in range(2, 1000):
        deduped = candidate.with_name(f"{candidate.stem}-{idx}{candidate.suffix}")
        if not deduped.exists():
            return deduped
    raise FileExistsError(f"Could not find a free report path near {candidate}")


def _switch_to_bool(raw: str) -> Optional[bool]:
    if raw == "auto":
        return None
    return raw == "on"


def _apply_rag_runtime_overrides(args, rag_config, rag_module) -> None:
    """Apply CLI ablation flags to config and rag module globals in this process."""

    bool_overrides = {
        "RAG_HYBRID_RETRIEVAL_ENABLED": _switch_to_bool(args.hybrid_retrieval),
        "RAG_PDF_FINE_CHUNKING_ENABLED": _switch_to_bool(args.pdf_fine_chunking),
        "RAG_PDF_PARENT_EXPANSION_ENABLED": _switch_to_bool(args.pdf_parent_expansion),
        "RAG_PDF_PARENT_RESCUE_ENABLED": _switch_to_bool(args.pdf_parent_rescue),
        "RAG_PDF_PARENT_RERANK_CONTEXT_ENABLED": _switch_to_bool(
            args.pdf_parent_rerank_context
        ),
        "RAG_PDF_PARENT_SCORE_FUSION_ENABLED": _switch_to_bool(
            args.pdf_parent_score_fusion
        ),
    }
    for name, value in bool_overrides.items():
        if value is None:
            continue
        setattr(rag_config, name, value)
        setattr(rag_module, name, value)

    numeric_overrides = {
        "RAG_BM25_TOP_K": args.bm25_top_k,
        "RAG_BM25_SCORE_WEIGHT": args.bm25_score_weight,
        "RAG_PDF_FINE_CHUNK_MAX_CHARS": args.pdf_fine_chunk_max_chars,
        "RAG_PDF_SECTION_SCORE_WEIGHT": args.pdf_section_score_weight,
        "RAG_PDF_PARENT_RESCUE_LOOKAHEAD": args.pdf_parent_rescue_lookahead,
        "RAG_PDF_PARENT_RESCUE_MIN_PDF_CANDIDATES": (
            args.pdf_parent_rescue_min_pdf_candidates
        ),
        "RAG_PDF_PARENT_RESCUE_MIN_PARENT_SCORE": (
            args.pdf_parent_rescue_min_parent_score
        ),
        "RAG_RERANK_POOL_MULTIPLIER": args.rag_rerank_pool_multiplier,
        "RAG_RERANK_POOL_MAX": args.rag_rerank_pool_max,
        "RAG_PDF_RERANK_PARENT_CONTEXT_CHARS": args.pdf_rerank_parent_context_chars,
    }
    for name, value in numeric_overrides.items():
        if value is None:
            continue
        setattr(rag_config, name, value)
        setattr(rag_module, name, value)


def _rag_runtime_config(rag_module) -> Dict[str, object]:
    return {
        "hybrid_retrieval_enabled": rag_module.RAG_HYBRID_RETRIEVAL_ENABLED,
        "bm25_top_k": rag_module.RAG_BM25_TOP_K,
        "bm25_score_weight": rag_module.RAG_BM25_SCORE_WEIGHT,
        "rag_rerank_pool_multiplier": rag_module.RAG_RERANK_POOL_MULTIPLIER,
        "rag_rerank_pool_max": rag_module.RAG_RERANK_POOL_MAX,
        "pdf_fine_chunking_enabled": rag_module.RAG_PDF_FINE_CHUNKING_ENABLED,
        "pdf_fine_chunk_max_chars": rag_module.RAG_PDF_FINE_CHUNK_MAX_CHARS,
        "pdf_parent_expansion_enabled": rag_module.RAG_PDF_PARENT_EXPANSION_ENABLED,
        "pdf_parent_rescue_enabled": rag_module.RAG_PDF_PARENT_RESCUE_ENABLED,
        "pdf_parent_rescue_lookahead": rag_module.RAG_PDF_PARENT_RESCUE_LOOKAHEAD,
        "pdf_parent_rescue_min_pdf_candidates": (
            rag_module.RAG_PDF_PARENT_RESCUE_MIN_PDF_CANDIDATES
        ),
        "pdf_parent_rescue_min_parent_score": (
            rag_module.RAG_PDF_PARENT_RESCUE_MIN_PARENT_SCORE
        ),
        "pdf_parent_rerank_context_enabled": (
            rag_module.RAG_PDF_PARENT_RERANK_CONTEXT_ENABLED
        ),
        "pdf_parent_score_fusion_enabled": rag_module.RAG_PDF_PARENT_SCORE_FUSION_ENABLED,
        "pdf_section_score_weight": rag_module.RAG_PDF_SECTION_SCORE_WEIGHT,
        "pdf_rerank_parent_context_chars": rag_module.RAG_PDF_RERANK_PARENT_CONTEXT_CHARS,
        "pdf_neighbor_chunks": rag_module.RAG_PDF_NEIGHBOR_CHUNKS,
    }


def _pretty_print_summary(report: Dict[str, object]) -> None:
    cfg = report["config"]
    print("=" * 72)
    print(f"[RAG Eval] samples = {cfg['sample_count']}")
    print(f"[RAG Eval] agents = {cfg.get('agents', [])}")
    print(f"[RAG Eval] stage1 ks = {cfg['stage1_ks']} | stage1_pool = {cfg['stage1_pool']}")
    print(f"[RAG Eval] stage2 ks = {cfg['stage2_ks']}")
    print("-" * 72)
    print("[Embedding Stage  (dense retrieve + PDF section expansion)]")
    print(json.dumps(report["embedding_stage"], ensure_ascii=False, indent=2))
    print("-" * 72)
    print("[Rerank Stage  (cross-encoder on dense top-N)]")
    print(json.dumps(report["rerank_stage"], ensure_ascii=False, indent=2))
    print("-" * 72)
    if report.get("embedding_stage_relaxed") or report.get("rerank_stage_relaxed"):
        print("[Relaxed PDF/Long-doc Metrics]")
        print(
            json.dumps(
                {
                    "embedding_stage_relaxed": report.get("embedding_stage_relaxed"),
                    "rerank_stage_relaxed": report.get("rerank_stage_relaxed"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print("-" * 72)
    if report["rerank_uplift_vs_embedding"]:
        print("[Uplift: rerank - embedding  (positive = rerank helps)]")
        print(json.dumps(report["rerank_uplift_vs_embedding"], ensure_ascii=False, indent=2))
        print("-" * 72)
    if report.get("metadata_summary"):
        print("[Metadata Summary]")
        for name, summary in report["metadata_summary"].items():
            counts = {
                group: data.get("sample_count", 0)
                for group, data in summary.items()
            }
            print(f"{name}: {counts}")
    print("=" * 72)


def main():
    from pathlib import Path as _Path
    from health_guide import config as rag_config
    from health_guide.config import KNOWLEDGE_BASE_DIR, KNOWLEDGE_BASE_AGENT_SUBDIRS
    import health_guide.rag as rag_module
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
        help=(
            "Base path for output evaluation report JSON. A timestamp is inserted "
            "before the suffix to avoid overwriting previous reports."
        ),
    )
    parser.add_argument(
        "--report-label",
        default="",
        help="Optional label appended to report_title for ablation runs.",
    )
    parser.add_argument(
        "--hybrid-retrieval",
        choices=["auto", "on", "off"],
        default="auto",
        help="Override dense+BM25 hybrid retrieval for this run.",
    )
    parser.add_argument(
        "--bm25-top-k",
        type=int,
        default=None,
        help="Override lexical BM25 candidate count for this run.",
    )
    parser.add_argument(
        "--bm25-score-weight",
        type=float,
        default=None,
        help="Override normalized BM25 score weight for candidate fusion.",
    )
    parser.add_argument(
        "--pdf-fine-chunking",
        choices=["auto", "on", "off"],
        default="auto",
        help="Override finer PDF child chunking for this run.",
    )
    parser.add_argument(
        "--pdf-fine-chunk-max-chars",
        type=int,
        default=None,
        help="Override PDF fine chunk max chars for this run.",
    )
    parser.add_argument(
        "--pdf-parent-expansion",
        choices=["auto", "on", "off"],
        default="auto",
        help="Override PDF section parent-child candidate expansion for this run.",
    )
    parser.add_argument(
        "--pdf-parent-rescue",
        choices=["auto", "on", "off"],
        default="auto",
        help="Override gated PDF parent rescue for this run.",
    )
    parser.add_argument(
        "--pdf-parent-rescue-lookahead",
        type=int,
        default=None,
        help="Override how many current candidates the parent rescue gate inspects.",
    )
    parser.add_argument(
        "--pdf-parent-rescue-min-pdf-candidates",
        type=int,
        default=None,
        help="Override minimum PDF candidates before parent rescue is skipped.",
    )
    parser.add_argument(
        "--pdf-parent-rescue-min-parent-score",
        type=float,
        default=None,
        help="Override minimum section-parent score required by gated rescue.",
    )
    parser.add_argument(
        "--pdf-parent-rerank-context",
        choices=["auto", "on", "off"],
        default="auto",
        help="Override injecting parent excerpt into reranker input for this run.",
    )
    parser.add_argument(
        "--pdf-parent-score-fusion",
        choices=["auto", "on", "off"],
        default="auto",
        help="Override parent score fusion in candidate/final scoring for this run.",
    )
    parser.add_argument(
        "--pdf-section-score-weight",
        type=float,
        default=None,
        help="Override parent score fusion weight for this run.",
    )
    parser.add_argument(
        "--pdf-rerank-parent-context-chars",
        type=int,
        default=None,
        help="Override max parent excerpt chars injected into reranker input.",
    )
    parser.add_argument(
        "--rag-rerank-pool-multiplier",
        type=int,
        default=None,
        help="Override rerank pool multiplier for this run.",
    )
    parser.add_argument(
        "--rag-rerank-pool-max",
        type=int,
        default=None,
        help="Override rerank pool cap for this run.",
    )
    parser.add_argument(
        "--skip-ground-truth-validation",
        action="store_true",
        help=(
            "Skip checking whether relevant_sources/relevant_chunk_ids exist in the "
            "current KB. Useful only for exploratory/stale dataset debugging."
        ),
    )
    args = parser.parse_args()
    _apply_rag_runtime_overrides(args, rag_config, rag_module)

    stage1_ks = parse_ks(args.stage1_ks)
    stage2_ks = parse_ks(args.stage2_ks)
    dataset = load_dataset(Path(args.dataset))
    dataset_agents = sorted({sample.agent for sample in dataset})
    configured_agents = set(KNOWLEDGE_BASE_AGENT_SUBDIRS.keys())
    unknown_agents = sorted(set(dataset_agents) - configured_agents)
    if unknown_agents:
        raise ValueError(
            "Dataset references agent(s) not configured in "
            f"KNOWLEDGE_BASE_AGENT_SUBDIRS: {unknown_agents}. "
            f"Configured agents: {sorted(configured_agents)}"
        )

    kb_root = _Path(args.kb_dir)
    agent_kbs = {}
    for agent in dataset_agents:
        subdir = KNOWLEDGE_BASE_AGENT_SUBDIRS[agent]
        kb_path = kb_root / subdir
        if not kb_path.exists():
            raise FileNotFoundError(
                f"Knowledge base directory for agent '{agent}' does not exist: {kb_path}"
            )
        kb = LocalKnowledgeBase(
            kb_dir=str(kb_path),
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            boundary_look_back=args.boundary_look_back,
            min_chunk_chars=args.min_chunk_chars,
        )
        agent_kbs[agent] = kb

    validation = validate_dataset_against_kbs(dataset, agent_kbs)
    if not args.skip_ground_truth_validation and (
        validation["unknown_sources"] or validation["unknown_chunk_ids"]
    ):
        preview = {
            "unknown_sources": validation["unknown_sources"][:10],
            "unknown_chunk_ids": validation["unknown_chunk_ids"][:10],
        }
        raise ValueError(
            "Dataset ground truth does not match the current knowledge_base. "
            "Regenerate or update the dataset, or rerun with "
            "--skip-ground-truth-validation to inspect retrieval anyway. "
            f"Preview: {json.dumps(preview, ensure_ascii=False)}"
        )

    for kb in agent_kbs.values():
        kb.build(force_rebuild=False)

    report = evaluate(
        samples=dataset,
        agent_kbs=agent_kbs,
        stage1_ks=stage1_ks,
        stage2_ks=stage2_ks,
        stage1_pool=args.stage1_pool,
    )
    report["config"]["agents"] = dataset_agents
    report["config"]["rag_runtime"] = _rag_runtime_config(rag_module)
    report["ground_truth_validation"] = validation

    run_dt = datetime.now().astimezone()
    timestamp = run_dt.strftime("%Y%m%d-%H%M%S")
    out = _timestamped_report_path(Path(args.out), timestamp, "rag_eval_report", args.out)
    label = args.report_label.strip()
    report["report_title"] = (
        f"RAG Eval Report {label} {timestamp}" if label else f"RAG Eval Report {timestamp}"
    )
    report["generated_at"] = run_dt.isoformat(timespec="seconds")
    report["requested_out"] = args.out
    report["report_path"] = str(out)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _pretty_print_summary(report)
    print(f"[RAG Eval] Report written to: {out}")


if __name__ == "__main__":
    main()
