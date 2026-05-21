"""用 LLM 自动生成 RAG 评测集，并尽量降低“贴脸样本”的污染。

核心思路:
对知识库里的高质量 chunk,让 LLM 反向生成“中文用户可能会怎么问”
的问题；每条样本同时带 answer / supporting_span / question_type /
difficulty，方便后续人工抽检、分层评测和可回答性验证。

注意: LLM 反向生成的 `(query, chunk_id)` 不是天然正确的真值。脚本会
在生成前过滤低质 chunk，在生成时约束问题形态，在生成后做字面泄漏、
supporting_span 回溯、去重和可选 LLM 二次验证。

这个脚本**对知识库结构没有任何硬编码**:
- 遍历 `KNOWLEDGE_BASE_AGENT_SUBDIRS` 中声明的各 agent 私有目录;
- 对每一个 .md / .txt / .pdf 文件按和生产管线完全一致的方式切分 chunk;
- 默认每个源文件先最多抽 10 个 chunk,避免长 PDF/书籍在生成池里压倒短文档;
- 默认输出带 namespace 的 ground truth,例如
  `nutritionist:who_healthy_diet.md#chunk-1` 和 `nutritionist/who_healthy_diet.md`;
- `evaluate_rag.py` 会把 namespace 形式规范化后匹配运行时返回的 per-agent chunk_id。

用法:

    python scripts/generate_eval_dataset.py \\
        --max-chunks 20 \\
        --dry-run

    python scripts/generate_eval_dataset.py \\
        --max-chunks 50 \\
        --questions-per-chunk 4 \\
        --fast \\
        --out eval/rag_eval_dataset_generated.quick.jsonl

典型迭代节奏:
- 第一次先跑 `--max-chunks 10 --dry-run` 看 chunk 过滤分布;
- 然后用 `--fast` 生成一小批样本,人工抽查 10 到 20 条 query / answer / supporting_span;
- 确认样本形态后,再按需去掉 `--fast` 打开二次 LLM 质检;
- 再跑 `python scripts/evaluate_rag.py --dataset eval/rag_eval_dataset_generated.jsonl`
  得到 Recall@k / MRR / nDCG / MAP 报告，以及按 question_type / difficulty
  分层的指标。
- 如果要和 `evaluate_rag.py` 默认评测完全对齐,不要改 `--chunk-size`、
  `--overlap`、`--boundary-look-back`、`--min-chunk-chars`。
"""

import argparse
import math
import json
import random
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Chunk collection (no embedding models needed)
# --------------------------------------------------------------------------- #


def _chunks_from_store(store, agent: str) -> List[Dict]:
    """调用 LocalKnowledgeBase 的文档读取 + 切分,但**不触发 embedding 加载**。

    生产代码里 `build()` 会尝试 lazy-load embed 模型,脚本这里完全不需要向量,
    所以直接复用 `_read_documents` / `_split_text` 两个纯 IO/字符串的私有方法。
    这样连 torch 都不用 import,生成脚本在一台没 GPU 的机器上也能跑。
    """
    docs = store._read_documents()
    collected: List[Dict] = []
    for d in docs:
        pieces = store._split_text(d["text"])
        for i, (text, page_range) in enumerate(pieces):
            if page_range:
                chunk_id = f"{d['source']}#p{page_range}-chunk-{i+1}"
            else:
                chunk_id = f"{d['source']}#chunk-{i+1}"
            collected.append(
                {
                    "agent": agent,
                    "chunk_id": chunk_id,
                    "source": d["source"],
                    "text": text,
                    "page_range": page_range,
                }
            )
    return collected


def collect_all_chunks(
    kb_root: Path,
    chunk_size: int,
    overlap: int,
    boundary_look_back: int,
    min_chunk_chars: int,
    agents: Optional[List[str]] = None,
) -> List[Dict]:
    """遍历所有 agent 私有知识库, 返回统一结构的 chunk 列表。"""
    from wellness_copilot.config import KNOWLEDGE_BASE_AGENT_SUBDIRS
    from wellness_copilot.rag import LocalKnowledgeBase

    kb_root = Path(kb_root)
    all_chunks: List[Dict] = []
    configured_agents = set(KNOWLEDGE_BASE_AGENT_SUBDIRS.keys())
    selected_agents = agents or list(KNOWLEDGE_BASE_AGENT_SUBDIRS.keys())
    unknown_agents = sorted(set(selected_agents) - configured_agents)
    if unknown_agents:
        raise ValueError(
            "Unknown agent namespace(s): "
            f"{unknown_agents}. Configured: {sorted(configured_agents)}"
        )

    # Agent-specific namespaces(动态读取 config, 不硬编码 agent 名)
    for agent in selected_agents:
        subdir = KNOWLEDGE_BASE_AGENT_SUBDIRS[agent]
        agent_path = kb_root / subdir
        if not agent_path.exists():
            print(f"[Generate][warn] 跳过不存在的知识库目录: {agent_path}")
            continue
        agent_store = LocalKnowledgeBase(
            kb_dir=str(agent_path),
            chunk_size=chunk_size,
            overlap=overlap,
            recursive=True,
            boundary_look_back=boundary_look_back,
            min_chunk_chars=min_chunk_chars,
        )
        all_chunks.extend(_chunks_from_store(agent_store, agent=agent))

    return all_chunks


def _parse_agents(raw_agents: Optional[List[str]]) -> Optional[List[str]]:
    if not raw_agents:
        return None
    agents: List[str] = []
    seen = set()
    for raw in raw_agents:
        for item in raw.split(","):
            agent = item.strip().lower()
            if agent and agent not in seen:
                seen.add(agent)
                agents.append(agent)
    return agents or None


def _chunk_stats(chunks: List[Dict]) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for chunk in chunks:
        agent = str(chunk.get("agent") or "")
        stats[agent] = stats.get(agent, 0) + 1
    return dict(sorted(stats.items()))


def _cap_chunks_per_file(
    chunks: List[Dict],
    max_chunks_per_file: int,
    seed: int,
) -> Tuple[List[Dict], Dict[str, object]]:
    """Pre-sample each source file before global sampling.

    Long PDFs can produce hundreds of chunks. If they enter the global pool as-is,
    they dominate both LLM cost and eval distribution. This cap is applied before
    quality filtering and before --max-chunks so each source file gets a bounded
    chance to contribute.
    """
    if max_chunks_per_file <= 0:
        return chunks, {
            "enabled": False,
            "file_count": 0,
            "capped_file_count": 0,
            "dropped_chunk_count": 0,
            "max_chunks_per_file": max_chunks_per_file,
        }

    by_file: Dict[Tuple[str, str], List[Dict]] = {}
    for chunk in chunks:
        key = (str(chunk.get("agent") or ""), str(chunk.get("source") or ""))
        by_file.setdefault(key, []).append(chunk)

    rng = random.Random(seed)
    selected_ids = set()
    capped_file_count = 0
    dropped_chunk_count = 0
    capped_preview = []

    for key, group in sorted(by_file.items()):
        if len(group) > max_chunks_per_file:
            capped_file_count += 1
            dropped_chunk_count += len(group) - max_chunks_per_file
            sampled = rng.sample(group, max_chunks_per_file)
            selected_ids.update(id(chunk) for chunk in sampled)
            if len(capped_preview) < 8:
                capped_preview.append(
                    {
                        "agent": key[0],
                        "source": key[1],
                        "before": len(group),
                        "after": max_chunks_per_file,
                    }
                )
        else:
            selected_ids.update(id(chunk) for chunk in group)

    capped = [chunk for chunk in chunks if id(chunk) in selected_ids]
    return capped, {
        "enabled": True,
        "file_count": len(by_file),
        "capped_file_count": capped_file_count,
        "dropped_chunk_count": dropped_chunk_count,
        "max_chunks_per_file": max_chunks_per_file,
        "capped_preview": capped_preview,
    }


def _sample_chunks(chunks: List[Dict], max_chunks: int, seed: int) -> List[Dict]:
    """Stratified-ish sampling so small namespaces are not drowned out."""
    if not max_chunks or max_chunks >= len(chunks):
        return chunks

    rng = random.Random(seed)
    by_agent: Dict[str, List[Dict]] = {}
    for chunk in chunks:
        by_agent.setdefault(chunk["agent"], []).append(chunk)

    for group in by_agent.values():
        rng.shuffle(group)

    selected: List[Dict] = []
    active_agents = sorted(by_agent.keys())
    while len(selected) < max_chunks and active_agents:
        next_active = []
        for agent in active_agents:
            group = by_agent[agent]
            if group and len(selected) < max_chunks:
                selected.append(group.pop())
            if group:
                next_active.append(agent)
        active_agents = next_active

    rng.shuffle(selected)
    return selected


# --------------------------------------------------------------------------- #
# Quality helpers
# --------------------------------------------------------------------------- #


@dataclass
class GeneratedSample:
    query: str
    answer: str = ""
    supporting_span: str = ""
    question_type: str = "factual"
    difficulty: str = "medium"


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_+-]{1,}")
_SENTENCE_END_RE = re.compile(r"[。！？!?]|[.](?=\s|$)")
_META_QUESTION_RE = re.compile(r"(这段|文本|文中|上述|上面|原文|资料|片段|知识库)")
_LEADING_CONTEXT_PRONOUN_RE = re.compile(r"^\s*(它|其|这个|这种|这些|上述|上面|该)")
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").lower())


def _semantic_chars(text: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", str(text or "").lower()))


def _char_ngrams(text: str, n: int) -> set:
    compact = _semantic_chars(text)
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def _longest_common_substring_ratio(needle: str, haystack: str) -> float:
    """Return LCS-substring length divided by the normalized needle length."""
    a = _semantic_chars(needle)
    b = _semantic_chars(haystack)
    if not a or not b:
        return 0.0

    prev = [0] * (len(b) + 1)
    best = 0
    for ca in a:
        curr = [0]
        for j, cb in enumerate(b, start=1):
            value = prev[j - 1] + 1 if ca == cb else 0
            if value > best:
                best = value
            curr.append(value)
        prev = curr
    return best / max(1, len(a))


def _literal_overlap_score(query: str, chunk_text: str) -> float:
    """Estimate whether the question is too literally copied from the chunk.

    The score combines character n-gram containment and longest copied span.
    It is deliberately language-light so it works for Chinese queries against
    Chinese or English source text without adding tokenizer dependencies.
    """
    normalized_query = _semantic_chars(query)
    if not normalized_query:
        return 1.0
    n = 3 if len(normalized_query) < 16 else 4
    q_grams = _char_ngrams(query, n)
    c_grams = _char_ngrams(chunk_text, n)
    containment = len(q_grams & c_grams) / max(1, len(q_grams))
    copied_span = _longest_common_substring_ratio(query, chunk_text)
    return round(max(containment, copied_span), 4)


def _supporting_span_trace_score(span: str, chunk_text: str) -> float:
    span_norm = _semantic_chars(span)
    if not span_norm:
        return 0.0
    chunk_norm = _semantic_chars(chunk_text)
    if span_norm in chunk_norm:
        return 1.0
    return round(_longest_common_substring_ratio(span, chunk_text), 4)


def _lexical_tokens(text: str) -> List[str]:
    words = [w.lower() for w in _WORD_RE.findall(text or "") if len(w) >= 2]
    cjk_chars = _CJK_RE.findall(text or "")
    cjk_bigrams = [
        "".join(cjk_chars[i : i + 2])
        for i in range(max(0, len(cjk_chars) - 1))
    ]
    cjk_trigrams = [
        "".join(cjk_chars[i : i + 3])
        for i in range(max(0, len(cjk_chars) - 2))
    ]
    return words + cjk_bigrams + cjk_trigrams


class CharNgramBM25:
    """Tiny dependency-free BM25 used only to flag overly lexical eval items."""

    def __init__(self, chunks: List[Dict]):
        self.keys: List[Tuple[str, str]] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_freq: Counter = Counter()
        for chunk in chunks:
            tokens = _lexical_tokens(chunk.get("text") or "")
            if not tokens:
                continue
            self.keys.append((str(chunk.get("agent") or ""), str(chunk.get("chunk_id") or "")))
            self.doc_tokens.append(tokens)
            self.doc_freq.update(set(tokens))
        self.n_docs = len(self.doc_tokens)
        self.avgdl = (
            sum(len(tokens) for tokens in self.doc_tokens) / self.n_docs
            if self.n_docs
            else 0.0
        )

    def _score_doc(self, query_counts: Counter, doc_tokens: List[str]) -> float:
        if not query_counts or not doc_tokens or not self.n_docs:
            return 0.0
        k1 = 1.5
        b = 0.75
        doc_counts = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        score = 0.0
        for token, qf in query_counts.items():
            tf = doc_counts.get(token, 0)
            if tf <= 0:
                continue
            df = self.doc_freq.get(token, 0)
            idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * doc_len / max(1.0, self.avgdl))
            score += idf * ((tf * (k1 + 1)) / denom) * min(1.0, 0.5 + 0.5 * qf)
        return score

    def target_stats(self, query: str, agent: str, chunk_id: str) -> Dict[str, object]:
        if not self.n_docs:
            return {"rank": 0, "score": 0.0, "top_score": 0.0, "score_ratio": 0.0}

        query_counts = Counter(_lexical_tokens(query))
        scores = [
            (self._score_doc(query_counts, tokens), key)
            for key, tokens in zip(self.keys, self.doc_tokens)
        ]
        scores.sort(key=lambda item: item[0], reverse=True)
        top_score = scores[0][0] if scores else 0.0
        second_score = scores[1][0] if len(scores) > 1 else 0.0
        target_key = (agent, chunk_id)
        rank = 0
        target_score = 0.0
        for idx, (score, key) in enumerate(scores, start=1):
            if key == target_key:
                rank = idx
                target_score = score
                break
        ratio = target_score / max(second_score, 1e-9) if rank == 1 else 0.0
        return {
            "rank": rank,
            "score": round(target_score, 4),
            "top_score": round(top_score, 4),
            "score_ratio": round(ratio, 4),
        }


def _chunk_quality_check(chunk: Dict, args) -> Tuple[bool, str, Dict[str, float]]:
    text = str(chunk.get("text") or "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    nonspace_len = len(re.sub(r"\s+", "", text))
    content_chars = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text))
    content_ratio = content_chars / max(1, nonspace_len)
    sentence_count = len(_SENTENCE_END_RE.findall(text))

    table_lines = [
        line
        for line in lines
        if line.startswith("|")
        or line.endswith("|")
        or line.count("|") >= 2
        or re.fullmatch(r"[-:| ]{6,}", line)
    ]
    table_ratio = len(table_lines) / max(1, len(lines))
    duplicate_ratio = 0.0
    if lines:
        duplicate_ratio = 1 - (len(set(lines)) / len(lines))
    url_ratio = len(re.findall(r"https?://|www\.", text, flags=re.I)) / max(1, len(lines))
    referenceish = bool(re.search(r"\b(references?|bibliography|contents)\b|参考文献|目录", text, re.I))

    metrics = {
        "chars": float(len(text)),
        "content_ratio": round(content_ratio, 4),
        "sentence_count": float(sentence_count),
        "table_line_ratio": round(table_ratio, 4),
        "duplicate_line_ratio": round(duplicate_ratio, 4),
        "url_ratio": round(url_ratio, 4),
    }

    if len(text) < args.min_chunk_len:
        return False, "too_short", metrics
    if not args.chunk_quality_filter:
        return True, "ok", metrics
    if table_ratio > args.max_table_line_ratio:
        return False, "mostly_table_or_rule_lines", metrics
    if content_ratio < args.min_content_char_ratio:
        return False, "low_content_char_ratio", metrics
    if duplicate_ratio > args.max_duplicate_line_ratio:
        return False, "duplicated_lines", metrics
    if referenceish and sentence_count <= 2:
        return False, "reference_or_toc_like", metrics
    if sentence_count == 0 and len(text) > 180:
        return False, "no_sentence_boundary", metrics
    if url_ratio > 0.35:
        return False, "mostly_urls", metrics
    return True, "ok", metrics


def _filter_chunks_for_generation(chunks: List[Dict], args) -> Tuple[List[Dict], Counter]:
    kept: List[Dict] = []
    reasons: Counter = Counter()
    for chunk in chunks:
        keep, reason, metrics = _chunk_quality_check(chunk, args)
        chunk["chunk_quality"] = metrics
        reasons[reason] += 1
        if keep:
            kept.append(chunk)
    return kept, reasons


def _query_is_bad_shape(query: str, args) -> Optional[str]:
    compact = _compact_text(query)
    if len(compact) < args.min_question_len:
        return "question_too_short"
    if len(compact) > args.max_question_len:
        return "question_too_long"
    if _META_QUESTION_RE.search(query):
        return "meta_question"
    if _LEADING_CONTEXT_PRONOUN_RE.search(query):
        return "context_dependent_pronoun"
    if query.count("?") + query.count("？") > 1:
        return "multi_question"
    return None


def _question_key(query: str) -> str:
    return _semantic_chars(query)


def _is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _verdict_by_index(verdicts: List[Dict[str, object]]) -> Dict[int, Dict[str, object]]:
    by_index: Dict[int, Dict[str, object]] = {}
    for verdict in verdicts:
        try:
            idx = int(verdict.get("index"))
        except (TypeError, ValueError):
            continue
        by_index[idx] = verdict
    return by_index


def _filter_generated_samples(
    candidates: List[GeneratedSample],
    chunk: Dict,
    args,
    bm25: CharNgramBM25,
    seen_questions: set,
    llm_verdicts: Optional[List[Dict[str, object]]] = None,
    apply_llm_verdicts: bool = True,
    mark_seen: bool = True,
    limit: Optional[int] = None,
) -> Tuple[List[Tuple[GeneratedSample, Dict[str, object]]], Counter]:
    accepted: List[Tuple[GeneratedSample, Dict[str, object]]] = []
    rejected: Counter = Counter()
    verdicts = _verdict_by_index(llm_verdicts or [])
    batch_seen = set()
    sample_limit = args.questions_per_chunk if limit is None else limit
    enforce_llm = bool(args.llm_verify and apply_llm_verdicts)

    for idx, sample in enumerate(candidates):
        query = sample.query.strip()
        quality: Dict[str, object] = {
            "literal_overlap": _literal_overlap_score(query, chunk["text"]),
            "supporting_span_trace": _supporting_span_trace_score(
                sample.supporting_span,
                chunk["text"],
            ),
        }
        bm25_stats = bm25.target_stats(query, chunk["agent"], chunk["chunk_id"])
        quality["bm25"] = bm25_stats

        reason = _query_is_bad_shape(query, args)
        if reason:
            rejected[reason] += 1
            continue

        key = _question_key(query)
        if key in seen_questions or key in batch_seen:
            rejected["duplicate_question"] += 1
            continue

        if quality["literal_overlap"] > args.max_literal_overlap:
            rejected["literal_overlap_high"] += 1
            continue

        if args.require_supporting_span and not sample.supporting_span:
            rejected["missing_supporting_span"] += 1
            continue

        if sample.supporting_span and quality["supporting_span_trace"] < args.min_support_trace:
            rejected["supporting_span_not_traceable"] += 1
            continue

        if (
            args.downsample_easy_bm25
            and bm25_stats["rank"] == 1
            and bm25_stats["score_ratio"] >= args.easy_bm25_score_ratio
            and quality["literal_overlap"] >= args.easy_bm25_min_overlap
        ):
            rejected["too_easy_bm25"] += 1
            continue

        verdict = verdicts.get(idx)
        if enforce_llm and verdict is not None:
            quality["llm_verdict"] = verdict
            if not _is_truthy(verdict.get("answerable")):
                rejected["llm_not_answerable"] += 1
                continue
            if not _is_truthy(verdict.get("supporting_span_present")):
                rejected["llm_support_missing"] += 1
                continue
            if str(verdict.get("leakage") or "").strip().lower() == "high":
                rejected["llm_leakage_high"] += 1
                continue
        elif enforce_llm and args.require_llm_verdict:
            quality["llm_verdict"] = {"reason": "missing_verdict"}
            rejected["llm_missing_verdict"] += 1
            continue
        elif enforce_llm:
            quality["llm_verdict"] = {"reason": "verification_unavailable_local_filters_only"}

        accepted.append((sample, quality))
        batch_seen.add(key)
        if mark_seen:
            seen_questions.add(key)

        if len(accepted) >= sample_limit:
            break

    return accepted, rejected


# --------------------------------------------------------------------------- #
# LLM question generation
# --------------------------------------------------------------------------- #


SYSTEM_PROMPT = """你是一名评测集构造助手, 专门为健康 / 训练 / 营养 / 心理 / 医学 / 安全边界领域的 RAG 检索系统生成评测 query。
输入是一段来自知识库的文本片段(可能是中文, 也可能是 WHO / USDA 等英文权威语料)。
你的任务是: 从这段文本中, 设计若干个中文用户问题, 每个问题都必须能被这段文本明确回答。

硬性要求:
1) 问题要像真实中文用户, 可以短、口语化、带轻微模糊表达, 但必须清楚;
2) 不要照抄原文里的长词组、专有短语、数字组合或完整句式; 尽量用同义词、上位词、生活化说法;
3) 不要把答案塞进问题里, 尤其不要泄漏关键数字、阈值、结论性术语;
4) 一个问题只问一件事, 不要把多个子问题塞在一起;
5) 不要依赖外部上下文的代词, 不要出现"这段文字/文中/上述/资料里"这类元说法;
6) 如果文本是英文的, 仍然用中文提问;
7) supporting_span 必须是原文中能支撑答案的原句或短句, 尽量逐字摘录, 不要改写。

输出格式严格要求: 只输出一个 JSON 数组, 每个元素是对象, 不要 markdown 代码块、不要编号。
对象字段:
- query: 中文问题
- answer: 可由该片段回答的简短答案
- supporting_span: 原文支撑句或短句
- question_type: factual | reasoning | comparison | boundary | scenario
- difficulty: easy | medium | hard

示例:
[
  {
    "query": "最近压力大睡不踏实, 睡前可以先做什么放松?",
    "answer": "可以先做几分钟呼吸放松或正念练习。",
    "supporting_span": "睡前进行呼吸放松、正念练习或渐进式肌肉放松有助于入睡。",
    "question_type": "scenario",
    "difficulty": "medium"
  }
]
"""


def _build_user_prompt(chunk_text: str, agent: str, n_questions: int) -> str:
    type_hint = "factual、reasoning、boundary、scenario"
    if n_questions >= 4:
        type_hint = "factual、reasoning、comparison、boundary、scenario"
    return (
        f"知识片段所属命名空间: {agent}\n"
        f"需要生成的候选样本数量: {n_questions}\n"
        f"问题类型尽量覆盖: {type_hint}\n"
        f"知识片段文本(不要在 query 里直接复用里面的长词组、关键数字或原句):\n"
        f"<<<\n{chunk_text}\n>>>\n"
        f"请生成 {n_questions} 个候选样本, 严格按 JSON 数组对象格式输出。"
    )


VERIFY_SYSTEM_PROMPT = """你是 RAG 评测集质检员。你会看到一个知识片段和若干候选样本。
请只根据这个知识片段判断每个候选样本是否适合作为正样本。

判定标准:
- answerable: 仅凭该片段能否回答 query, 不能依赖常识或片段外信息;
- supporting_span_present: supporting_span 是否确实能在片段中定位到, 且足以支撑 answer;
- leakage: low | medium | high, high 表示 query 明显抄了片段关键短语/数字/答案;
- reason: 用很短中文说明主要问题, 没问题写 "ok"。

输出格式严格要求: 只输出 JSON 数组, 每个元素:
{"index": 0, "answerable": true, "supporting_span_present": true, "leakage": "low", "reason": "ok"}
"""


LLM_MAX_ATTEMPTS = 3
LLM_MAX_RETRIES = LLM_MAX_ATTEMPTS - 1


def _trim_chunk_for_prompt(chunk_text: str, max_chars: int) -> str:
    """Keep LLM prompts bounded while local validation still uses the full chunk."""
    text = str(chunk_text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    head = text[:max_chars].rstrip()
    min_keep = max(120, int(max_chars * 0.65))
    sentence_ends = [
        match.end()
        for match in re.finditer(r"[。！？!?]\s*|[.](?=\s|$)", head)
        if match.end() >= min_keep
    ]
    if sentence_ends:
        return head[: sentence_ends[-1]].rstrip()
    return head


def _collect_text_parts(value, parts: List[str]):
    if value is None:
        return
    if hasattr(value, "content"):
        _collect_text_parts(value.content, parts)
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            parts.append(text)
        return
    if isinstance(value, list):
        for item in value:
            _collect_text_parts(item, parts)
        return
    if isinstance(value, dict):
        text_value = value.get("text")
        if isinstance(text_value, str):
            text = text_value.strip()
            if text:
                parts.append(text)
            return
        if text_value is not None:
            _collect_text_parts(text_value, parts)
            return
        for key in ("content", "output_text", "value"):
            if key in value and value[key] is not None:
                _collect_text_parts(value[key], parts)
        return


def _extract_text_content(message_or_content) -> str:
    parts: List[str] = []
    _collect_text_parts(message_or_content, parts)
    return "\n".join(parts)


def _parse_json_array(raw_text: str) -> List:
    """从 LLM 输出里抽取 JSON 数组。

    为什么需要抽取而不是直接 json.loads:
    - 一些模型倾向于给返回包一层 markdown 代码块 (```json ... ```), 或者在前后加一两句
      解释("好的, 以下是生成的问题: [...]"), 这些都会让 json.loads 失败;
    - 只要找到第一个 `[` 到最后一个 `]` 之间的切片再 parse, 就能兼容大多数情况。
    """
    if not raw_text:
        return []
    match = _JSON_ARRAY_RE.search(raw_text)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return parsed


def _normalize_question_type(value: object) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "fact": "factual",
        "事实": "factual",
        "推理": "reasoning",
        "compare": "comparison",
        "对比": "comparison",
        "边界": "boundary",
        "场景": "scenario",
        "情景": "scenario",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in {"factual", "reasoning", "comparison", "boundary", "scenario"}:
        return "factual"
    return normalized


def _normalize_difficulty(value: object) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "简单": "easy",
        "容易": "easy",
        "中等": "medium",
        "普通": "medium",
        "困难": "hard",
        "较难": "hard",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in {"easy", "medium", "hard"}:
        return "medium"
    return normalized


def _coerce_generated_sample(item: object) -> Optional[GeneratedSample]:
    if isinstance(item, str):
        query = item.strip()
        return GeneratedSample(query=query) if query else None
    if not isinstance(item, dict):
        return None

    query = str(item.get("query") or item.get("question") or "").strip()
    if not query:
        return None
    answer = str(item.get("answer") or "").strip()
    supporting_span = str(
        item.get("supporting_span")
        or item.get("supporting_text")
        or item.get("support")
        or item.get("evidence")
        or item.get("supporting_sentence")
        or ""
    ).strip()
    return GeneratedSample(
        query=query,
        answer=answer,
        supporting_span=supporting_span,
        question_type=_normalize_question_type(item.get("question_type") or item.get("type")),
        difficulty=_normalize_difficulty(item.get("difficulty")),
    )


def _parse_generated_samples(raw_text: str) -> List[GeneratedSample]:
    parsed = _parse_json_array(raw_text)
    samples: List[GeneratedSample] = []
    for item in parsed:
        sample = _coerce_generated_sample(item)
        if sample:
            samples.append(sample)
    return samples


def _generate_samples_for_chunk(
    llm,
    chunk_text: str,
    agent: str,
    n_questions: int,
    retries: int = 2,
) -> List[GeneratedSample]:
    """调用 LLM 反向生成候选样本, 失败时重试 `retries` 次。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    user_prompt = _build_user_prompt(chunk_text, agent, n_questions)
    last_err: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
            content = _extract_text_content(response)
            samples = _parse_generated_samples(content)
            if samples:
                return samples[:n_questions]
            last_err = f"LLM 返回无法解析为 JSON 样本数组, 原始内容: {repr(content[:200])}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{e.__class__.__name__}: {e}"
        time.sleep(0.5 * (attempt + 1))

    print(f"[Generate][warn] 放弃该 chunk: {last_err}")
    return []


def _verify_samples_for_chunk(
    llm,
    chunk_text: str,
    samples: List[GeneratedSample],
    retries: int = 2,
) -> List[Dict[str, object]]:
    if not samples:
        return []

    from langchain_core.messages import HumanMessage, SystemMessage

    payload = [
        {
            "index": idx,
            "query": sample.query,
            "answer": sample.answer,
            "supporting_span": sample.supporting_span,
        }
        for idx, sample in enumerate(samples)
    ]
    user_prompt = (
        "知识片段:\n"
        f"<<<\n{chunk_text}\n>>>\n\n"
        "候选样本 JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n\n"
        "请逐条质检, 严格输出 JSON 数组。"
    )
    last_err: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=VERIFY_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
            content = _extract_text_content(response)
            parsed = _parse_json_array(content)
            verdicts = [v for v in parsed if isinstance(v, dict)]
            if verdicts:
                return verdicts
            last_err = f"LLM 质检返回无法解析为 JSON 数组, 原始内容: {repr(content[:200])}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{e.__class__.__name__}: {e}"
        time.sleep(0.5 * (attempt + 1))
    print(f"[Generate][warn] 该 chunk 的 LLM 质检失败, 将仅使用本地过滤: {last_err}")
    return []


def _create_eval_dataset_llm():
    """Create the LLM used only by this dataset-generation script."""
    from wellness_copilot import config
    from langchain_openai import ChatOpenAI

    profile = {
        "label": "EVAL_DATASET_LLM",
        "base_url": config.EVAL_DATASET_LLM_BASE_URL,
        "api_key": config.EVAL_DATASET_LLM_API_KEY,
        "model": config.EVAL_DATASET_LLM_MODEL,
        "api_mode": config.EVAL_DATASET_LLM_API_MODE,
        "output_version": config.EVAL_DATASET_LLM_OUTPUT_VERSION,
        "disable_thinking": config.EVAL_DATASET_LLM_DISABLE_THINKING,
    }
    print(
        "[Generate] LLM: "
        f"model={profile['model']} "
        f"mode={profile['api_mode']} "
        f"base_url={profile['base_url']}"
    )
    if not profile.get("model"):
        raise ValueError(
            "EVAL_DATASET_LLM_MODEL is not set. Set it in .env, or leave it empty "
            "only when LLM_MODEL is configured for fallback."
        )
    if not profile.get("api_key"):
        raise ValueError(
            "EVAL_DATASET_LLM_API_KEY is not set. Set it in .env, or leave it empty "
            "only when LLM_API_KEY is configured for fallback."
        )

    kwargs = {
        "model": profile["model"],
        "base_url": profile["base_url"],
        "api_key": profile["api_key"],
        "max_retries": LLM_MAX_RETRIES,
    }
    if profile.get("disable_thinking"):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    if profile["api_mode"] != "responses":
        return ChatOpenAI(**kwargs)

    responses_kwargs = {
        "use_responses_api": True,
        "output_version": profile["output_version"],
    }
    try:
        return ChatOpenAI(**kwargs, **responses_kwargs)
    except TypeError:
        return ChatOpenAI(
            **kwargs,
            model_kwargs=responses_kwargs,
        )


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #


def _namespaced_chunk_id(agent: str, chunk_id: str) -> str:
    return f"{agent}:{chunk_id}"


def _namespaced_source(agent: str, source: str) -> str:
    return f"{agent}/{source}"


def _write_sample(
    fh,
    sample: GeneratedSample,
    chunk: Dict,
    quality: Dict[str, object],
):
    agent = chunk["agent"]
    chunk_id = chunk["chunk_id"]
    source = chunk["source"]
    row = {
        "query": sample.query,
        "agent": agent,
        "relevant_chunk_ids": [_namespaced_chunk_id(agent, chunk_id)],
        # 冗余写一份 source 级 ground truth, 便于 source-level recall 报告
        "relevant_sources": [_namespaced_source(agent, source)],
        "answer": sample.answer,
        "supporting_span": sample.supporting_span,
        "question_type": sample.question_type,
        "difficulty": sample.difficulty,
        "sample_kind": "single_hop_positive",
        "quality": quality,
    }
    if chunk.get("page_range"):
        row["page_range"] = chunk["page_range"]
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()


def _validate_generated_dataset(path: Path, kb_root: Path, args) -> None:
    from wellness_copilot.config import KNOWLEDGE_BASE_AGENT_SUBDIRS
    from wellness_copilot.rag import LocalKnowledgeBase
    from scripts.evaluate_rag import load_dataset, validate_dataset_against_kbs

    samples = load_dataset(path)
    agent_kbs = {}
    for agent in sorted({sample.agent for sample in samples}):
        subdir = KNOWLEDGE_BASE_AGENT_SUBDIRS[agent]
        agent_kbs[agent] = LocalKnowledgeBase(
            kb_dir=str(kb_root / subdir),
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            boundary_look_back=args.boundary_look_back,
            min_chunk_chars=args.rag_min_chunk_chars,
        )

    validation = validate_dataset_against_kbs(samples, agent_kbs)
    if validation["unknown_sources"] or validation["unknown_chunk_ids"]:
        preview = {
            "unknown_sources": validation["unknown_sources"][:10],
            "unknown_chunk_ids": validation["unknown_chunk_ids"][:10],
        }
        raise ValueError(
            "Generated dataset failed ground-truth validation. "
            f"Preview: {json.dumps(preview, ensure_ascii=False)}"
        )
    print(f"[Generate] ground truth 校验通过: {validation['checked_agents']}")


def _print_generation_cost_plan(chunks: List[Dict], args) -> None:
    target_candidates = args.questions_per_chunk * args.candidate_multiplier
    trimmed_chars = sum(
        len(_trim_chunk_for_prompt(chunk.get("text") or "", args.max_prompt_chars))
        for chunk in chunks
    )
    full_chars = sum(len(chunk.get("text") or "") for chunk in chunks)
    verify_calls = len(chunks) if args.llm_verify else 0
    verify_mode = "off"
    verify_candidate_cap = 0
    if args.llm_verify:
        if args.verify_all_candidates:
            verify_mode = "all_candidates"
            verify_candidate_cap = target_candidates
        else:
            verify_mode = f"local_pass_first(+{args.verify_candidate_buffer})"
            verify_candidate_cap = min(
                target_candidates,
                args.questions_per_chunk + args.verify_candidate_buffer,
            )

    print(
        "[Generate] 成本预估: "
        f"generation_calls={len(chunks)}, "
        f"verify_calls<={verify_calls}, "
        f"target_candidates/chunk={target_candidates}, "
        f"verify_mode={verify_mode}, "
        f"verify_candidates/chunk<={verify_candidate_cap}, "
        f"prompt_chars={trimmed_chars}/{full_chars}"
    )


def main():
    from wellness_copilot.config import KNOWLEDGE_BASE_DIR

    parser = argparse.ArgumentParser(
        description=(
            "Use the project's LLM to synthesize a RAG evaluation set by "
            "reverse-generating Chinese questions from every chunk in the "
            "knowledge base."
        )
    )
    parser.add_argument("--kb-dir", default=KNOWLEDGE_BASE_DIR, help="Knowledge base root")
    parser.add_argument("--chunk-size", type=int, default=420, help="Must match evaluate_rag.py.")
    parser.add_argument("--overlap", type=int, default=100, help="Must match evaluate_rag.py.")
    parser.add_argument(
        "--boundary-look-back",
        type=int,
        default=120,
        help="Must match evaluate_rag.py.",
    )
    parser.add_argument(
        "--rag-min-chunk-chars",
        "--min-chunk-chars",
        dest="rag_min_chunk_chars",
        type=int,
        default=30,
        help="LocalKnowledgeBase min_chunk_chars. Must match evaluate_rag.py.",
    )
    parser.add_argument(
        "--agent",
        action="append",
        help=(
            "Optional agent namespace(s) to generate. Can be repeated or comma-separated, "
            "e.g. --agent trainer,nutritionist --agent doctor. Default: all configured KBs."
        ),
    )
    parser.add_argument(
        "--questions-per-chunk",
        type=int,
        default=2,
        help="Number of accepted questions to keep per chunk (default: 2).",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=3,
        help=(
            "Generate this many candidates per requested question, then filter. "
            "Default: 3."
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Cost-saving preset for iteration: set --candidate-multiplier 1 and "
            "skip the second LLM verification pass."
        ),
    )
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=1800,
        help=(
            "Maximum source characters sent to the LLM per chunk. Local validation "
            "still uses the full chunk. 0 disables trimming (default: 1800)."
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Cap on how many chunks to process. 0 = all. Useful for dry runs.",
    )
    parser.add_argument(
        "--max-chunks-per-file",
        type=int,
        default=10,
        help=(
            "Pre-sample at most this many chunks from each source file before the "
            "global --max-chunks sampling. 0 disables this file-level cap "
            "(default: 10)."
        ),
    )
    parser.add_argument(
        "--min-chunk-len",
        type=int,
        default=120,
        help=(
            "Generation-quality filter: skip generated-source chunks shorter than "
            "this many characters after RAG chunking. This is separate from "
            "--min-chunk-chars, which controls LocalKnowledgeBase chunk creation."
        ),
    )
    parser.add_argument(
        "--no-chunk-quality-filter",
        dest="chunk_quality_filter",
        action="store_false",
        default=True,
        help="Disable extra low-quality chunk filters except --min-chunk-len.",
    )
    parser.add_argument(
        "--min-content-char-ratio",
        type=float,
        default=0.55,
        help="Skip chunks with too little alphanumeric/CJK content (default: 0.55).",
    )
    parser.add_argument(
        "--max-table-line-ratio",
        type=float,
        default=0.65,
        help="Skip chunks dominated by table/rule lines (default: 0.65).",
    )
    parser.add_argument(
        "--max-duplicate-line-ratio",
        type=float,
        default=0.55,
        help="Skip chunks with many repeated non-empty lines (default: 0.55).",
    )
    parser.add_argument(
        "--max-literal-overlap",
        type=float,
        default=0.72,
        help=(
            "Reject generated queries whose character n-gram / copied-span overlap "
            "with the source chunk is above this threshold (default: 0.72)."
        ),
    )
    parser.add_argument(
        "--min-support-trace",
        type=float,
        default=0.78,
        help=(
            "If supporting_span is present, require it to trace back to the chunk "
            "at least this strongly (default: 0.78)."
        ),
    )
    parser.add_argument(
        "--no-require-supporting-span",
        dest="require_supporting_span",
        action="store_false",
        default=True,
        help="Allow samples without supporting_span. Not recommended for final datasets.",
    )
    parser.add_argument(
        "--no-llm-verify",
        dest="llm_verify",
        action="store_false",
        default=True,
        help="Skip the second LLM pass that checks answerability and leakage.",
    )
    parser.add_argument(
        "--verify-all-candidates",
        action="store_true",
        help=(
            "Legacy/strict mode: send every generated candidate to the second LLM "
            "verification pass. By default only locally passing finalists are verified."
        ),
    )
    parser.add_argument(
        "--verify-candidate-buffer",
        type=int,
        default=2,
        help=(
            "When LLM verification is enabled, verify this many extra locally passing "
            "candidates beyond --questions-per-chunk (default: 2)."
        ),
    )
    parser.add_argument(
        "--require-llm-verdict",
        action="store_true",
        help=(
            "Drop candidates when the second LLM verification call fails or omits "
            "a verdict. By default the script falls back to local filters."
        ),
    )
    parser.add_argument(
        "--no-downsample-easy-bm25",
        dest="downsample_easy_bm25",
        action="store_false",
        default=True,
        help="Keep samples that a lexical BM25 baseline can trivially top-1.",
    )
    parser.add_argument(
        "--easy-bm25-score-ratio",
        type=float,
        default=2.4,
        help="BM25 top-1 target/runner-up ratio considered too easy (default: 2.4).",
    )
    parser.add_argument(
        "--easy-bm25-min-overlap",
        type=float,
        default=0.38,
        help=(
            "Only reject easy-BM25 samples when literal overlap is at least this "
            "value, avoiding over-pruning genuinely semantic matches (default: 0.38)."
        ),
    )
    parser.add_argument(
        "--min-question-len",
        type=int,
        default=6,
        help="Reject generated questions shorter than this after whitespace removal.",
    )
    parser.add_argument(
        "--max-question-len",
        type=int,
        default=90,
        help="Reject generated questions longer than this after whitespace removal.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for chunk sampling when --max-chunks is set.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep (seconds) between LLM calls for rate limiting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect/filter/sample chunks and print stats without calling the LLM.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validating the generated dataset against the current KB.",
    )
    parser.add_argument(
        "--out",
        default="eval/rag_eval_dataset_generated.jsonl",
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    # 1) 收集所有 chunks
    agents = _parse_agents(args.agent)
    chunks = collect_all_chunks(
        kb_root=Path(args.kb_dir),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        boundary_look_back=args.boundary_look_back,
        min_chunk_chars=args.rag_min_chunk_chars,
        agents=agents,
    )
    print(f"[Generate] 原始 chunk 总数: {len(chunks)}")
    print(f"[Generate] 原始 chunk 分布: {_chunk_stats(chunks)}")

    # 2) 文件级预抽样 —— 避免长 PDF/书籍在后续全局池里压倒短文档。
    before_file_cap = len(chunks)
    chunks, file_cap_stats = _cap_chunks_per_file(
        chunks,
        max_chunks_per_file=args.max_chunks_per_file,
        seed=args.seed,
    )
    if file_cap_stats["enabled"]:
        print(
            "[Generate] 文件级预抽样后: "
            f"{len(chunks)} (每文件最多 {file_cap_stats['max_chunks_per_file']} 个, "
            f"命中文件 {file_cap_stats['capped_file_count']}/"
            f"{file_cap_stats['file_count']}, "
            f"去掉 {before_file_cap - len(chunks)} 条)"
        )
        if file_cap_stats.get("capped_preview"):
            print(
                "[Generate] 文件级预抽样示例: "
                f"{json.dumps(file_cap_stats['capped_preview'], ensure_ascii=False)}"
            )
        print(f"[Generate] 文件级预抽样后分布: {_chunk_stats(chunks)}")

    # 3) 生成前过滤 —— 丢掉短片段、目录/参考文献、低信息密度或表格噪声片段。
    before = len(chunks)
    chunks, chunk_filter_reasons = _filter_chunks_for_generation(chunks, args)
    print(f"[Generate] chunk 过滤后: {len(chunks)} (去掉 {before - len(chunks)} 条)")
    print(f"[Generate] chunk 过滤原因: {dict(sorted(chunk_filter_reasons.items()))}")
    print(f"[Generate] 过滤后 chunk 分布: {_chunk_stats(chunks)}")

    if not chunks:
        print("[Generate] 没有可用 chunk, 退出。")
        return

    # 4) 如果设置了 max-chunks, 做均匀随机抽样(跨命名空间更均衡)
    if args.max_chunks and args.max_chunks < len(chunks):
        chunks = _sample_chunks(chunks, args.max_chunks, args.seed)
        print(f"[Generate] 抽样到: {len(chunks)} (seed={args.seed})")
        print(f"[Generate] 抽样后 chunk 分布: {_chunk_stats(chunks)}")

    if args.fast:
        args.candidate_multiplier = 1
        args.llm_verify = False
        args.verify_all_candidates = False
        print("[Generate] fast 模式: candidate_multiplier=1, llm_verify=off")

    if args.candidate_multiplier < 1:
        raise ValueError("--candidate-multiplier must be >= 1")
    if args.questions_per_chunk < 1:
        raise ValueError("--questions-per-chunk must be >= 1")
    if args.max_prompt_chars < 0:
        raise ValueError("--max-prompt-chars must be >= 0")
    if args.max_chunks_per_file < 0:
        raise ValueError("--max-chunks-per-file must be >= 0")
    if args.verify_candidate_buffer < 0:
        raise ValueError("--verify-candidate-buffer must be >= 0")
    if not args.llm_verify and args.require_llm_verdict:
        print("[Generate][warn] --require-llm-verdict 在 --no-llm-verify/--fast 下会被忽略。")
        args.require_llm_verdict = False

    _print_generation_cost_plan(chunks, args)

    if args.dry_run:
        print("[Generate] dry-run: 不调用 LLM, 不写入数据集。")
        return

    # 5) 懒加载评测集生成专用 LLM —— dry-run 不需要任何 API key。
    llm = _create_eval_dataset_llm()
    bm25 = CharNgramBM25(chunks)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        backup = out_path.with_suffix(".jsonl.bak")
        shutil.copy2(out_path, backup)
        print(f"[Generate] 已备份旧数据集到: {backup}")

    n_success = 0
    n_written = 0
    generated_candidates = 0
    verified_candidates = 0
    rejected_reasons: Counter = Counter()
    seen_questions = set()
    with out_path.open("w", encoding="utf-8") as fh:
        for idx, chunk in enumerate(chunks, start=1):
            prompt_text = _trim_chunk_for_prompt(chunk["text"], args.max_prompt_chars)
            print(
                f"[Generate] ({idx}/{len(chunks)}) "
                f"{chunk['source']} | len={len(chunk['text'])} prompt_len={len(prompt_text)}"
            )
            target_candidates = args.questions_per_chunk * args.candidate_multiplier
            candidates = _generate_samples_for_chunk(
                llm=llm,
                chunk_text=prompt_text,
                agent=chunk["agent"],
                n_questions=target_candidates,
            )
            generated_candidates += len(candidates)

            accepted: List[Tuple[GeneratedSample, Dict[str, object]]] = []
            rejected: Counter = Counter()
            if args.llm_verify and candidates and not args.verify_all_candidates:
                prefilter_limit = args.questions_per_chunk + args.verify_candidate_buffer
                prefiltered, pre_rejected = _filter_generated_samples(
                    candidates=candidates,
                    chunk=chunk,
                    args=args,
                    bm25=bm25,
                    seen_questions=seen_questions,
                    apply_llm_verdicts=False,
                    mark_seen=False,
                    limit=prefilter_limit,
                )
                rejected.update(pre_rejected)
                finalists = [sample for sample, _quality in prefiltered]
                verdicts = []
                if finalists:
                    verdicts = _verify_samples_for_chunk(
                        llm=llm,
                        chunk_text=prompt_text,
                        samples=finalists,
                    )
                    verified_candidates += len(finalists)
                accepted, final_rejected = _filter_generated_samples(
                    candidates=finalists,
                    chunk=chunk,
                    args=args,
                    bm25=bm25,
                    seen_questions=seen_questions,
                    llm_verdicts=verdicts,
                )
                rejected.update(final_rejected)
            else:
                verdicts = []
                if args.llm_verify and candidates:
                    verdicts = _verify_samples_for_chunk(
                        llm=llm,
                        chunk_text=prompt_text,
                        samples=candidates,
                    )
                    verified_candidates += len(candidates)

                accepted, rejected = _filter_generated_samples(
                    candidates=candidates,
                    chunk=chunk,
                    args=args,
                    bm25=bm25,
                    seen_questions=seen_questions,
                    llm_verdicts=verdicts,
                )
            rejected_reasons.update(rejected)

            if accepted:
                n_success += 1
                for sample, quality in accepted:
                    _write_sample(
                        fh,
                        sample=sample,
                        chunk=chunk,
                        quality=quality,
                    )
                    n_written += 1
            elif candidates:
                print(
                    "[Generate][warn] 该 chunk 候选均被过滤: "
                    f"{dict(sorted(rejected.items()))}"
                )

            if args.sleep > 0:
                time.sleep(args.sleep)

    print("=" * 72)
    print(f"[Generate] 完成: {n_success}/{len(chunks)} chunks 生成成功")
    print(f"[Generate] 生成候选数: {generated_candidates}")
    if args.llm_verify:
        print(f"[Generate] LLM 质检候选数: {verified_candidates}")
    print(f"[Generate] 写入样本数: {n_written}")
    print(f"[Generate] 样本过滤原因: {dict(sorted(rejected_reasons.items()))}")
    print(f"[Generate] 输出: {out_path}")
    print("=" * 72)
    if n_written and not args.no_validate:
        _validate_generated_dataset(out_path, Path(args.kb_dir), args)
    print("下一步: 用新生成的评测集跑一次分层评测")
    print(f"  python scripts/evaluate_rag.py --dataset {out_path}")


if __name__ == "__main__":
    main()
