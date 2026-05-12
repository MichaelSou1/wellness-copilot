"""用 LLM 自动生成 RAG 评测集。

核心思路:
对知识库里的每一个 chunk,让 LLM 反向生成"如果我是中文用户,
看不到原文,我会怎么问才会落到这一段"的问题。
生成的 `(query, chunk_id)` 天然自带 chunk 级 ground truth ——
因为 query 就是从这段 chunk 反推出来的,正确答案当然是它。

这个脚本**对知识库结构没有任何硬编码**:
- 遍历 `KNOWLEDGE_BASE_AGENT_SUBDIRS` 中声明的各 agent 私有目录;
- 对每一个 .md / .txt / .pdf 文件按和生产管线完全一致的方式切分 chunk;
- chunk_id 格式与 `LocalKnowledgeBase.retrieve_stages()` 返回值保持一致,
  生成的 ground truth 可被 `evaluate_rag.py` 原样匹配。

用法:

    python scripts/generate_eval_dataset.py \\
        --max-chunks 50 \\
        --questions-per-chunk 2 \\
        --out eval/rag_eval_dataset_generated.jsonl

典型迭代节奏:
- 第一次先跑 `--max-chunks 10` 看生成质量,确认 prompt 合理;
- 然后放开到全量或合适的上限,跑一次生成几十到几百条评测样本;
- 再跑 `python scripts/evaluate_rag.py --dataset eval/rag_eval_dataset_generated.jsonl`
  就能得到基于 LLM 自动标注评测集的 Recall@k / MRR / nDCG / MAP 报告。

TODO(quality-filter): 这里只做了最小长度过滤,没有做
  (1) 原文字面泄漏检测 —— LLM 有时会把原文关键短语直接抄进问题,
      这样的 query 不是在测语义,而是在测字符串匹配,应当被过滤掉;
  (2) 答案可定位性验证 —— 可以二次调用 LLM 让它"回答自己出的题",
      只保留能被目标 chunk 正确回答的样本。
  初版先不做,避免脚本过于重量级;有需要时可作为下一步迭代。
"""

import argparse
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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


def collect_all_chunks(kb_root: Path, chunk_size: int, overlap: int) -> List[Dict]:
    """遍历所有 agent 私有知识库, 返回统一结构的 chunk 列表。"""
    from health_guide.config import KNOWLEDGE_BASE_AGENT_SUBDIRS
    from health_guide.rag import LocalKnowledgeBase

    kb_root = Path(kb_root)
    all_chunks: List[Dict] = []

    # Agent-specific namespaces(动态读取 config, 不硬编码 agent 名)
    for agent, subdir in KNOWLEDGE_BASE_AGENT_SUBDIRS.items():
        agent_path = kb_root / subdir
        if not agent_path.exists():
            continue
        agent_store = LocalKnowledgeBase(
            kb_dir=str(agent_path),
            chunk_size=chunk_size,
            overlap=overlap,
            recursive=True,
        )
        all_chunks.extend(_chunks_from_store(agent_store, agent=agent))

    return all_chunks


# --------------------------------------------------------------------------- #
# LLM question generation
# --------------------------------------------------------------------------- #


SYSTEM_PROMPT = """你是一名评测集构造助手, 专门为健康 / 训练 / 营养 / 心理领域的 RAG 检索系统生成评测 query。
输入是一段来自知识库的文本片段(可能是中文, 也可能是 WHO / USDA 等英文权威语料)。
你的任务是: 从这段文本中, 设计若干个中文用户问题, 每个问题都必须满足:

1) 答案可以在这段文本中被明确地找到;
2) 问题不要照抄原文的关键词短语, 用同义改写 / 口语化表达, 模拟真实用户提问;
3) 问题要自然、清晰、一个问题只问一件事, 避免"……的好处和坏处分别是什么"这种多问;
4) 不要依赖外部上下文的代词(不允许出现"它是什么意思"、"上面这段话说的是什么");
5) 如果文本是英文的, 你仍然要用中文提问 —— 模拟中文用户查询英文语料的真实场景;
6) 避免生成"这段文字的主题是什么"这种元问题; 要问具体事实 / 数字 / 建议。

**输出格式严格要求**: 只输出一个 JSON 数组, 每个元素是一条中文字符串问题, 不要任何额外说明、
不要 markdown 代码块、不要编号。例如:
["成年人每晚建议睡眠多少小时?", "高压期应该如何调整训练量?"]
"""


def _build_user_prompt(chunk_text: str, agent: str, n_questions: int) -> str:
    return (
        f"知识片段所属命名空间: {agent}\n"
        f"需要生成的问题数量: {n_questions}\n"
        f"知识片段文本(不要在问题里直接复用里面的原词短语):\n"
        f"<<<\n{chunk_text}\n>>>\n"
        f"请生成 {n_questions} 个中文问题, 严格按 JSON 数组格式输出。"
    )


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_questions(raw_text: str) -> List[str]:
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
    return [str(q).strip() for q in parsed if isinstance(q, str) and q.strip()]


def _generate_questions_for_chunk(
    llm,
    chunk_text: str,
    agent: str,
    n_questions: int,
    retries: int = 2,
) -> List[str]:
    """调用 LLM 反向生成问题, 失败时重试 `retries` 次, 最终失败就返回空列表。"""
    from langchain_core.messages import HumanMessage, SystemMessage
    from health_guide.llm import extract_text_content

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
            content = extract_text_content(response)
            questions = _parse_questions(content)
            if questions:
                return questions[:n_questions]
            last_err = f"LLM 返回无法解析为 JSON 数组, 原始内容: {repr(content[:200])}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{e.__class__.__name__}: {e}"
        time.sleep(0.5 * (attempt + 1))

    print(f"[Generate][warn] 放弃该 chunk: {last_err}")
    return []


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #


def _write_sample(fh, query: str, agent: str, chunk_id: str, source: str):
    row = {
        "query": query,
        "agent": agent,
        "relevant_chunk_ids": [chunk_id],
        # 冗余写一份 source 级 ground truth, 便于 source-level recall 报告
        "relevant_sources": [source],
    }
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()


def main():
    from health_guide.config import KNOWLEDGE_BASE_DIR

    parser = argparse.ArgumentParser(
        description=(
            "Use the project's LLM to synthesize a RAG evaluation set by "
            "reverse-generating Chinese questions from every chunk in the "
            "knowledge base."
        )
    )
    parser.add_argument("--kb-dir", default=KNOWLEDGE_BASE_DIR, help="Knowledge base root")
    parser.add_argument("--chunk-size", type=int, default=420)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument(
        "--questions-per-chunk",
        type=int,
        default=2,
        help="Number of questions to generate per chunk (default: 2).",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Cap on how many chunks to process. 0 = all. Useful for dry runs.",
    )
    parser.add_argument(
        "--min-chunk-len",
        type=int,
        default=120,
        help=(
            "Skip chunks shorter than this (characters). Very short chunks are "
            "usually boilerplate (file headers, TOC lines) and don't yield good "
            "questions."
        ),
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
        "--out",
        default="eval/rag_eval_dataset_generated.jsonl",
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    # 1) 收集所有 chunks
    chunks = collect_all_chunks(Path(args.kb_dir), args.chunk_size, args.overlap)
    print(f"[Generate] 原始 chunk 总数: {len(chunks)}")

    # 2) 最小长度过滤 —— 丢掉明显是 boilerplate 的短片段
    before = len(chunks)
    chunks = [c for c in chunks if len(c["text"]) >= args.min_chunk_len]
    print(
        f"[Generate] 经过 min-chunk-len={args.min_chunk_len} 过滤后: "
        f"{len(chunks)} (去掉 {before - len(chunks)} 条)"
    )

    if not chunks:
        print("[Generate] 没有可用 chunk, 退出。")
        return

    # 3) 如果设置了 max-chunks, 做均匀随机抽样(跨命名空间更均衡)
    if args.max_chunks and args.max_chunks < len(chunks):
        rng = random.Random(args.seed)
        chunks = rng.sample(chunks, args.max_chunks)
        print(f"[Generate] 抽样到: {len(chunks)} (seed={args.seed})")

    # 4) 懒加载 LLM —— 在 argparse/dry-run 之前 import 会逼着用户必须配 API key
    from health_guide.llm import llm

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        backup = out_path.with_suffix(".jsonl.bak")
        shutil.copy2(out_path, backup)
        print(f"[Generate] 已备份旧数据集到: {backup}")

    n_success = 0
    n_written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for idx, chunk in enumerate(chunks, start=1):
            print(
                f"[Generate] ({idx}/{len(chunks)}) "
                f"{chunk['source']} | len={len(chunk['text'])}"
            )
            questions = _generate_questions_for_chunk(
                llm=llm,
                chunk_text=chunk["text"],
                agent=chunk["agent"],
                n_questions=args.questions_per_chunk,
            )
            if questions:
                n_success += 1
                for q in questions:
                    _write_sample(
                        fh,
                        query=q,
                        agent=chunk["agent"],
                        chunk_id=chunk["chunk_id"],
                        source=chunk["source"],
                    )
                    n_written += 1

            if args.sleep > 0:
                time.sleep(args.sleep)

    print("=" * 72)
    print(f"[Generate] 完成: {n_success}/{len(chunks)} chunks 生成成功")
    print(f"[Generate] 写入样本数: {n_written}")
    print(f"[Generate] 输出: {out_path}")
    print("=" * 72)
    print("下一步: 用新生成的评测集跑一次分层评测")
    print(f"  python scripts/evaluate_rag.py --dataset {out_path}")


if __name__ == "__main__":
    main()
