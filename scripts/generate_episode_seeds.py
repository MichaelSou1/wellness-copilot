#!/usr/bin/env python3
"""Generate labeled episodic-memory seed data for the output / episode evals.

Part of todos/rag-diagnosis-plan.md (Phase 1 & 2). For each sample in the
output-eval dataset we fabricate a coherent PRIOR conversation history of N
episodes for that same user persona, labeled for relevance to the sample's
CURRENT question:

  * 3 "relevant": semantically close to the current question AND carrying durable
    personal context / prior decisions a good assistant could weave into a
    personalized answer (NOT a verbatim answer to the current question).
  * 6 "distractor": other realistic wellness topics for the same persona,
    semantically distinct from the current question.

Output: eval/episode_seeds.jsonl, one line per sample:
    {"id": <sample_id>, "episodes": [{query, experts, gist, facts?, relevant}, ...]}

The eval harness (scripts/evaluate_output.py, EVAL_SEED_EPISODES=1) and the
episodic-retrieval eval (scripts/evaluate_episode_memory.py) both consume this.

Usage:
    python scripts/generate_episode_seeds.py                  # all samples
    python scripts/generate_episode_seeds.py --ids nutrition_001,training_003
    python scripts/generate_episode_seeds.py --limit 5 --verbose
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from wellness_copilot.llm import create_llm, extract_text_content  # noqa: E402

DATASET = PROJECT_ROOT / "eval" / "output_eval_dataset.jsonl"
OUT_PATH = PROJECT_ROOT / "eval" / "episode_seeds.jsonl"

N_RELEVANT = 3
N_DISTRACTOR = 6
N_TOTAL = N_RELEVANT + N_DISTRACTOR

VALID_EXPERTS = {"Nutritionist", "Trainer", "Psychologist", "Doctor"}

_SYSTEM = """\
你是健康陪伴多智能体系统的数据合成助手。你的任务是为一个用户"伪造"出他过去若干轮对话的"情景记忆"（episodic memory），用于评测跨轮记忆检索与个性化能力。

每条情景记忆是该用户过去某一轮对话的摘要，字段为：
  - query：用户当时的提问（第一人称中文，<=110字）
  - experts：当时回答的专家，取值只能是 ["Nutritionist","Trainer","Psychologist","Doctor"] 的子集
  - gist：当时给出的建议摘要（中文，<=380字，具体、专业、医学上稳妥）
  - facts：可选的稳定个人事实字典，如 {"weight":82,"goal":"增肌"}；没有就省略该字段
  - relevant：布尔值，见下方规则

你只输出一个 JSON 对象：{"episodes":[ ... ]}，不要任何额外文字或 markdown 代码块。"""

_USER_TMPL = """\
请为下面这位用户伪造他过去的【{n_total}】条情景记忆。

## 该用户的画像（所有情景记忆都必须与之一致）
{profile_block}

## 该用户当前正要问的问题（情景记忆都是这个问题"之前"发生的）
"{current_query}"

## 该问题理想回答涉及的知识点（仅供你理解话题，不要照抄进情景记忆）
{reference_answer}

## 硬性要求
1. 正好 {n_total} 条，其中正好 {n_relevant} 条 relevant=true，正好 {n_distractor} 条 relevant=false。
2. relevant=true 的 {n_relevant} 条：必须与"当前问题"语义高度相关（让向量检索能把它们排到前面），并且携带可被融入个性化回答的"持久个人信息/既往决策"——例如之前共同定下但执行不到位的目标、为既往伤病做过的调整、测过的基线数据。**绝不能直接把当前问题的答案（尤其是当前问题要问的那个具体数字）泄露出来。**
3. relevant=false 的 {n_distractor} 条：是这位用户聊过的其它真实健康话题，主题与当前问题明显不同（因此不应被当作最相关结果检索到）。
4. 每条都要与画像一致：复用真实的年龄/体重/身高/伤病/目标；在合适处把这些数值放进 facts。
5. 全程中文。建议安全：不下诊断、不开处方、不与用户的伤病/过敏冲突。
6. {n_total} 条内容要具体、互不重复、各自是不同的一次对话；experts 要与话题匹配（饮食营养→Nutritionist，训练动作→Trainer，睡眠情绪压力→Psychologist，症状用药→Doctor）。
7. 只输出 {{"episodes":[...]}} 这个 JSON 对象。"""


def _load_samples() -> list[dict]:
    out = []
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _current_query(sample: dict) -> str:
    users = [t["content"] for t in sample.get("turns", []) if t.get("role") == "user"]
    return users[-1] if users else ""


def _profile_block(profile: dict) -> str:
    return json.dumps(profile or {}, ensure_ascii=False, indent=2)


def _extract_json(text: str) -> dict:
    """Pull the first {...} JSON object out of a model reply."""
    text = text.strip()
    # Strip ```json fences if present.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: grab the outermost brace span.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no JSON object found in reply")


def _validate(episodes: list[dict]) -> list[str]:
    errs: list[str] = []
    if len(episodes) != N_TOTAL:
        errs.append(f"expected {N_TOTAL} episodes, got {len(episodes)}")
    n_rel = sum(1 for e in episodes if e.get("relevant") is True)
    if n_rel != N_RELEVANT:
        errs.append(f"expected {N_RELEVANT} relevant=true, got {n_rel}")
    for i, e in enumerate(episodes):
        if not str(e.get("query", "")).strip():
            errs.append(f"ep[{i}] empty query")
        if not str(e.get("gist", "")).strip():
            errs.append(f"ep[{i}] empty gist")
        experts = e.get("experts")
        if not isinstance(experts, list) or not experts:
            errs.append(f"ep[{i}] experts must be a non-empty list")
        elif any(x not in VALID_EXPERTS for x in experts):
            errs.append(f"ep[{i}] invalid expert in {experts}")
    return errs


def _clean(episodes: list[dict]) -> list[dict]:
    cleaned = []
    for e in episodes:
        ep = {
            "query": str(e.get("query", "")).strip()[:120],
            "experts": [x for x in (e.get("experts") or []) if x in VALID_EXPERTS],
            "gist": str(e.get("gist", "")).strip()[:400],
            "relevant": bool(e.get("relevant")),
        }
        facts = e.get("facts")
        if isinstance(facts, dict) and facts:
            ep["facts"] = facts
        cleaned.append(ep)
    return cleaned


def generate_for_sample(llm, sample: dict, retries: int = 3, verbose: bool = False) -> list[dict] | None:
    user_prompt = _USER_TMPL.format(
        n_total=N_TOTAL,
        n_relevant=N_RELEVANT,
        n_distractor=N_DISTRACTOR,
        profile_block=_profile_block(sample.get("profile", {})),
        current_query=_current_query(sample),
        reference_answer=sample.get("reference_answer", ""),
    )
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            resp = llm.invoke([
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=user_prompt),
            ])
            text = extract_text_content(resp)
            obj = _extract_json(text)
            episodes = obj.get("episodes") if isinstance(obj, dict) else None
            if not isinstance(episodes, list):
                raise ValueError("missing 'episodes' list")
            errs = _validate(episodes)
            if errs:
                raise ValueError("; ".join(errs))
            return _clean(episodes)
        except Exception as exc:
            last_err = str(exc)
            if verbose:
                print(f"    attempt {attempt}/{retries} failed: {last_err}")
    print(f"  [FAIL] {sample['id']}: {last_err}")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="", help="comma-separated sample ids (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of samples")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    samples = _load_samples()
    if args.ids:
        wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        samples = [s for s in samples if s["id"] in wanted]
    if args.limit:
        samples = samples[: args.limit]

    # Merge into any existing file so re-runs of subsets are additive.
    out_path = Path(args.out)
    existing: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rec = json.loads(line)
                existing[rec["id"]] = rec

    order = [s["id"] for s in _load_samples()]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _flush() -> None:
        # Persist after every sample so long/parallel runs are crash-safe.
        with out_path.open("w", encoding="utf-8") as f:
            written = set()
            for sid in order:
                if sid in existing:
                    f.write(json.dumps(existing[sid], ensure_ascii=False) + "\n")
                    written.add(sid)
            for sid, rec in existing.items():
                if sid not in written:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    llm = create_llm()
    print(f"[INFO] generating episode seeds for {len(samples)} sample(s) → {out_path}")

    ok = 0
    for idx, sample in enumerate(samples):
        sid = sample["id"]
        print(f"[{idx+1:>2}/{len(samples)}] {sid} ...", flush=True)
        episodes = generate_for_sample(llm, sample, retries=args.retries, verbose=args.verbose)
        if episodes is None:
            continue
        existing[sid] = {"id": sid, "episodes": episodes}
        ok += 1
        _flush()

    print(f"[DONE] {ok}/{len(samples)} generated; file now has {len(existing)} sample(s).")


if __name__ == "__main__":
    main()
