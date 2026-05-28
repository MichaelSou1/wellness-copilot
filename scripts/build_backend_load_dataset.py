"""Build backend load-test JSONL from the end-to-end eval dataset."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build backend load dataset from output eval cases")
    parser.add_argument("--input", default="eval/output_eval_dataset.jsonl")
    parser.add_argument("--output", default="eval/backend_load_dataset.jsonl")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the source set N times")
    return parser.parse_args()


def _profile_hint(profile: dict) -> str:
    stats = profile.get("physical_stats") or {}
    diet = profile.get("dietary_context") or {}
    mental = profile.get("mental_state") or {}
    pieces = []
    for key, label in (("age", "年龄"), ("height", "身高"), ("weight", "体重")):
        value = stats.get(key)
        if value:
            unit = "岁" if key == "age" else ("cm" if key == "height" else "kg")
            pieces.append(f"{label}{value}{unit}")
    injuries = stats.get("injuries") or []
    if injuries:
        pieces.append("伤病/限制：" + "、".join(str(x) for x in injuries))
    if diet.get("goal"):
        pieces.append(f"饮食目标：{diet['goal']}")
    preferences = diet.get("preferences") or []
    if preferences:
        pieces.append("饮食偏好：" + "、".join(str(x) for x in preferences))
    stress = mental.get("stress_sources") or []
    if stress:
        pieces.append("压力源：" + "、".join(str(x) for x in stress))
    return "；".join(pieces)


def _message_from_turns(turns: list[dict], profile: dict) -> str:
    user_turns = [str(t.get("content") or "").strip() for t in turns if t.get("role") == "user" and str(t.get("content") or "").strip()]
    if not user_turns:
        return ""
    hint = _profile_hint(profile or {})
    if len(user_turns) == 1:
        message = user_turns[0]
    else:
        history = "\n".join(f"用户第{i + 1}轮：{text}" for i, text in enumerate(user_turns[:-1]))
        message = f"下面是同一用户的历史问题：\n{history}\n\n现在用户继续问：{user_turns[-1]}"
    if hint:
        return f"我的已知情况：{hint}。\n{message}"
    return message


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        message = _message_from_turns(case.get("turns") or [], case.get("profile") or {})
        if not message:
            continue
        records.append(
            {
                "id": case.get("id"),
                "category": case.get("category"),
                "user_id": f"load_{case.get('category', 'case')}_{case.get('id', len(records))}",
                "message": message,
                "expected_experts": (case.get("criteria") or {}).get("expected_experts") or [],
            }
        )

    repeated = []
    for round_idx in range(max(1, args.repeat)):
        for item in records:
            cloned = dict(item)
            if args.repeat > 1:
                cloned["id"] = f"{item['id']}#r{round_idx + 1}"
                cloned["user_id"] = f"{item['user_id']}_r{round_idx + 1}"
            repeated.append(cloned)

    out.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in repeated) + "\n",
        encoding="utf-8",
    )
    counts = Counter(item["category"] for item in repeated)
    print(json.dumps({"output": str(out), "count": len(repeated), "by_category": counts}, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
