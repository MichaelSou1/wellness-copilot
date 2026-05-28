"""Lightweight backend load test and markdown report generator."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wellness_copilot import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load-test Wellness Copilot backend API")
    parser.add_argument("--base-url", default=os.environ.get("BACKEND_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=config.BACKEND_API_KEY)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--mode", choices=["chat", "jobs"], default="chat")
    parser.add_argument("--message", default="FAKE压测：给我一句健康建议。")
    parser.add_argument("--dataset", default="", help="JSONL load dataset with message/text/query fields")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before selecting requests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--per-category",
        type=int,
        default=0,
        help="Use up to N samples per category before applying --requests",
    )
    parser.add_argument(
        "--categories",
        default="",
        help="Comma-separated category allowlist, e.g. nutrition,training,multi_expert",
    )
    parser.add_argument("--poll-timeout", type=float, default=120)
    return parser.parse_args()


def _request(base_url: str, path: str, *, api_key: str = "", payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(base_url.rstrip("/") + path, data=data, headers=headers, method="POST" if payload else "GET")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[idx]


def _load_dataset(path: str) -> list[dict]:
    if not path:
        return []
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            message = item.get("message") or item.get("text") or item.get("query")
            if not message:
                raise ValueError(f"{path}:{line_no}: missing message/text/query")
            items.append(item)
    if not items:
        raise ValueError(f"{path}: no load samples found")
    return items


def _select_dataset(items: list[dict], args: argparse.Namespace) -> list[dict]:
    if not items:
        return []
    selected = list(items)
    if args.categories:
        allowed = {part.strip() for part in args.categories.split(",") if part.strip()}
        selected = [item for item in selected if str(item.get("category") or "default") in allowed]
        if not selected:
            raise ValueError(f"No dataset samples matched categories={sorted(allowed)}")
    rng = random.Random(args.seed)
    if args.per_category > 0:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for item in selected:
            grouped[str(item.get("category") or "default")].append(item)
        sampled: list[dict] = []
        for category in sorted(grouped):
            bucket = list(grouped[category])
            if args.shuffle:
                rng.shuffle(bucket)
            sampled.extend(bucket[: args.per_category])
        selected = sampled
    if args.shuffle:
        rng.shuffle(selected)
    if not selected:
        raise ValueError("Dataset selection is empty")
    return selected


def _payload_for(args: argparse.Namespace, idx: int) -> dict:
    dataset = getattr(args, "_dataset_items", []) or []
    if dataset:
        item = dataset[idx % len(dataset)]
        message = item.get("message") or item.get("text") or item.get("query")
        user_id = item.get("user_id") or f"load_user_{idx % max(1, args.concurrency)}"
        return {
            "user_id": str(user_id),
            "message": str(message),
            "source": "load",
        }
    return {
        "user_id": f"load_user_{idx % max(1, args.concurrency)}",
        "message": f"{args.message} #{idx}",
        "source": "load",
    }


def _chat_once(args: argparse.Namespace, idx: int) -> dict:
    start = time.perf_counter()
    payload = _payload_for(args, idx)
    dataset = getattr(args, "_dataset_items", []) or []
    item = dataset[idx % len(dataset)] if dataset else {}
    result = _request(args.base_url, "/v1/chat", api_key=args.api_key, payload=payload)
    accepted_async = bool(result.get("job_id") and not result.get("answer"))
    if accepted_async:
        return {
            "ok": True,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "trace_id": result.get("trace_id"),
            "queue_lag_ms": 0,
            "case_id": item.get("id", ""),
            "category": item.get("category", ""),
            "accepted_async": True,
            "error": "",
        }
    return {
        "ok": bool(result.get("answer")),
        "latency_ms": (time.perf_counter() - start) * 1000,
        "trace_id": result.get("trace_id"),
        "queue_lag_ms": 0,
        "case_id": item.get("id", ""),
        "category": item.get("category", ""),
        "accepted_async": False,
        "error": "",
    }


def _job_once(args: argparse.Namespace, idx: int) -> dict:
    start = time.perf_counter()
    payload = _payload_for(args, idx)
    dataset = getattr(args, "_dataset_items", []) or []
    item = dataset[idx % len(dataset)] if dataset else {}
    created = _request(args.base_url, "/v1/jobs", api_key=args.api_key, payload=payload)
    job_id = created["job_id"]
    deadline = time.time() + args.poll_timeout
    job = {}
    while time.time() < deadline:
        job = _request(args.base_url, f"/v1/jobs/{job_id}", api_key=args.api_key)
        if job.get("status") in {"succeeded", "dead"}:
            break
        time.sleep(0.5)
    created_at = float(job.get("created_at") or 0)
    started_at = float(job.get("started_at") or 0)
    return {
        "ok": job.get("status") == "succeeded",
        "latency_ms": (time.perf_counter() - start) * 1000,
        "trace_id": created.get("trace_id"),
        "queue_lag_ms": max(0, (started_at - created_at) * 1000) if created_at and started_at else 0,
        "case_id": item.get("id", ""),
        "category": item.get("category", ""),
        "accepted_async": False,
        "error": job.get("error") or "",
    }


def main() -> None:
    args = parse_args()
    raw_dataset_items = _load_dataset(args.dataset) if args.dataset else []
    dataset_items = _select_dataset(raw_dataset_items, args) if raw_dataset_items else []
    args._dataset_items = dataset_items
    worker = _chat_once if args.mode == "chat" else _job_once
    results: list[dict] = []
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(worker, args, idx) for idx in range(args.requests)]
        for fut in concurrent.futures.as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:
                results.append({"ok": False, "latency_ms": 0, "queue_lag_ms": 0, "error": f"{type(exc).__name__}: {exc}"})
    elapsed = time.perf_counter() - started

    health = {}
    try:
        health = _request(args.base_url, "/healthz")
    except Exception:
        pass

    latencies = [r["latency_ms"] for r in results if r["latency_ms"]]
    queue_lags = [r["queue_lag_ms"] for r in results if r["queue_lag_ms"]]
    ok_count = sum(1 for r in results if r.get("ok"))
    failures = [r for r in results if not r.get("ok")]
    accepted_async_count = sum(1 for r in results if r.get("accepted_async"))
    by_category: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        grouped[str(result.get("category") or "default")].append(result)
    for category, items in sorted(grouped.items()):
        cat_latencies = [x["latency_ms"] for x in items if x.get("latency_ms")]
        by_category[category] = {
            "count": len(items),
            "success": sum(1 for x in items if x.get("ok")),
            "p50_ms": round(_percentile(cat_latencies, 50), 2),
            "p95_ms": round(_percentile(cat_latencies, 95), 2),
        }
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "dataset": args.dataset or "",
        "dataset_size": len(raw_dataset_items),
        "selected_dataset_size": len(dataset_items),
        "dataset_categories": dict(Counter(str(item.get("category") or "default") for item in raw_dataset_items)),
        "selected_dataset_categories": dict(Counter(str(item.get("category") or "default") for item in dataset_items)),
        "shuffle": bool(args.shuffle),
        "seed": args.seed,
        "per_category": args.per_category,
        "categories": args.categories,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "elapsed_sec": round(elapsed, 3),
        "success_count": ok_count,
        "failure_count": len(failures),
        "accepted_async_count": accepted_async_count,
        "success_rate": round(ok_count / max(1, len(results)), 4),
        "latency_ms": {
            "p50": round(_percentile(latencies, 50), 2),
            "p95": round(_percentile(latencies, 95), 2),
            "max": round(max(latencies or [0]), 2),
            "avg": round(statistics.mean(latencies), 2) if latencies else 0,
        },
        "queue_lag_ms": {
            "p50": round(_percentile(queue_lags, 50), 2),
            "p95": round(_percentile(queue_lags, 95), 2),
            "max": round(max(queue_lags or [0]), 2),
        },
        "backend_counts": (health.get("checks") or {}).get("backend", {}),
        "by_category": by_category,
        "sample_errors": [
            {"case_id": r.get("case_id"), "category": r.get("category"), "error": r.get("error")}
            for r in failures[:5]
        ],
    }

    out = Path("reports") / f"backend_upgrade_report_{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# Backend Upgrade Load Report\n\n"
        f"```json\n{json.dumps(report, ensure_ascii=False, indent=2)}\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[report] {out}")


if __name__ == "__main__":
    main()
