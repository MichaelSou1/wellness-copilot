"""Consume Agent jobs from SQLite and enqueue reply side effects."""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wellness_copilot import config
from wellness_copilot.backend_queue import (
    claim_next_job,
    complete_job,
    enqueue_outbox_event,
    fail_job,
)
from wellness_copilot.backend_runtime import run_agent_turn
from wellness_copilot.backend_telemetry import json_log
from wellness_copilot.tools import prewarm_knowledge_bases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wellness Copilot Agent job worker")
    parser.add_argument("--once", action="store_true", help="Process available jobs once and exit")
    parser.add_argument("--limit", type=int, default=20, help="Max jobs to process in --once mode")
    parser.add_argument("--interval", type=float, default=config.BACKEND_WORKER_IDLE_SEC)
    return parser.parse_args()


def _enqueue_wechat_reply(job: dict, result: dict) -> None:
    payload = job.get("input") or {}
    wechat_context = payload.get("wechat_context") or {}
    target = wechat_context.get("user_wxid") or ""
    context_token = wechat_context.get("context_token") or ""
    if not target and not context_token:
        return
    enqueue_outbox_event(
        kind="wechat_reply",
        payload={
            "target_wxid": target,
            "context_token": context_token,
            "text": result.get("answer") or "",
            "user_id": job.get("user_id") or "",
        },
        idempotency_key=f"wechat_reply:{job['job_id']}",
        job_id=job["job_id"],
        trace_id=job.get("trace_id") or "",
    )


def process_one() -> bool:
    job = claim_next_job()
    if not job:
        return False
    payload = job.get("input") or {}
    trace_id = job.get("trace_id") or ""
    json_log(
        "agent_job_start",
        trace_id=trace_id,
        job_id=job["job_id"],
        user_id=job.get("user_id"),
        thread_id=job.get("thread_id"),
        attempts=job.get("attempts"),
    )
    try:
        result = run_agent_turn(
            user_id=job["user_id"],
            thread_id=job["thread_id"],
            message=payload.get("message") or "",
            content=payload.get("content"),
            source=payload.get("source") or job.get("source") or "job",
            trace_id=trace_id,
            wechat_context=payload.get("wechat_context") or {},
        ).to_dict()
        complete_job(job["job_id"], result)
        _enqueue_wechat_reply(job, result)
        json_log("agent_job_done", trace_id=trace_id, job_id=job["job_id"], route=result.get("route"))
    except Exception as exc:
        updated = fail_job(job["job_id"], f"{type(exc).__name__}: {exc}")
        json_log(
            "agent_job_failed",
            trace_id=trace_id,
            job_id=job["job_id"],
            status=(updated or {}).get("status"),
            error=type(exc).__name__,
            detail=str(exc)[:300],
        )
    return True


def main() -> None:
    args = parse_args()
    if config.BACKEND_PREWARM_RAG:
        started = time.perf_counter()
        results = prewarm_knowledge_bases(config.BACKEND_PREWARM_RAG_QUERY)
        json_log(
            "worker_rag_prewarm_done",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            results=results,
        )
    processed = 0
    while True:
        did_work = process_one()
        if did_work:
            processed += 1
        if args.once and (not did_work or processed >= args.limit):
            return
        if not did_work:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
