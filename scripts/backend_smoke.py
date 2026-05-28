"""Smoke-test the FastAPI backend endpoints."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wellness_copilot import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Wellness Copilot backend API")
    parser.add_argument("--base-url", default=os.environ.get("BACKEND_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=config.BACKEND_API_KEY)
    parser.add_argument("--message", default="你好，用一句话介绍你能做什么。")
    parser.add_argument("--user-id", default="smoke_user")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--run-worker-once", action="store_true", help="Run one local job worker pass after creating a job")
    return parser.parse_args()


def _request(base_url: str, path: str, *, api_key: str = "", payload: dict | None = None, stream: bool = False):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(base_url.rstrip("/") + path, data=data, headers=headers, method="POST" if payload else "GET")
    try:
        resp = urllib.request.urlopen(req, timeout=300)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"{path} failed: HTTP {exc.code} {detail}") from exc
    if stream:
        return resp
    body = resp.read().decode("utf-8")
    return json.loads(body) if body else {}


def main() -> None:
    args = parse_args()
    payload = {"user_id": args.user_id, "message": args.message, "source": "smoke"}

    health = _request(args.base_url, "/healthz")
    print("[healthz]", json.dumps(health, ensure_ascii=False))

    chat = _request(args.base_url, "/v1/chat", api_key=args.api_key, payload=payload)
    print("[chat]", json.dumps({k: chat.get(k) for k in ("trace_id", "route", "latency_ms", "answer")}, ensure_ascii=False)[:1000])
    if not chat.get("trace_id") or not chat.get("answer"):
        raise SystemExit("/v1/chat did not return trace_id and answer")

    stream = _request(args.base_url, "/v1/chat/stream", api_key=args.api_key, payload=payload, stream=True)
    raw = b""
    start = time.time()
    while time.time() - start < args.timeout and b"event: final" not in raw and b"event: error" not in raw:
        raw += stream.read(512)
        if not raw:
            break
    text = raw.decode("utf-8", errors="ignore")
    print("[stream]", text[:1200].replace("\n", "\\n"))
    if "event: final" not in text:
        raise SystemExit("/v1/chat/stream did not emit final event")

    created = _request(args.base_url, "/v1/jobs", api_key=args.api_key, payload=payload)
    print("[job:create]", json.dumps(created, ensure_ascii=False))
    job_id = created["job_id"]
    if args.run_worker_once:
        subprocess.run([sys.executable, "scripts/agent_job_worker.py", "--once", "--limit", "1"], cwd=ROOT, check=False)

    deadline = time.time() + args.timeout
    job = {}
    while time.time() < deadline:
        job = _request(args.base_url, f"/v1/jobs/{job_id}", api_key=args.api_key)
        if job.get("status") in {"succeeded", "dead"}:
            break
        time.sleep(2)
    print("[job:final]", json.dumps({k: job.get(k) for k in ("job_id", "status", "attempts", "trace_id", "error", "result")}, ensure_ascii=False)[:1200])
    if job.get("status") != "succeeded":
        raise SystemExit(f"job did not succeed: {job.get('status')}")


if __name__ == "__main__":
    main()
