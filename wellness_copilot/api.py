"""FastAPI backend for Wellness Copilot."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from . import config
from .backend_metrics import (
    AGENT_QUEUE_LAG,
    CONTENT_TYPE_LATEST,
    HTTP_LATENCY,
    HTTP_REQUESTS,
    metrics_body,
)
from .backend_queue import (
    backend_counts,
    claim_new_agent_job,
    complete_job,
    enqueue_agent_job,
    ensure_backend_tables,
    fail_job,
    get_job,
    queue_capacity,
)
from .backend_runtime import run_agent_turn, stream_agent_turn
from .backend_telemetry import instrument_fastapi, json_log, new_trace_id
from .integrations.local_logs import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_backend_tables()
    if config.BACKEND_PREWARM_RAG:
        from .tools import prewarm_knowledge_bases

        started = time.perf_counter()
        try:
            result = await asyncio.to_thread(
                prewarm_knowledge_bases,
                config.BACKEND_PREWARM_RAG_QUERY,
            )
            json_log(
                "rag_prewarm_done",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                result=result,
            )
        except Exception as exc:
            json_log("rag_prewarm_failed", error=type(exc).__name__, detail=str(exc)[:300])
    yield


app = FastAPI(title="Wellness Copilot Backend", version="0.1.0", lifespan=lifespan)
instrument_fastapi(app)
_OBSERVED_QUEUE_LAG_JOB_IDS: set[str] = set()


class ChatRequest(BaseModel):
    user_id: str = Field(default="default_user", min_length=1)
    thread_id: str = ""
    message: str = Field(min_length=1)
    source: str = "api"


class JobRequest(ChatRequest):
    idempotency_key: str = ""


def _job_response(job: dict, *, reason: str = "", accepted: bool = True) -> dict[str, Any]:
    return {
        "status": job.get("status") or ("pending" if accepted else "rejected"),
        "accepted": accepted,
        "reason": reason,
        "job_id": job.get("job_id"),
        "thread_id": job.get("thread_id"),
        "trace_id": job.get("trace_id"),
        "poll_url": f"/v1/jobs/{job.get('job_id')}" if job.get("job_id") else "",
    }


def _ensure_capacity() -> dict[str, Any]:
    capacity = queue_capacity()
    if capacity.get("limited"):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "queue_backpressure",
                "message": "Agent job queue is full; retry later.",
                "capacity": capacity,
            },
        )
    return capacity


def _run_claimed_job_once(job: dict) -> dict[str, Any]:
    payload = job.get("input") or {}
    job_id = job["job_id"]
    try:
        result = run_agent_turn(
            user_id=job["user_id"],
            thread_id=job["thread_id"],
            message=payload.get("message") or "",
            content=payload.get("content"),
            source=payload.get("source") or job.get("source") or "api",
            trace_id=job.get("trace_id") or "",
            wechat_context=payload.get("wechat_context") or {},
        ).to_dict()
        complete_job(job_id, result)
        return result
    except Exception as exc:
        fail_job(job_id, f"{type(exc).__name__}: {exc}")
        raise


def _thread_id(user_id: str, supplied: str = "") -> str:
    return supplied or f"api:{user_id}:{uuid.uuid4().hex[:12]}"


def _route_path(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", request.url.path)


async def require_api_key(x_api_key: str = Header(default="")) -> None:
    expected = config.BACKEND_API_KEY
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid X-API-Key")


@app.middleware("http")
async def record_http_metrics(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        path = _route_path(request)
        elapsed = time.perf_counter() - start
        HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
        HTTP_LATENCY.labels(request.method, path).observe(elapsed)
        json_log(
            "http_request",
            method=request.method,
            path=path,
            status=status,
            latency_ms=round(elapsed * 1000, 2),
        )


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    checks: dict[str, Any] = {"api": "ok"}
    try:
        init_db()
        ensure_backend_tables()
        checks["sqlite"] = "ok"
        checks["backend"] = backend_counts()
    except Exception as exc:
        checks["sqlite"] = f"error:{type(exc).__name__}"
    try:
        path = Path(config.SQLITE_DB_PATH)
        if path.parent and str(path.parent) not in {"", "."}:
            path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        conn.execute("SELECT 1")
        conn.close()
        checks["checkpoint"] = "ok"
    except Exception as exc:
        checks["checkpoint"] = f"error:{type(exc).__name__}"
    ok = all(str(value).startswith("ok") or isinstance(value, dict) for value in checks.values())
    return {"ok": ok, "checks": checks}


@app.post("/v1/chat", dependencies=[Depends(require_api_key)])
async def chat(req: ChatRequest) -> dict[str, Any]:
    _ensure_capacity()
    trace_id = new_trace_id()
    thread_id = _thread_id(req.user_id, req.thread_id)
    job = await asyncio.to_thread(
        claim_new_agent_job,
        user_id=req.user_id,
        thread_id=thread_id,
        message=req.message,
        source=req.source or "api",
        trace_id=trace_id,
        idempotency_key=f"sync-chat:{trace_id}",
    )
    task = asyncio.create_task(
        asyncio.to_thread(
            _run_claimed_job_once,
            job,
        )
    )
    try:
        result = await asyncio.wait_for(
            asyncio.shield(task),
            timeout=max(1.0, config.BACKEND_SYNC_TIMEOUT_SEC),
        )
        return result
    except asyncio.TimeoutError:
        json_log(
            "chat_sync_timeout_enqueued",
            trace_id=trace_id,
            user_id=req.user_id,
            thread_id=thread_id,
            job_id=job.get("job_id"),
            timeout_sec=config.BACKEND_SYNC_TIMEOUT_SEC,
        )
        return _job_response(job, reason="sync_timeout")


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


@app.post("/v1/chat/stream", dependencies=[Depends(require_api_key)])
def chat_stream(req: ChatRequest) -> StreamingResponse:
    trace_id = new_trace_id()
    thread_id = _thread_id(req.user_id, req.thread_id)

    def _iter():
        try:
            for item in stream_agent_turn(
                user_id=req.user_id,
                thread_id=thread_id,
                message=req.message,
                source=req.source or "api",
                trace_id=trace_id,
            ):
                yield _sse(item.event, {"trace_id": item.trace_id, "node": item.node, **item.data})
        except Exception as exc:
            yield _sse(
                "error",
                {
                    "trace_id": trace_id,
                    "error": type(exc).__name__,
                    "detail": str(exc)[:300],
                },
            )

    return StreamingResponse(_iter(), media_type="text/event-stream")


@app.post("/v1/jobs", dependencies=[Depends(require_api_key)])
def create_job(req: JobRequest) -> dict[str, Any]:
    _ensure_capacity()
    trace_id = new_trace_id()
    thread_id = _thread_id(req.user_id, req.thread_id)
    job = enqueue_agent_job(
        user_id=req.user_id,
        thread_id=thread_id,
        message=req.message,
        source=req.source or "api",
        trace_id=trace_id,
        idempotency_key=req.idempotency_key,
    )
    return {
        "job_id": job["job_id"],
        "thread_id": job["thread_id"],
        "status": job["status"],
        "trace_id": job["trace_id"],
    }


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def read_job(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("started_at") and job_id not in _OBSERVED_QUEUE_LAG_JOB_IDS:
        created_at = int(job.get("created_at") or 0)
        started_at = int(job.get("started_at") or 0)
        if created_at and started_at:
            AGENT_QUEUE_LAG.observe(max(0, started_at - created_at))
            _OBSERVED_QUEUE_LAG_JOB_IDS.add(job_id)
    return job


@app.get("/metrics")
def metrics() -> Response:
    return Response(metrics_body(), media_type=CONTENT_TYPE_LATEST)
