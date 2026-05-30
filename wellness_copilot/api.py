"""FastAPI backend for Wellness Copilot."""
from __future__ import annotations

import asyncio
import base64
import binascii
from io import BytesIO
import json
import re
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
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
from .integrations.local_logs import (
    bind_wechat_user,
    default_wechat_project_user_id,
    get_wechat_binding,
    init_db,
    list_wechat_bindings,
)
from .integrations.wechat_ilink import (
    WeChatILinkClient,
    WeChatILinkError,
    runtime_login_status,
    save_runtime_login,
)


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
_FRONTEND_DIR = config.PROJECT_ROOT / "frontend"
_ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
_DATA_URL_RE = re.compile(r"^data:(image/(?:jpe?g|png|webp));base64,(.+)$", re.IGNORECASE | re.DOTALL)
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend_static")


class ChatImage(BaseModel):
    data: str = Field(min_length=1)
    mime_type: str = "image/jpeg"
    filename: str = ""


class ChatRequest(BaseModel):
    user_id: str = Field(default="default_user", min_length=1)
    thread_id: str = ""
    message: str = Field(min_length=1)
    source: str = "api"
    image: ChatImage | None = None


class JobRequest(ChatRequest):
    idempotency_key: str = ""


class WeChatBindingRequest(BaseModel):
    wechat_wxid: str = Field(min_length=1)
    user_id: str = ""
    display_name: str = ""


class WeChatLoginPollRequest(BaseModel):
    qrcode: str = Field(min_length=1)
    poll_base_url: str = ""


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


def _normalize_mime_type(raw: str) -> str:
    mime_type = (raw or "image/jpeg").strip().lower()
    if mime_type == "image/jpg":
        mime_type = "image/jpeg"
    if mime_type not in _ALLOWED_IMAGE_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail={
                "error": "unsupported_image_type",
                "message": "Only JPEG, PNG, and WebP images are supported.",
                "allowed": sorted(_ALLOWED_IMAGE_MIME_TYPES),
            },
        )
    return mime_type


def _normalize_chat_image(image: ChatImage | None) -> dict[str, Any] | None:
    if image is None:
        return None
    data = (image.data or "").strip()
    mime_type = _normalize_mime_type(image.mime_type)
    match = _DATA_URL_RE.match(data)
    if match:
        mime_type = _normalize_mime_type(match.group(1))
        data = match.group(2)
    encoded = "".join(data.split())
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_image_base64", "message": "Image data must be base64 encoded."},
        ) from None
    if not raw:
        raise HTTPException(
            status_code=422,
            detail={"error": "empty_image", "message": "Image data is empty."},
        )
    if len(raw) > config.WEB_CHAT_MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "image_too_large",
                "message": f"Image must be <= {config.WEB_CHAT_MAX_IMAGE_BYTES} bytes.",
                "max_bytes": config.WEB_CHAT_MAX_IMAGE_BYTES,
            },
        )
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
        "mime_type": mime_type,
        "filename": (image.filename or "").strip()[:160],
    }


def _chat_payload(req: ChatRequest) -> tuple[str, Any]:
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(
            status_code=422,
            detail={"error": "empty_message", "message": "message must not be empty."},
        )
    image_part = _normalize_chat_image(req.image)
    if image_part is None:
        return message, message
    return message, [
        {"type": "text", "text": message},
        image_part,
    ]


def _pick(data: dict, *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value:
            return str(value)
        nested = data.get("data")
        if isinstance(nested, dict) and nested.get(key):
            return str(nested[key])
    return ""


def _looks_like_data_image(value: str) -> bool:
    return value.strip().lower().startswith("data:image/")


def _looks_like_base64_image(value: str) -> bool:
    text = value.strip()
    return len(text) > 120 and bool(re.fullmatch(r"[A-Za-z0-9+/=\s]+", text))


def _qr_data_url(payload: str) -> str:
    try:
        import qrcode
    except Exception:
        return ""
    buf = BytesIO()
    img = qrcode.make(payload)
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _qrcode_display_payload(qr: dict) -> dict[str, str]:
    image_value = _pick(
        qr,
        "qrcode_base64",
        "qr_base64",
        "image_base64",
        "qrcode_image",
        "qr_image",
        "image",
    )
    qrcode_url = _pick(qr, "qrcode_img_content", "qrcode_url", "qr_url", "url")
    payload = _pick(qr, "qr_code", "qr", "content", "payload")
    if qrcode_url:
        payload = payload or qrcode_url
    if image_value:
        if _looks_like_data_image(image_value):
            return {"image_data_url": image_value, "qrcode_url": qrcode_url, "payload": payload}
        if _looks_like_base64_image(image_value):
            return {
                "image_data_url": f"data:image/png;base64,{''.join(image_value.split())}",
                "qrcode_url": qrcode_url,
                "payload": payload,
            }
    if payload:
        return {"image_data_url": _qr_data_url(payload), "qrcode_url": qrcode_url, "payload": payload}
    if qrcode_url:
        return {"image_data_url": _qr_data_url(qrcode_url), "qrcode_url": qrcode_url, "payload": qrcode_url}
    fallback = _pick(qr, "qrcode")
    if fallback:
        return {"image_data_url": _qr_data_url(fallback), "qrcode_url": qrcode_url, "payload": fallback}
    return {"image_data_url": "", "qrcode_url": qrcode_url, "payload": payload}


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


@app.get("/", include_in_schema=False)
def frontend_index() -> FileResponse:
    index_path = _FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="frontend not built")
    return FileResponse(index_path)


@app.get("/app", include_in_schema=False)
def frontend_app() -> FileResponse:
    return frontend_index()


@app.get("/v1/frontend/config")
def frontend_config() -> dict[str, Any]:
    return {
        "api_key_required": bool(config.BACKEND_API_KEY),
        "allowed_image_mime_types": sorted(_ALLOWED_IMAGE_MIME_TYPES),
        "max_image_bytes": config.WEB_CHAT_MAX_IMAGE_BYTES,
    }


@app.get("/v1/wechat/login/status", dependencies=[Depends(require_api_key)])
def wechat_login_state() -> dict[str, Any]:
    return runtime_login_status()


@app.post("/v1/wechat/login/qrcode", dependencies=[Depends(require_api_key)])
async def create_wechat_login_qrcode() -> dict[str, Any]:
    client = WeChatILinkClient(bot_token="")
    try:
        qr = await asyncio.to_thread(client.get_bot_qrcode)
    except WeChatILinkError as exc:
        raise HTTPException(
            status_code=502,
            detail={"error": "wechat_qrcode_failed", "message": str(exc)},
        ) from None
    qrcode_id = _pick(qr, "qrcode", "qrcode_id", "qr_id", "ticket", "id")
    if not qrcode_id:
        raise HTTPException(
            status_code=502,
            detail={"error": "wechat_qrcode_missing_id", "message": "QR response did not contain qrcode id."},
        )
    display = _qrcode_display_payload(qr)
    return {
        "qrcode": qrcode_id,
        "image_data_url": display.get("image_data_url") or "",
        "qrcode_url": display.get("qrcode_url") or "",
        "payload": display.get("payload") or "",
        "status": "created",
    }


@app.post("/v1/wechat/login/poll", dependencies=[Depends(require_api_key)])
async def poll_wechat_login(req: WeChatLoginPollRequest) -> dict[str, Any]:
    client = WeChatILinkClient(bot_token="")
    try:
        status = await asyncio.to_thread(
            client.poll_qrcode_status,
            req.qrcode,
            10,
            base_url=(req.poll_base_url or None),
        )
    except WeChatILinkError as exc:
        if "timed out" in str(exc).lower():
            return {
                "authorized": False,
                "state": "pending",
                "next_poll_base_url": req.poll_base_url,
                "login": runtime_login_status(),
            }
        raise HTTPException(
            status_code=502,
            detail={"error": "wechat_login_poll_failed", "message": str(exc)},
        ) from None
    state = _pick(status, "status", "state")
    token = _pick(status, "bot_token", "token", "access_token")
    redirect_host = _pick(status, "redirect_host")
    next_poll_base_url = f"https://{redirect_host}" if state == "scaned_but_redirect" and redirect_host else ""
    if token:
        saved = save_runtime_login(
            bot_token=token,
            base_url=_pick(status, "baseurl", "base_url"),
            account_id=_pick(status, "ilink_bot_id", "bot_id", "account_id"),
            login_user_id=_pick(status, "ilink_user_id", "user_id"),
        )
        return {
            "authorized": True,
            "state": state or "authorized",
            "next_poll_base_url": next_poll_base_url,
            "login": saved,
        }
    return {
        "authorized": False,
        "state": state or "pending",
        "next_poll_base_url": next_poll_base_url,
        "login": runtime_login_status(),
    }


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
    message, content = _chat_payload(req)
    trace_id = new_trace_id()
    thread_id = _thread_id(req.user_id, req.thread_id)
    job = await asyncio.to_thread(
        claim_new_agent_job,
        user_id=req.user_id,
        thread_id=thread_id,
        message=message,
        content=content,
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
    message, content = _chat_payload(req)
    trace_id = new_trace_id()
    thread_id = _thread_id(req.user_id, req.thread_id)

    def _iter():
        try:
            for item in stream_agent_turn(
                user_id=req.user_id,
                thread_id=thread_id,
                message=message,
                content=content,
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
    message, content = _chat_payload(req)
    trace_id = new_trace_id()
    thread_id = _thread_id(req.user_id, req.thread_id)
    job = enqueue_agent_job(
        user_id=req.user_id,
        thread_id=thread_id,
        message=message,
        content=content,
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


@app.get("/v1/wechat/bindings", dependencies=[Depends(require_api_key)])
def wechat_bindings() -> dict[str, Any]:
    return {"bindings": list_wechat_bindings()}


@app.post("/v1/wechat/bindings", dependencies=[Depends(require_api_key)])
def upsert_wechat_binding(req: WeChatBindingRequest) -> dict[str, Any]:
    wxid = (req.wechat_wxid or "").strip()
    if not wxid:
        raise HTTPException(
            status_code=422,
            detail={"error": "empty_wechat_wxid", "message": "wechat_wxid must not be empty."},
        )
    user_id = (req.user_id or "").strip() or default_wechat_project_user_id(wxid)
    previous = get_wechat_binding(wxid)
    bound_user_id = bind_wechat_user(wxid, user_id, display_name=req.display_name)
    return {
        "wechat_wxid": wxid,
        "project_user_id": bound_user_id,
        "display_name": (req.display_name or "").strip(),
        "previous_project_user_id": previous.get("project_user_id") if previous else "",
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(metrics_body(), media_type=CONTENT_TYPE_LATEST)
