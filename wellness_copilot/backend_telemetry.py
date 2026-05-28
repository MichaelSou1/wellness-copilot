"""Small backend telemetry helpers.

The MVP keeps telemetry optional: Prometheus/OpenTelemetry dependencies are in
requirements, but local scripts still run with a clear fallback if they have
not been installed yet.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator


def new_trace_id() -> str:
    return uuid.uuid4().hex


def now_epoch() -> int:
    return int(time.time())


def json_log(event: str, **fields: Any) -> None:
    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
    }
    payload.update({k: v for k, v in fields.items() if v not in (None, "")})
    print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)


_SPANS_ENABLED = os.environ.get("OTEL_SPANS_ENABLED", "").strip().lower() in {"1", "true", "yes"}

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    if _SPANS_ENABLED:
        _provider = TracerProvider()
        if os.environ.get("OTEL_CONSOLE_EXPORTER", "").strip().lower() in {"1", "true", "yes"}:
            _provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(_provider)
        _tracer = trace.get_tracer("wellness-copilot-backend")
    else:
        _tracer = None
except Exception:  # pragma: no cover - dependency fallback
    _tracer = None


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[None]:
    if _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name) as active:
        for key, value in attrs.items():
            if value not in (None, ""):
                active.set_attribute(key, str(value))
        yield


def instrument_fastapi(app: Any) -> None:
    if not _SPANS_ENABLED:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except Exception:
        return
