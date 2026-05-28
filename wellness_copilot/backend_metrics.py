"""Prometheus metrics for the backend MVP."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
    HTTP_LATENCY = Histogram("http_request_latency_seconds", "HTTP request latency", ["method", "path"])
    AGENT_TURN_LATENCY = Histogram("agent_turn_latency_seconds", "Agent turn latency", ["source"])
    AGENT_NODE_LATENCY = Histogram("agent_node_latency_seconds", "LangGraph node latency", ["node", "source"])
    AGENT_JOBS = Counter("agent_jobs_total", "Agent jobs by status", ["status"])
    AGENT_QUEUE_LAG = Histogram("agent_job_queue_lag_seconds", "Agent job queue lag")
    AGENT_QUEUE_DEPTH = Histogram("agent_job_queue_depth", "Agent queue depth sampled at enqueue", ["status"])
    AGENT_JOB_FAILURES = Counter("agent_job_failures_total", "Agent job failures")
    OUTBOX_EVENTS = Counter("outbox_events_total", "Outbox events by kind/status", ["kind", "status"])
    OUTBOX_FAILURES = Counter("outbox_delivery_failures_total", "Outbox delivery failures", ["kind"])
    RAG_HITS = Counter("rag_retrieval_hits_total", "RAG retrieval hits")
    RAG_RETRIEVAL_LATENCY = Histogram("rag_retrieval_latency_seconds", "RAG retrieval latency", ["agent"])
    TOOL_CALLS = Counter("tool_calls_total", "Tool calls", ["tool"])

    def metrics_body() -> bytes:
        return generate_latest()

except Exception:  # pragma: no cover - dependency fallback
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _NoopMetric:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            return None

        def observe(self, *args, **kwargs):
            return None

    HTTP_REQUESTS = _NoopMetric()
    HTTP_LATENCY = _NoopMetric()
    AGENT_TURN_LATENCY = _NoopMetric()
    AGENT_NODE_LATENCY = _NoopMetric()
    AGENT_JOBS = _NoopMetric()
    AGENT_QUEUE_LAG = _NoopMetric()
    AGENT_QUEUE_DEPTH = _NoopMetric()
    AGENT_JOB_FAILURES = _NoopMetric()
    OUTBOX_EVENTS = _NoopMetric()
    OUTBOX_FAILURES = _NoopMetric()
    RAG_HITS = _NoopMetric()
    RAG_RETRIEVAL_LATENCY = _NoopMetric()
    TOOL_CALLS = _NoopMetric()

    def metrics_body() -> bytes:
        return b"# prometheus_client is not installed\n"


@contextmanager
def observe_latency(metric, *label_values: str) -> Iterator[None]:
    import time

    start = time.perf_counter()
    try:
        yield
    finally:
        metric.labels(*label_values).observe(time.perf_counter() - start)
