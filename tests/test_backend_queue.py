import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_agent_job_retry_schedule(tmp_path, monkeypatch):
    db_path = tmp_path / "backend.db"
    monkeypatch.setenv("BACKEND_DB_PATH", str(db_path))

    from wellness_copilot.backend_queue import (
        claim_next_job,
        enqueue_agent_job,
        fail_job,
        get_job,
    )

    job = enqueue_agent_job(
        user_id="u1",
        thread_id="t1",
        message="hello",
        trace_id="trace",
    )
    assert job["status"] == "pending"

    base = 2_000_000_000
    claimed = claim_next_job(now=base)
    assert claimed["job_id"] == job["job_id"]
    assert claimed["attempts"] == 1
    retrying = fail_job(job["job_id"], "boom", now=base + 1)
    assert retrying["status"] == "retrying"
    assert retrying["available_at"] == base + 31

    assert claim_next_job(now=base + 30) is None
    claimed = claim_next_job(now=base + 31)
    assert claimed["attempts"] == 2
    retrying = fail_job(job["job_id"], "boom", now=base + 32)
    assert retrying["available_at"] == base + 92

    claimed = claim_next_job(now=base + 92)
    assert claimed["attempts"] == 3
    retrying = fail_job(job["job_id"], "boom", now=base + 93)
    assert retrying["available_at"] == base + 213

    claimed = claim_next_job(now=base + 213)
    assert claimed["attempts"] == 4
    dead = fail_job(job["job_id"], "boom", now=base + 214)
    assert dead["status"] == "dead"
    assert get_job(job["job_id"])["error"] == "boom"


def test_agent_job_lease_expiry_allows_reclaim(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKEND_DB_PATH", str(tmp_path / "backend.db"))

    from wellness_copilot.backend_queue import claim_next_job, enqueue_agent_job

    enqueue_agent_job(user_id="u1", thread_id="t1", message="hello", trace_id="trace")
    base = 2_000_000_000
    first = claim_next_job(lease_seconds=10, now=base)
    assert first["status"] == "running"
    assert claim_next_job(lease_seconds=10, now=base + 5) is None
    second = claim_next_job(lease_seconds=10, now=base + 10)
    assert second["job_id"] == first["job_id"]
    assert second["attempts"] == 2


def test_outbox_idempotency_and_retry_schedule(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKEND_DB_PATH", str(tmp_path / "backend.db"))

    from wellness_copilot.backend_queue import (
        claim_next_outbox,
        enqueue_outbox_event,
        fail_outbox_event,
    )

    first = enqueue_outbox_event(
        kind="wechat_reply",
        payload={"text": "hi"},
        idempotency_key="idem-1",
        trace_id="trace",
    )
    duplicate = enqueue_outbox_event(
        kind="wechat_reply",
        payload={"text": "hi again"},
        idempotency_key="idem-1",
        trace_id="trace",
    )
    assert duplicate["event_id"] == first["event_id"]

    base = 2_000_000_000
    claimed = claim_next_outbox(now=base)
    assert claimed["attempts"] == 1
    retrying = fail_outbox_event(claimed["event_id"], "send failed", now=base + 1)
    assert retrying["status"] == "retrying"
    assert retrying["available_at"] == base + 31
