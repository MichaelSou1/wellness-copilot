"""Optional iCloud CalDAV integration.

This stretch integration is kept dependency-light: install `caldav` and
`icalendar`, then wire schedule_workout into Trainer when you are ready to
demo calendar sync.
"""
from __future__ import annotations

import hashlib
import os


def schedule_workout(
    title: str,
    start_iso: str,
    duration_min: int,
    description: str = "",
    idempotency_key: str = "",
) -> dict:
    """Create a workout event in iCloud Calendar when optional deps are present."""
    try:
        import caldav
        from icalendar import Calendar, Event
    except ImportError as exc:
        return {
            "ok": False,
            "action": "schedule_workout",
            "error": "caldav/icalendar not installed",
            "exception": type(exc).__name__,
        }

    username = os.environ.get("ICLOUD_USERNAME", "")
    password = os.environ.get("ICLOUD_APP_SPECIFIC_PASSWORD", "")
    if not username or not password:
        return {"ok": False, "action": "schedule_workout", "error": "iCloud credentials not configured"}

    # Full CalDAV event creation is intentionally deferred until credentials and
    # target calendar selection are confirmed on the deployment machine.
    uid = hashlib.sha256((idempotency_key or f"{title}:{start_iso}").encode("utf-8")).hexdigest()
    return {
        "ok": False,
        "action": "schedule_workout",
        "uid": uid,
        "error": "CalDAV credentials detected, but calendar selection is not configured yet",
    }
