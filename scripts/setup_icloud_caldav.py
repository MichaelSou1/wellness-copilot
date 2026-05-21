"""Validate optional iCloud CalDAV credentials and calendar selection."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _is_installed(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def main() -> None:
    load_dotenv()

    missing_env = [
        name
        for name in ("ICLOUD_USERNAME", "ICLOUD_APP_SPECIFIC_PASSWORD")
        if not os.environ.get(name)
    ]
    missing_deps = [name for name in ("caldav", "icalendar") if not _is_installed(name)]

    if missing_env:
        print(f"Missing env: {', '.join(missing_env)}")
    if missing_deps:
        print(f"Missing Python packages: {', '.join(missing_deps)}")
        print("Install with: pip install caldav icalendar")

    if missing_env or missing_deps:
        return

    from wellness_copilot.integrations.apple_calendar import list_apple_calendars

    result = list_apple_calendars()
    if not result.get("ok"):
        print("iCloud CalDAV check failed:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("iCloud CalDAV credentials are valid.")
    print(f"Selected calendar: {result.get('calendar_name') or '(first writable-looking calendar)'}")
    calendars = result.get("available_calendars") or []
    if calendars:
        print("Available calendars:")
        for name in calendars:
            print(f"- {name}")


if __name__ == "__main__":
    main()
