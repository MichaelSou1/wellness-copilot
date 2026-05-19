"""Validate optional iCloud CalDAV credentials."""
from __future__ import annotations

import os


def main() -> None:
    missing = [
        name
        for name in ("ICLOUD_USERNAME", "ICLOUD_APP_SPECIFIC_PASSWORD")
        if not os.environ.get(name)
    ]
    if missing:
        print(f"Missing: {', '.join(missing)}")
        return
    try:
        import caldav  # noqa: F401
        import icalendar  # noqa: F401
    except ImportError:
        print("Install optional dependencies first: pip install caldav icalendar")
        return
    print("iCloud CalDAV credentials and optional dependencies are present.")


if __name__ == "__main__":
    main()
