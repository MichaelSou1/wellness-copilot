"""Build semantic episode indices for existing episode_store.json data.

Run:
    python scripts/migrate_episode_index.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wellness_copilot.episode_memory import EpisodeMemory  # noqa: E402
from wellness_copilot.episode_store import _read_store  # noqa: E402


def main():
    store = _read_store()
    if not store:
        print("No episodes found.")
        return
    total = 0
    for user_id in sorted(store.keys()):
        count = EpisodeMemory(user_id).rebuild_from_store()
        total += count
        print(f"{user_id}: indexed {count} episode(s)")
    print(f"Done. Indexed {total} episode(s).")


if __name__ == "__main__":
    main()
