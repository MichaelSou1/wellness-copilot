"""Restore a local backup snapshot into a target directory.

For cloud restore, download the OSS/COS date folder first, then point this
script at the local backup directory.
"""
from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore Wellness Copilot backup")
    parser.add_argument("backup_dir", help="Directory containing *.gz backup files")
    parser.add_argument("--target", default="restore_tmp", help="Directory to restore into")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target files")
    return parser.parse_args()


def _restore_gz(src: Path, dst: Path, overwrite: bool) -> None:
    name = src.name[:-3] if src.name.endswith(".gz") else src.name
    out = dst / name
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
    with gzip.open(src, "rb") as gz, out.open("wb") as fh:
        shutil.copyfileobj(gz, fh)


def main() -> None:
    args = parse_args()
    backup_dir = Path(args.backup_dir)
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in backup_dir.glob("*.gz"):
        _restore_gz(src, target, args.overwrite)
        count += 1
    print(f"Restored {count} files to {target}")


if __name__ == "__main__":
    main()
