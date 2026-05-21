"""Daily backup loop for SQLite/JSON/report artifacts."""
from __future__ import annotations

import gzip
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wellness_copilot import config as cfg


def _backup_sqlite(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return
    source = sqlite3.connect(str(src))
    try:
        target = sqlite3.connect(str(dst))
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()


def _gzip_file(path: Path) -> Path:
    gz_path = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    path.unlink(missing_ok=True)
    return gz_path


def _candidate_files() -> list[Path]:
    paths = [
        Path(os.environ.get("SQLITE_DB_PATH", cfg.SQLITE_DB_PATH)),
        Path(os.environ.get("HEALTH_LOGS_DB_PATH", cfg.HEALTH_LOGS_DB_PATH)),
        Path(os.environ.get("OBSERVABILITY_DB_PATH", cfg.OBSERVABILITY_DB_PATH)),
        Path(os.environ.get("PROFILE_STORE_PATH", cfg.PROFILE_STORE_PATH)),
        Path(os.environ.get("EPISODE_STORE_PATH", cfg.EPISODE_STORE_PATH)),
        Path(os.environ.get("SESSION_STORE_PATH", "session_store.json")),
    ]
    reports = Path("reports")
    if reports.exists():
        paths.extend(p for p in reports.glob("*.json") if p.is_file())
    return paths


def _upload_oss(files: list[Path], date_key: str) -> None:
    if not (cfg.OSS_ACCESS_KEY_ID and cfg.OSS_ACCESS_KEY_SECRET and cfg.OSS_BUCKET and cfg.OSS_ENDPOINT):
        print("[backup] OSS credentials not configured; kept local backup only")
        return
    try:
        import oss2
    except ImportError:
        print("[backup] oss2 is not installed; kept local backup only")
        return
    auth = oss2.Auth(cfg.OSS_ACCESS_KEY_ID, cfg.OSS_ACCESS_KEY_SECRET)
    bucket = oss2.Bucket(auth, cfg.OSS_ENDPOINT, cfg.OSS_BUCKET)
    for file_path in files:
        key = f"{cfg.OSS_PREFIX}/{date_key}/{file_path.name}"
        bucket.put_object_from_file(key, str(file_path))
        print(f"[backup] uploaded oss://{cfg.OSS_BUCKET}/{key}")


def run_backup_once() -> Path:
    date_key = datetime.now().strftime("%Y%m%d")
    out_dir = Path(cfg.BACKUP_DIR) / date_key
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    for src in _candidate_files():
        if not src.exists() or src.is_dir():
            continue
        dst = out_dir / src.name
        if src.suffix in {".db", ".sqlite", ".sqlite3"}:
            _backup_sqlite(src, dst)
        else:
            shutil.copy2(src, dst)
        produced.append(_gzip_file(dst))

    manifest = out_dir / "manifest.txt"
    manifest.write_text("\n".join(path.name for path in produced) + "\n", encoding="utf-8")
    produced.append(manifest)
    _upload_oss(produced, date_key)
    print(f"[backup] completed {len(produced)} files in {out_dir}")
    return out_dir


def cleanup_old_backups() -> None:
    root = Path(cfg.BACKUP_DIR)
    if not root.exists():
        return
    cutoff = time.time() - cfg.BACKUP_RETENTION_DAYS * 86400
    for child in root.iterdir():
        if child.is_dir() and child.stat().st_mtime < cutoff:
            shutil.rmtree(child, ignore_errors=True)


def main() -> None:
    while True:
        try:
            run_backup_once()
            cleanup_old_backups()
        except Exception as exc:
            print(f"[backup] failed: {type(exc).__name__}: {exc}")
        time.sleep(max(1, cfg.BACKUP_INTERVAL_HOURS) * 3600)


if __name__ == "__main__":
    main()
