# Migration Playbook

Use this before the current server expires or when moving to a new machine.

## Source Machine

```bash
cd health-guide
python - <<'PY'
from scripts.backup_loop import run_backup_once
print(run_backup_once())
PY
ls data/backups
```

Confirm the latest backup contains at least:

- `checkpoints.db.gz`
- `health_logs.db.gz`
- `observability.db.gz`
- `profile_store.json.gz`
- `episode_store.json.gz`
- `session_store.json.gz`

## Target Machine

```bash
git clone <repo-url> health-guide
cd health-guide
cp .env.example .env
vim .env
mkdir -p data logs reports tmp
python scripts/restore_backup.py /path/to/latest-backup --target data --overwrite
docker compose up -d --build
```

Then validate:

```bash
docker exec hga-worker python -c "from health_guide.integrations.local_logs import query_logs; print(query_logs.invoke({'kind':'all','days_back':30})[:500])"
docker exec hga-worker python scripts/setup_icloud_caldav.py
docker compose logs --tail=100 worker dispatcher
```

After the worker responds to a WeChat message and the dispatcher can scan
reminders, the migration is complete.

If you need to refresh WeChat login on the new server:

```bash
docker compose exec worker python scripts/wechat_login.py --env /app/.env --qr-path /app/tmp/wechat_qrcode.png --terminal-qr --no-open
docker compose restart worker
```
