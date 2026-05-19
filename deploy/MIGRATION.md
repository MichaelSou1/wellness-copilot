# Migration Playbook

Use this before the current server expires or when moving to a new machine.

## Source Machine

```bash
cd health-guide
python scripts/backup_loop.py
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
mkdir -p data
python scripts/restore_backup.py /path/to/latest-backup --target data --overwrite
docker compose up -d --build
```

Then validate:

```bash
docker exec hga-worker python -c "from health_guide.integrations.local_logs import query_logs; print(query_logs.invoke({'kind':'all','days_back':30})[:500])"
docker compose logs --tail=100 worker dispatcher
```

After the worker responds to a WeChat message and the dispatcher can scan
reminders, the migration is complete.
