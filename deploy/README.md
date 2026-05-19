# Health Guide Agent Deployment

This deploys the WeChat worker, reminder dispatcher, and backup loop as three
Docker Compose services sharing the same `./data` volume.

## 1. Prepare Server

Recommended baseline: Ubuntu 22.04, 2C4G, domestic region for lower WeChat iLink
latency.

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

## 2. Configure

```bash
git clone <repo-url> health-guide
cd health-guide
cp .env.example .env
vim .env
```

At minimum set:

- `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`
- `WECHAT_APP_ID`, `WECHAT_APP_SECRET`
- `VISION_*` if meal-photo grounding should call a real model
- `OSS_*` if cloud backups should upload to object storage

For Docker, keep persistent paths under `/app/data`:

```env
SQLITE_DB_PATH=/app/data/checkpoints.db
HEALTH_LOGS_DB_PATH=/app/data/health_logs.db
OBSERVABILITY_DB_PATH=/app/data/observability.db
PROFILE_STORE_PATH=/app/data/profile_store.json
EPISODE_STORE_PATH=/app/data/episode_store.json
SESSION_STORE_PATH=/app/data/session_store.json
```

## 3. Start

```bash
docker compose up -d --build
docker compose logs -f worker dispatcher backup
```

First-time WeChat binding:

```bash
docker exec -it hga-worker python scripts/wechat_login.py
docker compose restart worker
```

## 4. Operate

Upgrade:

```bash
git pull
docker compose up -d --build
```

Smoke checks:

```bash
docker exec hga-worker python -c "from health_guide.integrations.local_logs import init_db; init_db(); print('ok')"
docker exec hga-worker python scripts/evaluate_output.py --no-judge
```

Reminder dry run:

```bash
docker exec hga-dispatcher python scripts/reminder_dispatcher.py --once
```

Backups are written under `./data/backups/YYYYMMDD/` and uploaded to OSS when
`OSS_*` variables are configured.
