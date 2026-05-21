# Wellness Copilot Deployment

This deploys the WeChat worker, reminder dispatcher, and backup loop as three
Docker Compose services sharing the same `./data` volume.

## 1. Prepare Server

Recommended baseline: Ubuntu 22.04/24.04, 2C4G minimum. For RAG embedding and
reranking inside the container, 4C8G plus a small swap file is safer. The app
does not expose an HTTP port; it only needs outbound HTTPS for LLM, WeChat
iLink, iCloud CalDAV, HuggingFace/model mirrors, and optional OSS.

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
docker compose version
```

## 2. Configure

```bash
git clone <repo-url> wellness-copilot
cd wellness-copilot
cp .env.example .env
vim .env
mkdir -p data logs reports tmp
chmod 600 .env
```

At minimum set:

- `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` for non-orchestrator text nodes
- `ORCHESTRATOR_LLM_*` if the parent agent should use a stronger/slower model
- `MULTIMODAL_LLM_*` if image grounding should call a real VLM
- `WECHAT_BOT_TOKEN` after running the WeChat login helper
- `ICLOUD_USERNAME`, `ICLOUD_APP_SPECIFIC_PASSWORD`, `ICLOUD_CALDAV_URL`,
  `ICLOUD_CALENDAR_NAME` if Apple Calendar sync is enabled
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

The Compose file also bind-mounts host `.env` to `/app/.env` so
`scripts/wechat_login.py` can write the token back to the host file. Restart the
worker after login because environment variables are read when the container
starts.

## 3. Start

```bash
docker compose up -d --build
docker compose logs -f worker dispatcher backup
```

First-time WeChat binding:

```bash
docker compose exec worker python scripts/wechat_login.py --env /app/.env --qr-path /app/tmp/wechat_qrcode.png --terminal-qr --no-open
docker compose restart worker
```

Apple Calendar check:

```bash
docker compose exec worker python scripts/setup_icloud_caldav.py
```

## 4. Operate

Upgrade:

```bash
docker compose down
git pull
docker compose up -d --build
```

If the ECS region cannot reach the default China mirrors, override build args:

```bash
docker compose build \
  --build-arg PYTHON_BASE_IMAGE=python:3.11-slim \
  --build-arg DEBIAN_MIRROR=http://deb.debian.org/debian \
  --build-arg DEBIAN_SECURITY_MIRROR=http://deb.debian.org/debian-security \
  --build-arg PIP_INDEX_URL=https://pypi.org/simple
docker compose up -d
```

Smoke checks:

```bash
docker compose config --quiet
docker exec wellness-copilot-worker python -c "from wellness_copilot.integrations.local_logs import init_db; init_db(); print('ok')"
docker exec wellness-copilot-worker python scripts/evaluate_output.py --no-judge
docker exec wellness-copilot-worker python scripts/setup_icloud_caldav.py
```

Reminder dry run:

```bash
docker exec wellness-copilot-dispatcher python scripts/reminder_dispatcher.py --once
```

Backups are written under `./data/backups/YYYYMMDD/` and uploaded to OSS when
`OSS_*` variables are configured.

## 5. Portability Notes

- `.dockerignore` excludes `.env`, SQLite files, WAL/SHM files, model caches,
  logs, reports, backups, and `tmp/` so local runtime state is not baked into
  images.
- Persistent runtime state lives in `./data`; copy or restore that directory
  when moving machines.
- `./knowledge_base` is bind-mounted, so edits to knowledge documents on the
  server are visible without rebuilding the image.
- HuggingFace cache is mounted at `./data/.hf_cache`; first boot may still take
  time to download models.
- If enabling MCP servers in Docker, install or mount the Node-based MCP server
  scripts inside the container and set `MCP_*_SCRIPT_PATH` to container paths.
  The default deployment keeps MCP disabled.
