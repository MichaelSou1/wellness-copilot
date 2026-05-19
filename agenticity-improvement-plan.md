# Health-Guide-Agent: 微信个人号 Bot + 多模态 + 真实世界 Actuation 升级

## Context

**为什么做这次升级。** 这个 LangGraph 多 agent 项目当前架构本身已经相当扎实：Planner / Dispatcher / 4 个专家（Trainer / Nutritionist / Wellness / General）/ ReplanJudge / Aggregator / Critic 全套，配 bge-m3 RAG、profile + episode 双层记忆、SQLite 可观测性。**它不是 LLM wrapper**。

但简历层面的核心问题是：**输入输出都是文本、agent 无任何真实世界 side effect、跑在命令行里**——所以"为什么不直接问 ChatGPT"这个问题在面试官眼里依旧成立。

本次升级目标：让 ChatGPT 永远做不到的四件事成立：

1. **多模态 grounding**：用户拍餐盘照 → Vision 抽出菜品 + 估算宏量营养素 → 进入 agent 决策上下文。
2. **真正的个人微信号 bot**：用户在自己的微信里直接和"健康助手"私聊（拍照、文字、语音、群聊都行）；朋友家人扫码加好友就能体验。
3. **主动推送 + 闭环**：agent 不只是被动回复，可以定时主动推送（早安复盘、晚间加餐提醒）；下一轮再读回前一轮写入的真实日志做数据驱动诊断。
4. **7×24 云端常驻**：Docker Compose 部署到国内轻量云服务器，worker / dispatcher 进程常驻；本机关机也不影响用户和朋友家人在任意时间发消息体验 bot；SQLite 数据卷定时备份到对象存储。这一项让 demo 从"本地玩具"变成"真上线的产品"。

> **关键技术选型**：用**微信 ClawBot / iLink 协议**（腾讯 2026.3.22 官方发布）。这是微信对**真正的个人微信号**开放的 Bot API，走官方插件体系**无封号风险**。和企业微信智能机器人是不同产品；ClawBot 更"个人化"、面试演示更有冲击力。
>
> ClawBot 只提供消息层（私聊/群聊、多媒体、主动推送），**没有日程/待办**。所以数据持久化用**本地 SQLite**（自己管，简单可靠），日程 actuation 可选 iCloud CalDAV 作为 stretch。

---

## Recommended Approach

### 1. 整体架构

```
                   ┌──────────  国内轻量云 (Lighthouse 2C2G)  ──────────┐
[用户在微信发餐盘照 / 文字]        │                                                    │
        ↓ (iLink 长轮询 getupdates)│  docker-compose:                                   │
  wechat_ilink_worker.py (常驻)   │  ├── worker        (wechat_ilink_worker.py)        │
        ↓ (构造 HumanMessage)     │  ├── dispatcher    (reminder_dispatcher.py)        │
  TurnStart → QueryRewriter →     │  └── backup-cron   (每日 SQLite → COS/OSS)         │
  MultiModalPreprocessor →        │                                                    │
  Planner → Dispatcher →          │  共享 volume: ./data/*.db, *.json, .env            │
    [Analyst | Trainer | ...]     │                                                    │
  → ReplanJudge → Aggregator      │  对外只暴露 healthz (8080)，不暴露 SQLite          │
  → Critic                        │                                                    │
        ↓ (Critic PASS 后)         └────────────────────────────────────────────────────┘
  ┌─→ wechat_ilink 回复用户
  ├─→ 各专家 tool 写本地 SQLite (meals/workouts/wellness/reminders)
  ├─→ (stretch) iCloud CalDAV 排训练事件
  └─→ dispatcher 每分钟扫 reminders → 主动推微信
      (stretch) daily_morning_briefing cron 每天 7am 主动推早安
```

> 本机关机不影响任何流程：worker 长轮询挂在云上 24h 接消息；dispatcher 准点推提醒；备份 cron 每日凌晨打包数据库到对象存储。本地只在开发时跑 `docker compose up` 调试，开发完 push image 到云端 pull 一下重启即可。

### 2. 拓扑改动（最小侵入）

在 `QueryRewriter → Planner` 之间插一个新节点 **`MultiModalPreprocessor`**：

- 节点逻辑：若 `messages[-1].content` 含 image part，调 Vision API 抽取结构化数据写入 state；否则秒过（~1ms no-op）。
- 不进 TurnStart：避免每轮都付 Vision 延迟。
- 不放在 Nutritionist 内部 tool：让 **Planner 在路由前就看到结构化食物数据**，能据此把 Trainer 也拉进来。

**改文件**：`health_guide/graph.py`（加 node + edge）。

### 3. AgentState 扩展（`health_guide/state.py`）

沿用现有 reducer 模式（`_turn_dict` / `_turn_list` + `RESET_SENTINEL`），新增字段：

```python
# turn-scoped（TurnStart 通过 RESET_SENTINEL 清空）
image_inputs:        Annotated[List[Dict], _turn_list]   # 本轮图片 (media_id 或 url)
vision_extractions:  Annotated[Dict[str, Dict], _turn_dict]  # {"meal": {...}, "form": {...}}
actuation_log:       Annotated[List[Dict], _turn_list]   # 本轮 side-effect 流水

# 持久（不进 reset 列表）
recent_logs_summary: Annotated[str, _take_last_str]      # TurnStart 写入的 7 日日志摘要
wechat_context:      Annotated[Dict, _take_last_str]     # {bot_token, context_token, chat_type, user_wxid}
```

**改文件**：`state.py`、`agents/turn_start.py`（在 RESET 字典里加上面前三个 turn-scoped 键名）。

### 4. 新增 integrations 包

新建 `health_guide/integrations/__init__.py` + 以下模块：

#### `integrations/vision.py`（内部 helper，不是 LLM tool）
- `analyze_meal_image(image_bytes_or_url) -> dict`：返回 `{items:[...], kcal, protein_g, carbs_g, fat_g, confidence}`。
- `analyze_form_image(image_bytes, exercise_hint) -> dict`（stretch）。
- 读 `VISION_PROVIDER` 配置：openai / anthropic；从 `config.py` 取 key。
- 仅由 `MultiModalPreprocessor` 调用，**不暴露给 LLM tool 列表**，避免被专家二次调用。

#### `integrations/wechat_ilink.py`（内部 helper，微信 iLink 协议客户端）
- 走 `https://ilinkai.weixin.qq.com` 纯 HTTP/JSON，**不需要 WebSocket**。
- 核心方法：
  - `get_bot_qrcode()` → 拿登录二维码 url
  - `poll_qrcode_status()` → 轮询扫码状态，扫码后拿到持久 `bot_token`（一次性，落 .env / 本地 keyring）
  - `get_updates(timeout=30)` → 长轮询拿新消息（messages, group_msgs, mentions）
  - `send_message(context_token, text|image|voice|file)` → 回复（带 `context_token` 标识对话上下文）
  - `push_to_user(wxid, text)` → 主动推送（用于早安 briefing / 提醒）
  - `download_media(media_id) -> bytes` → 下载用户发的图片到 Vision
- 凭证：`WECHAT_BOT_TOKEN`（首次扫码后写入 .env）。
- 模块级单例 + 心跳监控；断线指数退避重连。
- 不是 `@tool`，由 main loop 和 `push_reminder` 工具共用。

#### `integrations/local_logs.py`（暴露为 `@tool`，**本地 SQLite 持久化**）

新建一个轻量本地日志库（沿用项目已有 SQLite 风格，与 `observability.db` / `checkpoints.db` 同级，命名 `health_logs.db`）。表结构：

```sql
CREATE TABLE meals    (id INTEGER PK, user_id TEXT, date_iso TEXT, items_json TEXT,
                        kcal INT, protein_g INT, carbs_g INT, fat_g INT,
                        source TEXT, idempotency_key TEXT UNIQUE, created_at INT);
CREATE TABLE workouts (id INTEGER PK, user_id TEXT, date_iso TEXT, plan_json TEXT,
                        status TEXT, idempotency_key TEXT UNIQUE, created_at INT);
CREATE TABLE wellness (id INTEGER PK, user_id TEXT, date_iso TEXT, sleep_h REAL,
                        mood TEXT, notes TEXT, idempotency_key TEXT UNIQUE, created_at INT);
CREATE INDEX ix_meals_user_date    ON meals(user_id, date_iso);
CREATE INDEX ix_workouts_user_date ON workouts(user_id, date_iso);
CREATE INDEX ix_wellness_user_date ON wellness(user_id, date_iso);
```

工具：
- `log_meal(date_iso, items_json, kcal, protein_g, carbs_g, fat_g, source, idempotency_key)` — Nutritionist
- `log_workout(date_iso, plan_json, status, idempotency_key)` — Trainer
- `log_wellness_checkin(date_iso, sleep_h, mood, notes, idempotency_key)` — Wellness
- `query_logs(kind, days_back)` — read-only，给 Analyst + TurnStart

**幂等性天然由 SQLite `UNIQUE` 约束保证**：`INSERT OR IGNORE` 命中重复 key 直接静默返回已有 id。LangGraph checkpoint replay 不会重复写。

> **为什么用本地 SQLite 而不是某个云 SaaS**：项目本来就用 SQLite 做 checkpoint + observability，新增一个 logs.db 是最自然的延伸；零部署成本；面试 demo 时不需要联网、不需要别人的账号；如果想加 SaaS sync 是后续 stretch（如 `scripts/sync_logs_to_feishu.py`）。

#### `integrations/push_reminder.py`（暴露为 `@tool`）
- `push_reminder(remind_at_iso, text, idempotency_key)` — 任意专家可调，主要给 Nutritionist 和 Wellness。
- 实现：写入本地 SQLite `reminders` 表（带 `delivered=0`），由 `scripts/reminder_dispatcher.py` cron 每分钟扫一次到点的、调 `wechat_ilink.push_to_user` 发出去并标记 `delivered=1`。
- 幂等同样 SQLite UNIQUE。

#### `integrations/apple_calendar.py`（**stretch**，可选）
- `schedule_workout(title, start_iso, duration_min, description, idempotency_key)` — Trainer
- 走 CalDAV (`pip install caldav icalendar`)，端点 `https://caldav.icloud.com/`。
- 凭证：`ICLOUD_USERNAME` + `ICLOUD_APP_SPECIFIC_PASSWORD`。
- 幂等：iCalendar `UID` 字段 = hash(idempotency_key)。
- 用于展示"个人设备生态联动"：训练事件秒同步到用户 iPhone / Mac 日历，演示效果好。

**新文件**：5-6 个 integrations 模块 + `scripts/wechat_ilink_worker.py`（长轮询 main loop）+ `scripts/wechat_login.py`（首次扫码绑定）+ `scripts/reminder_dispatcher.py`（cron 推送提醒）+ `scripts/daily_morning_briefing.py`（stretch，cron 早安推送）+ `scripts/setup_icloud_caldav.py`（stretch）。

### 5. 新增 `Analyst` 专家（推荐做，MVP 可省）

新文件 `health_guide/agents/analyst.py`，模仿现有专家结构：
- 工具：`query_logs` + `get_user_profile`。
- 职责：纯**数据复盘**——读 7-30 天 SQLite 日志，输出 "本周蛋白质均值 78g、目标 110g、缺口 4/7 天" 这种**真实数字**驱动的诊断，不开处方。
- 排序：在 `_PRIORITY_ORDER` 插到 0（分析先于处方）。
- Planner 系统提示词加规则："问题含 '最近 / 这周 / 进展 / 复盘 / 趋势' → 把 Analyst 加进 plan"。

**改文件**：`graph.py`（加 node 和 edge）、`agents/planner.py`（提示词 + `_VALID_EXPERTS`）、`agents/replan_judge.py`、`agents/aggregator.py`、`agents/critic.py`。

### 6. Critic 新规则（`health_guide/agents/critic.py`）

复用现有 P0/P1/P2 框架，加 4 条新规则：

| 等级 | 名称 | 触发条件 | 处置 |
|------|------|----------|------|
| P1 | actuation 真实性 | 草稿宣称"已记录 / 已设提醒 / 已加日历"，但 `state.actuation_log` 没有对应成功条目 | REVISE：删除虚假声明 |
| P1 | 训练负荷冲突 | Trainer 排了 workout 且本周已 ≥3 次力量训练 | REVISE：要求显式说明加量理由或建议换训练日 |
| P2 | 数据驱动蛋白缺口 | `recent_logs_summary` 近 5 日蛋白 < 1.2 g/kg，且草稿推大重量 / 增肌训练但没提补蛋白 | REVISE：补营养护栏 |
| P2 | Vision 置信度 | `vision_extractions.meal.confidence < 0.5` 但草稿用确定语气给数字 | REVISE：改为"估算"区间表述 |

Critic 的 review prompt builder 需扩展，把 `actuation_log` 和 `vision_extractions` 也注入上下文。

### 7. TurnStart 增强：7 日日志摘要回灌

在 `turn_start_node` 现有 episode 加载之后追加一步：
- 调 `local_logs.query_logs("meal", 7)` + `("workout", 7)` + `("wellness", 7)`，本地缓存 key=`(user_id, date.today())`（每用户每自然日只查一次内存缓存）。
- 用 LLM 浓缩成 ≤300 字 `recent_logs_summary` 写入 state。
- 全程 best-effort（外抛 try/except），SQLite 挂了不影响主流程。

Planner 系统提示词加："如有 `recent_logs_summary` 且查询涉及饮食/训练复盘，优先把 Analyst 加进 plan"。

### 8. Demo 录屏剧本（README 用）

**所有交互都发生在用户自己的微信里**（朋友家人都能验证）：

1. 用户掏出手机，在微信里找到"健康助手"私聊，发烤鸡饭照片 + 文字 "这餐够支撑我增肌目标吗？下次练腿安排周四晚 7 点行不行？"
2. `wechat_ilink_worker.py` `getupdates` 长轮询拿到消息，`download_media` 下载图片，构造 LangGraph 输入（含 image part）。
3. TurnStart：load `recent_logs_summary` = "近 7 日蛋白均值 82g/目标 110g，缺口 4/7 天"。
4. MultiModalPreprocessor：Vision → `meal={items:["烤鸡180g","白米饭250g","西兰花100g"], kcal:720, protein_g:52, confidence:0.78}`。
5. Planner：识别 photo + 蛋白缺口 + 排日历意图 → `plan=[Analyst, Trainer, Nutritionist]`。
6. Analyst：`query_logs("workouts",7)` → "本周已 3 次力量，差腿训"，写 scratchpad。
7. Trainer：读 Analyst note → `log_workout(date="周四", plan="腿训", status="planned")` 入 SQLite；(stretch) `apple_calendar.schedule_workout(...)` 同步到 iPhone 日历。
8. Nutritionist：基于 Vision macros + 7 日缺口 → 建议加 25g 蛋白；`log_meal(...)` 入 SQLite；`push_reminder("晚 8 点补 25g 蛋白", "20:00")` 入 SQLite reminders 表。
9. Aggregator 合并、Critic 审核 P1-actuation 真实性 + P2-蛋白缺口 → PASS。
10. bot 在微信里 `send_message` 把最终回复发回用户（带饮食分析 + 训练已记录 + 晚 8 点提醒已设置）。
11. **真到晚 8 点**：`reminder_dispatcher.py` cron 触发，bot 在微信主动私聊推 "🍳 该补 25g 蛋白啦！"。
12. **第二天早 7 点**：(stretch) `daily_morning_briefing.py` cron，bot 主动推 "昨日复盘：蛋白达成 108g，目标对齐；今晚腿训记得带护膝"。
13. 录屏切到 iPhone 日历 app（stretch，出现周四晚训练）。

**整段录屏的杀伤力**：所有交互在用户日常用的微信里、由 agent 主动推送提醒、有真实数据回灌——任何一条 ChatGPT 都做不到。

### 9. 云部署架构

#### 9.1 服务器选型

**首选：国内云商学生认证免费档**（用户为在校学生）

| 方案 | 配置 | 价格 | 区域 | 备注 |
|------|------|------|------|------|
| **阿里云"高校计划" / 飞天加速** | 2C4G ECS / Lighthouse | **学生认证免费 1 年** | 国内全部 | 力度最大、文档全；到期续费 ¥9.5/月 |
| 腾讯云"校园计划" | 2C4G Lighthouse | **¥1/月**（学生认证） | 国内全部 | 等同免费；新人活动叠加更划算 |
| 华为云鲲鹏学生 | 2C4G | 学生免费 1 年 | 国内全部 | ARM 镜像，注意兼容性 |

> 阿里云"高校计划"和腾讯云"校园计划"在所有可行方案里**适配度最高**：国内节点 → iLink 域名 RTT ≤ 30ms；2C4G 比腾讯/阿里付费起步档（2C2G）资源还多；免费 1 年覆盖整个求职 + 实习周期。**强烈推荐这条路**。

**备选（毕业后或当下不方便认证）**：
- Oracle Cloud Always Free：东京/首尔区 ARM 4C24G 永久免费；需国际信用卡注册；RTT 40-80ms 可接受
- 腾讯云新人首单 ¥9.9/3 月 Lighthouse 2C2G

**不推荐**：
- GCP Always Free：免费档仅美区，iLink 跨太平洋长轮询 RTT 200-300ms；出站流量到中国**不**在免费额度内，长轮询持续连接易超额
- AWS Free Tier：仅 12 个月，到期失效

**操作系统**：Ubuntu 22.04 LTS（学生镜像通常默认即可）。
**端口**：worker 是出向长轮询，不需开任何入站端口；防火墙只放行 SSH 即可。

#### 9.2 Docker 化

**`Dockerfile`**（项目根）：
```dockerfile
FROM python:3.11-slim
WORKDIR /app
# 系统依赖（pillow/pypdf 偶尔需要的 libs）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libjpeg-dev zlib1g-dev curl tini \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["/usr/bin/tini", "--"]
# 默认 cmd 由 compose 覆盖
CMD ["python", "scripts/wechat_ilink_worker.py"]
```

**`docker-compose.yml`**（项目根）：
```yaml
version: "3.9"
services:
  worker:
    build: .
    container_name: hga-worker
    restart: always
    env_file: .env
    volumes:
      - ./data:/app/data         # SQLite + JSON + WeChat 缓存
      - ./logs:/app/logs
    command: ["python", "scripts/wechat_ilink_worker.py"]
    healthcheck:
      test: ["CMD", "python", "-c", "from health_guide.integrations.local_logs import _ensure_init; _ensure_init()"]
      interval: 30s
      timeout: 5s
      retries: 3
  dispatcher:
    build: .
    container_name: hga-dispatcher
    restart: always
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - worker
    command: ["python", "scripts/reminder_dispatcher.py"]
  backup:
    build: .
    container_name: hga-backup
    restart: always
    env_file: .env
    volumes:
      - ./data:/app/data
    command: ["python", "scripts/backup_loop.py"]   # 内部跑每日定时
```

> **健康检查**用 `python -c` 简单调 `_ensure_init` 验证 SQLite 可写；不引入额外 healthz HTTP 服务避免占端口。如果想要更严肃的 healthz，stretch 加一个 FastAPI `/healthz` 监 127.0.0.1:8080。

**`.env` 路径全部改为 `data/` 卷下**（让数据持久化跨容器重启）：
- `SQLITE_DB_PATH=/app/data/checkpoints.db`
- `HEALTH_LOGS_DB_PATH=/app/data/health_logs.db`
- `OBSERVABILITY_DB_PATH=/app/data/observability.db`
- `PROFILE_STORE_PATH=/app/data/profile_store.json`
- `EPISODE_STORE_PATH=/app/data/episode_store.json`
- `SESSION_STORE_PATH=/app/data/session_store.json`
- 同时 `config.py` 已有的 path 默认值改为读环境变量优先（项目本来就这么做了，确认即可）。

#### 9.3 数据备份（Wave 5 一并出）

`scripts/backup_loop.py`：每天凌晨 3 点（用 `apscheduler` 或简单 `time.sleep` 循环）执行：
1. `sqlite3 /app/data/health_logs.db ".backup /tmp/health_logs.YYYYMMDD.db"`（在线备份，不锁库）
2. 同理打包其他 `*.db` 和 `*.json`
3. 用 `cos-python-sdk-v5`（腾讯云）或 `oss2`（阿里云）上传到对象存储 bucket，按日期路径：`health-guide-backup/{date}/{file}.gz`
4. 保留策略：最近 14 天 + 每月 1 号永久。
5. 失败时通过 dispatcher 主动推送告警到自己微信（"备份失败：…"）。

**凭证**：`COS_SECRET_ID` / `COS_SECRET_KEY` / `COS_BUCKET` / `COS_REGION`（腾讯云）或对应阿里云 OSS 变量；都进 `.env`。

成本估算：每天 ~10MB × 30 天 = 300MB；标准存储 ¥0.12/GB/月，约 ¥0.04/月。**忽略不计**。

#### 9.4 部署运维

`deploy/README.md` 文档化：
0. **学生认证**：去阿里云"高校计划"或腾讯云"校园计划"页面，按指引上传学信网在线验证报告，等待 1-2 工作日审核通过；通过后在控制台领取免费 2C4G 实例。**这步是 0 成本的关键**。
1. 装 Docker / Compose（一行 `curl get.docker.com | sh`）
2. `git clone` 项目 → 填 `.env`
3. `docker compose up -d --build`
4. `docker exec -it hga-worker python scripts/wechat_login.py`（首次扫码绑定，写 `bot_token` 入 .env，重启 worker 生效）
5. `docker compose logs -f worker dispatcher` 看流水
6. 升级流程：`git pull && docker compose up -d --build`，SQLite volume 不丢
7. **学生套餐到期前 1 个月**：要么续费 ¥9.5/月 转付费 / 重新认证（如还在校），要么把数据从对象存储恢复到新服务器（备份 cron 保证可迁移）。`.env` 里所有路径和凭证可变量化，搬家无需改代码。

`deploy/systemd/*.service`（**备选**，给不想用 Docker 的人）：两个 service 文件（worker / dispatcher）+ Restart=always，文档化但不强推。

### 10. 实施顺序（已确认现有代码模式兼容）

读完 `state.py / graph.py / tools.py / turn_start.py / nutritionist.py / trainer.py / wellness.py / planner.py / critic.py / config.py / llm.py / main.py` 后确认：项目用的是 `RESET_SENTINEL` reducer 模式、`@tool` 装饰器统一注册、SQLite + WAL 的本地持久化风格、`extract_text_content` 已支持 list-content（image part 不会 crash）。新增模块全部沿用现有约定。

**按依赖关系分 4 个 wave 提交，每个 wave 一个 commit：**

- **Wave 1 — 多模态 + 本地日志**（零外部 IM 依赖，立即可单元测试）
  - `state.py` 加 5 字段；`agents/turn_start.py` reset 新字段
  - `integrations/__init__.py` + `integrations/vision.py`
  - `integrations/local_logs.py`（SQLite，UNIQUE 幂等）
  - `agents/multimodal_preprocessor.py` 新节点
  - `graph.py` 插入新节点
  - `tools.py` 注册 `log_meal / log_workout / log_wellness_checkin / query_logs`
  - 三个专家文件加新工具到 `tools` 列表
  - `agents/critic.py` 加 P1-actuation 真实性 + P2-vision 置信度规则
  - `config.py` 加 vision + SQLite 路径环境变量

- **Wave 2 — 提醒推送**（依赖 Wave 1 的 SQLite 基础）
  - `integrations/push_reminder.py`（写 reminders 表）
  - `scripts/reminder_dispatcher.py`（cron 入口；启动时若无 wechat 客户端则只打日志）

- **Wave 3 — 微信 iLink 接入**
  - `integrations/wechat_ilink.py`（HTTP/JSON 客户端，含 graceful 降级）
  - `scripts/wechat_login.py`（扫码绑定）
  - `scripts/wechat_ilink_worker.py`（长轮询 main loop）
  - `main.py` 改造为 `--mode cli|wechat` 双入口
  - `requirements.txt` 加 `pillow`, `requests`

- **Wave 4 — 容器化 + 云部署**
  - `Dockerfile` + `docker-compose.yml`（worker + dispatcher + backup 三服务，共享 ./data volume）
  - `scripts/backup_loop.py`（每日凌晨打包 SQLite + JSON 上传对象存储；失败告警走 dispatcher 推微信）
  - `deploy/README.md`（买服务器、装 Docker、首次扫码、升级流程的傻瓜文档）
  - `deploy/systemd/*.service`（**备选**，给不想用 Docker 的人）
  - `.env.example` 加云部署相关变量（`HEALTH_LOGS_DB_PATH=/app/data/...`、`COS_*`、`BACKUP_RETENTION_DAYS=14` 等）
  - 真在云上跑一遍 demo 流程；让朋友扫码加 bot 任意时间发消息验证 7×24 可用

- **Wave 5 — 收尾**
  - `README.md` 重写（架构图含云边界 + demo 流程 + 一键部署 quick start）
  - 简历项目描述定稿
  - （可选）录 60s demo 视频

每 wave 完成后 commit 并 push，再开下一 wave。Stretch（Analyst / iCloud / morning briefing / 群聊 / 姿势识别）作为后续独立 PR 处理。

### 11. MVP 切片 vs Stretch

**MVP（必做，1-2 周可完成）**：
- AgentState 5 个新字段
- MultiModalPreprocessor 节点 + Vision meal 抽取（不做 form）
- `wechat_ilink.py` 客户端 + `wechat_login.py` 扫码绑定 + `wechat_ilink_worker.py` 长轮询入口（核心 plumbing，**最关键**）
- `local_logs.py` 三个写工具 + `query_logs`（SQLite，零外部依赖）
- `push_reminder.py` + `reminder_dispatcher.py` cron
- Critic P1 actuation 真实性 + P2 vision 置信度
- **Dockerfile + docker-compose.yml + backup 容器 + 一台国内轻量云上跑通 7×24**（不上云就少了"产品"层级）
- 重写 README，加 demo 视频（手机微信屏幕录制）

**Stretch（出 MVP 后再加）**：
- 新 Analyst 专家
- TurnStart 7 日摘要回灌
- Critic 训练负荷冲突 + 蛋白缺口规则
- `apple_calendar.py` + iCloud CalDAV 同步
- `daily_morning_briefing.py` 早安主动推送
- 群聊支持（拉 bot 进健身打卡群，群成员 @bot 都能聊）
- 运动姿势照片识别

---

## Critical Files

需要修改：
- `health_guide/state.py` — 新增 5 个字段及对应 reducer 使用
- `health_guide/graph.py` — 加入 `MultiModalPreprocessor` 节点；加入 `Analyst` 节点和边（stretch）
- `health_guide/agents/turn_start.py` — RESET 字典补 turn-scoped 键；7 日摘要回灌（stretch）
- `health_guide/agents/planner.py` — `_VALID_EXPERTS` 加 Analyst（stretch）；提示词加图片 / 日志规则
- `health_guide/agents/critic.py` — 4 条新规则注入 system prompt + review prompt builder
- `health_guide/agents/{trainer,nutritionist,wellness}.py` — 各自 `tools` 列表 append 新工具
- `health_guide/tools.py` — append 新 `@tool`
- `health_guide/config.py` — 加 Vision / 微信 iLink / SQLite 路径 / (stretch) iCloud 环境变量声明
- `health_guide/llm.py` — 确认 `extract_text_content` 处理 list content 中带 image part 的消息（不要 crash）
- `requirements.txt` — 加 `pillow`, `requests`（如未有）, (stretch) `caldav`, `icalendar`
- `README.md` — 重写架构图，强调微信 bot + 多模态 + 主动推送闭环
- `main.py` — 改造为 `--mode cli|wechat` 双入口，默认 wechat 长轮询

需要新增：
- `health_guide/agents/multimodal_preprocessor.py`
- `health_guide/agents/analyst.py`（stretch）
- `health_guide/integrations/__init__.py`
- `health_guide/integrations/vision.py`
- `health_guide/integrations/wechat_ilink.py`
- `health_guide/integrations/local_logs.py`
- `health_guide/integrations/push_reminder.py`
- `health_guide/integrations/apple_calendar.py`（stretch）
- `scripts/wechat_login.py`
- `scripts/wechat_ilink_worker.py`
- `scripts/reminder_dispatcher.py`
- `scripts/backup_loop.py`（每日 SQLite/JSON 打包上传对象存储 + 失败告警）
- `scripts/daily_morning_briefing.py`（stretch）
- `scripts/setup_icloud_caldav.py`（stretch）
- `migrations/001_create_health_logs.sql`（SQLite schema）
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `.env.example`（含云部署变量模板）
- `deploy/README.md`（从 0 到上线的傻瓜部署文档）
- `deploy/systemd/hga-worker.service`（备选）
- `deploy/systemd/hga-dispatcher.service`（备选）

可复用的现有实现：
- `state.py` 的 `_turn_dict / _turn_list / _take_last_str` reducer + `RESET_SENTINEL` 模式
- `tools.py` 的 `@tool` 装饰器 + 模块级单例缓存
- `turn_start_node` 现有 episode_context 加载流程（直接 mirror 它写 `recent_logs_summary`）
- `critic.py` 现有 P0/P1/P2 评分框架（追加规则即可，不需要重写）
- `agents/*` 专家的 ReAct loop（直接复用，只是各自 tools 列表加新工具）
- `observability.py` 的 SQLite metrics（加新列：`actuation_count`, `vision_calls`, `wechat_msgs_in/out`）
- 现有 SQLite 风格（`checkpoints.db` / `observability.db` 共用 sqlite3 + WAL，新增 `health_logs.db` 同款）

---

## Verification

### 一次性准备（本地开发）
```bash
# 1. 微信侧（扫码绑定）
python scripts/wechat_login.py
# → 终端打印二维码 URL，浏览器打开后微信扫码确认 → bot_token 写入 .env

# 2. 本地 SQLite 表初始化
python -c "from health_guide.integrations.local_logs import init_db; init_db()"

# 3. 启动 worker 和 cron
python scripts/wechat_ilink_worker.py &           # 长轮询消息
python scripts/reminder_dispatcher.py &           # 每分钟扫提醒
```

### 云端部署（Wave 4 完成后正式上线）
```bash
# 服务器上一次性
ssh ubuntu@<server-ip>
curl -fsSL https://get.docker.com | sh
git clone <repo> health-guide && cd health-guide
cp .env.example .env && vim .env                  # 填 LLM / 微信 / COS 凭证
docker compose up -d --build                      # 起 worker + dispatcher + backup
docker exec -it hga-worker python scripts/wechat_login.py   # 扫码绑定，bot_token 写回 .env
docker compose restart worker                     # 让 bot_token 生效

# 后续观察
docker compose logs -f worker dispatcher backup
docker exec hga-worker sqlite3 /app/data/health_logs.db "SELECT count(*) FROM meals;"

# 升级
git pull && docker compose up -d --build          # SQLite volume 不丢
```

### 端到端手动验证（演示流程）
1. **单轮多模态**：在微信和 bot 私聊发餐盘照 + "这餐够吗？周四晚 7 点排腿训"，确认：
   - bot 文字回复出现在微信
   - `health_logs.db` `meals` + `workouts` 表各出现新行（含 idempotency_key）
   - `health_logs.db` `reminders` 表出现晚 8 点条目（delivered=0）
2. **真实定时主动推送**：把电脑时间调到 19:59 等 1 分钟，确认 bot 在微信主动发"该补蛋白"消息；reminders 表 delivered=1。
3. **跨轮闭环**：第二天发 "我昨天吃得怎么样？"，确认 Analyst 拉出昨天写入的真实日志，输出数字诊断。
4. **幂等回归**：worker kill 后重启，旧消息重投不会重复写 SQLite（UNIQUE 约束起作用）。
5. **(stretch) iCloud 同步**：iPhone 日历"健康助手"子日历出现周四晚训练事件。

### 检查清单
- [ ] 微信扫码绑定成功，`getupdates` 长轮询正常收消息
- [ ] `health_logs.db` 三张表实际出现新行（含 idempotency_key 列）
- [ ] reminders cron 准时推送，微信端收到主动消息
- [ ] 同一 idempotency_key 重试不重复写
- [ ] Critic actuation 真实性规则：人为篡改 actuation_log 为空、保留草稿声明 → 必须 REVISE
- [ ] Vision 失败注入（断网 / 假图）：流程降级到 "请用文字描述"，不 crash
- [ ] `observability.db` 出现 `actuation_count` / `vision_calls` / `wechat_msgs_in/out` 新列指标
- [ ] **云端 7×24 验证**：本机关机 24h 后，让朋友在任意时间发消息，bot 仍正常回复
- [ ] **容器自愈**：`docker kill hga-worker` 后 `restart: always` 自动起来，未读消息不丢（iLink 服务端保留 + 重启后 `getupdates` 拿到）
- [ ] **数据持久化**：`docker compose down && docker compose up -d` 后 SQLite 数据不丢
- [ ] **备份验证**：第二天看对象存储 bucket 出现昨日的备份文件；手动 `pip install cos-python-sdk-v5 && python -m scripts.restore_backup <date>` 能恢复到一个临时目录
- [ ] (stretch) iPhone 日历出现新事件

### 简历层面验证
- README 顶部一张系统拓扑图，红色高亮 MultiModalPreprocessor + 微信 iLink + 本地日志 + cron 主动推送
- 一段 60 秒 demo 视频：**手机微信屏幕录制**——
  - 0:00 微信里加 bot 好友 / 进入对话
  - 0:05 发餐盘照 + 排训意图
  - 0:15 bot 回复（含分析 + 训练已记录 + 提醒已设置）
  - 0:30 跳到 18:00（剪辑），bot 主动推"该补蛋白"
  - 0:45 第二天发"我昨天吃得怎么样" → bot 用真实数据复盘
- 简历项目描述加一行硬指标：
  - "接入微信 ClawBot iLink 协议实现真实 IM bot，Docker Compose 部署到国内学生免费云服务器实现 **7×24 在线**；新增多模态食物识别 + 本地结构化日志 + 主动定时推送闭环；SQLite 数据卷定时备份到对象存储；agent side-effect 流水可审计、幂等保证 100%"
- 一句锦上添花的"工程性价比"亮点（可放项目描述末尾或面试谈到时说）：
  - "全栈 0 元搭建：阿里云/腾讯云学生免费档 2C4G + 对象存储 ¥0.04/月备份；同等线上 always-on 商业方案月均 ¥30-60，体现资源运用与成本意识"

---

## 关键 gotchas

- **微信 ClawBot 是 2026 年 3 月 22 日新功能**，文档可能更新；以官方插件 / npm `@tencent-weixin` 包为准，社区教程可能落后或不准。集成第一步先跑通官方 "扫码 + getupdates + sendmessage" hello world，再往项目里搬。
- **iLink 协议是纯 HTTP/JSON**：长轮询 `getupdates` 超时建议 25-30s，比 WebSocket 简单，但要处理空响应、网络抖动。worker 必须支持 graceful restart；状态都落到 SQLite，重启不丢上下文。
- **bot_token 是一次性扫码后持久凭证**：必须 `.env` + gitignore，丢了要重新扫码。建议加密落 keyring。
- **图片消息只给 media_id**：iLink 回调的图片消息要先调 `download_media` 拿 bytes 再喂 Vision，封装重试。
- **群聊支持**：iLink 区分私聊 / 群聊；MVP 只做私聊，stretch 才支持群（需处理 @机器人 解析、群成员 wxid 映射）。
- **主动推送限流**：iLink 主动推送有频率限制（避免骚扰），cron 推送时控制单用户单日 ≤ 3-5 条；reminders 表加 `priority` 字段防刷屏。
- **协议升级风险**：因为是新产品，未来 3-6 个月 iLink 协议可能 breaking change；`wechat_ilink.py` 客户端做好版本声明 + 兼容层。简历可以把"对接最新发布的官方协议"作为亮点写出来。
- **跨时区**：reminders 用 UTC + tzid 存；提醒发出前用 `pytz` 转用户本地时间（默认 Asia/Shanghai）。
- **隐私自检**：SQLite 都在你自己的云服务器 + 自己的对象存储 bucket 里，agent 不向第三方上报（除 LLM / Vision API 调用外）。README 写明数据边界 + .env 模板说明权限。

### 云部署额外 gotchas

- **服务器选址**：必须**国内**（北京/上海/广州/成都都行）。`ilinkai.weixin.qq.com` 走境外服务器会高延迟甚至连不上；国内 LLM（DeepSeek / 通义 / 智谱 OpenAI 兼容）同理。
- **境外 LLM 用了境内服务器**：如果你 LLM 是 OpenAI / Anthropic 直连，国内服务器需要走代理（HTTP_PROXY 环境变量）；推荐项目默认切到国内 OpenAI 兼容服务避免代理。
- **bot_token 持久化**：扫码后写入 `.env`（位于宿主机 `./` 目录），volume mount 到容器；千万别只写容器内 `/tmp`，否则容器重建就丢。
- **重启不丢消息**：iLink 服务端会保留未拉取的更新（offset 概念），worker 重启后 `getupdates(offset=last_seen)` 会重新拿到；`last_seen_offset` 必须落 SQLite 持久化、不能只在内存。
- **同一 bot_token 同时只能一个 worker**：iLink 不支持多实例同 token 并发拉取，会互相抢消息。Docker Compose 配置确保 `worker` 服务 replicas=1；不要无脑横向扩容。
- **时区**：容器默认 UTC，会导致 cron / 日志时间和你本地差 8h。compose 加 `environment: TZ=Asia/Shanghai` 或 Dockerfile 装 `tzdata`。
- **磁盘膨胀**：`checkpoints.db` 会随对话累积；`observability.db` 同理。Wave 4 备份脚本顺手做 90 天前 checkpoint 软清理（保留摘要、删原文 messages），避免几个月后撑爆 40GB 系统盘。
- **secret 不要进 git**：`.env` 必须在 `.gitignore`；`.env.example` 留模板。对象存储 access key 走子账号 + 只授权对应 bucket 的写权限，最小权限原则。
- **流量费**：Lighthouse 套餐内含 4M 带宽 + 一般 500GB 流量/月；个人 bot 用不到 1GB。无需担心。
- **备份恢复演练**：上线第二周一定演练一次"从对象存储拉昨日备份还原到一个新容器"，否则备份等于没做。`scripts/restore_backup.py` 必须随 backup 脚本同时交付。
- **学生认证有效期**：阿里 / 腾讯学生免费档通常按学年发，**毕业前 1 个月**就要规划迁移：要么转付费档（¥9.5-30/月）、要么搬到 Oracle 东京区 Always Free（要国际信用卡）、要么搬回个人轻量机。`docker-compose.yml` + `.env` 抽象到位的话，迁移只需 `git clone + cp .env + docker compose up -d + 还原最新备份`，半小时搞定。**这条对长期维护重要**。
- **不要选 GCP / 不要跨太平洋**：再强调一遍——`ilinkai.weixin.qq.com` 是国内域名，worker 长轮询要常态保持。任何海外服务器（尤其美区）都会面临 RTT 高 + 跨境流量计费 + GFW 抖动三重劣势，免费档省的钱可能被流量账单反咬一口。学生认证国内节点是当前最优选。
