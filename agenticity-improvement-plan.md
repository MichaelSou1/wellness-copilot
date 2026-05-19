# Health-Guide-Agent: 微信个人号 Bot + 多模态 + 真实世界 Actuation 升级

## Context

**为什么做这次升级。** 这个 LangGraph 多 agent 项目当前架构本身已经相当扎实：

- 完整的 Plan-and-Execute + 动态 Replan 协作（Planner / Dispatcher / 4 个专家 / ReplanJudge / Aggregator / Critic）
- **两阶段 RAG**:bge-m3 dense retrieve + bge-reranker-v2-m3 cross-encoder rerank（506 条评测集实测 MRR 0.9677，首位命中 94.3%）
- **完整的评测工程闭环**:RAG 召回分层评测（Embedding / Rerank 两阶段独立指标 + Δ uplift 分析）、端到端输出质量评测（30 条 8 类场景 + LLM-as-Judge + 断言）、A/B embedding 模型对比脚本
- profile_store（语义记忆）+ episode_store（情节记忆）双层跨 thread 持久化
- TurnStart 轮边界清理 + 长历史 LLM 摘要压缩 + 跨进程会话恢复（SqliteSaver checkpoint + session_store）
- SQLite 可观测性（路由 / 工具 / 时延 / 引用率 / 自动导出 reports/latest_metrics.json）

**它不是 LLM wrapper。**

但简历层面的核心问题是：**输入输出都是文本、agent 无任何真实世界 side effect、跑在命令行里**——所以”为什么不直接问 ChatGPT”这个问题在面试官眼里依旧成立。

本次升级目标：让 ChatGPT 永远做不到的四件事成立：

1. **多模态 grounding**：用户拍餐盘照 → Vision 抽出菜品 + 估算宏量营养素 → 进入 agent 决策上下文。
1. **真正的个人微信号 bot**：用户在自己的微信里直接和”健康助手”私聊（拍照、文字、语音、群聊都行）；朋友家人扫码加好友就能体验。
1. **主动推送 + 闭环**：agent 不只是被动回复，可以定时主动推送（早安复盘、晚间加餐提醒）；下一轮再读回前一轮写入的真实日志做数据驱动诊断。
1. **7×24 云端常驻**：Docker Compose 部署到国内轻量云服务器，worker / dispatcher 进程常驻；本机关机也不影响用户和朋友家人在任意时间发消息体验 bot；SQLite 数据卷定时备份到对象存储。这一项让 demo 从”本地玩具”变成”真上线的产品”。

> **关键技术选型**：用**微信 ClawBot / iLink 协议**（腾讯 2026.3.22 官方发布）。这是微信对**真正的个人微信号**开放的 Bot API，走官方插件体系**无封号风险**。和企业微信智能机器人是不同产品；ClawBot 更”个人化”、面试演示更有冲击力。
>
> ClawBot 只提供消息层（私聊/群聊、多媒体、主动推送），**没有日程/待办**。所以数据持久化用**本地 SQLite**（自己管，简单可靠），日程 actuation 可选 iCloud CalDAV 作为 stretch。

> **评测回归原则**:项目已经有了相当完整的评测体系（RAG 召回 + 端到端质量 + A/B 模型对比）。本次升级**每个 Wave 完成后必须跑一次端到端评测脚本**，确保新 Critic 规则、新节点、新工具没有让 `eval/output_eval_dataset.jsonl` 的 30 条样本评分掉点。这是这次升级和”瞎改一通”的本质区别。

-----

## Recommended Approach

### 1. 整体架构

```
                   ┌──────────  国内轻量云 (阿里云 ECS 2C4G 学生免费档)  ──────────┐
[用户在微信发餐盘照 / 文字]        │                                                    │
        ↓ (iLink 长轮询 getupdates)│  docker-compose:                                   │
  wechat_ilink_worker.py (常驻)   │  ├── worker        (wechat_ilink_worker.py)        │
        ↓ (构造 HumanMessage)     │  ├── dispatcher    (reminder_dispatcher.py)        │
  TurnStart → QueryRewriter →     │  └── backup-cron   (每日 SQLite → OSS/COS)         │
  MultiModalPreprocessor →        │                                                    │
  Planner → Dispatcher →          │  共享 volume: ./data/*.db, *.json, .env            │
    [Analyst | Trainer | ...]     │                          + .index_cache/          │
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

> **RAG 索引缓存的特殊处理**:项目的 RAG 已经实现了 `knowledge_base/<namespace>/.index_cache/`（embeddings.npy + index.faiss + chunks/meta）来避免冷启动重新编码。**Docker 部署时务必把这个目录纳入 volume mount，否则容器每次重启都要重新跑 bge-m3 embedding，启动 30s+**。Wave 4 的 compose 文件已经考虑了这一点（见下方）。

### 2. 拓扑改动（最小侵入）

在 `QueryRewriter → Planner` 之间插一个新节点 **`MultiModalPreprocessor`**：

- 节点逻辑：若 `messages[-1].content` 含 image part，调 Vision API 抽取结构化数据写入 state；否则秒过（~1ms no-op）。
- 不进 TurnStart：避免每轮都付 Vision 延迟。
- 不放在 Nutritionist 内部 tool：让 **Planner 在路由前就看到结构化食物数据**，能据此把 Trainer 也拉进来。

**改文件**：`health_guide/graph.py`（加 node + edge）。

> **评测影响**:`output_eval_dataset.jsonl` 当前 30 条样本都是纯文本输入。MultiModalPreprocessor 节点对纯文本输入是 no-op（不调 Vision），所以不会影响现有 30 条样本的评分。**但 Wave 1 完成后必须验证一次**:跑 `python scripts/evaluate_output.py --no-judge` 确认路由准确率仍 ≥93.3%（当前 baseline）。

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
- 读 `VISION_PROVIDER` 配置：openai / anthropic / 阿里云通义千问 VL / 智谱 GLM-4V；从 `config.py` 取 key。
- 仅由 `MultiModalPreprocessor` 调用，**不暴露给 LLM tool 列表**，避免被专家二次调用。

> **国内 Vision 服务推荐**:既然 ECS 选了国内节点，Vision API 也尽量走国内供应商避免代理：
>
> - **通义千问 VL-Max**:阿里云原生，调用最快；输入约 ¥0.02/千 tokens
> - **智谱 GLM-4V**:国内可调，质量接近 GPT-4V
> - 仅当无国内方案时再回退 OpenAI / Anthropic（需要 ECS 配 HTTP_PROXY）

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
- 用于展示”个人设备生态联动”：训练事件秒同步到用户 iPhone / Mac 日历，演示效果好。

**新文件**：5-6 个 integrations 模块 + `scripts/wechat_ilink_worker.py`（长轮询 main loop）+ `scripts/wechat_login.py`（首次扫码绑定）+ `scripts/reminder_dispatcher.py`（cron 推送提醒）+ `scripts/daily_morning_briefing.py`（stretch，cron 早安推送）+ `scripts/setup_icloud_caldav.py`（stretch）。

### 5. 新增 `Analyst` 专家（推荐做，MVP 可省）

新文件 `health_guide/agents/analyst.py`，模仿现有专家结构：

- 工具：`query_logs` + `get_user_profile`。
- 职责：纯**数据复盘**——读 7-30 天 SQLite 日志，输出 “本周蛋白质均值 78g、目标 110g、缺口 4/7 天” 这种**真实数字**驱动的诊断，不开处方。
- 排序：在 `_PRIORITY_ORDER` 插到 0（分析先于处方）。
- Planner 系统提示词加规则：“问题含 ‘最近 / 这周 / 进展 / 复盘 / 趋势’ → 把 Analyst 加进 plan”。

**改文件**：`graph.py`（加 node 和 edge）、`agents/planner.py`（提示词 + `_VALID_EXPERTS`）、`agents/replan_judge.py`、`agents/aggregator.py`、`agents/critic.py`。

> **评测影响**:Analyst 加入后，`_VALID_EXPERTS` 集合变化、`routing_accuracy` 计算会受影响。**必须在 `eval/output_eval_dataset.jsonl` 里追加至少 3-5 条 `analyst` / `progress_review` 类样本**（如”我这周吃得怎么样”、“我训练有进步吗”），作为 Analyst 路由的回归测试。否则 multi_turn 类别评分可能虚高（Analyst 没被路由到反而装得像”没漏掉”）。

### 6. Critic 新规则（`health_guide/agents/critic.py`）

复用现有 P0/P1/P2 框架，加 4 条新规则：

|等级|名称           |触发条件                                                         |处置                      |
|--|-------------|-------------------------------------------------------------|------------------------|
|P1|actuation 真实性|草稿宣称”已记录 / 已设提醒 / 已加日历”，但 `state.actuation_log` 没有对应成功条目     |REVISE：删除虚假声明           |
|P1|训练负荷冲突       |Trainer 排了 workout 且本周已 ≥3 次力量训练                             |REVISE：要求显式说明加量理由或建议换训练日|
|P2|数据驱动蛋白缺口     |`recent_logs_summary` 近 5 日蛋白 < 1.2 g/kg，且草稿推大重量 / 增肌训练但没提补蛋白|REVISE：补营养护栏            |
|P2|Vision 置信度   |`vision_extractions.meal.confidence < 0.5` 但草稿用确定语气给数字       |REVISE：改为”估算”区间表述       |

Critic 的 review prompt builder 需扩展，把 `actuation_log` 和 `vision_extractions` 也注入上下文。

> **评测影响**:这是最危险的一组改动。Critic 规则收紧后，原有 30 条样本中 **safety / personalization 类目可能有少数样本被新规则误触发 REVISE**（特别是 P2 数据驱动蛋白缺口规则——历史样本里没 recent_logs_summary 字段，正确行为应是规则**跳过**而不是误触发）。
>
> **强制验证步骤**:
>
> 1. Wave 1 完成后跑 `python scripts/evaluate_output.py`，**对比 personalization 和 safety 维度评分**，确保不低于 baseline（3.667 / 4.767）
> 1. 任何评分掉点都要在 critic prompt 里加 “若 recent_logs_summary 为空则不触发该规则” 之类的 guard
> 1. 善用 `--rerun reports/output_eval_report.json --rerun-bad` 只重跑掉点样本，省 LLM judge 成本

### 7. TurnStart 增强：7 日日志摘要回灌

在 `turn_start_node` 现有 episode 加载之后追加一步：

- 调 `local_logs.query_logs("meal", 7)` + `("workout", 7)` + `("wellness", 7)`，本地缓存 key=`(user_id, date.today())`（每用户每自然日只查一次内存缓存）。
- 用 LLM 浓缩成 ≤300 字 `recent_logs_summary` 写入 state。
- 全程 best-effort（外抛 try/except），SQLite 挂了不影响主流程。

Planner 系统提示词加：“如有 `recent_logs_summary` 且查询涉及饮食/训练复盘，优先把 Analyst 加进 plan”。

### 8. Demo 录屏剧本（README 用）

**所有交互都发生在用户自己的微信里**（朋友家人都能验证）：

1. 用户掏出手机，在微信里找到”健康助手”私聊，发烤鸡饭照片 + 文字 “这餐够支撑我增肌目标吗？下次练腿安排周四晚 7 点行不行？”
1. `wechat_ilink_worker.py` `getupdates` 长轮询拿到消息，`download_media` 下载图片，构造 LangGraph 输入（含 image part）。
1. TurnStart：load `recent_logs_summary` = “近 7 日蛋白均值 82g/目标 110g，缺口 4/7 天”。
1. MultiModalPreprocessor：Vision → `meal={items:["烤鸡180g","白米饭250g","西兰花100g"], kcal:720, protein_g:52, confidence:0.78}`。
1. Planner：识别 photo + 蛋白缺口 + 排日历意图 → `plan=[Analyst, Trainer, Nutritionist]`。
1. Analyst：`query_logs("workouts",7)` → “本周已 3 次力量，差腿训”，写 scratchpad。
1. Trainer：读 Analyst note → `log_workout(date="周四", plan="腿训", status="planned")` 入 SQLite；(stretch) `apple_calendar.schedule_workout(...)` 同步到 iPhone 日历。
1. Nutritionist：基于 Vision macros + 7 日缺口 → 建议加 25g 蛋白；`log_meal(...)` 入 SQLite；`push_reminder("晚 8 点补 25g 蛋白", "20:00")` 入 SQLite reminders 表。
1. Aggregator 合并、Critic 审核 P1-actuation 真实性 + P2-蛋白缺口 → PASS。
1. bot 在微信里 `send_message` 把最终回复发回用户（带饮食分析 + 训练已记录 + 晚 8 点提醒已设置）。
1. **真到晚 8 点**：`reminder_dispatcher.py` cron 触发，bot 在微信主动私聊推 “🍳 该补 25g 蛋白啦！”。
1. **第二天早 7 点**：(stretch) `daily_morning_briefing.py` cron，bot 主动推 “昨日复盘：蛋白达成 108g，目标对齐；今晚腿训记得带护膝”。
1. 录屏切到 iPhone 日历 app（stretch，出现周四晚训练）。

**整段录屏的杀伤力**：所有交互在用户日常用的微信里、由 agent 主动推送提醒、有真实数据回灌——任何一条 ChatGPT 都做不到。

### 9. 云部署架构

#### 9.1 服务器选型

**当前确定方案**:**阿里云 ECS 华东1（杭州），2C4G，200Mbps，学生免费档 3 个月**，弹性公网 IP 47.96.235.135（节省停机模式，开机即用）。

> 选阿里云杭州区是最优解:`ilinkai.weixin.qq.com` 国内域名 RTT 低；2C4G 比腾讯/阿里付费起步档（2C2G）资源还多；3 个月覆盖整个开发 + 上线 + 求职演示窗口。

**到期前 2 周（建议 8/5 前）必做一次完整迁移演练**:

1. 把 OSS / COS 上的最新备份恢复到一个新的临时 ECS（或者本地 Docker）
1. 验证恢复后的环境能正常接收微信消息、所有 SQLite 数据完整
1. 演练通过后才能确认”3 个月到期就算被回收也能 10 分钟搬家”

**备选方案**（毕业后或学生认证失效）：

- 腾讯云”校园计划”：2C4G Lighthouse，¥1/月（学生认证），等同免费
- Oracle Cloud Always Free：东京/首尔区 ARM 4C24G 永久免费；需国际信用卡注册；RTT 40-80ms 可接受
- 腾讯云新人首单 ¥9.9/3 月 Lighthouse 2C2G

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
# 用国内 pip 镜像加速（关键，否则 ECS 上 pip install 慢/超时）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai \
    HF_ENDPOINT=https://hf-mirror.com
ENTRYPOINT ["/usr/bin/tini", "--"]
# 默认 cmd 由 compose 覆盖
CMD ["python", "scripts/wechat_ilink_worker.py"]
```

> **bge-m3 / bge-reranker-v2-m3 模型镜像处理**:这两个模型加起来约 2.5GB，**不要打进 Docker 镜像**（会让镜像膨胀到 5GB+，每次部署都慢）。两种方案：
>
> - **方案 A（推荐）**:首次启动容器时让 `scripts/download_rag_models.py` 自动从 hf-mirror 下载到 `data/.hf_cache/`（mount 为 volume），后续启动复用
> - **方案 B**:开发机 `huggingface-cli download` 下到本地 `.hf_cache/`，`scp` 到 ECS 的 `./data/.hf_cache/`，volume mount 进容器

> **`.index_cache` 持久化**:项目已有 `knowledge_base/<namespace>/.index_cache/`（FAISS 索引 + embeddings.npy）。这部分**必须 mount 出来到 volume**，否则每次容器重启都要重跑全量 embedding（CPU 模式约 30-60s）。

**`docker-compose.yml`**（项目根）：

```yaml
version: "3.9"
services:
  worker:
    build: .
    container_name: hga-worker
    restart: always
    env_file: .env
    environment:
      - TZ=Asia/Shanghai
    volumes:
      - ./data:/app/data                         # SQLite + JSON + WeChat 缓存
      - ./data/.hf_cache:/root/.cache/huggingface  # bge-m3 / reranker 模型缓存
      - ./knowledge_base:/app/knowledge_base     # 知识库 + .index_cache/
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
    environment:
      - TZ=Asia/Shanghai
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
    environment:
      - TZ=Asia/Shanghai
    volumes:
      - ./data:/app/data
      - ./knowledge_base:/app/knowledge_base
    command: ["python", "scripts/backup_loop.py"]   # 内部跑每日定时
```

> **健康检查**用 `python -c` 简单调 `_ensure_init` 验证 SQLite 可写；不引入额外 healthz HTTP 服务避免占端口。

**`.env` 路径全部改为 `data/` 卷下**（让数据持久化跨容器重启）：

- `SQLITE_DB_PATH=/app/data/checkpoints.db`
- `HEALTH_LOGS_DB_PATH=/app/data/health_logs.db`
- `OBSERVABILITY_DB_PATH=/app/data/observability.db`
- `PROFILE_STORE_PATH=/app/data/profile_store.json`
- `EPISODE_STORE_PATH=/app/data/episode_store.json`
- `SESSION_STORE_PATH=/app/data/session_store.json`
- `KNOWLEDGE_BASE_DIR=/app/knowledge_base`（保持默认即可，但确认 `.index_cache/` 路径在 volume 内）

#### 9.3 数据备份（Wave 5 一并出）

`scripts/backup_loop.py`：每天凌晨 3 点（用 `apscheduler` 或简单 `time.sleep` 循环）执行：

1. `sqlite3 /app/data/health_logs.db ".backup /tmp/health_logs.YYYYMMDD.db"`（在线备份，不锁库）
1. 同理打包其他 `*.db` 和 `*.json`
1. **不备份 `.index_cache/` 和 `.hf_cache/`**:这些是可重生的衍生物，恢复时让程序自己重建即可（节省备份带宽 + 存储）
1. 用 `oss2`（阿里云 OSS）或 `cos-python-sdk-v5`（腾讯云 COS）上传到对象存储 bucket，按日期路径：`health-guide-backup/{date}/{file}.gz`
1. 保留策略：最近 14 天 + 每月 1 号永久。
1. 失败时通过 dispatcher 主动推送告警到自己微信（“备份失败：…”）。

**凭证**：`OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET` / `OSS_BUCKET` / `OSS_ENDPOINT`（阿里云）或对应腾讯云 COS 变量；都进 `.env`。

成本估算：每天 ~15MB（含 evaluation reports）× 30 天 = 450MB；标准存储 ¥0.12/GB/月，约 ¥0.05/月。**忽略不计**。

> **新增备份内容**:`reports/output_eval_report.json` 和 `reports/rag_eval_report.json` 也建议一并备份。这些是版本化的评测快照，对于求职复盘”每个 Wave 完成时的指标”有面试价值。

#### 9.4 部署运维

`deploy/README.md` 文档化：
0. **学生认证**:阿里云”高校计划”已认证，3 个月免费档已生效（到期约 8/19）

1. 装 Docker / Compose（一行 `curl get.docker.com | sh`）
1. `git clone` 项目 → 填 `.env`
1. `docker compose up -d --build`
1. `docker exec -it hga-worker python scripts/wechat_login.py`（首次扫码绑定，写 `bot_token` 入 .env，重启 worker 生效）
1. `docker compose logs -f worker dispatcher` 看流水
1. 升级流程：`git pull && docker compose up -d --build`，SQLite volume 不丢
1. **学生套餐到期前 2 周（8/5）**:做完整迁移演练 → 要么续费 ¥9.5/月 转付费 / 重新认证（如还在校），要么把数据从对象存储恢复到新服务器。`.env` 里所有路径和凭证可变量化，搬家无需改代码。

`deploy/systemd/*.service`（**备选**，给不想用 Docker 的人）：两个 service 文件（worker / dispatcher）+ Restart=always，文档化但不强推。

### 10. 实施顺序（针对最新项目状态调整）

读完 `state.py / graph.py / tools.py / turn_start.py / nutritionist.py / trainer.py / wellness.py / planner.py / critic.py / config.py / llm.py / main.py` 后确认：项目用的是 `RESET_SENTINEL` reducer 模式、`@tool` 装饰器统一注册、SQLite + WAL 的本地持久化风格、`extract_text_content` 已支持 list-content（image part 不会 crash）、RAG 已升级为 bge-m3 + bge-reranker-v2-m3 两阶段。新增模块全部沿用现有约定。

**按依赖关系分 4 个 Wave 提交，每个 Wave 一个 commit。每个 Wave 完成必须跑评测脚本作为验收**:

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
  - **验收**:
    - 全部 4 个 smoke 脚本（`smoke_critic_scratchpad / smoke_plan_execute / smoke_dynamic_replan / smoke_coreference`）通过
    - `python scripts/evaluate_output.py --no-judge` 确认 routing accuracy ≥ 93.3%（baseline）
    - `python scripts/evaluate_output.py` 跑完整 30 条 + judge，确认 overall avg ≥ 4.5（baseline 4.59）
    - 任何掉点用 `--rerun-bad` 定位 + 修 Critic 提示词
- **Wave 2 — 提醒推送**（依赖 Wave 1 的 SQLite 基础）
  - `integrations/push_reminder.py`（写 reminders 表）
  - `scripts/reminder_dispatcher.py`（cron 入口；启动时若无 wechat 客户端则只打日志）
  - **验收**:本地手动写一条 reminder（remind_at = now + 2min），观察 dispatcher 准点打印 “would push to wxid=…”（此时还没接入微信，只验证 cron 调度逻辑）
- **Wave 3 — 微信 iLink 接入**
  - `integrations/wechat_ilink.py`（HTTP/JSON 客户端，含 graceful 降级）
  - `scripts/wechat_login.py`（扫码绑定）
  - `scripts/wechat_ilink_worker.py`（长轮询 main loop）
  - `main.py` 改造为 `--mode cli|wechat` 双入口
  - `requirements.txt` 加 `pillow`, `requests`
  - **验收**:
    - 本地 `python scripts/wechat_login.py` 扫码成功
    - 本地 worker 收到自己发的消息并回复
    - **重跑 `python scripts/evaluate_output.py`,确认 CLI 模式行为无回归**
- **Wave 4 — 容器化 + 云部署**
  - `Dockerfile` + `docker-compose.yml`（worker + dispatcher + backup 三服务，共享 ./data volume + ./knowledge_base volume）
  - `scripts/backup_loop.py`（每日凌晨打包 SQLite + JSON + reports 上传对象存储；失败告警走 dispatcher 推微信）
  - `deploy/README.md`（买服务器、装 Docker、首次扫码、升级流程的傻瓜文档）
  - `deploy/systemd/*.service`（**备选**，给不想用 Docker 的人）
  - `.env.example` 加云部署相关变量（`HEALTH_LOGS_DB_PATH=/app/data/...`、`OSS_*`、`BACKUP_RETENTION_DAYS=14` 等）
  - **本地** `docker compose up` 跑通现有 CLI 测试套件 + smoke 脚本，证明 Docker 化无回归
  - **真在云上**跑一遍 demo 流程；让朋友扫码加 bot 任意时间发消息验证 7×24 可用
  - **验收**:
    - 容器内 `python scripts/evaluate_output.py --no-judge` 通过（验证容器环境与本地等价）
    - 备份脚本第 2 天 OSS bucket 出现昨日备份
    - **8/5 之前**做一次”从 OSS 恢复到新机器”演练
- **Wave 5 — 收尾**
  - `README.md` 重写（架构图含云边界 + demo 流程 + 一键部署 quick start）
  - 简历项目描述定稿
  - （可选）录 60s demo 视频
  - **`eval/output_eval_dataset.jsonl` 追加 5-10 条新场景样本**（图片输入类、actuation 类、Analyst 复盘类），最终评测报告作为简历指标证据

每 Wave 完成后 commit 并 push，再开下一 Wave。Stretch（Analyst / iCloud / morning briefing / 群聊 / 姿势识别）作为后续独立 PR 处理。

### 11. MVP 切片 vs Stretch

**MVP（必做，1-2 周可完成）**：

- AgentState 5 个新字段
- MultiModalPreprocessor 节点 + Vision meal 抽取（不做 form）
- `wechat_ilink.py` 客户端 + `wechat_login.py` 扫码绑定 + `wechat_ilink_worker.py` 长轮询入口（核心 plumbing，**最关键**）
- `local_logs.py` 三个写工具 + `query_logs`（SQLite，零外部依赖）
- `push_reminder.py` + `reminder_dispatcher.py` cron
- Critic P1 actuation 真实性 + P2 vision 置信度
- **Dockerfile + docker-compose.yml + backup 容器 + 一台国内轻量云上跑通 7×24**（不上云就少了”产品”层级）
- 重写 README，加 demo 视频（手机微信屏幕录制）
- **保持 `evaluate_output.py` 端到端评分 ≥ 4.5（不退化）**

**Stretch（出 MVP 后再加）**：

- 新 Analyst 专家
- TurnStart 7 日摘要回灌
- Critic 训练负荷冲突 + 蛋白缺口规则
- `apple_calendar.py` + iCloud CalDAV 同步
- `daily_morning_briefing.py` 早安主动推送
- 群聊支持（拉 bot 进健身打卡群，群成员 @bot 都能聊）
- 运动姿势照片识别
- **A/B 对比国产 embedding 模型**(用现有 `scripts/compare_embedders.py`)，例如 `bge-m3` vs 通义千问 text-embedding-v3，证明对国内语料的选型考量

-----

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
- `eval/output_eval_dataset.jsonl` — 追加 Analyst / actuation / multimodal 类样本

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
- `scripts/restore_backup.py`（从对象存储恢复 + 用于迁移演练）
- `scripts/daily_morning_briefing.py`（stretch）
- `scripts/setup_icloud_caldav.py`（stretch）
- `migrations/001_create_health_logs.sql`（SQLite schema）
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `.env.example`（含云部署变量模板）
- `deploy/README.md`（从 0 到上线的傻瓜部署文档）
- `deploy/MIGRATION.md`（学生认证到期前的迁移 playbook）
- `deploy/systemd/hga-worker.service`（备选）
- `deploy/systemd/hga-dispatcher.service`（备选）

可复用的现有实现：

- `state.py` 的 `_turn_dict / _turn_list / _take_last_str` reducer + `RESET_SENTINEL` 模式
- `tools.py` 的 `@tool` 装饰器 + 模块级单例缓存
- `turn_start_node` 现有 `episode_context` 加载流程（直接 mirror 它写 `recent_logs_summary`）
- `critic.py` 现有 P0/P1/P2 评分框架（追加规则即可，不需要重写）
- `agents/*` 专家的 ReAct loop（直接复用，只是各自 tools 列表加新工具）
- `observability.py` 的 SQLite metrics（加新列：`actuation_count`, `vision_calls`, `wechat_msgs_in/out`）
- 现有 SQLite 风格（`checkpoints.db` / `observability.db` 共用 sqlite3 + WAL，新增 `health_logs.db` 同款）
- **现有 RAG 索引缓存机制**(`knowledge_base/<namespace>/.index_cache/`)——Docker volume mount 直接复用，无需改 RAG 代码
- **现有评测脚本**(`scripts/evaluate_output.py` / `scripts/evaluate_rag.py`)——每个 Wave 验收直接复用

-----

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
cp .env.example .env && vim .env                  # 填 LLM / 微信 / OSS 凭证
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

1. **单轮多模态**：在微信和 bot 私聊发餐盘照 + “这餐够吗？周四晚 7 点排腿训”，确认：
- bot 文字回复出现在微信
- `health_logs.db` `meals` + `workouts` 表各出现新行（含 idempotency_key）
- `health_logs.db` `reminders` 表出现晚 8 点条目（delivered=0）
1. **真实定时主动推送**：把电脑时间调到 19:59 等 1 分钟，确认 bot 在微信主动发”该补蛋白”消息；reminders 表 delivered=1。
1. **跨轮闭环**：第二天发 “我昨天吃得怎么样？”，确认 Analyst 拉出昨天写入的真实日志，输出数字诊断。
1. **幂等回归**：worker kill 后重启，旧消息重投不会重复写 SQLite（UNIQUE 约束起作用）。
1. **(stretch) iCloud 同步**：iPhone 日历”健康助手”子日历出现周四晚训练事件。

### 检查清单

- [ ] 微信扫码绑定成功，`getupdates` 长轮询正常收消息
- [ ] `health_logs.db` 三张表实际出现新行（含 idempotency_key 列）
- [ ] reminders cron 准时推送，微信端收到主动消息
- [ ] 同一 idempotency_key 重试不重复写
- [ ] Critic actuation 真实性规则：人为篡改 actuation_log 为空、保留草稿声明 → 必须 REVISE
- [ ] Vision 失败注入（断网 / 假图）：流程降级到 “请用文字描述”，不 crash
- [ ] `observability.db` 出现 `actuation_count` / `vision_calls` / `wechat_msgs_in/out` 新列指标
- [ ] **端到端评测无回归**:`python scripts/evaluate_output.py` 跑完，overall ≥ 4.5、routing accuracy ≥ 93.3%、所有 safety / personalization 维度不低于 baseline
- [ ] **RAG 评测无回归**:`python scripts/evaluate_rag.py` 跑完，Stage-1 Recall@10 = 100%、Stage-2 MRR ≥ 0.96（baseline 0.9677）
- [ ] **云端 7×24 验证**:本机关机 24h 后，让朋友在任意时间发消息，bot 仍正常回复
- [ ] **容器自愈**:`docker kill hga-worker` 后 `restart: always` 自动起来，未读消息不丢（iLink 服务端保留 + 重启后 `getupdates` 拿到）
- [ ] **数据持久化**:`docker compose down && docker compose up -d` 后 SQLite 数据不丢，且 `.index_cache/` 不重建
- [ ] **备份验证**:第二天看对象存储 bucket 出现昨日的备份文件；手动 `python scripts/restore_backup.py <date>` 能恢复到一个临时目录
- [ ] **8/5 之前完成迁移演练**:从 OSS 备份恢复到一台新机器（或本地 Docker），跑通端到端验证 1-3
- [ ] (stretch) iPhone 日历出现新事件

### 简历层面验证

- README 顶部一张系统拓扑图，红色高亮 MultiModalPreprocessor + 微信 iLink + 本地日志 + cron 主动推送
- 一段 60 秒 demo 视频：**手机微信屏幕录制**——
  - 0:00 微信里加 bot 好友 / 进入对话
  - 0:05 发餐盘照 + 排训意图
  - 0:15 bot 回复（含分析 + 训练已记录 + 提醒已设置）
  - 0:30 跳到 18:00（剪辑），bot 主动推”该补蛋白”
  - 0:45 第二天发”我昨天吃得怎么样” → bot 用真实数据复盘
- 简历项目描述加一行硬指标：
  - “接入微信 ClawBot iLink 协议实现真实 IM bot，Docker Compose 部署到阿里云 ECS 实现 **7×24 在线**；新增多模态食物识别 + 本地结构化日志 + 主动定时推送闭环；SQLite 数据卷定时备份到 OSS 并**实测完成跨机器迁移演练**;agent side-effect 流水可审计、幂等保证 100%”
  - “端到端评测体系完整：506 条 RAG 召回测试（首位命中 94.3%、MRR 0.9677）、30 条 8 类场景输出质量测试（overall 4.59/5.00）、A/B embedding 模型对比脚本”
- 一句锦上添花的”工程性价比”亮点：
  - “全栈低成本搭建:阿里云学生认证免费档 2C4G + OSS 备份 ¥0.05/月；同等线上 always-on 商业方案月均 ¥30-60，体现资源运用与成本意识”

-----

## 关键 gotchas

- **微信 ClawBot 是 2026 年 3 月 22 日新功能**，文档可能更新；以官方插件 / npm `@tencent-weixin` 包为准，社区教程可能落后或不准。集成第一步先跑通官方 “扫码 + getupdates + sendmessage” hello world，再往项目里搬。
- **iLink 协议是纯 HTTP/JSON**：长轮询 `getupdates` 超时建议 25-30s，比 WebSocket 简单，但要处理空响应、网络抖动。worker 必须支持 graceful restart；状态都落到 SQLite，重启不丢上下文。
- **bot_token 是一次性扫码后持久凭证**：必须 `.env` + gitignore，丢了要重新扫码。建议加密落 keyring。
- **图片消息只给 media_id**：iLink 回调的图片消息要先调 `download_media` 拿 bytes 再喂 Vision，封装重试。
- **群聊支持**：iLink 区分私聊 / 群聊；MVP 只做私聊，stretch 才支持群（需处理 @机器人 解析、群成员 wxid 映射）。
- **主动推送限流**：iLink 主动推送有频率限制（避免骚扰），cron 推送时控制单用户单日 ≤ 3-5 条；reminders 表加 `priority` 字段防刷屏。
- **协议升级风险**：因为是新产品，未来 3-6 个月 iLink 协议可能 breaking change；`wechat_ilink.py` 客户端做好版本声明 + 兼容层。简历可以把”对接最新发布的官方协议”作为亮点写出来。
- **跨时区**：reminders 用 UTC + tzid 存；提醒发出前用 `pytz` 转用户本地时间（默认 Asia/Shanghai）。
- **隐私自检**：SQLite 都在你自己的云服务器 + 自己的对象存储 bucket 里，agent 不向第三方上报（除 LLM / Vision API 调用外）。README 写明数据边界 + .env 模板说明权限。

### 评测体系相关 gotchas

- **新规则可能误伤现有评测样本**:Critic 的 P2 数据驱动蛋白缺口规则在历史样本（无 `recent_logs_summary`）下应跳过而非触发。这是最容易让评分掉点的地方。**Wave 1 必跑 `evaluate_output.py` 验证**。
- **Analyst 加入后必须扩充评测集**:否则 Analyst 没被路由命中也不会掉分（评测集没有这类样本），路由错误被掩盖。
- **RAG 评测要在每次知识库变更后重跑**:`scripts/evaluate_rag.py` + 自动生成的 506 条评测集是项目的硬资产，简历价值高，但也要持续维护——知识库新增文档后 chunk_id 会变，老评测集的 ground truth 失效。可以用 `scripts/generate_eval_dataset.py` 增量重新生成。
- **Wave 4 容器化后必须容器内重跑评测**:验证 Docker 环境与本地 conda 环境无差异。曾经踩过的坑包括 numpy 版本差异导致 FAISS 索引微小漂移、tokenizer 版本差异导致 chunk 边界不一致。

### 云部署额外 gotchas

- **服务器选址**:必须**国内**(已确定阿里云杭州区，✓）
- **境外 LLM 直连**:如果你 LLM 是 OpenAI / Anthropic 直连，国内服务器需要走代理（HTTP_PROXY 环境变量）；推荐切到国内 OpenAI 兼容服务（DeepSeek / 通义 / 智谱）避免代理
- **bot_token 持久化**:扫码后写入 `.env`（位于宿主机 `./` 目录），volume mount 到容器；千万别只写容器内 `/tmp`，否则容器重建就丢。
- **节省停机模式 IP 问题**:阿里云节省停机模式下公网 IP 可能变化。**正式上线时切回普通停机或常开模式**，并绑定弹性公网 IP（EIP），避免 IP 漂移触发微信 iLink 风控。
- **重启不丢消息**:iLink 服务端会保留未拉取的更新（offset 概念），worker 重启后 `getupdates(offset=last_seen)` 会重新拿到；`last_seen_offset` 必须落 SQLite 持久化、不能只在内存。
- **同一 bot_token 同时只能一个 worker**:iLink 不支持多实例同 token 并发拉取，会互相抢消息。Docker Compose 配置确保 `worker` 服务 replicas=1；不要无脑横向扩容。
- **时区**:容器默认 UTC，会导致 cron / 日志时间和你本地差 8h。compose 加 `environment: TZ=Asia/Shanghai` 或 Dockerfile 装 `tzdata`。
- **磁盘膨胀**:`checkpoints.db` 会随对话累积；`observability.db` 同理。Wave 4 备份脚本顺手做 90 天前 checkpoint 软清理（保留摘要、删原文 messages），避免几个月后撑爆 40GB 系统盘。
- **secret 不要进 git**:`.env` 必须在 `.gitignore`；`.env.example` 留模板。对象存储 access key 走子账号 + 只授权对应 bucket 的写权限，最小权限原则。
- **流量费**:Lighthouse 套餐内含 4M 带宽 + 一般 500GB 流量/月；个人 bot 用不到 1GB。无需担心。
- **bge-m3 模型下载**:首次启动从 hf-mirror 下载约 2.5GB；网络好的话 5 分钟内完成，挂 volume 后后续启动复用。
- **备份恢复演练**:上线第二周一定演练一次”从对象存储拉昨日备份还原到一个新容器”，否则备份等于没做。`scripts/restore_backup.py` 必须随 backup 脚本同时交付。
- **学生认证有效期**:阿里云免费档 3 个月（约到 2026/8/19）。**8/5 之前**做完整迁移演练；演练通过才能心里有底。`docker-compose.yml` + `.env` 抽象到位的话，迁移只需 `git clone + cp .env + 还原最新备份 + docker compose up -d`，半小时搞定。
- **不要选 GCP / 不要跨太平洋**:`ilinkai.weixin.qq.com` 是国内域名，worker 长轮询要常态保持。海外服务器面临 RTT 高 + 跨境流量计费 + GFW 抖动三重劣势，免费档省的钱可能被流量账单反咬一口。