# Health Guide Agent

Health Guide Agent 是一个面向个人健康管理的 LangGraph 多 Agent 系统。它不是“把问题丢给一个 LLM 回答”的聊天 Demo，而是把多专家协作、长期记忆、RAG 证据检索、通用多模态图片理解、结构化日志、真实提醒、Apple Calendar 日程写入、微信入口、备份和 Docker 常驻部署串成一个可实际运行的健康助手。

一句话面试版：

> 这是一个有真实 side effect 的健康管理 Agent。它能在 CLI 或微信里理解文本/图片问题，按需调训练、营养、心理、医学、日志分析等专家，读写长期画像和健康日志，设置微信提醒或写入 Apple Calendar，并用 Critic 校验安全性、个性化落地和“有没有真的执行工具”。

## 技术亮点与工程深度

这一节放在最前面，方便面试或简历项目介绍时直接讲重点。

### 1. 从“聊天机器人”推进到“可执行 Agent 系统”

普通健康问答只能给建议，这个项目把建议落到真实执行面：

- `log_meal`、`log_workout`、`log_wellness_checkin` 写入 SQLite 健康日志。
- `push_reminder` 写入 durable reminder 队列，由独立 dispatcher 到点主动推送微信。
- `schedule_workout` / `schedule_calendar_event` 通过 iCloud CalDAV 写入 Apple Calendar。
- 所有真实 side effect 都返回统一的 `[ACTUATION]` JSON 流水。
- Aggregator 和 Critic 只有在 `actuation_log` 中看到 `ok=true` 的对应流水时，才允许最终回答说“已记录 / 已设提醒 / 已加入日历”。

这个设计解决了 Agent 常见的“嘴上说做了，实际没做”的问题。LLM 的语言承诺必须和工具执行结果一致。

### 2. 真正的父 Agent 调子 Agent，而不是静态路由表

Orchestrator 是父 Agent，它把子专家封装成 LangChain tools：

- `consult_analyst`
- `consult_trainer`
- `consult_nutritionist`
- `consult_psychologist`
- `consult_doctor`

父 Agent 可以在同一个 tool-use loop 里连续调用多个专家，也可以对简单问题直接回答。代码里还保留 deterministic guards：身体不适、医疗边界、心理危机、提醒/日历、日志复盘等高确定性场景会直接触发对应专家，降低纯 LLM 路由漂移。

面试可讲的点：这个项目不是“Planner 输出专家名，然后 Dispatcher 硬执行”的模板，而是让父 Agent 在真实工具调用循环里完成任务分解、信息收集和结果整合。

### 3. 面向健康场景的多专家职责隔离

健康问题天然跨域，例如“这餐够不够支撑今晚腿训，晚上提醒我补蛋白”至少涉及：

- 餐食营养估算：Nutritionist。
- 训练负荷和恢复：Trainer。
- 真实提醒写入：push_reminder。
- 历史日志复盘：Analyst。
- 最终安全审核：Critic。

项目把不同专家的工具权限、RAG namespace、提示词边界和安全职责拆开：

- Trainer 负责训练、运动动作、TDEE/BMR、比赛训练、伤病/康复期负荷。
- Nutritionist 负责饮食、热量、蛋白质、补剂、食谱和餐食记录。
- Psychologist 负责压力、睡眠、焦虑、动力下降、心理危机边界。
- Doctor 负责一般医学建议、症状风险分层、就医建议、用药/处方边界。
- Analyst 只读结构化日志，输出趋势和复盘。

这种职责隔离让复杂 prompt 变成多个可控子系统，降低单 Agent 失控概率。

### 4. 子 Agent 输入隔离与 scratchpad 汇总

子 Agent 不直接读取完整历史，也不互相读取原始工具 trace。Orchestrator 给每个专家构造裁剪后的上下文：

- 本轮独立问题：来自 `QueryRewriter` 的 `contextualized_query`。
- 用户画像快照：来自 `TurnStart` 构造的 `personalization_ctx`。
- 情节记忆：来自 `episode_context`。
- 近期日志摘要：来自 `recent_logs_summary`。
- 图片 grounding：来自 `vision_extractions`。
- 同伴要点：通过 `format_peer_notes` 传递 scratchpad。

这样减少 token 成本，也防止某个专家看到不该看的领域信息。Aggregator 和 Critic 看到的是结构化专家结果和 scratchpad，而不是一堆混乱 tool trace。

### 5. LangGraph 持久化状态的“轮边界清理”

LangGraph checkpoint 会跨进程保存完整状态，这很强，但也容易出 bug：如果 reducer 只是 append 或 merge，上轮的 `agent_notes`、`expert_responses`、`actuation_log` 会残留到下一轮，导致 Aggregator/Critic 误判。

项目在 `health_guide/state.py` 里实现了 turn-scoped reducer：

- `_turn_dict`：支持 `{ "__RESET__": true }` 清空本轮 dict。
- `_turn_list`：支持 list 首项 `RESET_SENTINEL` 清空本轮 list。
- `_turn_int`：支持 `("__RESET__", 0)` 重置计数器。

`TurnStart` 每轮清空：

- `agent_notes`
- `expert_responses`
- `last_tools`
- `image_inputs`
- `vision_extractions`
- `actuation_log`
- `retrieval_hits`
- `plan / executed / next / replan_count`
- `draft_answer / critic_verdict / replan_context`

这是长会话 Agent 工程里非常实际的坑：不做轮边界清理，系统会“带着上一轮的幻觉继续判断”。

### 6. 个性化不是“贴画像”，而是可检查的决策点

项目不会只把 profile 粘进 prompt，而是生成 personalization decision points，让专家、Aggregator 和 Critic 都围绕这些点检查是否落地：

- 年龄、身高、体重：用于 TDEE、心率区间、蛋白质 g/kg、热量缺口等推导。
- 伤病、术后、康复期：用于动作禁忌、替代动作、进阶门槛和就医边界。
- 饮食偏好、过敏、乳糖不耐：用于食材替换和风险提醒。
- 压力源、睡眠状态：用于睡前流程、压力管理和心理支持。

Critic 会检查回答是否真正把画像转成了方案差异，而不是只说“根据你的情况”。

### 7. RAG 是本地两阶段检索，不依赖云向量数据库

`health_guide/rag.py` 实现了本地知识库：

- 支持 `.md`、`.txt`、`.pdf`、`.docx`。
- 每个专家独立 namespace：trainer、nutritionist、psychologist、doctor、safety。
- Stage 1：`BAAI/bge-m3` dense retrieval + FAISS。
- Stage 2：`BAAI/bge-reranker-v2-m3` cross-encoder rerank。
- 索引缓存包含 chunks、meta、FAISS index 和 fingerprint。
- fingerprint 绑定文档内容、chunk 参数、embedding 模型，避免错误复用旧索引。
- RAG 工具是 on-demand 的，专家决定是否检索；寒暄、纯记录和低风险直接回答不会乱打检索。

这套设计体现的是可控、可离线、可回归测试的 RAG，而不是单纯调用外部 search。

### 8. 通用多模态 grounding，而不是只识别餐盘

图片不会在一进图时就强行调用 VLM。流程是：

1. `MultiModalPreprocessor` 只提取本轮图片输入，写入 `image_inputs`。
2. Orchestrator 判断是否需要看图。
3. 如需要，调用 `multimodal_processor` 工具，让 VLM 根据用户 query 生成面向问题的文字 grounding。

它可以处理：

- 餐食图片：估算食物、热量、蛋白、碳水、脂肪和置信度。
- 训练动作/姿势：交给 Trainer 分析动作和风险。
- 身体部位、伤口、皮疹、化验单、药品包装：交给 Doctor 做风险边界。
- 配料表、营养成分表：交给 Nutritionist 解释。

Vision 置信度会进入 Critic：当 `confidence < 0.5`，最终回答不能用确定语气输出精确热量或宏量营养素。

### 9. 微信入口不是简单转发，而是碎片输入聚合

微信用户经常先发图，再发一句“帮我看看”；或者把一句话拆成多条。项目在两层做了处理：

- `scripts/wechat_ilink_worker.py` 有 inbox FIFO，先把微信消息持久化到 SQLite。
- `InputAccumulator` 在 LangGraph 内缓存碎片输入，等问题/命令完整后再进入正式流程。

因此：

- 单独发图不会立刻浪费 VLM 调用。
- “图 + 说明”可以合并成一轮完整 HumanMessage。
- `/bind <user_id>` 可把微信 wxid 绑定到项目内部用户 ID，复用已有 profile/memory/logs。
- worker offset、context_token 等轻量状态写入 `kv` 表，支持重启恢复。

### 10. 部署是常驻服务，而不是 notebook demo

Docker Compose 提供三个长期运行服务：

- `worker`：微信长轮询，下载图片，调用 graph，回复用户。
- `dispatcher`：扫描 `reminders`，到点主动推送。
- `backup`：周期性备份 SQLite/JSON/reports，可选上传 OSS。

持久化策略：

- `./data` 保存 checkpoints、health logs、profile、episode、session、backups。
- `./data/.hf_cache` 保存 HuggingFace 模型缓存。
- `./knowledge_base` bind mount，服务器上修改知识库不用重建镜像。
- `.dockerignore` 排除 `.env`、SQLite、WAL/SHM、日志、缓存、tmp、备份，避免本地运行态进入镜像。

### 11. 可观测性、评测和回归

项目保留了多层评测脚本：

- `scripts/evaluate_output.py`：端到端输出评测，支持 deterministic assertions 和 LLM-as-Judge。
- `scripts/evaluate_rag.py`：RAG 两阶段检索评测。
- `scripts/evaluate_architecture.py`：架构专项评测。
- smoke tests：coreference、dynamic replan、critic scratchpad、plan execute 等。

这让项目可以讲清楚“怎么证明系统没退化”，而不是只靠主观 demo。

## 开箱使用指引

下面按“最小可跑 -> 可选能力 -> Docker/ECS”组织。

### 0. 准备环境

推荐：

- Python 3.10+，本地开发建议用 conda。
- Docker Compose v2，用于服务器部署。
- 出站 HTTPS 网络，用于 LLM、WeChat iLink、iCloud CalDAV、HuggingFace/model mirror、可选 OSS。

克隆后先复制配置：

```bash
cp .env.example .env
```

`.env` 包含密钥，请不要提交到 Git。

### 1. 本地最小文本版

只需要 LLM，就能跑 CLI 纯文本健康咨询：

```bash
conda env create -f environment.yml
conda activate hga
pip install -r requirements.txt
```

编辑 `.env`，至少填写：

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_api_key
LLM_MODEL=gpt-5.5-mini
LLM_API_MODE=responses
```

如果服务商只兼容 Chat Completions：

```env
LLM_API_MODE=chat_completions
```

初始化 SQLite 健康日志：

```bash
python -c "from health_guide.integrations.local_logs import init_db; init_db()"
```

启动 CLI：

```bash
python main.py --mode cli --detail
```

`--detail` 会打印专家调用和工具使用，适合调试和面试演示。

### 2. 可选：给 Orchestrator 单独配置更强模型

默认 Orchestrator 继承 `LLM_*`。如果想让父 Agent 用更强模型做工具决策：

```env
ORCHESTRATOR_LLM_BASE_URL=https://api.openai.com/v1
ORCHESTRATOR_LLM_API_KEY=your_api_key
ORCHESTRATOR_LLM_MODEL=gpt-5.5
ORCHESTRATOR_LLM_API_MODE=responses
ORCHESTRATOR_LLM_OUTPUT_VERSION=responses/v1
```

### 3. 可选：启用多模态图片理解

```env
MULTIMODAL_LLM_ENABLED=true
MULTIMODAL_LLM_PROVIDER=openai
MULTIMODAL_LLM_BASE_URL=https://api.openai.com/v1
MULTIMODAL_LLM_API_KEY=your_api_key
MULTIMODAL_LLM_MODEL=gpt-4o-mini
MULTIMODAL_LLM_API_MODE=chat_completions
```

未配置时，图片能力会安全降级，不影响纯文本运行。

### 4. 可选：启用 Apple Calendar / iCloud CalDAV

Apple Calendar 使用 Apple ID 的 App 专用密码，不要填写 Apple ID 主密码。生成入口：

```text
https://account.apple.com/account/manage
```

`.env` 示例：

```env
ICLOUD_USERNAME=your_apple_id@example.com
ICLOUD_APP_SPECIFIC_PASSWORD=xxxx-xxxx-xxxx-xxxx
ICLOUD_CALDAV_URL=https://caldav.icloud.com
ICLOUD_CALENDAR_NAME=Calendar
```

验证连接和日历选择：

```bash
python scripts/setup_icloud_caldav.py
```

可用后，用户说“帮我把明晚 7 点跑步加入 Apple Calendar”，Trainer 会调用 `schedule_workout` 真正写入日历。

### 5. 可选：微信入口和主动提醒

首次登录：

```bash
python scripts/wechat_login.py --terminal-qr
```

启动 worker 和提醒 dispatcher：

```bash
python scripts/wechat_ilink_worker.py
python scripts/reminder_dispatcher.py
```

说明：

- `wechat_login.py` 会扫码获取 `WECHAT_BOT_TOKEN` 并写回 `.env`。
- worker 长轮询微信消息，调用 LangGraph，并回复用户。
- dispatcher 扫描 `reminders` 表，到点主动推送。
- 微信用户会自动绑定到稳定项目用户 ID，默认形如 `wechat_<hash>`。
- 如需手动绑定当前微信到已有项目用户，可发送 `/bind Michael`，或本地执行：

```bash
python scripts/wechat_bind_user.py --wxid '<user_wxid>' --user-id Michael
python scripts/wechat_bind_user.py --list
```

### 6. Docker Compose 本地或 ECS 部署

先准备目录和配置：

```bash
cp .env.example .env
vim .env
mkdir -p data logs reports tmp
chmod 600 .env
docker compose config --quiet
```

启动：

```bash
docker compose up -d --build
docker compose logs -f worker dispatcher backup
```

容器内首次微信扫码：

```bash
docker compose exec worker python scripts/wechat_login.py --env /app/.env --qr-path /app/tmp/wechat_qrcode.png --terminal-qr --no-open
docker compose restart worker
```

容器内 Apple Calendar 校验：

```bash
docker compose exec worker python scripts/setup_icloud_caldav.py
```

容器内 smoke checks：

```bash
docker compose exec worker python -c "from health_guide.integrations.local_logs import init_db; init_db(); print('db ok')"
docker compose exec worker python -c "from health_guide.graph import graph; print('graph ok')"
docker compose exec dispatcher python scripts/reminder_dispatcher.py --once
```

如果 ECS 区域访问默认国内镜像源不顺，可以覆盖 build args：

```bash
docker compose build \
  --build-arg PYTHON_BASE_IMAGE=python:3.11-slim \
  --build-arg DEBIAN_MIRROR=http://deb.debian.org/debian \
  --build-arg DEBIAN_SECURITY_MIRROR=http://deb.debian.org/debian-security \
  --build-arg PIP_INDEX_URL=https://pypi.org/simple
docker compose up -d
```

更详细部署和迁移见：

- `deploy/README.md`
- `deploy/MIGRATION.md`

## 核心架构

```mermaid
flowchart TD
    WX[微信文本/图片<br/>或 CLI 输入] --> Worker[scripts/wechat_ilink_worker.py<br/>长轮询 Worker]
    Worker --> IA[InputAccumulator<br/>微信碎片输入聚合]
    IA -->|READY| TS[TurnStart<br/>轮边界清理<br/>画像/情节记忆/近7日日志]
    IA -->|WAITING| END0[等待下一条输入<br/>不回复]
    TS --> QR[QueryRewriter<br/>多轮指代消解]
    QR --> MMP[MultiModalPreprocessor<br/>图片输入抽取]
    MMP --> ORC[Orchestrator<br/>父 Agent / 工具调用子 Agent]
    ORC -. tool call .-> VLM[multimodal_processor<br/>VLM 图片 grounding]

    subgraph ChildAgents[专业子 Agent]
      ANA[Analyst<br/>结构化日志复盘]
      TR[Trainer<br/>训练/运动/康复]
      NU[Nutritionist<br/>饮食/营养/补剂]
      PS[Psychologist<br/>压力/睡眠/情绪]
      DR[Doctor<br/>医学边界/就医建议]
    end

    ORC -. tool call .-> ANA
    ORC -. tool call .-> TR
    ORC -. tool call .-> NU
    ORC -. tool call .-> PS
    ORC -. tool call .-> DR

    ChildAgents --> AGG[Aggregator<br/>多专家融合]
    AGG --> CR[Critic<br/>安全 + 个性化 + Actuation + Vision 置信度]
    CR -->|PASS/REVISE| FINAL[最终回答]
    CR -->|REPLAN| ORC
    ORC -->|DIRECT| FINAL
    FINAL --> Worker
    Worker --> WX

    ChildAgents -. tools .-> Logs[(health_logs.db<br/>meals/workouts/wellness/reminders/kv/inbox)]
    Logs --> DISP[scripts/reminder_dispatcher.py<br/>到点主动推送]
    DISP --> WX
    ChildAgents -. tools .-> CAL[Apple Calendar<br/>iCloud CalDAV]
    Logs --> BK[scripts/backup_loop.py<br/>本地/OSS 备份]
```

## LangGraph 流程

入口定义在 `health_guide/graph.py`。

| 节点 | 作用 |
|---|---|
| `InputAccumulator` | 微信碎片输入聚合。CLI/评测直接 READY；微信可 WAITING。 |
| `TurnStart` | 清理本轮状态，加载 profile、episode、recent logs，必要时压缩长历史。 |
| `QueryRewriter` | 多轮指代消解，生成独立问题。 |
| `MultiModalPreprocessor` | 文本 no-op；图片轮只提取图片输入，不立刻调用 VLM。 |
| `Orchestrator` | 父 Agent，直接回答或调用子专家工具；也可调用 `multimodal_processor`。 |
| `Aggregator` | 多专家结果融合成统一口吻草稿。 |
| `Critic` | 安全、个性化、actuation、vision 置信度审核；可 PASS/REVISE/REPLAN。 |

Orchestrator 后的条件边：

- 如果是 direct answer，直接结束。
- 如果调用了子专家，进入 Aggregator。
- 如果已有草稿需要审核，进入 Critic。
- Critic 若写入 `replan_context`，回到 Orchestrator 补叫专家；否则结束。

## 关键模块

| 路径 | 作用 |
|---|---|
| `health_guide/graph.py` | LangGraph 拓扑、checkpoint、条件边。 |
| `health_guide/state.py` | AgentState 与 turn-scoped reducer。 |
| `health_guide/agents/input_accumulator.py` | 微信碎片输入聚合。 |
| `health_guide/agents/turn_start.py` | 轮边界清理、长历史摘要、画像/情节记忆/日志摘要加载。 |
| `health_guide/agents/query_rewriter.py` | 多轮问题改写。 |
| `health_guide/agents/multimodal_preprocessor.py` | 图片输入解析；VLM 调用由 Orchestrator 工具按需触发。 |
| `health_guide/agents/orchestrator.py` | 父 Agent、确定性安全 guard、子 Agent 工具封装。 |
| `health_guide/agents/analyst.py` | 读取健康日志，输出趋势与复盘。 |
| `health_guide/agents/trainer.py` | 训练、动作、运动恢复、TDEE/BMR、比赛训练。 |
| `health_guide/agents/nutritionist.py` | 饮食、营养、热量、蛋白质、补剂、食谱。 |
| `health_guide/agents/psychologist.py` | 压力、睡眠、情绪、动力、心理安全边界。 |
| `health_guide/agents/doctor.py` | 医学资料、症状风险、就医建议、用药边界。 |
| `health_guide/agents/aggregator.py` | 多专家回答融合。 |
| `health_guide/agents/critic.py` | 最终审核、REVISE、REPLAN、Actuation/Vision 规则。 |
| `health_guide/tools.py` | RAG 工具、画像工具、健康日志工具统一出口。 |
| `health_guide/rag.py` | 本地知识库、chunk、embedding、FAISS、rerank、缓存。 |
| `health_guide/integrations/local_logs.py` | SQLite 健康日志、提醒、微信 inbox、用户绑定。 |
| `health_guide/integrations/apple_calendar.py` | iCloud CalDAV / Apple Calendar 日程写入。 |
| `health_guide/integrations/vision.py` | OpenAI-compatible VLM helper。 |
| `health_guide/integrations/wechat_ilink.py` | 微信 iLink / ClawBot 风格 HTTP client。 |
| `scripts/wechat_ilink_worker.py` | 微信长轮询 worker。 |
| `scripts/reminder_dispatcher.py` | 到点提醒 dispatcher。 |
| `scripts/backup_loop.py` | SQLite/JSON/reports 备份，可选 OSS 上传。 |
| `docker-compose.yml` | worker / dispatcher / backup 常驻部署。 |

## 配置速查

| 配置 | 必填 | 说明 |
|---|---|---|
| `LLM_BASE_URL` | 是 | 默认文本 LLM 的 OpenAI-compatible endpoint。 |
| `LLM_API_KEY` | 是 | 默认文本 LLM key。 |
| `LLM_MODEL` | 是 | 默认文本模型。 |
| `LLM_API_MODE` | 建议 | `responses` 或 `chat_completions`。 |
| `ORCHESTRATOR_LLM_*` | 否 | 父 Agent 独立模型；留空继承 `LLM_*`。 |
| `MULTIMODAL_LLM_*` | 否 | VLM 图片 grounding。未配置可纯文本运行。 |
| `ICLOUD_*` | 否 | Apple Calendar / iCloud CalDAV。 |
| `WECHAT_*` | 否 | 微信入口和主动推送。 |
| `MCP_*` | 否 | 可选社区 MCP 工具服务器，默认关闭。 |
| `RAG_*` | 否 | 本地 embedding/rerank 配置。 |
| `OSS_*` | 否 | 备份上传对象存储。 |

Docker Compose 会覆盖持久化路径到 `/app/data`：

```env
SQLITE_DB_PATH=/app/data/checkpoints.db
HEALTH_LOGS_DB_PATH=/app/data/health_logs.db
OBSERVABILITY_DB_PATH=/app/data/observability.db
PROFILE_STORE_PATH=/app/data/profile_store.json
EPISODE_STORE_PATH=/app/data/episode_store.json
SESSION_STORE_PATH=/app/data/session_store.json
```

## 数据与记忆系统

### AgentState 关键字段

| 字段 | 生命周期 | 说明 |
|---|---|---|
| `messages` | checkpoint 持久 | LangGraph 消息历史，支持 `RemoveMessage` 摘要压缩。 |
| `contextualized_query` | 每轮覆盖 | QueryRewriter 输出的独立问题。 |
| `personalization_ctx` | 每轮覆盖 | 本轮统一用户画像快照。 |
| `episode_context` | 每轮覆盖 | 最近/语义相关对话摘要。 |
| `recent_logs_summary` | 每轮覆盖 | 近 7 日 SQLite 健康日志摘要。 |
| `pending_input_fragments` | 跨微信碎片 | 微信未完成输入的临时 buffer。 |
| `input_accumulator_status` | 每轮覆盖 | `WAITING` / `READY`。 |
| `image_inputs` | turn-scoped | 本轮图片列表。 |
| `vision_extractions` | turn-scoped | 本轮 VLM 图片 grounding / 餐食估算。 |
| `expert_responses` | turn-scoped | 子 Agent 回答。 |
| `agent_notes` | turn-scoped | 子 Agent scratchpad 要点。 |
| `actuation_log` | turn-scoped | 本轮真实 side effect 流水。 |
| `draft_answer` | 每轮覆盖 | Aggregator 或 Orchestrator 交给 Critic 的草稿。 |
| `critic_verdict` | 每轮覆盖 | PASS / REVISE / REPLAN / 规则命中原因。 |

### 三层记忆 + 一层行为日志

1. **Profile memory**：`profile_store.json`
   - 年龄、身高、体重、伤病、饮食偏好、压力源、回答风格。
   - 由 `set_physical_stats`、`add_injury`、`set_dietary_goal` 等工具更新。

2. **Episode memory**：`episode_store.json`
   - 每轮 query、experts、gist、facts。
   - 支持最近 N 轮召回和语义相似召回。

3. **Checkpoint memory**：`checkpoints.db`
   - LangGraph SqliteSaver 保存 thread 状态。
   - 支持跨进程恢复和长会话继续。

4. **Structured health logs**：`health_logs.db`
   - 餐食、训练、恢复/情绪、提醒、微信 inbox、用户绑定、kv。
   - 不混入 profile，避免长期画像和每日行为数据语义混乱。

### Health Logs Schema

主要表：

- `meals(id, user_id, date_iso, items_json, kcal, protein_g, carbs_g, fat_g, source, idempotency_key, created_at)`
- `workouts(id, user_id, date_iso, plan_json, status, idempotency_key, created_at)`
- `wellness(id, user_id, date_iso, sleep_h, mood, notes, idempotency_key, created_at)`
- `reminders(id, user_id, target_wxid, context_token, remind_at_iso, remind_at_epoch, text, priority, delivered, delivered_at, idempotency_key, created_at)`
- `kv(key, value, updated_at)`
- `wechat_inbox(update_id, user_wxid, context_token, chat_type, text, media_ids_json, raw_json, status, created_at, processed_at)`
- `wechat_user_bindings(wechat_wxid, project_user_id, display_name, created_at, updated_at)`

写入设计：

- mutating tool 都接受 `idempotency_key`。
- SQLite 使用 `UNIQUE(idempotency_key)` + `INSERT OR IGNORE`。
- 启用 WAL 和 `busy_timeout=5000`，适配 worker / dispatcher 同时访问。

## RAG 检索系统

知识库目录：

- `knowledge_base/trainer`
- `knowledge_base/nutritionist`
- `knowledge_base/psychologist`
- `knowledge_base/doctor`
- `knowledge_base/safety`

两阶段检索：

1. **文档读取**
   - 支持 `.md`、`.txt`、`.pdf`、`.docx`。
   - PDF 使用 `pypdf`，docx 提取段落和表格。

2. **Chunking**
   - 默认 `chunk_size=420`，`overlap=100`。
   - 中英文句末标点软边界，保留 source、chunk_id、PDF page_range。

3. **Dense Retrieval**
   - 默认 embedding：`BAAI/bge-m3`。
   - 默认 Top-K：`RAG_RETRIEVE_TOP_K=12`。
   - 使用 `faiss-cpu`。

4. **Cross-Encoder Rerank**
   - 默认 reranker：`BAAI/bge-reranker-v2-m3`。
   - 默认最终返回：`RAG_FINAL_TOP_K=4`。

5. **索引缓存**
   - 每个知识库目录维护 `.index_cache`。
   - 缓存 chunks/meta/FAISS/fingerprint。
   - fingerprint 绑定文档内容、chunk 参数和 embedding 模型。

仓库中保留的报告示例：

- `reports/rag_index_stats.json`
- `reports/rag_eval_report_small.json`
- `eval/rag_eval_dataset_v2.jsonl`

## 多模态与 Actuation

### 图片输入格式

`MultiModalPreprocessor` 支持：

- `image_url`
- `media_id`
- `image_bytes_b64`

微信 worker 会把图片下载为 bytes，再转成 `image_bytes_b64` 放入 HumanMessage content list。文本轮不调用 VLM；图片轮先收集 `image_inputs`，由 Orchestrator 决定是否调用 `multimodal_processor`。

### Vision 输出示例

```json
{
  "image_0": {
    "description": "图片中可见一份米饭、鸡肉和蔬菜，份量为视觉估算。",
    "query_focus": "用户询问这餐是否支持增肌，重点关注蛋白质来源和主食份量。",
    "health_relevance": "可作为营养师估算热量和宏量营养素的依据。",
    "uncertainty": "无法确认实际重量、烹调用油和隐藏调料。",
    "content_type": "meal",
    "confidence": 0.78
  },
  "meal": {
    "items": [{"name": "鸡胸肉", "estimated_amount": "约一掌心"}],
    "kcal": 720,
    "protein_g": 52,
    "carbs_g": 85,
    "fat_g": 18,
    "confidence": 0.78,
    "notes": "图片营养素为估算值。"
  }
}
```

### Actuation 返回格式

```text
[ACTUATION]{"ok":true,"action":"schedule_workout","table":"apple_calendar","uid":"...","start_iso":"2026-05-21T19:00:00+08:00",...}
Apple Calendar 日程已创建。
```

Critic 会读取 actuation log：

- 成功：允许最终回答“已记录 / 已设提醒 / 已加入日历”。
- 失败或没有工具流水：改成“可以记录 / 建议设置 / 需要先完成日历写入”。

## Docker / ECS 可移植性

Compose 服务：

| 服务 | 命令 | 作用 |
|---|---|---|
| `worker` | `python scripts/wechat_ilink_worker.py` | 接收微信消息、调用 graph、回复用户。 |
| `dispatcher` | `python scripts/reminder_dispatcher.py` | 扫描 reminders，到点主动推送。 |
| `backup` | `python scripts/backup_loop.py` | 定时备份 SQLite/JSON/reports，可选 OSS。 |

挂载：

```text
./.env:/app/.env
./data:/app/data
./data/.hf_cache:/root/.cache/huggingface
./knowledge_base:/app/knowledge_base
./logs:/app/logs
./reports:/app/reports
./tmp:/app/tmp
```

迁移到 Ubuntu ECS 时，通常只需要：

- 代码仓库。
- `.env`。
- `data/` 或 `data/backups/YYYYMMDD/`。
- 可选的 `reports/` 和 HuggingFace cache。

安全注意：

- `.env` 不进镜像，不进 Git。
- `.dockerignore` 排除了本地数据库、WAL/SHM、tmp 二维码、日志、缓存和备份。
- 应用不需要开放业务端口，只要出站 HTTPS 和 SSH 入站。

## 评测与回归

常用评测：

```bash
python scripts/evaluate_output.py --no-judge
python scripts/evaluate_output.py
python scripts/evaluate_rag.py --dataset eval/rag_eval_dataset_v2.jsonl
python scripts/evaluate_architecture.py
```

常用烟测：

```bash
python scripts/smoke_plan_execute.py
python scripts/smoke_dynamic_replan.py
python scripts/smoke_coreference.py
python scripts/smoke_critic_scratchpad.py
python scripts/smoke_mcp_tools.py
```

仓库保留的报告示例可用于面试说明回归方式：

- `reports/output_eval_report.json`
- `reports/rag_eval_report_small.json`
- `reports/architecture_eval_report.json`
- `reports/architecture_eval_round12_isolation_final.json`

说明：这些报告是仓库中的历史/阶段性回归产物；如果要在新服务器上展示最新结果，建议重新运行上述评测命令。

## 面试讲解速记

### 30 秒版本

我做的是一个可执行健康管理 Agent，而不是普通问答。它用 LangGraph 管理长会话状态，用父 Agent 调多个专家子 Agent，用本地 RAG 支撑专业知识，用 SQLite/Apple Calendar/微信提醒完成真实 side effect，再由 Critic 审核安全、个性化和工具执行真实性。项目能在微信里长期运行，支持图片、记忆、日志复盘、主动提醒和 Docker 部署。

### 2 分钟版本

这个项目的核心难点有三类。

第一是 Agent 架构。Orchestrator 不是静态路由器，而是父 Agent，把 Trainer、Nutritionist、Psychologist、Doctor、Analyst 封装成工具。它可以根据问题动态调用多个专家，也可以对寒暄、画像记录、医疗边界做 direct answer。

第二是状态和可信执行。LangGraph checkpoint 会跨轮保存状态，所以我实现了 turn-scoped reducer 和 TurnStart reset，避免上轮专家输出污染下一轮。所有真实写入工具返回 `[ACTUATION]` 流水，Critic 只有看到 `ok=true` 才允许最终回答声称“已记录 / 已设提醒 / 已加入日历”。

第三是产品闭环。系统不仅能回答，还能记录餐食/训练/睡眠，读取历史日志做复盘，写 Apple Calendar，微信主动推提醒，Docker Compose 常驻部署，backup 服务做数据备份。

### 面试官问“为什么不用一个 Agent？”

健康问题跨域明显，一个 Agent 很容易把训练、营养、心理和医学边界混在一起。多 Agent 的价值不是数量，而是隔离：

- 工具权限隔离。
- RAG namespace 隔离。
- 提示词职责隔离。
- 安全边界隔离。
- 输出审核集中化。

### 面试官问“最难的工程 bug 是什么？”

可以讲两个：

1. LangGraph reducer 状态残留。解决方式是在 `state.py` 做 reset-aware reducer，并在 `TurnStart` 统一清理 turn-scoped 字段。
2. LLM 声称执行但工具没执行。解决方式是 `[ACTUATION]` 协议 + `actuation_log` + Critic deterministic rewrite。

### 面试官问“怎么证明没有退化？”

可以讲：

- deterministic assertions：检查必须出现/禁止出现的内容。
- route assertions：检查该叫的专家是否被叫。
- RAG eval：MRR、Recall、Hit Rate。
- architecture eval：context isolation、RAG on-demand、parallel fanout、replan cap。
- smoke tests：覆盖 coreference、dynamic replan、critic scratchpad、MCP 工具等。

## 常用命令

```bash
# CLI
python main.py --mode cli --detail

# 初始化健康日志数据库
python -c "from health_guide.integrations.local_logs import init_db; init_db()"

# Apple Calendar 校验
python scripts/setup_icloud_caldav.py

# 微信登录
python scripts/wechat_login.py --terminal-qr

# 微信 worker
python scripts/wechat_ilink_worker.py

# 提醒 dispatcher
python scripts/reminder_dispatcher.py

# 端到端快速回归
python scripts/evaluate_output.py --no-judge

# Docker 配置校验
docker compose config --quiet

# Docker 启动
docker compose up -d --build

# Docker 日志
docker compose logs -f worker dispatcher backup
```

## 医疗安全说明

本项目是健康管理与信息整理助手，不是医疗器械，也不替代医生诊断、处方或急救服务。涉及胸痛胸闷、呼吸困难、晕厥、持续疼痛、神经症状、药物剂量、处方、疾病诊断等问题时，系统会优先给出就医或医生评估建议。
