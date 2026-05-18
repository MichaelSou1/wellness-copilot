# 提升 agent 个性化回复表现 — 上下文层系统性重构

## Context

测试反馈：agent 回答太通用，没代入画像数值（说"建议适量有氧"而非"以你 24 岁、75kg、ACL 术后状态，每周 3 次 30min 低冲击"）。

完整阅读代码后的现状梳理：

**存储：**
- Profile（`profile_store.py`）：用户长期画像，按 user_id 存为 JSON；`profile_to_prompt_text()` 只是裸 `json.dumps`
- Episode（`episode_store.py`）：跨 thread 情节记忆 `{ts, query[≤120], experts, gist[≤150]}`，滑动窗口 10 条

**注入路径与问题：**

| 节点 | Profile | History | 关键问题 |
|---|---|---|---|
| TurnStart | — | 装载 episode_context | 不读 profile，无统一快照 |
| Planner | 仅 `_profile_summary` | episode_context | 丢失 name/identity/relaxation |
| Trainer / Nutritionist / Wellness / General | 整段 JSON dump（各自读盘） | 全量 messages | **裸 JSON 无重点圈出；零值占位被当真数据；prompt 说"代入数值"但无强制结构** |
| Aggregator | ❌ | ❌（仅 user_question） | **合成阶段洗掉 expert 已建立的个性化语境（重灾区）** |
| Critic | 整段 JSON dump | ❌（仅 scratchpad） | 仅做安全审，无个性化引用校验 |

九个具体弱点：JSON dump 无加工 / Aggregator 漏点 / episode gist 仅 150 字 / episode 只喂 Planner / Planner summary 丢字段 / 零值占位 / update_user_profile 是裸 JSON patch / 每个 agent 各自读盘 / schema 缺风格偏好。

**上下文窗口管理现状（与个性化弱点相关）：**
- 轮内 messages：> 20 条触发 LLM 改写为中文要点摘要，删原文，保留尾部 8 条 + `SystemMessage(__history_summary__)`；下轮压缩时已有 summary 与新 head 合并，避免重复（`turn_start.py:94-134`）
- 跨 thread episode：**chronological FIFO**，窗口 10、每条 ≤150 字、按时间取最近 5 条喂 Planner —— **完全没有语义检索**
- 没有 embedding-based 召回 / similarity 排序，所以"上次提到的膝盖"这类语义关联回忆能力为 0
- 复用基建：`health_guide/rag.py` 已有 bge-m3 embedder + bge-reranker-v2-m3 + FAISS + 模块级模型缓存（rag.py:49, 395），lazy-load 后驻留内存，单次 embed ~10–30ms + FAISS 搜索 <1ms。绕过它的文件读取/切 chunk 逻辑后，可直接复用 `_lazy_load_models / _embed_model / _build_faiss_index / _dense_topk`

## Proposal — 五阶段重构

每阶段独立可交付、可观测；先 ship A 保证无回归，B 是质量跃迁点，C/D 复利叠加，E 兜底。

### 架构决策

1. **新建 `health_guide/personalization.py`** 作为"系统对该用户已知信息 + 渲染成 prompt"的唯一来源
2. **双重渲染并存**：自然语言"用户卡片"喂 expert/Aggregator；保留 raw JSON 给 Critic 做安全审
3. **`_profile_summary` 上提**到新模块 `profile_routing_digest`，Planner 复用，避免两份近似 summarizer
4. **State 新增 `personalization_ctx`** 字段，TurnStart 构建一次，全节点读取，禁止再读盘
5. **向后兼容**：缺失字段渲染为"未提供"或跳过；现有 `profile_store.json` 直接可用

---

### Phase A — State 层统一（行为零变化）

**目标：** 把 profile 读盘集中到 TurnStart，缓存进 state，所有节点改读 state。渲染暂时复用现有 JSON dump，保证可观察等价。

**改动：**

- **新增 `health_guide/personalization.py`**：
  - `build_personalization_ctx(user_id) -> dict`：调用 `get_user_profile`，返回 `{user_card, active_constraints, routing_digest, raw_profile, raw_profile_json, has_meaningful_data}`
  - Phase A 里 `user_card = profile_to_prompt_text(profile)`（与现状等价）
  - `profile_routing_digest(profile)`：从 `planner._profile_summary` 搬过来
  - `is_meaningful(profile)`：过滤 height=0/weight=0/空数组

- **`health_guide/state.py`**：第 75 行后加 `personalization_ctx: Annotated[dict, _take_last_dict]`；在 37-38 行附近加 `_take_last_dict` reducer

- **`health_guide/agents/turn_start.py:66-92`**：在 episode_context 块后调 `build_personalization_ctx(user_id)`，写入 `update["personalization_ctx"]`

- **`health_guide/agents/planner.py`**：删除 `_profile_summary`（74-96）和 `_get_profile` import；`_fresh_plan`（119-133）改读 `state["personalization_ctx"]["routing_digest"]`

- **四个 expert（`trainer.py / nutritionist.py / wellness.py / general.py`）**：`_build_*_agent` 改签名为接收 `pctx: dict` 而非 `user_id`；从 `pctx["raw_profile_json"]` 拿 profile_text。注意 tool 还要写盘，`user_id` 仍由 `*_node` 经 `HEALTH_GUIDE_USER_ID` 环境变量传给 tools

- **`health_guide/agents/critic.py:121-124`**：同样切到 `pctx["raw_profile_json"]`

**验证：** 跑 `scripts/smoke_critic_scratchpad.py`、`scripts/smoke_plan_execute.py` 必须仍打印 OK；在 smoke 里 assert `final_state["personalization_ctx"]` 非空。

---

### Phase B — 自然语言渲染 + Aggregator 注入（质量跃迁）

**这是解决用户痛点的核心。**

#### B.1 用户卡片格式（推荐"混合式"）

评估了三种格式，推荐 **leading paragraph + active-constraints bullet list**：

```
【关于该用户】
你目前对话的对象是 Michael（40 岁），身高 180cm，体重 88kg，BMI 27.2，目标定位为「健康」。

【必须遵守的个性化约束】
- 右膝半月板损伤恢复中，禁止深蹲/跳跃/跑步等高冲击动作，需在理疗师监督下渐进
- 饮食偏好：不吃香菜；其余未提供
```

为什么这种格式：开头段给 LLM 一个"主语"自然引用（"以你 40 岁、88kg…"）；bullet list 是强制扫描的 checklist，是把"建议适量有氧"翻译成具体处方的关键。

slot table 像数据不像知识；纯段落容易被略过；混合式兼顾。

#### B.2 渲染器实现

在 `personalization.py`：
- `render_user_card(profile)`：组装上面两段。零值字段一律跳过；全空时输出"该用户（个人信息暂未填写，请在恰当处主动询问）"
- `render_active_constraints(profile)`：根据 injuries 走小型派遣表（ACL / 半月板 / 冠心病 / 腰椎 / 肩袖 …）生成医学上具体的约束句；未知伤病走通用兜底；增肌/减脂目标自动派生热量与蛋白质量化参考；过敏类 preference 单独高亮
- `_describe_injury_constraint(inj)` 关键字派遣到具体禁忌动作清单

#### B.3 四个 expert prompt 重写

替换现有 `f"当前用户画像：{profile_text}。"` 一行，改为新骨架（Trainer 示例）：

```
你是力量训练教练。

{user_card}

{peer_notes_text}{rag_section}

【输出硬性要求】
1. 回答开头必须自然引用画像中至少 1 个数值（年龄/体重/身高/BMI），
   例如「以你 40 岁、88kg 的当前状态…」。不允许只说「根据你的情况」。
2. 训练量必须给出具体数字（频次/组数/时长/强度），不允许「适量」「适度」。
3. 若上方约束列出了伤病，回答前两句之内必须点名该伤病并说出限制；
   不得推荐冲突动作；替代动作须注明「须在理疗师许可下进行」。
4. 如用户在本轮提供了新的身体信息，调用对应 set_/add_ 工具记录。
```

Nutritionist 同结构，规则 1 改为引用 weight+goal，规则 2 改为数字 kcal/蛋白质 g；Wellness 规则 1 改为引用 stress_sources 名称；General 不加硬性规则（处理寒暄），仅放 user_card 避免对话脱节。

#### B.4 堵 Aggregator 漏点

`health_guide/agents/aggregator.py`：

- `_SYSTEM_PROMPT`（12-22）追加两条：
  - 必须保留各专家已引用的画像数值，不得"为了更顺"洗成「根据你的情况」
  - 必须保留所有伤病/过敏硬约束，不得软化或省略
- `_SYNTHESIS_TEMPLATE`（24-36）顶部插入 `{user_card}` 块
- `aggregator_node`（39）单 expert 短路分支（49-50）保持，多 expert 路径里从 `state["personalization_ctx"]["user_card"]` 取值传入

#### B.5 验证

新建 `scripts/smoke_personalization.py`：种子 `{age:24, weight:75, height:178, injuries:["ACL 撕裂术后早期"], goal:"增肌"}`，跑一轮"我想增肌该怎么练和吃"，assert 最终答案含 `"24"`、`"75"`、`"ACL"` 或 `"韧带"`、`"理疗师"`。

扩 `eval/output_eval_dataset.jsonl`：为有 physical_stats 的样本加 `must_contain_one_of`（age / weight 关键词），扩 `scripts/evaluate_output.py:239-253` 的 `_check_assertions` 支持该字段，新增聚合指标 `personalization_quantification_rate`。

跑 evaluate_output.py 前后对比 `by_dimension.personalization`，预期从 ~3.x 跳到 4+。

---

### Phase C — Episode 记忆升级 + experts 也吃到 + embed-on-write 预热

- `episode_store.append_episode` 增加 `facts: Optional[Dict]` 参数，`gist` 上限 150→400，`MAX_EPISODES_PER_USER` 10→20
- **采用便宜路径**：让 `_scratchpad.build_scratchpad_note` 顺便返回 facts dict，Critic union-merge 后写入 episode；避免额外 LLM 调用
- `format_episodes_for_prompt`：有 facts 时附加`（已记录：key=value、…）`
- 四个 expert 的 `_build_*_agent` 新增 `episode_section` 段，把 `state["episode_context"]` 也拼进 system prompt（user_card 之后，peer_notes 之前）
- **embed-on-write 预热（为 Phase F 铺路）**：`append_episode` 写盘后调 `EpisodeMemory(user_id).index_episode(text)` 顺手 embed 到 per-user FAISS 索引；该调用容错（embed 失败仅记日志，不影响主流程）。Phase C 单独完成时该索引仅写不读；Phase F 上线后才开始查询
- 可选：`dedup_episodes` 基于 token overlap > 0.7 去重，TurnStart 调用

验证：smoke 改为两轮，turn1 用户暴露伤病，turn2 不提；assert turn2 答案仍引用该伤病。另 assert 第二轮后 per-user FAISS 索引文件存在且条数 ≥ 2。

---

### Phase D — Schema 扩展 + 结构化工具

- `config.py:19-37`：`DEFAULT_USER_PROFILE` 增 `response_style: {tone, humor, formality, language}`（值都默认 ""）
- `render_user_card` 在 style 非空时追加`【风格偏好】简洁/带轻度幽默/口语化`段
- `tools.py:74-86` 保留 `update_user_profile`（向后兼容），新增 6 个结构化工具：`set_physical_stats / add_injury / set_dietary_goal / add_dietary_preference(kind in {like,dislike,allergy}) / add_stress_source / set_response_style`；每个是 `update_profile_in_store` 的薄包装
- 各 expert 的工具清单按职责挂相应工具；prompt 里"请调用 update_user_profile"改成"调用对应的 set_/add_ 工具"

验证：smoke 加一轮「我喜欢你回答简洁一点」，assert `profile_store.json` 写入 `response_style.tone="concise"`；后续一轮 assert 输出 < 350 字。

---

### Phase E — Critic 个性化校验（可选兜底）

- `critic.py:37-81` 的 `_CRITIC_SYSTEM` 加 P3 块：若 `has_meaningful_data` 且草稿全文未自然引用任何画像数值/伤病名/压力源 → REVISE
- Critic 输入改用 `user_card`（非 raw JSON）以与 expert 看到的对齐
- 复用现有单次 REVISE 机制，不引入迭代修订循环

验证：smoke 用一个"故意通用"的 mock expert 回复，assert `critic_verdict == "REVISE"` 且最终答案含画像数值。

---

### Phase F — 语义 episode 召回（Hybrid: recency + semantic）

**前置：Phase C 已完成（episode 变厚 + embed-on-write 已在跑）。** 单做 F 而不做 C 没意义，因为 150 字 gist 召回不出有用细节；C 阶段写盘时已顺手 embed 到 per-user FAISS 索引，F 阶段开始读取。

**为什么 hybrid（recency + semantic）：** 纯语义召回会错过"刚刚提的事"（用户上一轮说"我换了新工作"，本轮问"压力大怎么办"，纯语义可能召回的是另一次压力对话）。Hybrid 保底"最近 2 条"做时间锚，再叠加 top-3 语义相关。

**新增 `health_guide/episode_memory.py`（~150 行）：**

```
class EpisodeMemory:
    def __init__(self, user_id: str): ...        # per-user FAISS index 路径
    def index_episode(self, episode_id: str, text: str) -> None
    def retrieve_similar(self, query: str, top_k: int = 3, exclude_ids: set = None) -> list[dict]
    def rebuild_from_store(self) -> None         # 一次性迁移：扫已有 episodes 全量 embed
```

实现要点：
- 复用 `rag._lazy_load_models()`（rag.py:385）共享 bge-m3 embedder；不再加载第二份模型
- 索引存 `~/.health_guide_indices/episodes/<user_id>/` 下的 `index.faiss + ids.json + vecs.npy`，与 `knowledge_base` 的 `.index_cache` 物理分离
- `retrieve_similar` 走 `_dense_topk`（rag.py:722）做向量 top-k；不引入 reranker（episode 召回延迟敏感 + 候选数小，简单 cosine 足够）
- `episode_id` = ts + query 的 short hash，去重稳定
- embed 内容 = `query + " | " + gist + " | " + facts_as_string`，让语义向量同时承载提问与已记录事实

**TurnStart 改造（`turn_start.py:66-92`）：**

```
recent_eps = get_recent_episodes(user_id, n=2)              # 保留最近 2 条
recent_ids = {ep_id_of(e) for e in recent_eps}
if total_episode_count(user_id) >= EPISODE_SEMANTIC_MIN_COUNT:  # 阈值：太少时不启用语义
    try:
        semantic_eps = EpisodeMemory(user_id).retrieve_similar(
            query=current_user_message, top_k=3, exclude_ids=recent_ids,
        )
    except Exception:
        semantic_eps = []
else:
    semantic_eps = []
merged = recent_eps + semantic_eps                          # recency 先 + semantic 后
episode_context = format_episodes_for_prompt(merged, mark_source=True)
```

`format_episodes_for_prompt` 新增 `mark_source`：
- recent 项前缀 `• [最近]`
- semantic 项前缀 `• [相关]`

让 Planner / expert 知道哪条是"刚发生的"、哪条是"语义相关召回的"，避免被 LLM 误当时间线。

**Planner / expert prompt 同步更新：** 在使用 `episode_context` 的位置加一句："列表中 [最近] 表示按时间最近的对话，[相关] 表示因话题相似而召回的更早对话；引用时请按其性质区分使用。"

**新增 env / config（在 `config.py` 末尾）：**

```
EPISODE_SEMANTIC_RETRIEVAL_ENABLED = env bool, default True
EPISODE_SEMANTIC_MIN_COUNT = env int, default 8      # archive 数下限
EPISODE_SEMANTIC_TOP_K = env int, default 3
EPISODE_INDEX_DIR = env str, default "~/.health_guide_indices/episodes"
```

**迁移：** 提供 `scripts/migrate_episode_index.py`：扫所有 user_id，对每个用户调 `EpisodeMemory.rebuild_from_store()`，把存量 episode 全量 embed 进新索引。运行一次即可。

**验证（`scripts/smoke_semantic_episode.py`）：**
1. 种子 12 条 episode：其中 1 条提到 `膝盖 ACL`，6 条无关日常话题，5 条最近的训练/营养相关
2. 用 "我膝盖最近有点疼，能跑步吗" 触发一轮
3. assert `episode_context` 包含那条 ACL episode（被语义召回），尽管它不在最近 2 条里
4. assert `episode_context` 同时包含最近 2 条（recency 保底）
5. 不启用语义召回时（`EPISODE_SEMANTIC_MIN_COUNT=999`）回归为 chronological，确认行为可降级

**性能 budget：** 单轮 turn 增加 ~15-40ms（一次 embed + FAISS top-k）。embed 模型在 Phase A 之前如未被 RAG 调用过则首轮 lazy-load 多 1-3s（一次性）。

**轮内 messages 维持现状：** 不为 messages 引入语义检索；现有 summary + 尾部保留 8 条已经够单 thread 用，引入 embedding 收益小、开销大。

---

## Critical files

- 新增：`health_guide/personalization.py`，`health_guide/episode_memory.py`（Phase F），`scripts/smoke_personalization.py`，`scripts/smoke_semantic_episode.py`（Phase F），`scripts/migrate_episode_index.py`（Phase F）
- 主改：`health_guide/agents/turn_start.py`，`health_guide/agents/trainer.py`（其余三 expert 同模板），`health_guide/agents/aggregator.py`，`health_guide/state.py`
- 联动：`health_guide/agents/nutritionist.py / wellness.py / general.py / planner.py / critic.py`，`health_guide/episode_store.py`，`health_guide/tools.py`，`health_guide/config.py`，`health_guide/agents/_scratchpad.py`，`scripts/evaluate_output.py`，`eval/output_eval_dataset.jsonl`
- 复用（只读不改）：`health_guide/rag.py` 的 `_lazy_load_models / _embed_model / _build_faiss_index / _dense_topk`（Phase F）

## 复用 vs 删除

- **删**：`planner._profile_summary`（74-96）— 上提到 `personalization.profile_routing_digest`
- **保留但收敛调用方**：`profile_store.profile_to_prompt_text` — 唯一消费者变为 `build_personalization_ctx`；其余 5 处 import 删除
- **保留并扩展**：`_scratchpad.build_scratchpad_note`（Phase C 加 facts 返回值）、`format_episodes_for_prompt`（Phase C 加 facts 渲染）
- **保留作兼容**：`update_user_profile` 工具（Phase D 后变 fallback）

## 端到端验证

1. Phase A 后跑 `smoke_critic_scratchpad.py` / `smoke_plan_execute.py` 验证零回归
2. Phase B 后跑新 `smoke_personalization.py` + 重跑 `evaluate_output.py` 看 `personalization` 维度跳分
3. 手动三个 query 肉眼检：
   - 40 岁/88kg/半月板用户问"我想练腿"
   - 28 岁/80kg/减脂用户问"该吃多少蛋白质"
   - ACL 康复用户问"能跑步吗"
4. Phase C/D/E/F 各自独立 smoke 已在各 Phase 内列出
