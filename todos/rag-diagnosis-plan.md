# RAG 诊断与路由决策 — 3-Phase 计划

> 背景：output_eval（53 条）显示 KB-RAG 调用率 ≈0（`avg_rag_calls=0.08`，仅 4/53 触发），
> 但 RAG 专项 eval 证明检索质量优秀（rerank recall@5≈0.99, MRR≈0.91）。
> 即「KB 不是大便，是 agent 不去叫它」。同时项目里有**两套独立 RAG**，不要混：
> - **KB RAG**（`knowledge_base/`，`retrieve_*_knowledge`）：pull，领域通识，喂 completeness/safety，是 MCP 的同级路由备选。
> - **情景记忆 RAG**（FAISS，`episode_memory.py`/`episode_store.py`）：push，自动注入 `episode_context`，喂 personalization，**不在 RAG-vs-MCP 路由里**。
>
> 关键 confound：`scripts/evaluate_output.py:1333` 每条样本用全新 user 只灌 profile、不种 episode，
> 而 `EPISODE_SEMANTIC_MIN_COUNT=8` → 情景记忆 RAG 在 eval 里**从未触发**，personalization=3.70 是在它缺席下打出的。
>
> 执行顺序：先 B 定因，再 C，最后压轴啃最重的 A（KB-RAG vs MCP 自进化路由）。

---

## Phase 1 — B：personalization 3.70 归因（最低分，先拍死成因）

**目标**：判定 personalization 偏低是 (b1) eval 没种 episode 导致情景记忆 RAG 空转，还是 (b2) glm-4.5-air 没把已注入的 profile/决策点织进正文。

- [x] 确认现状：在 output_eval 跑动时打印/断言 `episode_context` 是否为空（预期：恒空）。
      ✅ 确定性实验 + control 臂实跑：`episode_context_present_rate=0.094`（仅 multi_turn 第2轮的同线程"最近"注入），语义跨会话检索一次都没触发。
- [x] 改 eval：给测试 user 预种 ≥8 条 episode（`_seed_episodes` + `EVAL_SEED_EPISODES=1`，53 样本×9 条，见 `eval/episode_seeds.jsonl`）。
- [x] 重跑 output_eval（两臂 ×53，同 judge 背靠背），对比 personalization（基线 3.70）：
  - ✅ **成因 = b1（情景记忆 RAG 空转，eval 装置缺陷）**。开火率 0.094→1.0；personalization 3.731→3.923（全量 Δ+0.192，被自洽单轮稀释），
    增益精准集中在 **profile_personalization +1.25 / multi_turn +0.80 / progress_review +0.50**，自洽单轮类别零变化。夹带噪声级弱 b2。

**退出条件 ✅**：主因 b1，Δ 已记录。详见 `reports/rag_diagnosis_phase12_findings.md`。

---

## Phase 2 — C：情景记忆 RAG 本身的质量（现有 eval 测不到的盲区）

**目标**：专门验 per-user 情景记忆 RAG（FAISS `retrieve_similar`）召回的历史是否相关、是否真被 subagent 用进正文。Phase 1 指向 b1、或要正式上线情景记忆时优先做。

- [x] 造带历史的用户场景：53 user × 9 条结构化 episode（3 相关 + 6 干扰），带 relevance 标注（`eval/episode_seeds.jsonl`）。
- [x] 评情景记忆检索质量（`scripts/evaluate_episode_memory.py`）：**top-1 相关率 90.6%、MRR 0.945、recall@3 0.79 / @5 0.92、hit@3 0.98**——与 KB RAG 同档优秀。
- [x] 评下游使用：episode_context 注入率 100%，**下游采纳率 90.6%**（答案出现 episode 独有 token，平均 ~16 个/样本）→ 确被织进正文。
- [x] `EPISODE_SEMANTIC_TOP_K` 敏感性：默认 3 最优（hit@3=0.98）；升 5 召回 0.92 但 precision 跌 0.55；降 1 召回 0.31。
      `EPISODE_SEMANTIC_MIN_COUNT=8` 是 turn_start 放行门槛，是冷启动个性化弱+本 eval confound 的同根，建议下调 3-5。

**退出条件 ✅**：检索质量（top-1 90.6%/MRR 0.945）+ 下游采纳率（90.6%）齐备；真实贡献 = 在依赖历史的类别上 personalization +0.8~+1.25。详见 `reports/rag_diagnosis_phase12_findings.md`。

---

## Phase 3 — A：KB-RAG vs MCP（最初的自进化问题，最重，压轴）

**目标**：在「检索质量已被证明好」的前提下，用反事实判定 KB-RAG 被调多了到底改不改善质量。**这是决定要不要做 RAG-vs-MCP 路由的前置闸门。**

- [ ] 加 `FORCE_RAG` 开关（或在 subagent prompt 临时强制「先接地再回答」），让每个专家子调用强制 `retrieve_*_knowledge` 一次。
- [ ] 跑两臂 ×53：基线（自由调用）vs 强制 RAG，judge 对比 **completeness（基线 4.55）/ safety（4.98）/ relevance**（**不看 personalization**，KB 与它无关）。
  - 强制 RAG 顶起质量 → 低调用是 **bug**。修法是「改 tool 描述（加触发条件+示例）+ prompt 强制接地 + 可选 retrieval gate」，**不是**时延路由。修完后 RAG-vs-MCP 路由才谈得上。
  - 强制 RAG 不动质量（只加时延）→ 这个 workload **不需要 KB-RAG** → 考虑砍掉/改 opt-in，自进化路由的想法就地判死。
- [ ] 补前置埋点（无论哪种结论都要做）：per-backend 时延 + 调用计数（RAG / MCP × GPU/CPU），现状 MCP 完全没被统计、一臂全黑。
- [ ] 仅当上一步证明 RAG 值得调，再设计路由层（外部 policy，**不碰 model 权重**）：
      bandit reward = `quality − λ·latency`，离线 eval 种子冷启动 + 在线时延 EWMA，质量走周期性 A/B 闸门而非热路径 judge；
      施加方式优先「注入 prompt hint / 重排 tool list」，保留 LLM 语义判断；留 exploration 防锁死，按 (role, device) reset。
      ⚠️ 注意：本机是 GPU=RAG 快侧，要验 CPU 慢分支需 `CUDA_VISIBLE_DEVICES=""` 强制 reranker 上 CPU 模拟。

**退出条件**：明确「KB-RAG 该不该被多调」的数据结论；若该调，产出最小路由层设计 + per-backend 埋点落地。

---

## 决策图

```
Phase 1 (B) ──┬─ b1 (eval 空转) ──────────────► Phase 2 (C) 深入验情景记忆 RAG
              └─ b2 (模型织入弱) ─────────────► prompt/模型方向（与 RAG/MCP 无关）

Phase 3 (A) ──┬─ 强制 RAG 提质 ─► 修 tool 描述/prompt/gate ─► 再考虑 RAG-vs-MCP 路由层
              └─ 强制 RAG 无效 ─► KB-RAG 改 opt-in / 砍；自进化路由判死
```

## 不变量（任何 Phase 都成立的结论）
- KB 质量不是瓶颈，检索栈（hybrid dense+BM25+rerank+parent rescue）已验证优秀。
- 自进化「时延感知 RAG-vs-MCP 路由」**现在直接上会帮倒忙**——会把已≈0 的 KB-RAG 调用率压到更零（RAG 既是最慢臂又是最少用臂）。必须先过 Phase 3 闸门。
- 任何「学习/自进化」都在**模型外部的 policy 层**，不 fine-tune orchestrator / subagent 权重。
