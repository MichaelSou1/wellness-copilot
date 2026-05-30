# 子 Agent 上下文隔离 A/B 因果评测

**日期**：2026-05-30
**目的**：量化"subagent context isolation 能否提升子回答质量、减少 rotten context"这一常见论断在本系统中的真实效果，而不是停留在架构图或直觉。

---

## 1. 背景与动机

Wellness Copilot 的 Dispatcher 给每个专家子调用一个**隔离的上下文**（`wellness_copilot/agents/dispatcher.py`），由三处机制共同实现：

1. **profile** — role-cropped 画像：每个专家只看到自己角色白名单内的字段（`personalization.build_personalization_ctx` + `profile_store.profile_subset_for`）。
2. **peer** — 同批 peer notes 过滤：同一 plan batch 的专家并行执行，互相看不到对方的 scratchpad。
3. **history** — 不注入完整 history：专家只拿到 `QueryRewriter` 改写后的独立问题，而非父 Agent 的完整消息历史。

现有评测有三层，但**没有一层度量"隔离本身的因果效果"**：

- L4 `evaluate_architecture.py` 的 `context_isolation` 只断言隔离**有没有发生**（system_prompt 不含违禁字段），是结构正确性检查。
- L5 `evaluate_output.py` 测 E2E 质量，但绝对分已贴顶（safety 4.981 / coherence 4.981），**天花板效应**下测不出隔离开/关的细微差异。

本评测是一个**干预实验（A/B）**：同一输入分别在"隔离 ON（现状）"与"隔离 OFF"两个 arm 下跑全流程，用**成对偏好（pairwise）**判定差异，并**同时在 subagent 文本层和 E2E 层**度量。

## 2. 方法

### 2.1 运行期可切换隔离开关

`wellness_copilot/isolation.py` 提供运行期可变（非 import 时固化）的三维开关，A/B runner 用 `isolation_override(...)` 在**同一进程内**逐 arm 翻转。默认全 ON，生产行为零变化。三个接入点：

- profile → `build_personalization_ctx` 把 role 卡片从裁剪版换成完整画像；
- history → 各 `_build_*_agent` 通过 `isolation.noniso_history_section(pctx)` 注入完整 transcript（dispatcher 计算后经 pctx 传入，runner 签名不变）；
- peer → `dispatcher._run_plan` 改为顺序执行并把已完成专家的 scratchpad 喂给后续专家。

### 2.2 度量与数据集

- **确定性泄漏率**：数据集每条带 `leak_traps`（每个专家"不该出现"的跨域/过期词）。隔离 ON 应不泄漏，OFF 因注入更多上下文更可能泄漏。这是 rotten context 污染最干净的代理指标。
- **成对偏好（pairwise）judge**：subagent 层（role_relevance / focus / safety / usefulness / overall）与 E2E 层（relevance / completeness / safety / personalization / coherence / overall）。A/B 顺序随机化消除位置偏好。judge 用独立模型（deepseek-v4-pro）。
- **因果链分析**（`scripts/analyze_isolation_report.py`）：按 (样本, 专家) 关联"非隔离 arm 是否泄漏" × "judge 偏向哪个 arm"，输出 lift = P(偏向 ISO｜泄漏) − P(偏向 ISO｜干净)。

### 2.3 数据集

| 数据集 | 条数 | 形态 | 用途 |
|---|---:|---|---|
| `eval/isolation_hard_dataset.jsonl` | 50 | 多专家 + 跨域冲突画像 | 主评测（翻 profile+peer+history） |
| `eval/isolation_longhistory_dataset.jsonl` | 15 | 6 轮，植入作废/跑题内容 | 长历史维度（仅翻 history） |
| `eval/isolation_longhistory_xl_dataset.jsonl` | 12 | 12–14 轮（越过 20 条摘要阈值） | 长历史"腐烂区间"（仅翻 history） |

## 3. 结果

### 3.1 主评测（50 条，翻 profile+peer+history）

`reports/isolation_ab_report_20260530-014804.json`

| 指标 | 隔离 ON | 隔离 OFF |
|---|---:|---:|
| 跨域泄漏率 | **22.5%**（16/71） | 35.5%（22/62） |
| E2E pairwise 胜 | 14 | 25（tie 11） |
| subagent pairwise 胜 | 21 | 15（tie 25） |
| subagent focus 胜 | 16 | 11（tie 34） |

按专家拆分 subagent overall：**Psychologist 隔离胜率 81.8%（9/1/1）**，Trainer 8/9/8，Nutritionist 4/5/10，**Doctor 全平 0/0/6**（Doctor 的画像裁剪几乎涵盖所有字段，隔离改变不了它的输入——符合机制预期）。

**因果链分析（61 个 (样本,专家) 对，`analyze_isolation_report.py`）**：

| 条件 | overall 偏向 ISO | focus 偏向 ISO |
|---|---:|---:|
| 非隔离**发生泄漏**（n=22） | **63.6%**（14/22，0 平局） | 50.0% |
| 非隔离**保持干净**（n=39） | 17.9% | 12.8% |
| **lift** | **+45.7%** | **+37.2%** |

**核心结论**：隔离的质量优势几乎**全部集中在"非隔离确实被污染"的样本上**；未污染时两 arm 基本打平（focus 平局 30/39）。这把"rotten context → 子回答变差 → 隔离避免"的因果链直接画了出来。聚合 E2E 之所以偏向非隔离（多数样本未触发污染时，非隔离多出的上下文偶尔让 completeness 更讨喜），正是被这个条件分析揭示和解释的。

### 3.2 长历史：效应是长度依赖的

仅翻 history 维度（不动 Orchestrator 路由）：

| 实验 | 历史长度 | 摘要触发 | 泄漏 ON vs OFF |
|---|---|---|---:|
| 6 轮（`isolation_ab_history_*`） | 6 轮 / ~12 消息 | 否 | 14.3% vs 14.3%（**无差异**） |
| 12–14 轮（`isolation_ab_historyxl_*`） | 12–14 轮 / >20 消息 | **是**（history_summary 379 字） | **0% vs 25%** |

**6 轮时历史隔离没效果，推到 12–14 轮就出现了**：非隔离 arm 开始回吐过期/跑题内容（25%），隔离 arm 保持 0%。机制细节：长线程触发 `TurnStart` 摘要后，`render_transcript` 跳过摘要 SystemMessage、注入近期原文 tail，光是这部分 tail 里的 rot 就足以让非隔离 arm 泄漏。

### 3.3 token 成本：不是 RAG，是路由分歧

主评测聚合曾显示"隔离 arm input token 更高"（8854 vs 7346），与"隔离省 token"直觉相反。诊断（`scripts/diagnose_isolation_tokens.py`，6 条）与 50 条逐样本分析结论：

- **RAG 调用两 arm 均为 0**，token 基本相等（诊断 Δ −0.7%）。RAG 假设否定。
- 50 条逐样本：iso<non **29 条**、iso>non 14 条、持平 7 条；**中位 Δ = −54**（典型情况非隔离略高，符合理论）。
- "iso 更高"全由 **5 个离群**拉动，根因是两 arm **路由到的专家数不同**（如 iso_hard_032：隔离派 4 个专家=36.6k tok，非隔离 Orchestrator 直接答=2.5k tok）。因为 profile 隔离开关连带改了 Orchestrator 的输入，从而改了路由。
- 长历史 run（仅翻 history、不碰路由）印证：token 非隔离略高（95377 vs 93474），无离群。

## 4. 结论

针对"subagent context isolation 提升子回答质量、减少 rotten context"：

1. **减少 rotten context：成立。** 最强污染来源是**跨域画像/peer**（主评测 35.5%→22.5%）；**长对话历史**在约 12 轮以上也开始贡献（0% vs 25%），6 轮以下不明显。
2. **提升质量：条件性成立。** 质量优势几乎只在"确实发生污染"时出现（污染时 overall 偏向隔离 63.6% vs 36.4%、零平局；干净时打平），不是无差别提升。
3. **方法学副作用**：profile 隔离会"漏"到路由层，使两 arm 跑的不是同一计划，这同时影响 token 与质量对比。做最干净的"纯专家质量 A/B"应固定计划（仅在专家层翻隔离）。

## 5. 局限

- n 偏小（50 / 15 / 12），单 judge 模型。
- 隔离 arm 在主评测仍有 22.5% 泄漏：部分 trap 可能落在该角色本就可见的字段，或经检索带出；会拉低 ON/OFF 差距但不影响条件分析有效性。
- profiler 疑似漏计专家内部 LLM 调用，绝对 token 仅作粗略代理，已改用中位数 + 控专家数解读。
- 决策点（`build_personalization_decision_points`）存在有意的跨域耦合（如压力→训练负荷），属预期行为；L4 `arch_isolation_001` 断言已据此放宽（移除压力源、保留饮食偏好/过敏），3/3 通过。

## 6. 复现

```bash
# 主评测（翻全部三维）
python scripts/evaluate_isolation_ab.py --dataset eval/isolation_hard_dataset.jsonl

# 仅翻某一维做消融
python scripts/evaluate_isolation_ab.py --dataset eval/isolation_longhistory_xl_dataset.jsonl --ablate history

# 因果链分析
python scripts/analyze_isolation_report.py reports/isolation_ab_report_<ts>.json

# token 诊断（RAG / 路由）
python scripts/diagnose_isolation_tokens.py --n 6 --ablate all

# 隔离开关单测
python -m pytest tests/test_isolation_toggle.py -q
```

相关产物：`wellness_copilot/isolation.py`、`scripts/evaluate_isolation_ab.py`、`scripts/analyze_isolation_report.py`、`scripts/diagnose_isolation_tokens.py`、`tests/test_isolation_toggle.py`、`reports/isolation_ab_*.json`。
