"""Critic / Safety Reviewer.

Reads the Aggregator's draft answer along with the shared scratchpad notes
and the user's profile, and decides whether the draft is safe to ship.

Output protocol (the LLM must return one of):
  VERDICT: PASS
  VERDICT: REVISE
  ---
  <revised answer text, only present when REVISE>

If PASS, the draft is forwarded as-is. If REVISE, the revised version is
used as the final answer.
"""
import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..episode_store import append_episode
from ..llm import extract_text_content, llm
from ..mcp_client import MCP_REGISTRY
from ..profile_store import (
    get_user_profile as get_profile_from_store,
    profile_to_prompt_text,
)
from ..tools import retrieve_safety_guidelines
from .dispatcher import REPLAN_CAP
from .fallbacks import add_safety_warning, has_safety_risk
from .query_rewriter import get_user_question


# All experts the Critic can request as replan reinforcement. Kept in sync
# with dispatcher.EXPERT_RUNNERS — used to decide if any unexecuted expert
# is still available before firing REPLAN.
_ALL_EXPERTS = ("Trainer", "Nutritionist", "Wellness", "General")


# Trigger for pulling authoritative medical literature from medical-mcp.
# Tightened from single-char triggers ("片" matches "面包片") to multi-char
# clinical signals — drug names, symptom phrases, dosage patterns.
_MEDICAL_PATTERN = re.compile(
    r"(布洛芬|对乙酰氨基酚|阿司匹林|抗生素|药物相互作用|"
    r"剂量|mg|每日.*片|胸痛|胸闷|心率(?:过|不齐)|血压|血糖|"
    r"失眠|抑郁|焦虑|过敏|怀孕|哺乳|术后|韧带)"
)


_EXPERT_LABELS = {
    "Trainer": "训练教练",
    "Nutritionist": "营养师",
    "Wellness": "身心康复师",
    "General": "助理",
}


_CRITIC_SYSTEM = """\
你是健康团队的安全审核员。你的职责是对即将发给用户的回答做一次"安全 & 一致性"复核。

你可以输出三种结论：PASS、REVISE、REPLAN。优先级从高到低如下：

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【R0 · 缺人补员（REPLAN）— 仅在以下情况触发】
若同时满足：
  (a) 用户画像中存在活跃伤病/过敏/慢病/术后等关键风险因素（如 ACL/半月板/骨折/韧带/术后/食物过敏/心脏病/糖尿病），
  (b) 草稿与各专家 scratchpad 完全没有针对该风险因素给出任何具体建议或限制（不是只字未提"提醒"，而是连风险都未识别），
  (c) 系统提供的"已派出专家"列表中明显缺少对应领域的专家——
    · 伤病 / 术后 / 康复 → 缺少 Trainer
    · 过敏 / 慢病饮食限制 → 缺少 Nutritionist
    · 长期失眠/重度压力情绪 → 缺少 Wellness
→ 输出 VERDICT: REPLAN，并写明需要补叫哪位专家。
注意：如果对应专家已经在"已派出专家"列表中，但建议不够完整，请用 REVISE 而不是 REPLAN（避免重复派同一个专家）。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【P0 · 伤病安全硬限制（REVISE）】
若用户画像中记录了活跃伤病或康复中状态，下列任一情形 → REVISE：
  (P0a) 草稿对该伤病部位推荐了任何直接负重训练动作——哪怕加了"降级版""可控""轻度""箱式"等限定词——
        且未明确注明"须在运动医学医生或理疗师亲自评估许可并全程监督下进行"。
  (P0b) 草稿从头到尾**完全没有**承认或提及该伤病/康复状态（既未点名部位，也未给出与之相关的注意事项），
        即使内容本身（如纯饮食方案）没有直接危险，但忽略已知伤病等于浪费个性化机会，必须 REVISE 加上"基于你的 X 伤病/康复阶段..." 一段开头说明。
修订方向：(P0a) 移除直接负重动作建议，改为零负重替代方案；(P0b) 在开头补一段"考虑到你目前 X 状态..." 的衔接说明，把后续建议与伤病关联起来。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【P1 · 症状描述必须先导向就医（REVISE）】
若用户描述了身体症状（持续疼痛、心率异常、胸痛/胸闷、头晕、肿胀不消退等），
草稿在未首先明确建议就医的前提下，先给出了可能的病因分析、鉴别诊断或自我处理方案，
→ REVISE。正确结构：①先明确建议就医 → ②再提供辅助性信息。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【P2 · 其他健康安全风险（REVISE）】
- 极端热量限制：建议低于女性 1200 kcal/d 或男性 1500 kcal/d 且未说明医疗监督背景
- 已知慢病用户被建议高风险运动而未要求先获医生许可
- 越权用药：给出诊断、处方或具体药物剂量建议
- 常见膳食/运动补剂（如肌酸、乳清蛋白、咖啡因、鱼油、维生素D）可给一般推荐摄入范围、频率与注意事项；这不视为处方药剂量越权
- 跨专家事实性矛盾

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
判断克制原则：
- R0 与 P0/P1/P2 都不触发 → PASS
- 措辞不够漂亮、缺少边角细节 → PASS
- 优先级：R0 > P0 > P1 > P2。命中 R0 就输出 REPLAN，不要降级到 REVISE。
- 如系统注入了"安全知识库参考"，请把其中的红线作为硬约束。

输出格式（严格遵守，三选一）：
- 通过：仅一行 `VERDICT: PASS`
- 补员：两行
```
VERDICT: REPLAN
NEEDED: <Trainer|Nutritionist|Wellness>（恰好一个角色名，必须是已派出列表里没有的）
REASON: <一句话写明缺哪个领域、为什么需要补>
```
- 现地修订：
```
VERDICT: REVISE
REASON: <一句话写明触发修订的具体规则和风险>
---
<修订后的完整回答，以用户视角直接说话，不要提及"审核""修订"等元话语>
```
"""


def _retrieve_safety(user_question: str, draft: str) -> str:
    """Pull relevant red-lines from the safety KB.

    Returns a formatted prompt section, or empty string if KB unavailable or
    nothing useful comes back. Always best-effort — must not break review.
    """
    query = (user_question or "").strip() or (draft or "").strip()
    if not query:
        return ""
    try:
        hits = retrieve_safety_guidelines.invoke({"query": query, "top_k": 3})
    except Exception:
        return ""
    if not hits or hits.startswith("[RAG Error]") or "未命中本地知识库" in hits:
        return ""
    return f"安全知识库参考（按相关性排序，请作为硬约束）：\n{hits}\n\n"


def _retrieve_medical_context(user_question: str, draft: str) -> tuple[str, bool]:
    """Pull authoritative medical references from medical-mcp when the query
    mentions drugs/symptoms/conditions. Returns ``(prompt_section, hit)``.

    Mirrors ``_retrieve_safety`` — best-effort, never raises, empty string
    when the MCP isn't available or the trigger pattern misses.
    """
    text = f"{user_question or ''}\n{draft or ''}"
    if not _MEDICAL_PATTERN.search(text):
        return "", False
    tools = MCP_REGISTRY.get_tools("medical")
    if not tools:
        return "", False
    try:
        lit_tool = next(t for t in tools if t.name == "search-medical-literature")
    except StopIteration:
        return "", False
    try:
        hits = lit_tool.invoke({"query": (user_question or "")[:200], "max_results": 3})
    except Exception as e:
        print(f"[Critic][MCP] medical lookup failed: {type(e).__name__}: {e}")
        return "", False
    if not hits:
        return "", False
    hits_str = str(hits)
    if "[MCP Error]" in hits_str:
        return "", False
    return (
        f"权威医学参考（来自 PubMed/FDA/WHO/RxNorm，请作为硬约束）：\n{hits_str}\n\n",
        True,
    )


def _format_notes(agent_notes):
    if not agent_notes:
        return "（无）"
    lines = []
    for role, note in agent_notes.items():
        if not note:
            continue
        label = _EXPERT_LABELS.get(role, role)
        lines.append(f"- [{label}] {note}")
    return "\n".join(lines) if lines else "（无）"


def _parse_verdict(text: str):
    """Return (verdict, payload).

    - PASS    → ('PASS', None)
    - REVISE  → ('REVISE', revised_text)
    - REPLAN  → ('REPLAN', {'needed': <role>, 'reason': <str>})
    - else    → ('UNKNOWN', None)
    """
    if not text:
        return "UNKNOWN", None
    stripped = text.strip()
    first_line = stripped.splitlines()[0].strip().upper().replace(" ", "")
    if first_line.startswith("VERDICT:PASS"):
        return "PASS", None
    if first_line.startswith("VERDICT:REPLAN"):
        needed = ""
        reason = ""
        for line in stripped.splitlines()[1:]:
            ls = line.strip()
            up = ls.upper()
            if up.startswith("NEEDED:"):
                needed = ls.split(":", 1)[1].strip().strip("：:").strip()
            elif up.startswith("REASON:"):
                reason = ls.split(":", 1)[1].strip().strip("：:").strip()
        # Keep only the first valid role token (e.g., "Trainer" out of "Trainer 教练")
        for role in _ALL_EXPERTS:
            if role.lower() in needed.lower():
                needed = role
                break
        else:
            needed = ""
        return "REPLAN", {"needed": needed, "reason": reason or "审核员认为需补叫专家"}
    if first_line.startswith("VERDICT:REVISE"):
        parts = re.split(r"\n\s*---\s*\n", text, maxsplit=1)
        revised = parts[1].strip() if len(parts) == 2 else ""
        return "REVISE", revised or None
    return "UNKNOWN", None


def critic_node(state):
    draft = state.get("draft_answer", "") or ""
    if not draft:
        # Nothing to review (e.g., all experts produced empty). Fail safe.
        return {
            "messages": [AIMessage(content="抱歉，未能生成有效回答，请重试。")],
            "critic_verdict": "EMPTY",
        }

    user_id = state.get("profile_user_id", "default_user")
    try:
        profile_text = profile_to_prompt_text(get_profile_from_store(user_id))
    except Exception:
        profile_text = "（画像不可用）"

    notes_text = _format_notes(state.get("agent_notes") or {})
    user_question = get_user_question(state) or "（未获取到原始问题）"

    safety_section = _retrieve_safety(user_question, draft)
    medical_section, medical_hit = _retrieve_medical_context(user_question, draft)

    executed = list(state.get("executed") or [])
    replan_count = int(state.get("replan_count", 0) or 0)
    unexecuted = [r for r in _ALL_EXPERTS if r not in executed and r != "General"]
    can_replan = replan_count < REPLAN_CAP and bool(unexecuted)

    review_prompt = (
        f"用户画像：\n{profile_text}\n\n"
        f"用户本轮问题：\n{user_question}\n\n"
        f"本轮已派出专家：{', '.join(executed) if executed else '（无）'}\n"
        f"尚未派出但可补叫：{', '.join(unexecuted) if unexecuted else '（无可补叫）'}\n"
        f"已 replan 次数：{replan_count}（上限 {REPLAN_CAP}）\n\n"
        f"各专家共享 scratchpad 要点：\n{notes_text}\n\n"
        f"{safety_section}"
        f"{medical_section}"
        f"即将发送给用户的草稿回答：\n{draft}\n\n"
        + ("当前补员配额已用尽，请只在 PASS / REVISE 中二选一。\n" if not can_replan else "")
        + "请按指定格式给出审核结论。"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=_CRITIC_SYSTEM),
            HumanMessage(content=review_prompt),
        ])
        raw = extract_text_content(response)
    except Exception as e:
        guarded = has_safety_risk(user_question, draft, notes_text, profile_text)
        final_text = add_safety_warning(draft) if guarded else draft
        verdict = f"ERROR_GUARDED:{type(e).__name__}" if guarded else f"ERROR:{type(e).__name__}"
        if medical_hit:
            verdict = f"{verdict}+MED"
        try:
            append_episode(
                user_id=user_id,
                query=get_user_question(state) or "",
                experts=state.get("executed") or [],
                gist=final_text,
            )
        except Exception:
            pass
        return {
            "messages": [AIMessage(content=final_text)],
            "critic_verdict": verdict,
        }

    verdict, payload = _parse_verdict(raw)

    # ---- REPLAN path: hand back to Planner to bring in another expert ----
    if verdict == "REPLAN" and can_replan:
        info = payload or {}
        needed = info.get("needed") or ""
        reason = info.get("reason") or "审核员认为需补叫专家"
        # Sanitize: only honor REPLAN if the requested role is genuinely missing.
        if needed and needed not in executed:
            replan_marker = f"REPLAN+MED" if medical_hit else "REPLAN"
            return {
                # No messages — second Critic run will emit the final answer.
                "critic_verdict": replan_marker,
                "replan_context": f"安全审核员要求补叫 {needed}：{reason}",
                "replan_count": replan_count + 1,
                # Clear stale draft so Aggregator regenerates after the new expert returns.
                "draft_answer": "",
            }
        # REPLAN requested but target invalid → fall through to PASS/REVISE handling.
        verdict = "PASS"
        payload = None

    if verdict == "REVISE" and payload:
        final_text = payload
    else:
        # PASS / UNKNOWN / REPLAN-without-budget → ship the draft unchanged.
        final_text = draft
        if verdict == "UNKNOWN":
            verdict = "PASS_FALLBACK"
        elif verdict == "REPLAN":
            verdict = "PASS_REPLAN_BUDGETED"

    if medical_hit:
        verdict = f"{verdict}+MED"

    try:
        append_episode(
            user_id=user_id,
            query=get_user_question(state) or "",
            experts=state.get("executed") or [],
            gist=final_text,
        )
    except Exception:
        pass

    return {
        "messages": [AIMessage(content=final_text)],
        "critic_verdict": verdict,
    }
