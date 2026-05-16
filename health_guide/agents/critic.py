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
from .fallbacks import add_safety_warning, has_safety_risk
from .query_rewriter import get_user_question


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

以下三类问题必须触发 REVISE，优先级从高到低：

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【P0 · 伤病安全硬限制】
若用户画像中记录了活跃伤病或康复中状态（如 ACL 撕裂、半月板损伤、骨折、术后康复、韧带撕裂等），
草稿对该伤病部位推荐了任何直接负重训练动作——哪怕加了"降级版""可控""轻度""箱式"等限定词——
且未明确注明"须在运动医学医生或理疗师亲自评估许可并全程监督下进行"，
→ 必须 REVISE。
康复期原则：没有专业医疗许可 = 不推荐该部位直接负重动作，不论动作难度高低。
修订方向：移除所有直接负重动作建议，改为"请先获得理疗师许可"，并列出零负重替代方案（如直腿抬高、等长收缩、水中训练）。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【P1 · 症状描述必须先导向就医】
若用户描述了身体症状（持续疼痛、心率异常、胸痛/胸闷、头晕、肿胀不消退等），
草稿在未首先明确建议就医的前提下，先给出了可能的病因分析、鉴别诊断或自我处理方案，
→ 必须 REVISE。
正确结构：①先明确建议就医（"请尽快就医/检查"）→ ②再提供辅助性信息（如有）。
不允许以"列出可能原因 + 进一步问诊"的方式替代就医建议。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【P2 · 其他健康安全风险】
- 极端热量限制：建议低于女性 1200 kcal/d 或男性 1500 kcal/d 且未说明医疗监督背景
- 已知慢病用户（冠心病、糖尿病等）被建议高风险运动（HIIT 等）而未要求先获医生许可
- 越权用药：给出诊断、处方或具体药物剂量建议
- 跨专家事实性矛盾：综合回答与各专家核心要点存在直接冲突

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
判断克制原则：
- 措辞不够漂亮、缺少边角细节 → PASS
- 仅存在 P2 中的轻微风险（如建议略激进但未到危险级别）→ PASS
- 只有上述规则明确命中时才 REVISE
- 如系统注入了"安全知识库参考"，请把其中的红线作为硬约束，与上述 P0/P1/P2 共同执行

输出格式（严格遵守）：
- 如果通过，仅输出一行：`VERDICT: PASS`
- 如果需要修订，输出：
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
    """Return (verdict, revised_or_none). verdict ∈ {'PASS','REVISE','UNKNOWN'}."""
    if not text:
        return "UNKNOWN", None
    first_line = text.strip().splitlines()[0].strip().upper()
    if first_line.startswith("VERDICT: PASS"):
        return "PASS", None
    if first_line.startswith("VERDICT: REVISE"):
        # split on first '---' line
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

    review_prompt = (
        f"用户画像：\n{profile_text}\n\n"
        f"用户本轮问题：\n{user_question}\n\n"
        f"各专家共享 scratchpad 要点：\n{notes_text}\n\n"
        f"{safety_section}"
        f"{medical_section}"
        f"即将发送给用户的草稿回答：\n{draft}\n\n"
        "请按指定格式给出审核结论。"
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

    verdict, revised = _parse_verdict(raw)

    if verdict == "REVISE" and revised:
        final_text = revised
    else:
        # PASS or UNKNOWN → ship the draft unchanged.
        final_text = draft
        if verdict == "UNKNOWN":
            verdict = "PASS_FALLBACK"

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
