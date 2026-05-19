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
from ..personalization import (
    build_personalization_ctx,
    build_personalization_decision_points,
    check_decision_points_landed,
    format_decision_points_for_prompt,
    profile_anchor_terms,
)
from ..tools import retrieve_safety_guidelines
from .doctor import ensure_doctor_disclaimer
from .dispatcher import REPLAN_CAP
from .fallbacks import add_safety_warning, has_safety_risk
from .query_rewriter import get_user_question
from ._scratchpad import extract_facts_from_notes


# All specialist child agents the Critic can request as replan reinforcement. Kept in sync
# with dispatcher.EXPERT_RUNNERS — used to decide if any unexecuted expert
# is still available before firing REPLAN.
_ALL_EXPERTS = ("Analyst", "Trainer", "Nutritionist", "Psychologist", "Doctor")

_CRITICAL_REVIEW_PATTERN = re.compile(
    r"ACL|前交叉|韧带|半月板|术后|康复|撕裂|骨折|肩袖|腰椎|椎间盘|冠心|心脏|糖尿|"
    r"胸痛|胸闷|呼吸困难|心率|心悸|头晕|晕厥|血压|血糖|"
    r"布洛芬|对乙酰氨基酚|阿司匹林|抗生素|处方|剂量|怀孕|哺乳|过敏|"
    r"诊断|是什么病|轻生|自伤|极端低卡|低于\s*1200|低于\s*1500",
    re.IGNORECASE,
)

_DETERMINISTIC_SAFE_PATTERN = re.compile(
    r"以你目前.*(?:ACL|半月板|肩袖|腰椎|膝).*?(?:医生|理疗师|康复科).*?(?:评估|许可|门槛|确认)"
    r"|以你目前.*膝关节炎.*?(?:缓慢|渐进).*?(?:康复科|理疗师|咨询)"
    r"|以你目前.*运动后心跳.*?(?:检查|心内科|医生评估)"
    r"|熬夜后.*?(?:训练强度|RPE).*?(?:睡眠|作息)"
    r"|(?:10\s*(?:K|公里)|5\s*(?:K|公里)|半马|马拉松).*?(?:跑量|补给|减量)"
    r"|比赛补给.*?跑量"
    r"|我不能给出继续用药或具体剂量建议"
    r"|我不能给你开处方"
    r"|我不能根据文字判断你.*是什么病",
    re.IGNORECASE | re.DOTALL,
)

_FORBIDDEN_ACL_EXERCISE = re.compile(r"ACL|前交叉|韧带", re.IGNORECASE)
_OPEN_CHAIN_KNEE_EXTENSION = re.compile(r"坐姿(?:腿屈伸|伸膝)|腿屈伸", re.IGNORECASE)
_ACTUATION_RECORD_CLAIM = re.compile(r"已(?:经)?(?:帮你)?(?:记录|写入|保存)|已(?:经)?(?:安排|排好)", re.IGNORECASE)
_ACTUATION_REMINDER_CLAIM = re.compile(r"已(?:经)?(?:帮你)?(?:设置|设好|创建|安排).{0,8}提醒|提醒已(?:设置|创建|写入)", re.IGNORECASE)
_PRECISE_NUTRITION_NUMBER = re.compile(r"\d+(?:\.\d+)?\s*(?:kcal|千卡|大卡|g|克)", re.IGNORECASE)


_EXPERT_LABELS = {
    "Analyst": "数据分析师",
    "Trainer": "训练教练",
    "Nutritionist": "营养师",
    "Psychologist": "心理疗愈师",
    "Doctor": "医学顾问",
    "Orchestrator": "主助手",
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
    · 长期失眠/重度压力情绪/心理安全风险 → 缺少 Psychologist
    · 症状风险 / 用药 / 处方 / 诊断边界 → 缺少 Doctor
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
- 如果本轮已派出 Doctor，最终回答必须包含“仅供参考，如有不适请就医”；若缺失，请 REVISE 补上
- Actuation 真实性：草稿若声称“已记录 / 已保存 / 已写入 / 已设提醒 / 已安排”，必须能在系统提供的 actuation_log 中看到对应 ok=true 的成功流水；否则 REVISE，删除或改成“可以帮你记录/建议设置”。
- Vision 置信度：若系统提供的 vision_extractions.meal.confidence < 0.5，草稿不得用确定语气给出精确热量或宏量营养素；必须写成粗估/区间/需用户补充确认。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【P3 · 个性化落地缺失（REVISE）】
若用户画像中存在有意义数据，且本轮用户问题是在索要健康建议/方案/计划/评估，
必须按系统提供的“必须融入正文的个性化决策点”检查，而不是只看有没有提到画像。
- 若系统提供 2 个及以上决策点，草稿至少要自然落地其中 2 个；
- 若系统只提供 1 个决策点，草稿必须落地这 1 个；
- “以你 80kg/30岁来看”然后给通用模板，不算落地；
- 落地必须体现画像改变了具体数字、方案、动作限制、就医边界或压力/睡眠场景动作。
未满足 → REVISE。修订时必须保留安全边界，把缺失决策点融入对应正文段落，不要作为开头清单照搬。
纯寒暄、感谢、告别、空画像不触发 P3。危机/急症/处方场景优先安全，不要为了个性化添加无关营养或训练数字。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
判断克制原则：
- R0 与 P0/P1/P2/P3 都不触发 → PASS
- 措辞不够漂亮、缺少边角细节 → PASS
- 优先级：R0 > P0 > P1 > P2。命中 R0 就输出 REPLAN，不要降级到 REVISE。
- 如系统注入了"安全知识库参考"，请把其中的红线作为硬约束。

输出格式（严格遵守，三选一）：
- 通过：仅一行 `VERDICT: PASS`
- 补员：两行
```
VERDICT: REPLAN
NEEDED: <Trainer|Nutritionist|Psychologist|Doctor>（恰好一个角色名，必须是已派出列表里没有的）
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


def _profile_has_high_risk(profile: dict) -> bool:
    stats = profile.get("physical_stats") or {}
    injuries = " ".join(str(x) for x in (stats.get("injuries") or []))
    dietary = profile.get("dietary_context") or {}
    prefs = " ".join(str(x) for x in (dietary.get("preferences") or []))
    return bool(_CRITICAL_REVIEW_PATTERN.search(f"{injuries}\n{prefs}"))


def _decision_points_for_roles(pctx: dict, user_question: str, executed: list[str]) -> list:
    roles = [role for role in (executed or []) if role != "Orchestrator"] or ["Orchestrator"]
    points = []
    seen = set()
    for role in roles:
        for point in build_personalization_decision_points(pctx, user_question, role=role):
            if point.id in seen:
                continue
            seen.add(point.id)
            points.append(point)
    return sorted(points, key=lambda p: (p.priority, p.domain, p.id))


def _format_p3_check(check: dict) -> str:
    if not check or not check.get("required_total"):
        return "无本轮必检决策点。"
    status = "通过" if check.get("satisfied") else "未通过"
    missing = check.get("missing_instructions") or []
    missing_text = "\n".join(f"- {item}" for item in missing[:4]) if missing else "（无）"
    return (
        f"P3 预检{status}：需落地 {check.get('required_to_land')} / {check.get('required_total')} 个决策点，"
        f"当前落地 {check.get('landed_count')} 个。\n"
        f"已落地ID：{', '.join(check.get('landed_ids') or []) or '（无）'}\n"
        f"缺失或未充分落地的决策点：\n{missing_text}"
    )


def _format_json_section(value) -> str:
    if not value:
        return "（无）"
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(value)


def _has_successful_action(actuation_log: list[dict], *prefixes: str) -> bool:
    for event in actuation_log or []:
        if not isinstance(event, dict) or not event.get("ok"):
            continue
        action = str(event.get("action") or "")
        if any(action.startswith(prefix) for prefix in prefixes):
            return True
    return False


def _actuation_claim_issue(draft: str, actuation_log: list[dict]) -> str:
    if _ACTUATION_REMINDER_CLAIM.search(draft or "") and not _has_successful_action(actuation_log, "push_reminder"):
        return "REMINDER"
    if _ACTUATION_RECORD_CLAIM.search(draft or "") and not _has_successful_action(
        actuation_log,
        "log_meal",
        "log_workout",
        "log_wellness",
        "push_reminder",
    ):
        return "RECORD"
    return ""


def _soften_unverified_actuation_claims(draft: str, issue: str) -> str:
    text = draft or ""
    if issue == "REMINDER":
        text = re.sub(r"已(?:经)?(?:帮你)?(?:设置|设好|创建|安排).{0,8}提醒", "我建议你设置一个提醒", text)
        text = re.sub(r"提醒已(?:设置|创建|写入)", "提醒可以设置", text)
        return text
    if issue == "RECORD":
        text = re.sub(r"已(?:经)?(?:帮你)?(?:记录|写入|保存)", "建议记录", text)
        text = re.sub(r"已(?:经)?(?:安排|排好)", "可以安排", text)
    return text


def _low_confidence_meal(vision_extractions: dict) -> bool:
    meal = (vision_extractions or {}).get("meal")
    if not isinstance(meal, dict):
        return False
    try:
        return float(meal.get("confidence") or 0) < 0.5
    except Exception:
        return False


def _can_fast_pass(user_question: str, draft: str, profile: dict, executed: list[str], p3_check: dict) -> bool:
    """Skip the expensive LLM critic for straightforward low-risk answers."""
    if not draft or not executed:
        return False
    if p3_check and not p3_check.get("satisfied", True):
        return False
    stats = profile.get("physical_stats") or {}
    injuries_text = " ".join(str(x) for x in (stats.get("injuries") or []))
    if _FORBIDDEN_ACL_EXERCISE.search(f"{user_question or ''}\n{injuries_text}") and _OPEN_CHAIN_KNEE_EXTENSION.search(draft):
        return False
    if _DETERMINISTIC_SAFE_PATTERN.search(draft):
        return True
    if _profile_has_high_risk(profile):
        return False
    if _CRITICAL_REVIEW_PATTERN.search(f"{user_question or ''}\n{draft or ''}"):
        return False
    # Keep multi-specialist synthesis under the critic; single low-risk specialist
    # answers are where the cost/benefit is weakest.
    return len(executed) == 1


def critic_node(state):
    draft = state.get("draft_answer", "") or ""
    if not draft:
        # Nothing to review (e.g., all experts produced empty). Fail safe.
        return {
            "messages": [AIMessage(content="抱歉，未能生成有效回答，请重试。")],
            "critic_verdict": "EMPTY",
        }

    user_id = state.get("profile_user_id", "default_user")
    pctx = state.get("personalization_ctx") or {}
    if not pctx:
        try:
            pctx = build_personalization_ctx(user_id)
        except Exception:
            pctx = {}
    profile_text = pctx.get("raw_profile_json") or "（画像不可用）"
    profile = pctx.get("raw_profile") or {}
    user_card = pctx.get("user_card") or "（用户卡片不可用）"
    anchors = "、".join(profile_anchor_terms(pctx.get("raw_profile") or {})[:20])

    notes_text = _format_notes(state.get("agent_notes") or {})
    user_question = get_user_question(state) or "（未获取到原始问题）"
    actuation_log = state.get("actuation_log") or []
    vision_extractions = state.get("vision_extractions") or {}
    recent_logs_summary = state.get("recent_logs_summary") or ""

    executed = [role for role in (state.get("executed") or []) if role != "Orchestrator"]
    replan_count = int(state.get("replan_count", 0) or 0)
    unexecuted = [r for r in _ALL_EXPERTS if r not in executed]
    can_replan = replan_count < REPLAN_CAP and bool(unexecuted)
    doctor_executed = "Doctor" in executed

    decision_points = _decision_points_for_roles(pctx, user_question, executed)
    decision_section = format_decision_points_for_prompt(decision_points)
    p3_check = check_decision_points_landed(draft, decision_points)
    p3_check_text = _format_p3_check(p3_check)

    actuation_issue = _actuation_claim_issue(draft, actuation_log)
    if actuation_issue:
        final_text = _soften_unverified_actuation_claims(draft, actuation_issue)
        if doctor_executed:
            final_text = ensure_doctor_disclaimer(final_text)
        try:
            append_episode(
                user_id=user_id,
                query=get_user_question(state) or "",
                experts=executed,
                gist=final_text,
                facts=extract_facts_from_notes(state.get("agent_notes") or {}, get_user_question(state) or ""),
            )
        except Exception:
            pass
        return {
            "messages": [AIMessage(content=final_text)],
            "critic_verdict": f"REVISE_RULE_ACTUATION_{actuation_issue}",
        }

    if _low_confidence_meal(vision_extractions) and _PRECISE_NUTRITION_NUMBER.search(draft):
        final_text = (
            "图片识别置信度偏低，下面的热量和宏量营养素只能当作粗估范围，最好用食材重量或包装信息再确认。\n\n"
            f"{draft}"
        )
        if doctor_executed:
            final_text = ensure_doctor_disclaimer(final_text)
        try:
            append_episode(
                user_id=user_id,
                query=get_user_question(state) or "",
                experts=executed,
                gist=final_text,
                facts=extract_facts_from_notes(state.get("agent_notes") or {}, get_user_question(state) or ""),
            )
        except Exception:
            pass
        return {
            "messages": [AIMessage(content=final_text)],
            "critic_verdict": "REVISE_RULE_VISION_CONFIDENCE",
        }

    if _can_fast_pass(user_question, draft, profile, executed, p3_check):
        try:
            append_episode(
                user_id=user_id,
                query=get_user_question(state) or "",
                experts=executed,
                gist=draft,
                facts=extract_facts_from_notes(state.get("agent_notes") or {}, get_user_question(state) or ""),
            )
        except Exception:
            pass
        return {
            "messages": [AIMessage(content=draft)],
            "critic_verdict": "PASS_RULE",
        }

    stats = profile.get("physical_stats") or {}
    injuries_text = " ".join(str(x) for x in (stats.get("injuries") or []))
    if _FORBIDDEN_ACL_EXERCISE.search(f"{user_question}\n{injuries_text}") and _OPEN_CHAIN_KNEE_EXTENSION.search(draft):
        final_text = _OPEN_CHAIN_KNEE_EXTENSION.sub("股四头肌等长收缩", draft)
        final_text = (
            "ACL 术后 6 个月的膝关节训练要先经过运动医学医生或理疗师评估；我把可能增加前向剪切力的开链伸膝动作移除，"
            "改成更保守的等长和低冲击方案。\n\n"
            f"{final_text}"
        )
        if doctor_executed:
            final_text = ensure_doctor_disclaimer(final_text)
        try:
            append_episode(
                user_id=user_id,
                query=get_user_question(state) or "",
                experts=executed,
                gist=final_text,
                facts=extract_facts_from_notes(state.get("agent_notes") or {}, get_user_question(state) or ""),
            )
        except Exception:
            pass
        return {
            "messages": [AIMessage(content=final_text)],
            "critic_verdict": "REVISE_RULE_ACL",
        }

    safety_section = _retrieve_safety(user_question, draft)
    decision_section_text = decision_section or "【必须融入正文的个性化决策点】\n（无）\n"
    replan_budget_instruction = "" if can_replan else "当前补员配额已用尽，请只在 PASS / REVISE 中二选一。\n"

    review_prompt = (
        f"用户卡片（与专家看到的个性化上下文一致）：\n{user_card}\n\n"
        f"用户画像原始 JSON（安全审查用）：\n{profile_text}\n\n"
        f"可用于 P3 个性化校验的画像锚点：{anchors or '（无）'}\n\n"
        f"{decision_section_text}\n"
        f"确定性 P3 预检结果：\n{p3_check_text}\n\n"
        f"用户本轮问题：\n{user_question}\n\n"
        f"本轮图片/视觉抽取结果：\n{_format_json_section(vision_extractions)}\n\n"
        f"近7日结构化日志摘要：\n{recent_logs_summary or '（无）'}\n\n"
        f"本轮真实 side-effect / actuation_log：\n{_format_json_section(actuation_log)}\n\n"
        f"本轮已派出专家：{', '.join(executed) if executed else '（无）'}\n"
        f"尚未派出但可补叫：{', '.join(unexecuted) if unexecuted else '（无可补叫）'}\n"
        f"已 replan 次数：{replan_count}（上限 {REPLAN_CAP}）\n\n"
        f"各专家共享 scratchpad 要点：\n{notes_text}\n\n"
        f"{safety_section}"
        f"即将发送给用户的草稿回答：\n{draft}\n\n"
        f"{replan_budget_instruction}"
        "请按指定格式给出审核结论。"
    )

    try:
        last_error = None
        raw = ""
        for _ in range(2):
            try:
                response = llm.invoke([
                    SystemMessage(content=_CRITIC_SYSTEM),
                    HumanMessage(content=review_prompt),
                ])
                raw = extract_text_content(response)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
    except Exception as e:
        guarded = has_safety_risk(user_question, draft, notes_text, profile_text)
        final_text = add_safety_warning(draft) if guarded else draft
        if doctor_executed:
            final_text = ensure_doctor_disclaimer(final_text)
        verdict = f"ERROR_GUARDED:{type(e).__name__}" if guarded else f"ERROR:{type(e).__name__}"
        try:
            append_episode(
                user_id=user_id,
                query=get_user_question(state) or "",
                experts=[role for role in (state.get("executed") or []) if role != "Orchestrator"],
                gist=final_text,
                facts=extract_facts_from_notes(state.get("agent_notes") or {}, get_user_question(state) or ""),
            )
        except Exception:
            pass
        return {
            "messages": [AIMessage(content=final_text)],
            "critic_verdict": verdict,
        }

    verdict, payload = _parse_verdict(raw)

    # ---- REPLAN path: hand back to Orchestrator to bring in another expert ----
    if verdict == "REPLAN" and can_replan:
        info = payload or {}
        needed = info.get("needed") or ""
        reason = info.get("reason") or "审核员认为需补叫专家"
        # Sanitize: only honor REPLAN if the requested role is genuinely missing.
        if needed and needed not in executed:
            return {
                # No messages — second Critic run will emit the final answer.
                "critic_verdict": "REPLAN",
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

    if doctor_executed:
        final_text = ensure_doctor_disclaimer(final_text)

    try:
        append_episode(
            user_id=user_id,
            query=get_user_question(state) or "",
            experts=[role for role in (state.get("executed") or []) if role != "Orchestrator"],
            gist=final_text,
            facts=extract_facts_from_notes(state.get("agent_notes") or {}, get_user_question(state) or ""),
        )
    except Exception:
        pass

    return {
        "messages": [AIMessage(content=final_text)],
        "critic_verdict": verdict,
    }
