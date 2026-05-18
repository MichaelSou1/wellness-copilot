"""Orchestrator — the parent health-guide agent.

The orchestrator owns ordinary conversation and high-level delegation:

* direct replies for chitchat, capability questions, profile/style updates,
  clarification, and medical-boundary refusals;
* specialist plans for Trainer / Nutritionist / Wellness only;
* append-only replan decisions when Critic or ReplanJudge asks for one more
  specialist.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..episode_store import append_episode
from ..llm import extract_text_content, llm
from ..personalization import build_personalization_ctx, profile_routing_digest
from ..tools import (
    add_dietary_preference,
    add_injury,
    add_stress_source,
    set_dietary_goal,
    set_physical_stats,
    set_response_style,
    update_user_profile,
)
from ..utils import create_agent
from ._scratchpad import build_scratchpad_note, extract_facts_from_notes
from .fallbacks import empty_replan, orchestrator_error_answer
from .query_rewriter import get_user_question


# ---- Routing signals -------------------------------------------------------

_TRAINING_INTENT = re.compile(
    r"训练|锻炼|健身|动作|计划|频次|频率|强度|"
    r"减脂|增肌|塑形|燃脂|瘦身|减重|增重|健身房|"
    r"有氧|力量|跑步|跑量|游泳|骑行|跳绳|HIIT|拉伸|"
    r"运动量|活动量|腿|腰|肩|背|核心|腹|胸|臂|比赛|备赛|10K|半马|马拉松",
    re.IGNORECASE,
)

_FOOD_INTENT = re.compile(
    r"吃|食|餐|喝|饮食|营养|蛋白|碳水|脂肪|热量|kcal|大卡|"
    r"零食|早餐|午餐|晚餐|补剂|蛋白粉|奶|蔬菜|水果|食谱|菜谱|"
    r"补给|碳水加载|赛前餐|赛中",
    re.IGNORECASE,
)

_WELLNESS_INTENT = re.compile(
    r"睡|失眠|入睡|压力|焦虑|抑郁|情绪|疲劳|倦怠|心情|放松|休息|"
    r"恢复差|恢复不好|恢复不过来|恢复节奏",
    re.IGNORECASE,
)

_STYLE_INTENT = re.compile(
    r"简洁|详细|啰嗦|幽默|正式|口语|英文|中文|回答风格|语气|tone|concise|brief",
    re.IGNORECASE,
)

_PROFILE_UPDATE_INTENT = re.compile(
    r"记一下|记录|更新|保存|我的|我\s*(?:体重|身高|年龄)|"
    r"不吃|过敏|喜欢|偏好|目标|压力源|回答.*(?:简洁|详细|幽默|正式|英文|中文)",
    re.IGNORECASE,
)

_MEDICAL_BOUNDARY_INTENT = re.compile(
    r"是什么病|诊断|处方|开药|开一张|剂量|吃多少药|服药量|"
    r"布洛芬|对乙酰氨基酚|阿司匹林|抗生素|胸痛|胸闷|呼吸困难|"
    r"晕厥|头晕|心率异常|静息心率|血压|血糖",
    re.IGNORECASE,
)

_TRAINER_ONLY_TOPICS = re.compile(
    r"TDEE|BMR|基础代谢|总能量消耗|maintenance.?calories|代谢率",
    re.IGNORECASE,
)

_CARDIAC_EXERCISE_SIGNAL = re.compile(
    r"(?:运动|训练|跑步|健身|锻炼).{0,8}(?:胸闷|胸痛|胸口闷|胸口痛|心悸|心慌|头晕|晕|喘不上)"
    r"|(?:胸闷|胸痛|胸口闷|胸口痛|心悸|心慌|头晕|晕|喘不上).{0,8}(?:运动|训练|跑步|健身|锻炼)",
    re.IGNORECASE,
)

_ENDURANCE_RACE_SIGNAL = re.compile(
    r"10K|5K|半马|马拉松|越野赛|比赛|备赛|跑量|补给|碳水加载|赛前",
    re.IGNORECASE,
)

_VALID_EXPERTS = {"Trainer", "Nutritionist", "Wellness"}
_PRIORITY_ORDER = {"Trainer": 0, "Nutritionist": 1, "Wellness": 2}


_ROUTING_SYSTEM = """\
你是 Health Guide 的主 agent / orchestrator。你要判断当前用户消息是由你直接回应，还是需要调用专业子 agent。

可调用的专业子 agent 只有：
- Trainer：训练、运动动作、运动计划、伤病/康复期负荷、TDEE/BMR、比赛训练
- Nutritionist：饮食、营养、热量、蛋白质、补剂、食谱、比赛补给
- Wellness：睡眠、压力、焦虑/情绪、疲劳、恢复不过来/恢复节奏、心理安全边界；不要因为普通"运动后帮助恢复"就调用

直接由你回应（输出 DIRECT）的情况：
- 寒暄、感谢、告别、能力介绍、问名字
- 纯个人信息/偏好/回答风格更新
- 轻量澄清、范围说明、非专业闲聊
- 医疗边界/拒诊/拒开处方/用药剂量边界；除非问题同时明确涉及训练、饮食或心理危机，需要对应子 agent

调用子 agent 的情况：
- 用户要求具体训练/动作/运动安排 → Trainer
- 用户要求具体饮食/热量/营养/补剂/食谱 → Nutritionist
- 用户要求睡眠/压力/情绪/疲劳/恢复方案 → Wellness
- 同时涉及多个领域时输出多个角色，按 Trainer,Nutritionist,Wellness 排序

强制规则：
- TDEE/BMR/基础代谢/每日总消耗 → Trainer
- 10K/半马/马拉松/比赛/备赛/跑量 + 补给/饮食/碳水 → Trainer,Nutritionist
- 训练后睡不好/疲劳/恢复差/恢复不过来 → Trainer,Wellness
- 自伤/轻生/严重情绪危机 → Wellness
- 运动中胸闷/胸痛/心悸/头晕 → Trainer
- 若用户画像或近期记录中有伤病/术后，且当前问题会改变训练负荷或体型目标 → 至少包含 Trainer
- 若用户画像有过敏/饮食禁忌，且当前问题涉及吃喝/补剂 → 至少包含 Nutritionist

输出格式严格二选一：
- 直接回应：DIRECT
- 调用专家：Trainer / Nutritionist / Wellness，多个用英文逗号分隔，不加空格
不要输出解释。
"""

_REPLAN_SYSTEM = """\
你是 Health Guide 的 orchestrator。已有专家回答后，系统要求你判断是否追加专业子 agent。

可选角色: Trainer, Nutritionist, Wellness。
规则：
1. 不能重复已经派出过的角色。
2. 只追加确实必要的角色，默认 1 个，最多 2 个。
3. 如果不需要追加，直接输出 NONE。
4. 否则按 Trainer,Nutritionist,Wellness 输出，多个用英文逗号分隔，不加空格。
不要输出解释。
"""

_DIRECT_TOOLS = [
    set_physical_stats,
    add_injury,
    set_dietary_goal,
    add_dietary_preference,
    add_stress_source,
    set_response_style,
    update_user_profile,
]


def _sort_by_priority(roles: List[str]) -> List[str]:
    seen = []
    for role in roles:
        if role in _PRIORITY_ORDER and role not in seen:
            seen.append(role)
    seen.sort(key=lambda r: _PRIORITY_ORDER[r])
    return seen


def _parse_role_list(content: str) -> List[str]:
    content = (content or "").strip().replace("'", "").replace('"', "")
    if not content or content.upper() in {"DIRECT", "NONE"}:
        return []
    roles = [r.strip() for r in content.split(",")]
    return [r for r in roles if r in _VALID_EXPERTS]


def _profile_has_injury(profile: dict) -> bool:
    stats = profile.get("physical_stats") or {}
    return bool(stats.get("injuries"))


def _profile_has_dietary_restriction(profile: dict) -> bool:
    dietary = profile.get("dietary_context") or {}
    prefs = dietary.get("preferences") or []
    if not prefs:
        return False
    restriction_hits = re.compile(r"过敏|不耐|纯素|素食|忌|不吃|不喝|无糖|低糖|无麸质")
    return any(restriction_hits.search(str(p)) for p in prefs)


def _profile_has_chronic_condition(profile: dict, summary: str) -> bool:
    return bool(re.search(r"心脏|冠心|糖尿|高血压|哮喘|肾病", summary or ""))


def _heuristic_roles(profile: dict, profile_summary: str, query: str) -> List[str]:
    text = query or ""
    roles: List[str] = []

    if (
        _MEDICAL_BOUNDARY_INTENT.search(text)
        and not _CARDIAC_EXERCISE_SIGNAL.search(text)
        and not _TRAINING_INTENT.search(text)
        and not _WELLNESS_INTENT.search(text)
    ):
        return []

    if _TRAINING_INTENT.search(text):
        roles.append("Trainer")
    if _FOOD_INTENT.search(text):
        roles.append("Nutritionist")
    if _WELLNESS_INTENT.search(text):
        roles.append("Wellness")

    if _TRAINER_ONLY_TOPICS.search(text):
        roles.append("Trainer")

    if _ENDURANCE_RACE_SIGNAL.search(text):
        roles.append("Trainer")
        if _FOOD_INTENT.search(text) or re.search(r"补给|碳水|赛前|赛中|吃|喝", text):
            roles.append("Nutritionist")

    if _CARDIAC_EXERCISE_SIGNAL.search(text):
        roles.append("Trainer")

    if _profile_has_injury(profile) and (_TRAINING_INTENT.search(text) or re.search(r"减脂|增肌|塑形|瘦身", text)):
        roles.append("Trainer")

    if _profile_has_dietary_restriction(profile) and _FOOD_INTENT.search(text):
        roles.append("Nutritionist")

    if _profile_has_chronic_condition(profile, profile_summary) and _TRAINING_INTENT.search(text):
        roles.append("Trainer")

    if _STYLE_INTENT.search(text) and not (
        _TRAINING_INTENT.search(text) or _FOOD_INTENT.search(text) or _WELLNESS_INTENT.search(text)
    ):
        return []

    if _MEDICAL_BOUNDARY_INTENT.search(text) and not roles:
        return []

    if _PROFILE_UPDATE_INTENT.search(text) and not (
        _TRAINING_INTENT.search(text) or _FOOD_INTENT.search(text) or _WELLNESS_INTENT.search(text)
    ):
        return []

    return _sort_by_priority(roles)


def _enforce_rules(plan: List[str], profile: dict, profile_summary: str, query: str) -> List[str]:
    merged = list(plan)
    for role in _heuristic_roles(profile, profile_summary, query):
        if role not in merged:
            merged.append(role)
    text = query or ""
    if (
        "Wellness" in merged
        and not _WELLNESS_INTENT.search(text)
        and (_TRAINING_INTENT.search(text) or _FOOD_INTENT.search(text))
    ):
        merged = [role for role in merged if role != "Wellness"]
    return _sort_by_priority(merged)


def _format_responses(responses: Dict[str, str]) -> str:
    if not responses:
        return "（无）"
    parts = []
    for role, value in responses.items():
        snippet = (value or "").strip()
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"
        parts.append(f"[{role}]\n{snippet}")
    return "\n\n".join(parts)


def _episode_section(episode_context: str) -> str:
    if not episode_context:
        return ""
    return (
        "\n【近期/相关对话记录】\n"
        f"{episode_context}\n"
        "说明：[最近] 表示按时间最近的对话，[相关] 表示因话题相似而召回的更早对话；引用时请区分使用。\n"
    )


def _build_direct_agent(pctx: dict, episode_context: str = ""):
    user_card = pctx.get("user_card") or "【关于该用户】\n用户画像暂不可用。"
    system_prompt = (
        "你是 Health Guide 的主 agent / orchestrator，团队核心定位是健康管理：饮食、运动、睡眠、压力、伤病恢复。"
        f"\n\n{user_card}\n"
        f"{_episode_section(episode_context)}"
        "你负责直接处理寒暄、感谢、告别、能力介绍、澄清、画像/偏好记录、回答风格记录、"
        "通用健康常识和医疗边界说明。专业训练/营养/身心方案通常已由上游改派子 agent；"
        "如果这里仍遇到明显专业方案请求，先给简短方向并邀请用户继续细化，不要冒充医生或处方者。\n"
        "用户卡片就是本轮可用画像；不要说「我先看看/了解你的基本信息」，不要为了读取画像而调用工具。"
        "只有用户提供新信息时才调用结构化工具记录。"
        "若用户提供年龄、身高、体重、伤病、饮食目标/偏好、压力源或回答风格，请优先调用对应 set_/add_ 工具；"
        "update_user_profile 仅作兼容兜底。"
        "当用户明确要求「简洁/详细/幽默/正式/口语/英文/中文」等回答风格时，必须调用 set_response_style。"
        "对纯打招呼、感谢、告别、问名字、能力范围，简短自然回应，不要检索。"
        "当用户问「你能做什么」时，先说健康能力：训练与运动建议、饮食与营养规划、睡眠与压力管理、"
        "伤病/康复期注意事项、根据画像做个性化方案。"
        "对于「天气真好」「最近不想动」这类轻量消息，可以给 1 条低门槛行动建议，例如快走 20-30 分钟或轻松拉伸。"
        "【医疗边界】不能诊断疾病、开处方、给处方药剂量或保证「没事」。"
        "涉及胸痛/胸闷、呼吸困难、晕厥、明显心率异常、持续疼痛、用药剂量、处方或自我诊断时，"
        "必须先建议就医/医生评估；不要列未经确认的诊断猜测，也不要使用「可能是 X 病」替代就医建议。"
        "回答保持自然、简洁、像同一个主助手在说话。"
    )
    return create_agent(llm, _DIRECT_TOOLS, system_prompt)


def _record_episode(user_id: str, query: str, answer: str, experts: List[str]) -> None:
    try:
        notes = {"Orchestrator": answer}
        append_episode(
            user_id=user_id,
            query=query or "",
            experts=experts,
            gist=answer,
            facts=extract_facts_from_notes(notes, query or ""),
        )
    except Exception:
        pass


def _direct_answer(state, *, error: Exception | None = None) -> dict:
    user_id = state.get("profile_user_id", "default_user")
    user_question = get_user_question(state)
    pctx = state.get("personalization_ctx") or {}
    if not pctx:
        try:
            pctx = build_personalization_ctx(user_id)
        except Exception:
            pctx = {}
    episode_context = state.get("episode_context") or ""

    try:
        os.environ["HEALTH_GUIDE_USER_ID"] = user_id
        if error is not None:
            raise error
        agent = _build_direct_agent(pctx, episode_context)
        result = agent.invoke({"messages": [HumanMessage(content=user_question)]})

        used_tools: list[str] = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    used_tools.append(call.get("name", "Unknown"))

        retrieval_hits = sum(
            1 for tool_name in used_tools if "retrieve" in tool_name and "knowledge" in tool_name
        )
        answer = extract_text_content(result["messages"][-1])
        verdict = "REVISE_DIRECT" if _MEDICAL_BOUNDARY_INTENT.search(user_question or "") else "PASS_DIRECT"
    except Exception as exc:
        used_tools = [f"ERROR:Orchestrator:{type(exc).__name__}"]
        retrieval_hits = 0
        answer = orchestrator_error_answer(exc)
        verdict = f"ERROR_DIRECT:{type(exc).__name__}"

    _record_episode(user_id, user_question, answer, ["Orchestrator"])
    return {
        "messages": [AIMessage(content=answer)],
        "expert_responses": {"Orchestrator": answer},
        "agent_notes": {"Orchestrator": build_scratchpad_note("Orchestrator", answer)},
        "last_tools": used_tools,
        "retrieval_hits": retrieval_hits,
        "executed": ["Orchestrator"],
        "plan": [],
        "next": [],
        "replan_count": 0,
        "replan_context": "",
        "critic_verdict": verdict,
        "orchestrator_decision": "DIRECT",
    }


def _fresh_orchestrate(state) -> dict:
    user_id = state.get("profile_user_id", "default_user")
    pctx = state.get("personalization_ctx") or {}
    if not pctx:
        pctx = build_personalization_ctx(user_id)
    profile = pctx.get("raw_profile") or {}
    summary = pctx.get("routing_digest") or profile_routing_digest(profile)

    user_question = get_user_question(state)
    episode_ctx = (state.get("episode_context") or "").strip()

    parts = []
    if summary:
        parts.append(f"用户画像：{summary}")
    if episode_ctx:
        parts.append(
            "近期/相关对话记录：\n"
            f"{episode_ctx}\n"
            "说明：[最近] 表示按时间最近的对话，[相关] 表示因话题相似而召回的更早对话。"
        )
    parts.append(f"用户问题：{user_question}")

    try:
        response = llm.invoke([
            SystemMessage(content=_ROUTING_SYSTEM),
            HumanMessage(content="\n".join(parts)),
        ])
        content = extract_text_content(response).strip().replace("'", "").replace('"', "")
    except Exception as exc:
        roles = _heuristic_roles(profile, summary, user_question)
        if roles:
            return {
                "plan": roles,
                "executed": ["Orchestrator"],
                "replan_count": 0,
                "replan_context": "",
                "next": [],
                "orchestrator_decision": "PLAN",
            }
        return _direct_answer(state, error=exc)

    if content.upper() == "DIRECT":
        return _direct_answer(state)

    plan = _sort_by_priority(_parse_role_list(content))
    plan = _enforce_rules(plan, profile, summary, user_question)
    if not plan:
        return _direct_answer(state)

    return {
        "plan": plan,
        "executed": ["Orchestrator"],
        "replan_count": 0,
        "replan_context": "",
        "next": [],
        "orchestrator_decision": "PLAN",
    }


def _replan(state) -> dict:
    replan_ctx = state.get("replan_context", "")
    executed = state.get("executed") or []
    responses = state.get("expert_responses") or {}
    user_text = get_user_question(state)

    user_prompt = (
        f"用户最初的问题：\n{user_text or '（未获取到）'}\n\n"
        f"已经派出过的专家（按顺序）：{executed or '（无）'}\n\n"
        f"各专家已给出的回答摘要：\n{_format_responses(responses)}\n\n"
        f"补叫请求理由：\n{replan_ctx}\n\n"
        "请决定接下来要追加哪些专业子 agent。"
    )

    response = llm.invoke([
        SystemMessage(content=_REPLAN_SYSTEM),
        HumanMessage(content=user_prompt),
    ])
    content = extract_text_content(response).strip().replace("'", "").replace('"', "")

    if content.upper() == "NONE":
        return {
            "plan": [],
            "replan_context": "",
            "next": [],
            "orchestrator_decision": "NO_REPLAN",
        }

    candidates = _parse_role_list(content)
    filtered = [r for r in _sort_by_priority(candidates) if r not in executed]
    return {
        "plan": filtered,
        "replan_context": "",
        "next": [],
        "orchestrator_decision": "PLAN" if filtered else "NO_REPLAN",
    }


def orchestrator_node(state):
    try:
        if state.get("replan_context"):
            return _replan(state)
        return _fresh_orchestrate(state)
    except Exception as exc:
        if state.get("replan_context"):
            out = empty_replan()
            out["orchestrator_decision"] = "NO_REPLAN"
            return out
        return _direct_answer(state, error=exc)


# Backward compatibility for scripts importing the old planner symbol.
planner_node = orchestrator_node
