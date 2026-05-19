"""Orchestrator — the parent health-guide agent.

The orchestrator is a real parent agent: it owns the user-facing turn and can
call specialist child agents as tools. Specialist selection is therefore part
of the parent agent's tool-use loop, not a LangGraph routing step.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from ..episode_store import append_episode
from ..llm import extract_text_content, llm
from ..personalization import apply_personalization_boost, build_personalization_ctx
from ..profile_store import update_user_profile as store_update_user_profile
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
from ._scratchpad import build_scratchpad_note, extract_facts_from_notes, format_peer_notes
from .doctor import ensure_doctor_disclaimer, run_doctor
from .fallbacks import orchestrator_error_answer
from .nutritionist import run_nutritionist
from .query_rewriter import get_user_question
from .trainer import run_trainer
from .psychologist import run_psychologist


_ADVICE_REQUEST_SIGNAL = re.compile(
    r"能不能|能否|可不可以|可以.*吗|能.*吗|怎么|如何|建议|计划|安排|方案|给.*建议|帮我|应该|多少|[？?]",
    re.IGNORECASE,
)

_INJURY_RECORD_SIGNAL = re.compile(
    r"诊断出|被诊断|医生说|术后|重建|康复|恢复期|损伤|撕裂|拉伤|骨折|半月板|ACL|前交叉|肩袖|腰痛",
    re.IGNORECASE,
)

_MEDICAL_BOUNDARY_INTENT = re.compile(
    r"是什么病|诊断|处方|开药|开一张|剂量|吃多少药|服药量|"
    r"布洛芬|对乙酰氨基酚|阿司匹林|抗生素|胸痛|胸闷|呼吸困难|"
    r"晕厥|头晕|晕晕|眩晕|恶心|呕吐|心率异常|静息心率|血压|血糖",
    re.IGNORECASE,
)

_SELF_DIAGNOSIS_QUESTION = re.compile(
    r"是什么病|你觉得.*病|是不是.*病|帮我诊断|能诊断|诊断一下|确诊|可能是",
    re.IGNORECASE,
)

_MEDICATION_OR_PRESCRIPTION = re.compile(
    r"处方|开药|开一张|剂量|吃多少药|服药量|布洛芬|对乙酰氨基酚|阿司匹林|抗生素",
    re.IGNORECASE,
)

_CARDIAC_EXERCISE_SIGNAL = re.compile(
    r"(?:运动|训练|跑步|健身|锻炼).{0,20}(?:胸闷|胸痛|胸口闷|胸口痛|心悸|心慌|心跳|心率|头晕|晕晕|眩晕|晕|喘不上|恶心)"
    r"|(?:胸闷|胸痛|胸口闷|胸口痛|心悸|心慌|心跳|心率|头晕|晕晕|眩晕|晕|喘不上|恶心).{0,20}(?:运动|训练|跑步|健身|锻炼)"
    r"|运动后.{0,12}(?:十几分钟|10\s*分钟|很快|恢复正常)",
    re.IGNORECASE,
)

_PHYSICAL_DISCOMFORT_SIGNAL = re.compile(
    r"头晕|晕晕|眩晕|晕厥|昏厥|恶心|呕吐|胸痛|胸闷|胸口痛|胸口闷|"
    r"呼吸困难|喘不上|心悸|心慌|心跳异常|心率异常|发热|发烧|"
    r"剧痛|疼痛|持续疼|持续痛|刺痛|酸痛|肌肉痛|关节痛|膝盖痛|膝盖疼|腰痛|腰疼|"
    r"腿疼|腿痛|肌肉酸|腿酸|胳膊酸|肩酸|腰酸|酸到|酸得|"
    r"(?:头|胸|腹|胃|腰|背|肩|颈|膝盖|膝关节|关节|脚踝|小腿|大腿|手腕|手臂|胳膊|肌肉).{0,4}(?:痛|疼)|"
    r"(?:痛|疼).{0,4}(?:头|胸|腹|胃|腰|背|肩|颈|膝盖|膝关节|关节|脚踝|小腿|大腿|手腕|手臂|胳膊|肌肉)|"
    r"肿胀|麻木|无力|乏力|抽筋|痉挛|拉伤",
    re.IGNORECASE,
)

_EXERCISE_DISCOMFORT_SIGNAL = re.compile(
    r"疲劳|恢复差|恢复不过来|过度训练|练不动|体力不支",
    re.IGNORECASE,
)

_EXERCISE_CONTEXT_SIGNAL = re.compile(
    r"运动|训练|跑步|健身|锻炼|练腿|练胸|练背|力量|有氧|HIIT|肌肉|DOMS|比赛|跑量",
    re.IGNORECASE,
)

_PSYCH_CRISIS_SIGNAL = re.compile(
    r"活着没意思|不想活|轻生|自杀|自伤|结束生命|不想跟任何人说|不想和任何人说|"
    r"没有活下去|想死|消失算了",
    re.IGNORECASE,
)

_ROLE_TOOL_NAMES = {
    "Trainer": "consult_trainer",
    "Nutritionist": "consult_nutritionist",
    "Psychologist": "consult_psychologist",
    "Doctor": "consult_doctor",
}

_ROLE_DISPLAY_NAMES = {
    "Trainer": "训练教练",
    "Nutritionist": "营养师",
    "Psychologist": "心理疗愈师",
    "Doctor": "医学顾问",
}

_EXPERT_RUNNERS = {
    "Trainer": run_trainer,
    "Nutritionist": run_nutritionist,
    "Psychologist": run_psychologist,
    "Doctor": run_doctor,
}

_DIRECT_TOOLS = [
    set_physical_stats,
    add_injury,
    set_dietary_goal,
    add_dietary_preference,
    add_stress_source,
    set_response_style,
    update_user_profile,
]


@dataclass
class _ChildCallContext:
    user_id: str
    user_question: str
    pctx: dict
    episode_context: str = ""
    expert_responses: Dict[str, str] = field(default_factory=dict)
    agent_notes: Dict[str, str] = field(default_factory=dict)
    prior_agent_notes: Dict[str, str] = field(default_factory=dict)
    last_tools: List[str] = field(default_factory=list)
    retrieval_hits: int = 0
    executed: List[str] = field(default_factory=list)


def _profile_anchor_sentence(pctx: dict) -> str:
    profile = pctx.get("raw_profile") or {}
    stats = profile.get("physical_stats") or {}
    anchors = []
    age = stats.get("age")
    weight = stats.get("weight")
    height = stats.get("height")
    injuries = stats.get("injuries") or []
    if age:
        anchors.append(f"{int(age)}岁")
    if weight:
        anchors.append(f"{int(weight) if float(weight).is_integer() else weight}kg")
    if height:
        anchors.append(f"{int(height) if float(height).is_integer() else height}cm")
    if injuries:
        anchors.append("、".join(str(x) for x in injuries if str(x).strip()))
    return "；结合你的已知信息：" + "、".join(anchors) if anchors else ""


def _extract_injury_records(text: str) -> list[str]:
    records: list[str] = []
    raw = text or ""
    if re.search(r"半月板", raw):
        records.append("膝盖半月板损伤")
    if re.search(r"ACL|前交叉", raw, re.IGNORECASE):
        month = re.search(r"(\d+|一|二|三|四|五|六|七|八|九|十)\s*个?月|半年", raw)
        if month:
            token = month.group(1) if month.lastindex else ""
            records.append("ACL术后6个月" if "半年" in month.group(0) or token == "6" or token == "六" else f"ACL术后{month.group(0)}")
        elif re.search(r"术后|重建", raw):
            records.append("ACL术后")
        else:
            records.append("ACL损伤")
    if re.search(r"肩袖", raw):
        records.append("肩袖损伤恢复期" if re.search(r"恢复|康复", raw) else "肩袖损伤")
    if re.search(r"肌肉拉伤", raw):
        records.append("肌肉拉伤")
    if re.search(r"腰痛|腰疼", raw):
        records.append("腰痛")
    if re.search(r"膝盖|膝关节", raw) and not any("膝" in r or "ACL" in r for r in records):
        records.append("膝盖不适")

    seen = []
    for item in records:
        if item and item not in seen:
            seen.append(item)
    return seen


def _record_episode(user_id: str, query: str, answer: str, experts: List[str], notes: dict | None = None) -> None:
    try:
        source_notes = notes or {"Orchestrator": answer}
        append_episode(
            user_id=user_id,
            query=query or "",
            experts=experts,
            gist=answer,
            facts=extract_facts_from_notes(source_notes, query or ""),
        )
    except Exception:
        pass


def _doctor_used(executed: list[str] | None, tools: list[str] | None = None) -> bool:
    if "Doctor" in (executed or []):
        return True
    return any(str(t) == "consult_doctor" for t in (tools or []))


def _physical_discomfort_roles(query: str) -> list[str]:
    """Route body symptoms away from the psych-support agent.

    Doctor owns physical discomfort and symptom risk. Trainer should only own
    training plans when there is no body-symptom complaint in the current ask.
    """
    text = query or ""
    exercise_context = bool(_EXERCISE_CONTEXT_SIGNAL.search(text))
    has_discomfort = bool(_PHYSICAL_DISCOMFORT_SIGNAL.search(text)) or (
        exercise_context and bool(_EXERCISE_DISCOMFORT_SIGNAL.search(text))
    )
    if not has_discomfort:
        return []
    return ["Doctor"]


def _direct_state(
    user_id: str,
    user_question: str,
    answer: str,
    *,
    verdict: str = "PASS_DIRECT",
    used_tools: list[str] | None = None,
    retrieval_hits: int = 0,
    experts: list[str] | None = None,
    expert_responses: dict | None = None,
    agent_notes: dict | None = None,
    executed: list[str] | None = None,
) -> dict:
    notes = agent_notes or {"Orchestrator": build_scratchpad_note("Orchestrator", answer)}
    _record_episode(user_id, user_question, answer, experts or ["Orchestrator"], notes)
    return {
        "messages": [AIMessage(content=answer)],
        "expert_responses": expert_responses or {"Orchestrator": answer},
        "agent_notes": notes,
        "last_tools": used_tools or [],
        "retrieval_hits": retrieval_hits,
        "executed": executed or ["Orchestrator"],
        "plan": [],
        "next": [],
        "replan_count": 0,
        "replan_request": "",
        "replan_context": "",
        "critic_verdict": verdict,
        "orchestrator_decision": "DIRECT",
    }


def _pure_injury_record_answer(state, user_question: str, pctx: dict) -> dict | None:
    text = user_question or ""
    if not _INJURY_RECORD_SIGNAL.search(text):
        return None
    if _ADVICE_REQUEST_SIGNAL.search(text):
        return None
    injuries = _extract_injury_records(text)
    if not injuries:
        return None

    user_id = state.get("profile_user_id", "default_user")
    profile = pctx.get("raw_profile") or {}
    stats = profile.get("physical_stats") or {}
    existing = [str(x).strip() for x in (stats.get("injuries") or []) if str(x).strip()]
    merged = list(existing)
    for injury in injuries:
        if injury not in merged:
            merged.append(injury)
    try:
        store_update_user_profile(user_id, {"physical_stats": {"injuries": merged}})
    except Exception:
        pass

    answer = (
        f"已记录：{'、'.join(injuries)}。先按医生给出的治疗/康复要求执行，"
        "在没有医生或理疗师许可前，相关部位先不要自行加负重、跑跳或做疼痛诱发动作。"
    )
    return _direct_state(user_id, text, answer)


def _medical_boundary_answer(user_question: str, pctx: dict) -> str:
    text = user_question or ""
    anchor = _profile_anchor_sentence(pctx)
    doctor = "请尽快找医生、全科/骨科门诊或相应专科做面诊评估"

    if re.search(r"处方|开药|开一张", text):
        return (
            f"我不能给你开处方，也不能替医生决定具体药物和剂量{anchor}。"
            f"{doctor}，由医生根据疼痛部位、用药禁忌、肝肾功能和既往病史判断是否需要用药。\n\n"
            "在就诊前，可以先做非处方层面的保护：暂停诱发疼痛的训练，按需要休息、冰敷或热敷、避免继续负重刺激；"
            "如果疼痛加重、麻木无力、肿胀明显或影响日常活动，要更早就医。"
        )

    if re.search(r"布洛芬|对乙酰氨基酚|阿司匹林|抗生素|剂量|吃多少药|服药量", text):
        return (
            f"我不能给出继续用药或具体剂量建议{anchor}。"
            "布洛芬这类药连续使用、叠加其他药物或本身有胃肠道/肾脏/心血管风险时，需要医生或药师评估；"
            f"如果疼痛持续，{doctor}。\n\n"
            "在获得专业意见前，不要自行加量、延长疗程或把止痛药当作继续训练的许可。"
            "若出现黑便、胃痛明显、胸闷、呼吸困难、过敏反应、下肢麻木无力或大小便异常，请及时就医。"
        )

    return (
        f"我不能根据文字判断你“是什么病”，也不能做诊断{anchor}。"
        f"这类持续疼痛需要通过医生查体，必要时结合影像或实验室检查来明确；{doctor}。\n\n"
        "在明确原因前，先暂停剧烈运动和会诱发疼痛的动作，不要自行按诊断用药。"
        "如果伴随胸痛胸闷、呼吸困难、发热、晕厥、疼痛快速加重或神经症状，请优先急诊。"
    )


def _should_answer_medical_boundary_direct(query: str) -> bool:
    """Deterministic guard for diagnosis / prescription / medication questions."""
    text = query or ""
    if not _MEDICAL_BOUNDARY_INTENT.search(text):
        return False
    if _MEDICATION_OR_PRESCRIPTION.search(text) or _SELF_DIAGNOSIS_QUESTION.search(text):
        return True
    if re.search(r"持续|痛了|疼了|胸痛|胸闷|呼吸困难|晕厥|头晕|晕晕|眩晕|恶心|呕吐|心率异常|静息心率|血压|血糖", text):
        return not _CARDIAC_EXERCISE_SIGNAL.search(text)
    return False


def _simple_direct_answer(state, user_question: str, pctx: dict) -> dict | None:
    text = (user_question or "").strip()
    if not text:
        return None
    answer = ""
    if re.search(r"你叫什么名字|你是谁|名字", text):
        answer = "我是 Health Guide，你的健康管理助手，可以帮你拆解训练、饮食、睡眠、压力和康复相关问题。"
    elif re.search(r"天气.*好|今天天气|阳光|外面.*好", text):
        profile = pctx.get("raw_profile") or {}
        stats = profile.get("physical_stats") or {}
        age = stats.get("age")
        weight = stats.get("weight")
        goal = (profile.get("dietary_context") or {}).get("goal")
        anchor = ""
        if age or weight or goal:
            bits = []
            if age:
                bits.append(f"{int(age)}岁")
            if weight:
                bits.append(f"{int(weight) if float(weight).is_integer() else weight}kg")
            if goal and goal != "健康":
                bits.append(f"目标{goal}")
            anchor = f"按你目前{ '、'.join(bits) }，"
        answer = f"是啊，天气好很适合做一点低门槛活动：{anchor}出门轻松走 20-30 分钟，或者晒晒太阳、做 5 分钟拉伸就很好。"
    if not answer:
        return None
    return _direct_state(state.get("profile_user_id", "default_user"), text, answer)


def _episode_section(episode_context: str) -> str:
    if not episode_context:
        return ""
    return (
        "\n【近期/相关对话记录】\n"
        f"{episode_context}\n"
        "说明：[最近] 表示按时间最近的对话，[相关] 表示因话题相似而召回的更早对话；引用时请区分使用。\n"
    )


def _call_child_agent(role: str, ctx: _ChildCallContext) -> str:
    if role in ctx.executed:
        return f"[{role}] 本轮已经调用过该子 agent，请直接使用已有结果。"
    runner = _EXPERT_RUNNERS[role]
    peer_notes = format_peer_notes(
        {**ctx.prior_agent_notes, **ctx.agent_notes},
        self_role=role,
    )
    result = runner(ctx.user_id, ctx.user_question, peer_notes, ctx.pctx, ctx.episode_context)
    response = (result.get("expert_responses") or {}).get(role, "")
    ctx.expert_responses.update(result.get("expert_responses") or {})
    ctx.agent_notes.update(result.get("agent_notes") or {})
    ctx.last_tools.extend(result.get("last_tools") or [])
    ctx.retrieval_hits += int(result.get("retrieval_hits") or 0)
    ctx.executed.append(role)
    return response or f"[{role}] 子 agent 已完成，但没有返回可用文本。"


def _make_child_tool(role: str, ctx: _ChildCallContext):
    descriptions = {
        "Trainer": "调用 Trainer 子 agent，处理训练、运动动作、运动计划、TDEE/BMR、伤病/康复期负荷和比赛训练。",
        "Nutritionist": "调用 Nutritionist 子 agent，处理饮食、营养、热量、蛋白质、补剂、食谱和比赛补给。",
        "Psychologist": (
            "调用心理疗愈师子 agent，处理压力、焦虑/情绪、动力下降、倦怠、压力性进食、"
            "睡前心理放松和心理安全边界；不要用它处理身体症状、疼痛、头晕、恶心、伤病或医学不适。"
        ),
        "Doctor": "调用 Doctor 子 agent，处理一般医学建议、症状风险分层、就医建议、用药/处方边界和医学资料查询。",
    }

    def _consult_child() -> str:
        return _call_child_agent(role, ctx)

    return tool(
        _ROLE_TOOL_NAMES[role],
        description=descriptions[role],
    )(_consult_child)


def _orchestrator_tools(ctx: _ChildCallContext):
    return [
        _make_child_tool("Trainer", ctx),
        _make_child_tool("Nutritionist", ctx),
        _make_child_tool("Psychologist", ctx),
        _make_child_tool("Doctor", ctx),
        *_DIRECT_TOOLS,
    ]


def _build_parent_agent(pctx: dict, child_ctx: _ChildCallContext, episode_context: str = ""):
    user_card = pctx.get("user_card") or "【关于该用户】\n用户画像暂不可用。"
    system_prompt = (
        "你是 Health Guide 的父 agent / orchestrator。你直接面向用户，并通过工具调用专业子 agent；"
        "不要把专家选择当成文本路由输出，也不要输出 PLAN、DIRECT 或专家名单。\n\n"
        f"{user_card}\n"
        f"{_episode_section(episode_context)}"
        "你可以直接处理寒暄、感谢、告别、能力介绍、澄清、画像/偏好记录、回答风格记录、"
        "通用健康常识和医疗边界说明。只有用户提供新信息时才调用结构化画像工具；"
        "用户卡片就是本轮可用画像，不要为了读取画像而调用工具。\n\n"
        "可调用的专业子 agent 工具：\n"
        "- consult_trainer：训练、运动动作、运动计划、伤病/康复期负荷、TDEE/BMR、比赛训练。\n"
        "- consult_nutritionist：饮食、营养、热量、蛋白质、补剂、食谱、比赛补给。\n"
        "- consult_psychologist：心理疗愈师，仅处理压力、焦虑/情绪、动力下降、倦怠、压力性进食、睡前心理放松和心理安全边界；"
        "不处理身体症状、疼痛、头晕、恶心、伤病或医学不适。\n"
        "- consult_doctor：一般医学建议、症状风险分层、就医建议、用药/处方边界、医学资料查询。\n\n"
        "调用原则：\n"
        "- 用户要求具体训练/动作/运动安排时，调用 consult_trainer。\n"
        "- 用户要求具体饮食/热量/营养/补剂/食谱时，调用 consult_nutritionist。\n"
        "- 用户要求压力、焦虑、情绪困扰、心理倦怠、压力性进食、睡前脑子停不下来或动力下降时，调用 consult_psychologist。\n"
        "- 用户要求医学建议、症状判断、身体不适如何处理、持续疼痛/头晕/恶心/胸闷等症状、用药安全、处方/药物剂量、体检指标或疾病相关信息时，调用 consult_doctor。\n"
        "- 同时涉及多个领域时，按需要连续调用多个子 agent，然后把结果自然整合给用户。\n"
        "- TDEE/BMR/基础代谢/每日总消耗只调用 Trainer。\n"
        "- 10K/5K/半马/马拉松/跑步比赛/备赛/跑量/补给/碳水，通常同时调用 Trainer 和 Nutritionist。\n"
        "- 训练后睡不好：若重点是训练负荷/恢复，调用 Trainer；若重点是紧张、压力、反刍思维或焦虑，追加 Psychologist。\n"
        "- 疲劳、恢复差、肌肉酸痛、头晕、恶心、胸闷、心悸等身体不适，一律不要调用心理疗愈师或 Trainer；先交给 Doctor。\n"
        "- 运动中胸闷/胸痛/心悸/头晕，同时调用 Doctor 和 Trainer，并优先给出就医/停止高强度训练边界。\n"
        "- 若用户画像或近期记录中有伤病/术后，且当前问题会改变训练负荷或体型目标，至少调用 Trainer。\n"
        "- 若用户画像有过敏/饮食禁忌，且当前问题涉及吃喝/补剂，至少调用 Nutritionist。\n\n"
        "医疗边界：不能诊断疾病、开处方、给处方药剂量或保证“没事”。"
        "涉及胸痛/胸闷、呼吸困难、晕厥、头晕、恶心、明显心率异常、持续疼痛、用药剂量、处方或自我诊断时，"
        "必须调用 consult_doctor，并先建议就医/医生评估；不要列未经确认的诊断猜测。"
        "如果调用了 Doctor，最终回答必须包含「仅供参考，如有不适请就医」。\n\n"
        "回答要求：如果调用了子 agent，要吸收它们的结论，用一个统一助手的口吻回答，不要说“路由到/派发给”。"
        "如多个子 agent 观点有重叠，只保留最完整且更安全的表达。"
    )
    return create_agent(llm, _orchestrator_tools(child_ctx), system_prompt)


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
        child_ctx = _ChildCallContext(user_id, user_question, pctx, episode_context)

        if _should_answer_medical_boundary_direct(user_question):
            _call_child_agent("Doctor", child_ctx)
            answer = child_ctx.expert_responses.get("Doctor") or _medical_boundary_answer(user_question, pctx)
            answer = apply_personalization_boost(answer, pctx, user_question, max_notes=2)
            answer = ensure_doctor_disclaimer(answer)
            return _direct_state(
                user_id,
                user_question,
                answer,
                verdict="REVISE_DIRECT",
                used_tools=["consult_doctor", *child_ctx.last_tools],
                retrieval_hits=child_ctx.retrieval_hits,
                experts=["Doctor"],
                expert_responses=child_ctx.expert_responses or {"Doctor": answer},
                agent_notes=child_ctx.agent_notes or {"Doctor": build_scratchpad_note("Doctor", answer)},
                executed=["Doctor"],
            )

        agent = _build_parent_agent(pctx, child_ctx, episode_context)
        result = agent.invoke({"messages": [HumanMessage(content=user_question)]})

        used_tools: list[str] = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    used_tools.append(call.get("name", "Unknown"))
        used_tools.extend(child_ctx.last_tools)

        answer = extract_text_content(result["messages"][-1])
        answer = apply_personalization_boost(answer, pctx, user_question, max_notes=3)
        if _doctor_used(child_ctx.executed, used_tools):
            answer = ensure_doctor_disclaimer(answer)
        retrieval_hits = child_ctx.retrieval_hits
        verdict = "REVISE_DIRECT" if _MEDICAL_BOUNDARY_INTENT.search(user_question or "") else "PASS_DIRECT"
    except Exception as exc:
        used_tools = [f"ERROR:Orchestrator:{type(exc).__name__}"]
        retrieval_hits = 0
        answer = orchestrator_error_answer(exc)
        verdict = f"ERROR_DIRECT:{type(exc).__name__}"

    return _direct_state(
        user_id,
        user_question,
        answer,
        verdict=verdict,
        used_tools=used_tools,
        retrieval_hits=retrieval_hits,
    )


def _run_parent_agent(state) -> dict:
    user_id = state.get("profile_user_id", "default_user")
    pctx = state.get("personalization_ctx") or {}
    if not pctx:
        pctx = build_personalization_ctx(user_id)
    user_question = get_user_question(state)
    episode_context = state.get("episode_context") or ""

    record_update = _pure_injury_record_answer(state, user_question, pctx)
    if record_update:
        return record_update

    simple_direct = _simple_direct_answer(state, user_question, pctx)
    if simple_direct:
        return simple_direct

    if _should_answer_medical_boundary_direct(user_question):
        return _direct_answer(state)

    if _PSYCH_CRISIS_SIGNAL.search(user_question or ""):
        child_ctx = _ChildCallContext(
            user_id,
            user_question,
            pctx,
            episode_context,
            prior_agent_notes=dict(state.get("agent_notes") or {}),
        )
        _call_child_agent("Psychologist", child_ctx)
        return {
            "expert_responses": child_ctx.expert_responses,
            "agent_notes": child_ctx.agent_notes,
            "last_tools": [_ROLE_TOOL_NAMES["Psychologist"], *child_ctx.last_tools],
            "retrieval_hits": child_ctx.retrieval_hits,
            "executed": child_ctx.executed,
            "plan": [],
            "next": [],
            "replan_count": 0,
            "replan_request": "",
            "replan_context": "",
            "orchestrator_decision": "CALLED_CHILD",
        }

    physical_roles = _physical_discomfort_roles(user_question)
    if physical_roles:
        child_ctx = _ChildCallContext(
            user_id,
            user_question,
            pctx,
            episode_context,
            prior_agent_notes=dict(state.get("agent_notes") or {}),
        )
        used_tools: list[str] = []
        for role in physical_roles:
            _call_child_agent(role, child_ctx)
            used_tools.append(_ROLE_TOOL_NAMES[role])
        return {
            "expert_responses": child_ctx.expert_responses,
            "agent_notes": child_ctx.agent_notes,
            "last_tools": used_tools + child_ctx.last_tools,
            "retrieval_hits": child_ctx.retrieval_hits,
            "executed": child_ctx.executed,
            "plan": [],
            "next": [],
            "replan_count": 0,
            "replan_request": "",
            "replan_context": "",
            "orchestrator_decision": "CALLED_CHILD",
        }

    try:
        os.environ["HEALTH_GUIDE_USER_ID"] = user_id
        child_ctx = _ChildCallContext(
            user_id,
            user_question,
            pctx,
            episode_context,
            prior_agent_notes=dict(state.get("agent_notes") or {}),
        )

        if state.get("replan_context"):
            needed = ""
            context_text = str(state.get("replan_context") or "")
            executed_roles = state.get("executed") or []
            for role in _EXPERT_RUNNERS:
                if role in context_text and role not in executed_roles:
                    needed = role
                    break
            if not needed and re.search(r"医学|症状|就医|用药|处方|诊断|剂量|Doctor", context_text, re.IGNORECASE):
                if "Doctor" not in executed_roles:
                    needed = "Doctor"
            if needed:
                _call_child_agent(needed, child_ctx)
                return {
                    "expert_responses": child_ctx.expert_responses,
                    "agent_notes": child_ctx.agent_notes,
                    "last_tools": child_ctx.last_tools,
                    "retrieval_hits": child_ctx.retrieval_hits,
                    "executed": [role for role in executed_roles if role != "Orchestrator"] + child_ctx.executed,
                    "plan": [],
                    "next": [],
                    "replan_request": "",
                    "replan_context": "",
                    "orchestrator_decision": "CALLED_CHILD",
                }

        agent = _build_parent_agent(pctx, child_ctx, episode_context)
        result = agent.invoke({"messages": [HumanMessage(content=user_question)]})

        parent_tools: list[str] = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    parent_tools.append(call.get("name", "Unknown"))

        answer = extract_text_content(result["messages"][-1])
        answer = apply_personalization_boost(answer, pctx, user_question, max_notes=3)
        all_tools = parent_tools + child_ctx.last_tools
        if _doctor_used(child_ctx.executed, parent_tools):
            answer = ensure_doctor_disclaimer(answer)

        if child_ctx.executed:
            return {
                "draft_answer": answer,
                "expert_responses": child_ctx.expert_responses,
                "agent_notes": child_ctx.agent_notes,
                "last_tools": all_tools,
                "retrieval_hits": child_ctx.retrieval_hits,
                "executed": child_ctx.executed,
                "plan": [],
                "next": [],
                "replan_count": 0,
                "replan_request": "",
                "replan_context": "",
                "orchestrator_decision": "CALLED_CHILD",
            }

        answer = apply_personalization_boost(answer, pctx, user_question, max_notes=2)
        _record_episode(user_id, user_question, answer, ["Orchestrator"])
        return {
            "messages": [AIMessage(content=answer)],
            "expert_responses": {"Orchestrator": answer},
            "agent_notes": {"Orchestrator": build_scratchpad_note("Orchestrator", answer)},
            "last_tools": all_tools,
            "retrieval_hits": 0,
            "executed": ["Orchestrator"],
            "plan": [],
            "next": [],
            "replan_count": 0,
            "replan_request": "",
            "replan_context": "",
            "critic_verdict": "PASS_DIRECT",
            "orchestrator_decision": "DIRECT",
        }
    except Exception as exc:
        return _direct_answer(state, error=exc)


def orchestrator_node(state):
    return _run_parent_agent(state)


# Backward compatibility for scripts importing the old planner symbol.
planner_node = orchestrator_node
