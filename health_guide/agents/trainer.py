"""Trainer expert — invoked as a callable by the Dispatcher (no longer a graph node).

Receives an isolated, scoped input from the parent agent:
  - SystemMessage: role profile (cropped) + optional peer scratchpad
  - HumanMessage: contextualized user question

RAG is *on-demand*: `retrieve_trainer_knowledge` lives in the tool list so the
expert's ReAct loop decides whether to call it (skip greetings / pure personal
chat, fire for actual training questions).
"""
import os
import re

from langchain_core.messages import HumanMessage

from ..mcp_client import MCP_REGISTRY
from ..tools import (
    add_injury,
    calculate_tdee,
    retrieve_trainer_knowledge,
    set_dietary_goal,
    set_physical_stats,
    update_user_profile,
)
from ..utils import create_agent
from ..llm import extract_text_content, llm
from ..personalization import build_personalization_ctx
from ..detail import print_expert_end, print_expert_start, print_expert_trace
from .fallbacks import expert_error_update
from ._scratchpad import build_scratchpad_note


_TRAINER_TOOLS = [
    calculate_tdee,
    set_physical_stats,
    add_injury,
    set_dietary_goal,
    update_user_profile,
    retrieve_trainer_knowledge,
]


def _num(value):
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _fmt(value, suffix=""):
    n = _num(value)
    if n is None:
        return ""
    return f"{int(n) if n.is_integer() else round(n, 1)}{suffix}"


def _profile(pctx: dict) -> dict:
    return pctx.get("raw_profile") or {}


def _stats(profile: dict) -> dict:
    return profile.get("physical_stats") or {}


def _goal(profile: dict) -> str:
    return str((profile.get("dietary_context") or {}).get("goal") or "").strip()


def _answer_has_profile_anchor(answer: str, stats: dict, injuries: list[str]) -> bool:
    text = answer or ""
    for key in ("age", "weight", "height"):
        value = _fmt(stats.get(key))
        if value and value in text:
            return True
    return any(str(injury) and str(injury) in text for injury in injuries)


def _injury_intro_and_rules(injuries: list[str], question: str) -> str:
    if not injuries:
        return ""
    joined = "、".join(str(x) for x in injuries if str(x).strip())
    text = f"{joined}\n{question or ''}"
    lines = [f"先把安全边界放在前面：你记录里有 {joined}，训练必须按康复/疼痛反应来限制。"]

    if re.search(r"半月板", text, re.IGNORECASE):
        lines.append(
            "半月板/膝部场景先避免深屈膝负重、跑跳、扭转和冲刺；更稳的替代是直腿抬高 2-3 组x10-15 次、"
            "侧卧蚌式开合 2-3 组x12-15 次、低阻力固定单车或游泳 15-30 分钟，并以无痛为前提。"
        )
    elif re.search(r"膝盖|膝关节", text, re.IGNORECASE):
        lines.append(
            "膝部康复场景先避免跑跳、急停变向、深屈膝负重和疼痛诱发动作；更稳的替代是直腿抬高、臀桥、"
            "低阻力固定单车或游泳，并以无痛为前提。"
        )
    if re.search(r"acl|前交叉|韧带", text, re.IGNORECASE):
        lines.append(
            "ACL 术后/韧带康复要由运动医学医生或理疗师评估后再决定深蹲进阶；常见门槛包括屈伸活动度达标、"
            "单腿控制稳定、患侧股四头肌力量接近健侧 80% 以上。"
        )
    if re.search(r"肩袖|肩关节|肩", text):
        lines.append(
            "肩袖恢复期先让理疗师评估外旋力量和活动度；可优先弹力带外旋、Y-T-W、墙壁俯卧撑，暂避杠铃卧推、双杠臂屈伸和过头推举。"
        )
    if re.search(r"腰椎|椎间盘|腰痛|下背", text):
        lines.append(
            "腰部伤病先避开大重量硬拉、负重深蹲和负重扭转；更适合从死虫、鸟狗、侧桥等核心稳定动作做起。"
        )
    lines.append("上述动作也应以医生或理疗师许可为前提；疼痛、肿胀或不稳感加重就停。")
    return "\n".join(lines)


def _specific_knee_rehab_answer(pctx: dict, user_question: str) -> str:
    profile = _profile(pctx)
    stats = _stats(profile)
    injuries = [str(x).strip() for x in (stats.get("injuries") or []) if str(x).strip()]
    injury_text = "、".join(injuries)
    q = user_question or ""
    anchors = [x for x in (_fmt(stats.get("age"), "岁"), _fmt(stats.get("weight"), "kg"), _fmt(stats.get("height"), "cm")) if x]
    anchor_text = f"以你目前 {', '.join(anchors)}，且记录有 {injury_text} 来看，" if anchors else f"考虑到你记录有 {injury_text}，"

    is_training_query = re.search(r"练腿|腿部|深蹲|蹲|负重|跑步|训练|运动|计划|走|步数|开始做", q)
    if not injuries or not is_training_query:
        return ""

    if any("半月板" in injury for injury in injuries):
        return (
            f"{anchor_text}下周先不要做常规腿部训练计划，应把目标改成无痛康复和维持活动量。\n\n"
            "先避免深屈膝负重、跳跃、跑步、冲刺、扭转和靠墙静蹲这类可能增加膝关节压力的动作；"
            "更稳的选择是：直腿抬高 2-3 组x10-15 次、股四头肌等长收缩 2-3 组x10 次、"
            "侧卧髋外展 2-3 组x12-15 次、低阻力固定单车或游泳 15-30 分钟。\n\n"
            "进阶条件很简单：训练中无痛，训练后 24 小时没有肿胀、卡顿或疼痛反跳；"
            "最好先让理疗师确认可做范围，再逐步加阻力。"
        )

    if any(re.search(r"ACL|前交叉|韧带", injury, re.IGNORECASE) for injury in injuries) and re.search(r"深蹲|蹲|负重", q):
        stage_text = f"{injury_text}\n{q}"
        has_six_month = re.search(r"6\s*个?月|六\s*个?月|半年", stage_text)
        if has_six_month:
            return (
                f"{anchor_text}ACL 术后约 6 个月通常可能进入强化阶段，但能不能开始深蹲必须由运动医学医生或理疗师评估，"
                "不能只凭自我感觉决定。\n\n"
                "常见门槛包括：膝关节伸直可到 0 度、屈曲至少约 120 度且无痛，单腿蹲 30 度时膝盖不内扣不晃，"
                "患侧股四头肌力量达到健侧约 80% 以上，训练后没有肿胀或不稳感。\n\n"
                "在评估前，先继续做坐姿伸膝、闭链等长收缩、直腿抬高和低阻力固定单车等低风险动作；"
                "若评估通过，也应从徒手小范围、低次数开始，而不是直接上负重或深幅度。"
            )
        return (
            f"{anchor_text}现阶段应避免自行做深蹲、跳绳、跳箱、冲刺或急停变向。"
            "这些动作会增加膝关节剪切力和冲击，对 ACL/韧带康复不友好。\n\n"
            "更稳妥的康复替代是直腿抬高、股四头肌等长收缩、臀桥、水中训练或低阻力固定单车，"
            "每次 10-20 分钟从无痛范围开始；是否能进入下肢负重训练，必须由运动医学医生或理疗师按阶段评估。"
        )

    return ""


def _deterministic_trainer_answer(pctx: dict, user_question: str) -> str:
    profile = _profile(pctx)
    stats = _stats(profile)
    q = user_question or ""

    knee = _specific_knee_rehab_answer(pctx, q)
    if knee:
        return knee

    anchors = [x for x in (_fmt(stats.get("age"), "岁"), _fmt(stats.get("weight"), "kg"), _fmt(stats.get("height"), "cm")) if x]
    anchor_text = f"以你目前 {', '.join(anchors)} 来看，" if anchors else ""
    injuries = [str(x).strip() for x in (stats.get("injuries") or []) if str(x).strip()]

    if injuries and re.search(r"减脂|饮食方案|每日饮食|减脂餐", q):
        injury_text = "、".join(injuries)
        knee_word = "膝关节/膝盖" if re.search(r"ACL|前交叉|韧带|膝|半月板", injury_text, re.IGNORECASE) else "伤病部位"
        return (
            f"{anchor_text}记录里有 {injury_text}，减脂饮食要和 {knee_word} 康复限制联动。"
            "训练侧先避免深蹲、跳跃、跑步冲刺和急停变向，把热量赤字控制得温和，避免因恢复差影响康复。\n\n"
            "可以用低冲击活动维持消耗：低阻力固定单车、游泳或上肢力量训练，每次 15-30 分钟，"
            "并以医生或理疗师许可、训练后 24 小时无肿胀疼痛为前提。"
        )

    if re.search(r"TDEE|BMR|基础代谢|每日所需热量|每日.*热量|总消耗", q, re.IGNORECASE):
        age = _num(stats.get("age"))
        weight = _num(stats.get("weight"))
        height = _num(stats.get("height"))
        if age and weight and height:
            bmr = round(10 * weight + 6.25 * height - 5 * age + 5)
            sedentary = round(bmr * 1.2)
            active = round(bmr * 1.55)
            very_active = round(bmr * 1.725)
            deficit_low = max(1500, active - 500)
            deficit_high = max(1500, active - 300)
            protein_low = round(weight * 1.6)
            protein_high = round(weight * 2.0)
            return (
                f"{anchor_text}用 Mifflin-St Jeor 公式估算，BMR 约 {bmr} kcal/天。"
                f"活动量未知时可先看三个场景：久坐 TDEE 约 {sedentary} kcal/天，中等活动约 {active} kcal/天，"
                f"高活动量约 {very_active} kcal/天。\n\n"
                f"如果目标是减脂，建议先从中等活动估算值下调 300-500 kcal，也就是约 {deficit_low}-{deficit_high} kcal/天；"
                f"蛋白质按 {weight:g}kg 计算可放在 {protein_low}-{protein_high}克/天，帮助保肌。"
            )

    if injuries and any(re.search(r"肩袖|肩关节|肩", injury) for injury in injuries) and re.search(r"练胸|胸|卧推|推", q):
        injury_text = "、".join(injuries)
        return (
            f"{anchor_text}记录里有 {injury_text}，下周练胸不要从大重量或极限重量开始，先让理疗师评估肩关节活动度和外旋力量。\n\n"
            "可做的低风险起点：弹力带外旋 2-3 组x12-15 次、Y-T-W 2 组x8-12 次、墙壁俯卧撑 2-3 组x8-12 次。"
            "先避免杠铃卧推、双杠臂屈伸、过头推举和任何疼痛范围内的推举动作；连续 24 小时无痛无酸胀反跳后，再考虑逐步增加阻力。"
        )

    if injuries and any("膝关节炎" in injury or "膝" in injury for injury in injuries) and re.search(r"走多少步|多少步|步数", q):
        return (
            f"{anchor_text}考虑到你记录有 {'、'.join(injuries)}，步数要缓慢、渐进增加，不建议直接追求每天 15000 步。\n\n"
            "先从每天 6000-8000 步开始，分 2-3 次完成，优先选平地或塑胶跑道，避免上下坡和连续爬楼。"
            "如果走完后 24 小时内膝盖肿胀、夜间痛或疼痛明显加重，就把步数下调 20%-30%，并咨询康复科或理疗师。"
        )

    if re.search(r"运动后.*心跳|运动后.*心率|心跳.*十几分钟|心率.*十几分钟", q):
        return (
            f"{anchor_text}运动后心跳很快且要十几分钟才恢复，不建议简单判断为“正常”。"
            "请先降低训练强度，并尽快做医生评估或心内科/运动医学检查，至少包括心电图、血压和必要时的运动心肺相关检查。\n\n"
            "在检查排除风险前，先避免冲刺、HIIT 和大重量训练；只保留能轻松说话的低强度步行或固定单车。"
            "如果伴随胸痛、胸闷、头晕、气短或晕厥，要立即停止运动并及时就医。"
        )

    if re.search(r"熬夜|睡眠不足|没睡好", q):
        return (
            f"{anchor_text}熬夜后训练强度要主动下调，今天不要做 HIIT 或高强度训练。"
            "建议只做 RPE 5-6 的轻有氧或低重量全身活动 20-30 分钟，例如轻松走、低阻力固定单车、"
            "徒手深蹲替代为坐站练习、弹力带划船等；任何头晕、心悸或动作变形都立即停止。"
        )

    if re.search(r"10\s*(?:K|公里)|5\s*(?:K|公里)|半马|马拉松|跑步.*比赛|比赛.*跑", q, re.IGNORECASE):
        return (
            f"{anchor_text}下个月 10K 比赛不要再硬堆跑量，重点是稳定节奏和赛前减量。"
            "可以按每周 3-4 跑安排：1 次轻松跑 4-6km、1 次配速/节奏跑 3-5km、1 次长距离 7-9km，"
            "另加 1 次可选恢复跑或低强度交叉训练。\n\n"
            "赛前最后一周跑量降到平时约 50%，保留 2 次 20-30 分钟轻松跑和少量加速跑唤醒状态。"
            "强度以 RPE 4-6 为主，避免临赛前新增冲刺或大强度腿部训练。"
        )

    if re.search(r"新手|没有经验|每周.*几次|健身房几次", q):
        return (
            f"{anchor_text}纯新手建议每周 2-3 天去健身房，每次 45-60 分钟，两次训练之间至少隔 1 天。"
            "前 4-6 周先做全身训练：推、拉、蹲/髋、核心各 2-3 组，每组 8-12 次，RPE 6 左右。"
            "先把动作做稳，再逐步加重量或加到每周 3-4 天。"
        )

    if re.search(r"腿酸|酸痛|DOMS|走路.*困难|练腿.*酸", q, re.IGNORECASE):
        return (
            f"{anchor_text}这像延迟性肌肉酸痛，今天不要继续高强度练同一肌群。"
            "可以做 10-20 分钟轻松步行或固定单车、温和拉伸、泡沫轴放松和热水澡，帮助缓解。"
            "蛋白质和睡眠补足；如果疼痛像刺痛、有关节肿胀或越来越严重，就暂停训练并就医评估。"
        )

    if re.search(r"深蹲多少|深蹲.*合适", q):
        weight = _num(stats.get("weight"))
        if weight:
            start_low = round(weight * 0.45)
            start_high = round(weight * 0.55)
            return (
                f"{anchor_text}如果没有膝/腰伤且动作还不稳定，先用徒手深蹲 3 组x12 次找动作模式。"
                f"能稳定后，第一周负重可从体重的 45%-55% 估算，也就是约 {start_low}-{start_high}kg，做 4-5 组x5-8 次，"
                "组间休息 90-120 秒，RPE 控制在 6-7。\n\n"
                "每周只加 2.5kg 或每组多 1-2 次；如果膝盖内扣、腰背代偿或第二天关节痛，就先降重量。"
            )

    return ""


def _training_profile_intro(pctx: dict, user_question: str, answer: str) -> str:
    profile = _profile(pctx)
    stats = _stats(profile)
    injuries = [str(x).strip() for x in (stats.get("injuries") or []) if str(x).strip()]
    anchors = []
    for key, suffix in (("age", "岁"), ("weight", "kg"), ("height", "cm")):
        value = _fmt(stats.get(key), suffix)
        if value:
            anchors.append(value)
    goal = _goal(profile)
    if goal and goal != "健康":
        anchors.append(f"目标{goal}")

    add = []
    if anchors and not _answer_has_profile_anchor(answer, stats, injuries):
        add.append(f"以你目前 {', '.join(anchors)} 来看，下面的训练量先按保守起点安排，再根据恢复逐步加量。")
    injury_block = _injury_intro_and_rules(injuries, user_question)
    if injury_block and not any(str(x) in (answer or "") for x in injuries):
        add.append(injury_block)
    if not add:
        return answer
    return "\n\n".join(add + [answer])


def _episode_section(episode_context: str) -> str:
    if not episode_context:
        return ""
    return (
        "\n【近期/相关对话记录】\n"
        f"{episode_context}\n"
        "说明：[最近] 表示按时间最近的对话，[相关] 表示因话题相似而召回的更早对话；引用时请区分使用。\n"
    )


def _build_trainer_agent(pctx: dict, peer_notes_text: str, episode_context: str = ""):
    peer_section = peer_notes_text if peer_notes_text else ""
    user_card = (pctx.get("role_user_cards") or {}).get("Trainer") or pctx.get("user_card") or "【关于该用户】\n用户画像暂不可用。"
    wger_tools = MCP_REGISTRY.get_tools("wger")
    mcp_hint = (
        "如需查询具体动作百科（标准动作要领、目标肌群、所需器械），可调用 wger MCP 工具："
        "search_exercises / get_exercise_details / list_muscles / list_equipment / list_categories。"
        if wger_tools
        else ""
    )
    system_prompt = (
        "你是力量训练教练。\n\n"
        f"{user_card}\n"
        f"{_episode_section(episode_context)}"
        f"{peer_section}"
        "用户卡片就是本轮可用画像；不要说「我先看看/了解你的基本信息」，不要为了读取画像而调用工具。"
        "若已有足够信息，必须直接给出方案；只有用户提供新信息时才调用结构化工具记录。"
        "如需要训练/动作/恢复方面的知识库支持，可主动调用 retrieve_trainer_knowledge。"
        f"{mcp_hint}"
        "对于纯打招呼或与训练无关的内容，请直接回答，无需检索。"
        "若检索结果明确返回 '未命中本地知识库'，可凭通用训练知识给出保守兜底建议。"
        "对动作安全与伤病风险进行约束。"
        "如果用户提供了新的身体信息，请优先调用 set_physical_stats / add_injury / set_dietary_goal 做结构化更新；"
        "update_user_profile 仅作兼容兜底。"
        "【工具使用】当用户询问 TDEE、BMR、基础代谢、每日热量消耗或减脂/增肌热量起点时，"
        "若画像已有体重、身高、年龄，必须调用 calculate_tdee；回答中写明这是估算值，并给出活动系数/目标热量调整建议。"
        "若用户没有说明活动水平，不要只给默认久坐值；应说明活动水平未知，并给出久坐/中等活动/高活动量至少 2 个场景的 TDEE 估算或请用户补充活动量。"
        "当用户询问动作技术、训练计划、恢复、平台期或伤病训练时，优先调用 retrieve_trainer_knowledge。"
        "【训练处方格式】回答训练建议时按 FITT 思路落地：频次、时长、动作类型、组数/次数、强度（RPE 或心率区间）、组间休息、"
        "进阶规则和恢复安排。新手从低容量开始（例如每周 2-3 次全身训练），不要直接给高强度或高频计划。"
        "【恢复与平台期】肌肉酸痛/DOMS 先给主动恢复、轻活动、泡沫轴/拉伸、睡眠与补水；平台期必须强调训练日志和渐进超负荷"
        "（加重量/次数/组数或改善动作质量），并给出 1-2 周的具体进阶方式。"
        "【症状红线】若用户提到运动中胸闷/胸痛、心悸、头晕、晕厥、运动后心率十几分钟不降、明显肿胀或剧痛，"
        "必须先建议停止训练并做医学评估/就医检查；在排除风险前不要建议继续中高强度训练。"
        "回答尽量给出可执行计划（频次/组数/强度/恢复）。\n"
        "【输出硬性要求】\n"
        "1. 若用户卡片中有年龄/体重/身高/BMI，回答开头必须自然引用至少 1 个数值，"
        "例如「以你 40 岁、88kg 的当前状态…」。不允许只说「根据你的情况」；若数值缺失，先说明需要补充。\n"
        "2. 训练量必须给出具体数字（频次/组数/时长/强度），不允许只说「适量」「适度」。\n"
        "3. 若用户卡片列出了伤病，回答前两句之内必须点名该伤病并说出限制；"
        "不得推荐冲突动作；替代动作须注明「须在医生或理疗师许可下进行」。"
        "半月板损伤场景优先给直腿抬高、轻阻力坐姿伸膝、侧卧蚌式开合、游泳或固定单车低阻力；"
        "不得给靠墙静蹲、深屈膝、跑步、冲刺、跳跃或扭转动作。"
        "ACL 术后 6 个月不是一律禁止所有蹲类，而是必须由医生/理疗师评估后分阶段进入强化；"
        "回答需写出屈伸活动度、单腿控制、股四头肌力量等门槛，并给固定单车低阻力、坐姿伸膝、闭链等长收缩等替代。\n"
        "4. 至少包含 2 条由画像具体数值（体重/年龄/伤病/目标）推导出的可执行数字。"
    )
    return create_agent(llm, list(_TRAINER_TOOLS) + wger_tools, system_prompt)


def run_trainer(
    user_id: str,
    user_question: str,
    peer_notes_text: str = "",
    pctx: dict | None = None,
    episode_context: str = "",
) -> dict:
    """Execute the Trainer expert and return a state update dict."""
    try:
        os.environ["HEALTH_GUIDE_USER_ID"] = user_id
        print_expert_start("Trainer", user_question)
        pctx = pctx or build_personalization_ctx(user_id)
        deterministic = _deterministic_trainer_answer(pctx, user_question)
        if deterministic:
            print_expert_end("Trainer", [], deterministic)
            return {
                "expert_responses": {"Trainer": deterministic},
                "agent_notes": {"Trainer": build_scratchpad_note("Trainer", deterministic)},
                "last_tools": [],
                "retrieval_hits": 0,
            }

        agent = _build_trainer_agent(pctx, peer_notes_text, episode_context)
        result = agent.invoke({"messages": [HumanMessage(content=user_question)]})

        print_expert_trace("Trainer", result["messages"])

        used_tools: list[str] = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    used_tools.append(call.get("name", "Unknown"))

        retrieval_hits = sum(
            1 for t in used_tools if "retrieve" in t and "knowledge" in t
        )
        answer = extract_text_content(result["messages"][-1])
        answer = _training_profile_intro(pctx, user_question, answer)
        print_expert_end("Trainer", used_tools, answer)
        return {
            "expert_responses": {"Trainer": answer},
            "agent_notes": {"Trainer": build_scratchpad_note("Trainer", answer)},
            "last_tools": used_tools,
            "retrieval_hits": retrieval_hits,
        }
    except Exception as e:
        return expert_error_update("Trainer", e)
