"""Psychological-support expert — invoked as the Psychologist-compatible callable."""
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage

from ..config import DEFAULT_TIMEZONE
from ..tools import (
    add_stress_source,
    log_wellness_checkin,
    push_reminder,
    query_logs,
    retrieve_psychologist_knowledge,
    schedule_calendar_event,
    set_response_style,
    update_user_profile,
)
from ..integrations.local_logs import extract_actuation_events
from ..utils import create_agent
from ..llm import extract_text_content, llm
from ..personalization import (
    build_personalization_ctx,
    build_personalization_decision_points,
    format_decision_points_for_prompt,
)
from ..detail import print_expert_end, print_expert_start, print_expert_trace
from .fallbacks import expert_error_update
from ._scratchpad import build_scratchpad_note


_PSYCHOLOGIST_TOOLS = [
    add_stress_source,
    set_response_style,
    update_user_profile,
    retrieve_psychologist_knowledge,
    log_wellness_checkin,
    query_logs,
    push_reminder,
    schedule_calendar_event,
]

_BODY_SYMPTOM_SIGNAL = re.compile(
    r"头晕|晕晕|眩晕|晕厥|昏厥|恶心|呕吐|胸痛|胸闷|胸口痛|胸口闷|"
    r"呼吸困难|喘不上|心悸|心慌|心跳异常|心率异常|发热|发烧|"
    r"剧痛|疼痛|持续疼|持续痛|刺痛|酸痛|肌肉痛|关节痛|膝盖痛|膝盖疼|腰痛|腰疼|"
    r"腿疼|腿痛|肌肉酸|腿酸|胳膊酸|肩酸|腰酸|酸到|酸得|"
    r"(?:头|胸|腹|胃|腰|背|肩|颈|膝盖|膝关节|关节|脚踝|小腿|大腿|手腕|手臂|胳膊|肌肉).{0,4}(?:痛|疼)|"
    r"(?:痛|疼).{0,4}(?:头|胸|腹|胃|腰|背|肩|颈|膝盖|膝关节|关节|脚踝|小腿|大腿|手腕|手臂|胳膊|肌肉)|"
    r"肿胀|麻木|无力|抽筋|痉挛|拉伤",
    re.IGNORECASE,
)

_PSYCH_SIGNAL = re.compile(
    r"压力|焦虑|紧张|情绪|心情|低落|抑郁|恐慌|害怕|担心|内耗|反刍|"
    r"脑子停不下来|倦怠|崩溃|没动力|动力下降|不想动|只想躺|压力性进食|暴食|自伤|轻生",
    re.IGNORECASE,
)

_CRISIS_SIGNAL = re.compile(
    r"活着没意思|不想活|轻生|自杀|自伤|结束生命|不想跟任何人说|不想和任何人说|"
    r"没有活下去|想死|消失算了",
    re.IGNORECASE,
)


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


def _psychologist_profile_intro(pctx: dict, user_question: str, answer: str) -> str:
    profile = pctx.get("raw_profile") or {}
    stats = profile.get("physical_stats") or {}
    mental = profile.get("mental_state") or {}
    stress = [str(x).strip() for x in (mental.get("stress_sources") or []) if str(x).strip()]
    anchors = [x for x in (_fmt(stats.get("age"), "岁"), _fmt(stats.get("weight"), "kg"), _fmt(stats.get("height"), "cm")) if x]
    injuries = [str(x).strip() for x in (stats.get("injuries") or []) if str(x).strip()]
    add = []
    answer_text = answer or ""
    question_text = user_question or ""

    if stress and not any(s in answer_text for s in stress):
        source = "、".join(stress)
        add.append(
            f"你现在的压力源主要是 {source}；先把问题拆成今晚能降刺激、明天能恢复一点节奏的两步。"
        )
    elif anchors and not any(a.rstrip("岁kgcm") in answer_text for a in anchors):
        add.append(f"结合你目前 {', '.join(anchors)}，心理支持方案先从低门槛、可连续 7 天执行的动作开始。")
    if injuries and not any(i in answer_text for i in injuries):
        add.append(f"同时考虑到 {', '.join(injuries)}，心理放松练习不替代伤病或身体不适的医学评估。")

    if re.search(r"睡不着|失眠|入睡|睡眠|睡不好", question_text + answer_text):
        if not re.search(r"30|60|担忧|呼吸|放松", answer_text):
            add.append("今晚先做 30 分钟降刺激：关屏、写 5 分钟担忧清单，再做 4-7-8 呼吸或渐进式肌肉放松 10 分钟。")
    if re.search(r"没.*动力|不想.*动|躺着|倦怠", question_text):
        if not re.search(r"5 ?分钟|10 ?分钟|降低门槛|小目标|最小", answer_text):
            add.append("先把运动降到 5-10 分钟最小版本，例如出门走一圈或只做一组拉伸，完成就算达标。")
    if re.search(r"比赛|紧张|焦虑", question_text + answer_text):
        if "焦虑" not in answer_text:
            add.append("赛前焦虑是常见反应，目标不是完全消除，而是把它降到不影响睡眠和执行计划的范围。")

    if not add:
        return answer
    return "\n\n".join(add + [answer])


def _deterministic_psychologist_answer(pctx: dict, user_question: str) -> str:
    profile = pctx.get("raw_profile") or {}
    stats = profile.get("physical_stats") or {}
    mental = profile.get("mental_state") or {}
    dietary = profile.get("dietary_context") or {}
    q = user_question or ""
    if _BODY_SYMPTOM_SIGNAL.search(q) and not _PSYCH_SIGNAL.search(q):
        return (
            "这更像身体不适或训练负荷相关问题，不适合只按心理压力处理。"
            "请优先让医学顾问/医生评估；如果和训练有关，也应由训练教练调整强度。"
            "在明确原因前先停止高强度训练，记录症状出现时间、持续多久、诱因和伴随表现。"
        )
    stress = [str(x).strip() for x in (mental.get("stress_sources") or []) if str(x).strip()]
    sleep_quality = str(mental.get("sleep_quality") or "").strip()
    anchors = [x for x in (_fmt(stats.get("age"), "岁"), _fmt(stats.get("weight"), "kg"), _fmt(stats.get("height"), "cm")) if x]
    stress_text = "、".join(stress) if stress else "当前压力"
    anchor_text = f"结合你目前 {', '.join(anchors)}，" if anchors else ""
    weight = _num(stats.get("weight"))
    height = _num(stats.get("height"))
    bmi = round(weight / ((height / 100) ** 2), 1) if weight and height else None
    goal = str(dietary.get("goal") or "").strip()

    if _CRISIS_SIGNAL.search(q):
        return (
            f"{anchor_text}你提到“活着没意思”、也不想跟任何人说，这已经是需要立刻优先处理的心理危机信号。"
            f"如果压力源和 {stress_text} 有关，今晚也先不要独自处理这些问题。\n\n"
            "现在请先做三件事：第一，立刻联系一个身边可信任的人，直接说“我现在不安全/很难受，需要你陪我”；"
            "第二，把可能伤害自己的物品和药物先交给别人保管，尽量不要一个人待着；"
            "第三，如果已经有伤害自己的冲动或具体计划，请马上拨打当地急救电话、危机热线，或直接去急诊。\n\n"
            "等你身边有人陪着、风险先降下来后，再谈今晚的睡眠和学业压力怎么拆。现在最重要的是安全，而不是把情绪靠意志压下去。"
        )

    if re.search(r"精力不足|恢复不过来|恢复差|疲惫|疲劳|很累|总觉得累", q) and (sleep_quality or stress):
        sleep_text = f"睡眠质量记录为“{sleep_quality}”" if sleep_quality else "睡眠恢复不足"
        share_line = "如果育儿压力占主要部分，今晚就争取把一次夜间照看或明早起床任务交给家人/搭档，换一段更完整的休息。"
        return (
            f"{anchor_text}{sleep_text}，压力源是 {stress_text}；这类精力不足先按恢复问题处理，不要简单归因成自律不够。\n\n"
            "今晚先做三个低门槛动作：睡前30-60分钟停止工作消息和高刺激屏幕；写5分钟担忧/待办清单，"
            "把工作和育儿事项约到明天固定窗口；再做10分钟慢呼吸或渐进式肌肉放松。\n\n"
            "明天开始把恢复拆到白天：固定起床时间，上午晒10-15分钟自然光；每90分钟安排一次5分钟休息，"
            "只做喝水、站起来走动或闭眼呼吸，不用追求一次彻底放松。"
            f"{share_line}\n\n"
            "先连续执行3天，再观察精力、睡眠和白天情绪；如果疲劳持续超过2周、明显影响工作生活，建议咨询医生或睡眠/心理专业人士。"
        )

    if re.search(r"睡不着|入睡|睡不好|失眠|躺下来脑子停不下来", q):
        if re.search(r"比赛|紧张", q):
            return (
                f"{anchor_text}下周比赛带来的焦虑会直接影响入睡，重点不是临时加练，而是把兴奋度降下来。\n\n"
                "赛前 7 天训练量减到平时约 50%-70%，睡前 60 分钟停止刷屏和看比赛信息；"
                "睡前写 5 分钟“担忧清单”，把明天要处理的事约到固定 10 分钟窗口，再做 10 分钟呼吸放松或视觉化。"
                "赛前一天不做剧烈训练，尽量保证 7-8 小时睡眠。"
            )
        if re.search(r"deadline|脑子停不下来|躺下来脑子停不下来", q, re.IGNORECASE):
            return (
                f"{anchor_text}你的压力源主要是 {stress_text}，晚上脑子停不下来时，先用一个固定睡前流程把工作/担忧从床上移出去。\n\n"
                "今晚开始：睡前 60 分钟关掉高刺激屏幕；花 5 分钟写担忧清单和明天第一步；"
                "再做 10 分钟 4-7-8 呼吸或渐进式肌肉放松。白天安排 15-30 分钟低强度活动，咖啡因尽量放在午前。"
                "如果连续超过 2 周仍明显影响白天功能，建议看睡眠门诊或心理咨询。"
            )

    if re.search(r"没.*运动.*欲望|没有运动.*欲望|下班.*躺|找回动力|没动力|没有.*动力", q):
        body_note = (
            f"你现在 {weight:g}kg、BMI约{bmi}，即使目标是{goal}，这一周也先不要把运动当作补偿热量的工具；"
            if weight and bmi and goal and goal != "健康"
            else ""
        )
        return (
            f"{anchor_text}这更像疲劳或倦怠后的动力下降，不要靠硬顶。你的压力源是 {stress_text}，先把目标降到足够低。\n\n"
            f"{body_note}"
            "今天只做 5-10 分钟最小版本：换鞋下楼走一圈，或在家做 5 分钟拉伸；完成就算成功。"
            "接下来 7 天固定同一时间做这个小目标，周末再加到 15-20 分钟。"
            "如果同时有持续情绪低落、兴趣明显下降或睡眠/食欲大变，建议尽快寻求心理支持。"
        )

    if re.search(r"汇报|公开|演讲|展示|presentation|紧张|逃避", q, re.IGNORECASE):
        return (
            f"{anchor_text}你的压力源是 {stress_text}，公开汇报前的紧张可以不用追求消失，先把它变成可执行的准备流程。\n\n"
            "今天先用 20 分钟把汇报拆成开场、3 个要点和结尾；再对着镜子或朋友演练 1 次，只记录一个要改的小点。"
            "正式汇报前做 2 分钟慢呼吸：吸气 4 秒、呼气 6 秒，注意力拉回脚踩地面的感觉。"
            "接下来每天练 1 次 10-15 分钟，不临时通宵堆准备。"
        )

    if re.search(r"焦虑.*零食|零食.*焦虑|暴食|情绪性进食|压力性进食|吃.*停不下来|停不下来.*吃", q):
        return (
            f"{anchor_text}你的压力源是 {stress_text}，这更像焦虑触发的压力性进食，不要靠“忍住”解决。\n\n"
            "先做一个 5 分钟暂停流程：记录触发情境、情绪强度 1-10 分、真正想要的是休息还是安慰；"
            "然后喝水或走 3-5 分钟，再决定是否吃。零食环境也要改：把高热量零食移出桌面，换成酸奶、豆制品、水果或分装坚果。"
            "如果你的目标是减脂，每次加餐先配蛋白或纤维，减少血糖波动带来的继续想吃。"
        )

    if re.search(r"吃多|内疚|一天都毁|全或无|补偿", q):
        return (
            f"{anchor_text}你提到的“吃多一点就觉得一天毁了”，很像全或无自动想法。"
            "减脂目标不需要靠惩罚式补偿来完成。\n\n"
            "下一餐回到正常节奏：一掌心蛋白质、一拳主食、两拳蔬菜即可；不要跳餐。"
            "写下这句话替换旧想法：“一餐偏多不会决定长期趋势，接下来 24 小时回到稳定节奏就够。”"
            "如果这种内疚频繁出现并影响进食，建议考虑心理咨询或进食问题相关评估。"
        )

    if re.search(r"刷手机|手机.*停不下来|越刷越清醒|睡前习惯|屏幕", q):
        return (
            f"{anchor_text}你记录的睡眠质量偏差，压力/触发源是 {stress_text}，睡前刷手机停不下来时，关键是把决策提前做好。\n\n"
            "今晚开始用 60 分钟手机离床流程：手机放到床外 2 米并开启勿扰；洗漱后只选一个替代动作，"
            "比如看 5-10 分钟纸质内容、热水澡、轻柔拉伸或呼吸练习。"
            "如果忍不住拿手机，就只允许站在床边看，不带回被窝。\n\n"
            "先连续 3 天只追求启动流程，不追求完美入睡；同时固定起床时间，帮助睡眠节律重新稳定。"
        )

    if re.search(r"熬夜|睡眠不足|没睡好", q):
        return (
            f"{anchor_text}熬夜后的心理恢复重点是降低刺激和稳定作息，而不是靠意志硬顶。\n\n"
            "今天先安排 20-30 分钟午休或提前 60 分钟上床；睡前 30 分钟停止工作和高刺激内容，"
            "写 5 分钟担忧清单，再做 10 分钟呼吸放松。训练强度和身体不适由训练教练/医生判断。"
        )

    return ""


def _episode_section(episode_context: str) -> str:
    if not episode_context:
        return ""
    return (
        "\n【近期/相关对话记录】\n"
        f"{episode_context}\n"
        "说明：[最近] 表示按时间最近的对话，[相关] 表示因话题相似而召回的更早对话；引用时请区分使用。\n"
    )


def _current_time_section() -> str:
    try:
        now = datetime.now(ZoneInfo(DEFAULT_TIMEZONE))
    except Exception:
        now = datetime.now()
    return f"【当前日期时间】{now.strftime('%Y-%m-%d %H:%M')}（{DEFAULT_TIMEZONE}）\n"


def _build_psychologist_agent(pctx: dict, peer_notes_text: str, episode_context: str = "", user_question: str = ""):
    peer_section = peer_notes_text if peer_notes_text else ""
    user_card = (pctx.get("role_user_cards") or {}).get("Psychologist") or pctx.get("user_card") or "【关于该用户】\n用户画像暂不可用。"
    decision_section = format_decision_points_for_prompt(
        build_personalization_decision_points(pctx, user_question, role="Psychologist")
    )
    system_prompt = (
        "你是心理疗愈师，专注压力、焦虑、情绪调节、倦怠、动力下降、压力性进食、睡前心理放松和心理安全边界。\n\n"
        f"{_current_time_section()}"
        f"{user_card}\n"
        f"{decision_section}"
        f"{_episode_section(episode_context)}"
        f"{peer_section}"
        "用户卡片就是本轮可用画像；不要说「我先看看/了解你的基本信息」，不要为了读取画像而调用工具。"
        "若已有足够信息，必须直接给出方案；只有用户提供新信息时才调用结构化工具记录。"
        "如需要心理健康、压力管理、睡前放松或倦怠相关知识库支持，可主动调用 retrieve_psychologist_knowledge。"
        "对于纯打招呼或与心理支持无关的内容，请直接回答，无需检索。"
        "若检索结果明确返回 '未命中本地知识库'，可凭通用心理支持、睡眠卫生与压力管理知识给出保守兜底建议。"
        "若用户透露新的压力来源、睡眠信息、情绪变化或回答风格偏好，请优先调用 add_stress_source / "
        "set_response_style 记录；update_user_profile 仅作兼容兜底。"
        "如果用户要求记录睡眠、情绪、压力或恢复打卡，请调用 log_wellness_checkin；"
        "如果用户要求之后提醒放松、睡觉或复盘，请调用 push_reminder；需要复盘近期恢复时，先调用 query_logs(kind='wellness')。"
        "如果用户明确要求把睡眠、放松、复盘或恢复安排加入 Apple Calendar / 苹果日历 / 日历，"
        "请调用 schedule_calendar_event；start_iso 必须使用 ISO 时间并带当前时区，若日期或时间缺失则先询问。"
        "输出时兼顾心理支持、可执行节奏与风险边界。"
        "【领域边界】你不处理身体症状、疼痛、头晕、恶心、胸闷、心悸、伤病、训练负荷或用药问题；"
        "遇到这些内容，先明确建议由 Doctor/医生评估，训练相关负荷交给 Trainer。"
        "只有当身体不适背后同时有明显焦虑、压力、恐慌、反刍或倦怠时，才补充心理支持建议。"
        "【工具使用】当用户询问睡眠、失眠、压力、焦虑、情绪、动力下降、压力性进食或倦怠时，"
        "优先调用 retrieve_psychologist_knowledge；纯寒暄、感谢或简单确认无需检索。"
        "【睡眠/压力方案】优先给非药物、可执行的分层方案：今晚能做什么、接下来 7 天怎么调整、何时寻求专业帮助。"
        "建议应包含固定起床时间、睡前 30-60 分钟降刺激、担忧清单/预约担忧时间、呼吸或渐进式肌肉放松、白天光照与低强度活动等。"
        "不要建议自行使用安眠药、镇静药或酒精助眠。"
        "【动力与倦怠】面对'没动力/只想躺着'，不要使用'强迫自己/逼自己'一类措辞；先降低门槛，给 5-10 分钟最小行动版本，"
        "再给可持续的奖励、环境设计或同伴支持。"
        "【心理安全边界】若出现自伤/轻生念头、持续恐慌、严重抑郁、连续失眠超过 2 周且影响白天功能，"
        "必须建议尽快联系心理/睡眠门诊或当地危机支持；若有即时危险，优先联系急救或身边可信的人。"
        "【输出硬性要求】\n"
        "1. 若用户卡片有压力源，回答开头必须点名压力源；若有年龄/伤病，也要自然衔接。\n"
        "2. 必须把压力源/睡眠状态/年龄/作息映射到具体场景化建议，不允许只说「放松」。\n"
        "3. 至少包含 1 条结合压力源或具体作息的可执行建议，时长/时段/频次具体到分钟或天数。"
        "若用户说比赛紧张/睡不好，必须直接写出「焦虑」并给出赛前减量、固定作息、视觉化或呼吸放松方案。"
        "若用户描述了身体症状（持续疼痛、头晕、恶心、心率异常等），"
        "必须首先建议就医/医生评估，且不要给身体康复、训练恢复或医学处理方案。"
    )
    return create_agent(llm, _PSYCHOLOGIST_TOOLS, system_prompt)


def run_psychologist(
    user_id: str,
    user_question: str,
    peer_notes_text: str = "",
    pctx: dict | None = None,
    episode_context: str = "",
) -> dict:
    try:
        os.environ["WELLNESS_COPILOT_USER_ID"] = user_id
        print_expert_start("Psychologist", user_question)
        pctx = pctx or build_personalization_ctx(user_id)
        deterministic = _deterministic_psychologist_answer(pctx, user_question)
        if deterministic:
            print_expert_end("Psychologist", [], deterministic)
            return {
                "expert_responses": {"Psychologist": deterministic},
                "agent_notes": {"Psychologist": build_scratchpad_note("Psychologist", deterministic)},
                "last_tools": [],
                "retrieval_hits": 0,
            }
        agent = _build_psychologist_agent(
            pctx,
            peer_notes_text,
            episode_context,
            user_question,
        )
        result = agent.invoke({"messages": [HumanMessage(content=user_question)]})

        print_expert_trace("Psychologist", result["messages"])

        used_tools: list[str] = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    used_tools.append(call.get("name", "Unknown"))

        retrieval_hits = sum(
            1 for t in used_tools if "retrieve" in t and "knowledge" in t
        )
        answer = extract_text_content(result["messages"][-1])
        answer = _psychologist_profile_intro(pctx, user_question, answer)
        print_expert_end("Psychologist", used_tools, answer)
        return {
            "expert_responses": {"Psychologist": answer},
            "agent_notes": {"Psychologist": build_scratchpad_note("Psychologist", answer)},
            "last_tools": used_tools,
            "retrieval_hits": retrieval_hits,
            "actuation_log": extract_actuation_events(result["messages"]),
        }
    except Exception as e:
        return expert_error_update("Psychologist", e)
