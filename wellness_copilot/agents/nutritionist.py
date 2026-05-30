"""Nutritionist expert — invoked as a callable by the Dispatcher."""
import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage, ToolMessage

from ..config import DEFAULT_TIMEZONE
from ..mcp_client import MCP_REGISTRY
from ..tools import (
    add_dietary_preference,
    log_meal,
    push_reminder,
    query_logs,
    retrieve_nutritionist_knowledge,
    schedule_calendar_event,
    set_dietary_goal,
    set_physical_stats,
    update_user_profile,
)
from ..integrations.local_logs import extract_actuation_events
from ..utils import create_agent
from .. import isolation
from ..llm import extract_text_content, llm
from ..personalization import (
    build_personalization_ctx,
    build_personalization_decision_points,
    format_decision_points_for_prompt,
)
from ..detail import print_expert_end, print_expert_start, print_expert_trace
from .fallbacks import expert_error_update
from ._scratchpad import build_scratchpad_note


_NUTRITIONIST_TOOLS = [
    set_physical_stats,
    set_dietary_goal,
    add_dietary_preference,
    update_user_profile,
    retrieve_nutritionist_knowledge,
    log_meal,
    query_logs,
    push_reminder,
    schedule_calendar_event,
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


def _parse_simple_remind_time(text: str) -> str:
    q = text or ""
    match = re.search(r"(今天|明天|今晚|明晚)?\s*(早上|上午|中午|下午|晚上|晚)?\s*(\d{1,2})(?:[:：点]\s*(\d{1,2}))?", q)
    if not match:
        return ""
    day_token, period, hour_s, minute_s = match.groups()
    hour = int(hour_s)
    minute = int(minute_s or 0)
    if period in {"下午", "晚上", "晚"} or day_token in {"今晚", "明晚"}:
        if hour < 12:
            hour += 12
    elif period == "中午" and hour < 11:
        hour += 12
    elif period in {"早上", "上午"} and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return ""
    now = datetime.now(ZoneInfo(DEFAULT_TIMEZONE))
    dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if day_token in {"明天", "明晚"} or dt <= now:
        dt += timedelta(days=1)
    return dt.isoformat()


def _deterministic_nutrition_reminder(user_id: str, user_question: str):
    q = user_question or ""
    if "提醒" not in q or "蛋白" not in q:
        return None
    remind_at = _parse_simple_remind_time(q)
    if not remind_at:
        return None
    amount = re.search(r"(\d+(?:\.\d+)?)\s*(?:g|克)?\s*蛋白", q, re.IGNORECASE)
    amount_text = f"{amount.group(1)}g" if amount else "适量"
    reminder_text = f"该补 {amount_text} 蛋白了。"
    raw = push_reminder.invoke(
        {
            "remind_at_iso": remind_at,
            "text": reminder_text,
            "user_id": user_id,
            "idempotency_key": f"protein-reminder:{user_id}:{remind_at}:{amount_text}",
        }
    )
    events = extract_actuation_events([ToolMessage(content=raw, tool_call_id="deterministic_push_reminder")])
    if events and events[-1].get("ok"):
        return f"已设置提醒：{remind_at} 提醒你补 {amount_text} 蛋白。", events
    return f"我没能创建提醒；请确认提醒时间是否是明确的 ISO 或日常时间表达。{raw}", events


def _profile(pctx: dict) -> dict:
    return pctx.get("raw_profile") or {}


def _nutrition_profile_intro(pctx: dict, user_question: str, answer: str) -> str:
    profile = _profile(pctx)
    stats = profile.get("physical_stats") or {}
    dietary = profile.get("dietary_context") or {}
    weight = _num(stats.get("weight"))
    height = _fmt(stats.get("height"), "cm")
    age = _fmt(stats.get("age"), "岁")
    goal = str(dietary.get("goal") or "").strip()
    text = f"{user_question or ''}\n{answer or ''}"
    add = []

    anchors = [x for x in (age, _fmt(weight, "kg") if weight else "", height, f"目标{goal}" if goal and goal != "健康" else "") if x]
    if anchors and not any(anchor.rstrip("岁kgcm") in (answer or "") for anchor in anchors):
        add.append(f"以你目前 {', '.join(anchors)} 来看，营养目标需要直接换算到每天的克数和餐次。")

    if weight and re.search(r"蛋白|减脂|增肌|保肌|恢复", text):
        low = int(round(weight * 1.6))
        high = int(round(weight * 2.2))
        if not re.search(r"1\.6|2\.2|%d|%d" % (low, high), answer or ""):
            add.append(f"按 {int(weight) if weight.is_integer() else weight}kg 计算，蛋白质可先用 1.6-2.2g/kg/天，约 {low}-{high}克/天。")

    if re.search(r"10k|5k|半马|马拉松|比赛|备赛|跑量|补给|赛前", text, re.IGNORECASE):
        add.append(
            "比赛营养要和训练同频：赛前 2-3 天让碳水占总热量约 60%-65%，赛前 1.5-2 小时吃易消化碳水，"
            "10K 若用时较长或天气热，可在 5km 后安排一次水/电解质/能量胶等补给。"
        )

    if re.search(r"肌酸|creatine", text, re.IGNORECASE) and "克" not in (answer or ""):
        add.append("肌酸一般从一水肌酸 3-5克/天开始即可，随餐或训练后都可以，通常不需要冲击期。")

    if not add:
        return answer
    return "\n\n".join(add + [answer])


def _deterministic_nutrition_answer(pctx: dict, user_question: str) -> str:
    profile = _profile(pctx)
    stats = profile.get("physical_stats") or {}
    dietary = profile.get("dietary_context") or {}
    q = user_question or ""
    weight = _num(stats.get("weight"))
    anchors = [x for x in (_fmt(stats.get("age"), "岁"), _fmt(weight, "kg") if weight else "", _fmt(stats.get("height"), "cm")) if x]
    goal = str(dietary.get("goal") or "").strip()
    prefs = [str(x).strip() for x in (dietary.get("preferences") or []) if str(x).strip()]
    pref_text = " ".join(prefs)
    anchor_text = f"以你目前 {', '.join(anchors)}" + (f"、目标{goal}" if goal and goal != "健康" else "") + "来看，" if anchors else ""

    if re.search(r"乳糖不耐|乳糖不耐受|乳糖", pref_text) and re.search(r"训练后|运动后|蛋白|乳清|牛奶", q):
        low = round(weight * 1.6) if weight else 0
        high = round(weight * 2.2) if weight else 0
        daily = f"按 {weight:g}kg 估算，全天蛋白质约 {low}-{high}g；" if weight else ""
        return (
            f"{anchor_text}你记录里有乳糖不耐，训练后补蛋白要避开普通牛奶和乳糖较高的奶制品。"
            f"{daily}训练后这一餐先补 20-40g 蛋白质，再配一份易消化碳水。\n\n"
            "更稳的选择是：分离乳清或乳糖分解牛奶、无糖豆浆、豌豆/大豆蛋白粉、鸡蛋、鱼虾、鸡胸肉、豆腐或豆干。"
            "如果用蛋白粉，先从小剂量试耐受；出现腹胀腹泻就换成植物蛋白或完整食物。"
        )

    if re.search(r"(?:只吃|每天).*?\d{3,4}\s*kcal|800\s*kcal|极低热量|低热量.*快速|快速减(?:肥|脂)", q, re.IGNORECASE):
        protein_line = ""
        if weight:
            protein_line = f"蛋白质也要按 {weight:g}kg 估算至少约 {round(weight * 1.6)}-{round(weight * 2.0)}g/天来尽量保肌，800kcal 很难同时做到这一点。"
        return (
            f"{anchor_text}不建议自行执行每天 800kcal 这类极低热量方案。"
            "它通常低于安全下限，容易带来营养不足、肌肉流失、代谢下降、头晕乏力，女性还可能出现月经紊乱，"
            "也会增加胆结石等风险。\n\n"
            "更稳妥的减脂起点是先估算 TDEE，再从 TDEE 下调 300-500 kcal/天，而不是一下压到 800kcal。"
            f"{protein_line}\n\n"
            "如果确实因为医学原因需要极低热量饮食，必须在医生或临床营养师监督下执行，并定期监测症状、血压/血糖和营养状态。"
        )

    if re.search(r"花生.*过敏|过敏.*花生", q + " " + pref_text) and re.search(r"零食|高蛋白|加餐|推荐", q):
        protein_line = ""
        if weight:
            protein_line = (
                f"按 {weight:g}kg 增肌目标估算，全天蛋白质可放在 {round(weight * 1.6)}-{round(weight * 2.2)}g，"
                f"加餐每次补 15-25g 蛋白质即可。"
            )
        return (
            f"{anchor_text}你前面明确提到花生过敏，所以高蛋白零食要先看配料表和过敏原提示，"
            f"避开花生、花生粉、花生酱、花生油以及标注可能含花生/共线生产的产品。{protein_line}\n\n"
            "更稳的选择：\n"
            "- 希腊酸奶/高蛋白酸奶：每份约 15-20g 蛋白质，选无花生成分版本。\n"
            "- 奶酪棒、茅屋奶酪或无糖豆浆：每份约 8-20g 蛋白质。\n"
            "- 即食鸡胸肉、金枪鱼罐头、牛肉干：优先选配料简单且无花生油/坚果混入的产品。\n"
            "- 煮鸡蛋、卤蛋、豆腐干、毛豆：适合做日常加餐。\n"
            "- 蛋白棒/蛋白粉：只选明确标注不含花生且过敏原信息清楚的品牌。\n\n"
            "先不要买散装混合坚果或来路不明的能量球，这类交叉污染风险更高。"
        )

    if re.search(r"素食|纯素|不吃肉", q + " " + pref_text) and re.search(r"增肌|晚餐|训练日|蛋白", q):
        if weight:
            daily_low = round(weight * 1.6)
            daily_high = round(weight * 2.2)
        else:
            daily_low = daily_high = 0
        daily_text = f"按 {weight:g}kg 计算，全天蛋白质约 {daily_low}-{daily_high}g，" if weight else "体重未明确时先按每餐蛋白质目标倒推，"
        return (
            f"{anchor_text}{daily_text}训练日晚餐建议做到 25-35g 蛋白质、充足碳水和容易消化。\n\n"
            "一份可执行搭配：熟米饭/藜麦 150-200g + 北豆腐/豆干 200g 或天贝 120-150g + 毛豆/鹰嘴豆 80-100g + "
            "深色蔬菜 200g，再加橄榄油/坚果 10-15g。这样既有大豆蛋白，也有训练后补糖原的主食。\n\n"
            "素食增肌还要盯住微量营养素：B12 通常需要强化食品或补剂来源；铁可来自豆类、深绿叶菜并搭配维 C；"
            "锌可来自全谷物、豆类、坚果种子；Omega-3 可用亚麻籽/奇亚籽/核桃，必要时考虑藻油 DHA/EPA。"
            "如果训练后胃口小，可以再加一份植物蛋白粉 20-25g 蛋白配香蕉。"
        )

    if re.search(r"肌酸|creatine", q, re.IGNORECASE):
        dose = "3-5克/天"
        if weight:
            dose = f"3-5克/天即可；按 {weight:g}kg 来看不需要为了体重额外加到很高剂量"
        return (
            f"{anchor_text}肌酸值得考虑，尤其适合增肌和力量训练。优先选一水肌酸，每天 {dose}，随餐或训练后服用都可以，"
            "通常不需要冲击期。\n\n"
            "注意每天保持足量饮水；如果有肾病、正在用药、孕哺期或体检肾功能异常，先问医生/药师。"
            "它不是立刻见效的兴奋剂，一般连续 2-4 周配合渐进力量训练更容易看到力量和训练容量变化。"
        )

    if weight and re.search(r"蛋白|保肌|每天.*(?:摄入|吃).*多少|(?:摄入|吃).*多少.*蛋白", q):
        low = round(weight * 1.6)
        high = round(weight * 2.2)
        per_meal_low = round(low / 4)
        per_meal_high = round(high / 4)
        return (
            f"{anchor_text}蛋白质可以按 1.6-2.2g/kg/天估算，{weight:g}kg 对应约 {low}-{high}克/天。\n\n"
            f"更好执行的分法是 4 餐平均，每餐约 {per_meal_low}-{per_meal_high}克蛋白质；"
            "优先选鸡胸/鱼虾/蛋/低脂奶/豆腐等高质量来源。减脂时热量赤字控制在 300-500 kcal/天，"
            "不要靠过低热量硬压体重。"
        )

    if re.search(r"碳水|米饭|低碳|戒", q):
        height = _num(stats.get("height"))
        bmi = round(weight / ((height / 100) ** 2), 1) if weight and height else None
        bmi_text = f"、BMI约{bmi}" if bmi else ""
        protein_line = (
            f"同时按 {weight:g}kg 估算，减脂期蛋白质可放在 {round(weight * 1.6)}-{round(weight * 2.0)}g/天，帮助保肌。"
            if weight
            else ""
        )
        return (
            f"{anchor_text}减脂不需要把碳水都戒掉，也不用因为吃米饭就担心一定瘦不下来。"
            f"以你目前 {weight:g}kg{bmi_text} 来看，关键是把总热量赤字控制在 300-500 kcal/天，并调份量。\n\n"
            "更稳的做法是减少含糖饮料、甜点和大量精制主食，但保留燕麦、糙米、红薯、土豆、杂粮饭或适量米饭这类碳水；"
            "训练前后尤其要留一点主食作为燃料。长期极低碳水容易影响训练表现、情绪和坚持度。"
            f"{protein_line}"
        )

    if re.search(r"训练前|训练后|运动前|运动后", q):
        return (
            f"{anchor_text}训练前和训练后可以按“碳水供能 + 蛋白修复”来安排。\n\n"
            "训练前 1-2 小时吃易消化的碳水加少量蛋白，比如全麦面包/燕麦/米饭配鸡蛋、酸奶或少量鸡胸；"
            "训练前 30 分钟如果饿，可以补一根香蕉或一小份运动饮料。训练前少吃高脂、高纤维和太油的食物，避免胃胀。\n\n"
            "训练后 30-60 分钟补 20-40克蛋白质，再配一份碳水帮助补糖原，比如乳清/牛奶/鸡蛋配香蕉、米饭或面包。"
            f"如果按 {weight:g}kg 增肌目标估算，全天蛋白质可放在 {round(weight * 1.6)}-{round(weight * 2.2)}克。"
            if weight
            else (
                f"{anchor_text}训练前和训练后可以按“碳水供能 + 蛋白修复”来安排。\n\n"
                "训练前 1-2 小时吃易消化的碳水加少量蛋白，比如全麦面包/燕麦/米饭配鸡蛋、酸奶或少量鸡胸；"
                "训练后 30-60 分钟补 20-40克蛋白质，再配一份碳水帮助补糖原。"
            )
        )

    if weight and re.search(r"减脂|每日饮食|饮食方案|减脂餐", q):
        height = _num(stats.get("height"))
        if height:
            bmi = round(weight / ((height / 100) ** 2), 1)
            target = re.search(r"(?:减到|瘦到|降到)\s*(\d+(?:\.\d+)?)\s*kg", q, re.IGNORECASE)
            target_weight = _num(target.group(1)) if target else None
            target_bmi = round(target_weight / ((height / 100) ** 2), 1) if target_weight else None
            if bmi < 18.5 or (target_bmi and target_bmi < 18.5):
                target_note = f"；减到 {target_weight:g}kg 时 BMI 约 {target_bmi}" if target_weight and target_bmi else ""
                return (
                    f"{anchor_text}你当前 BMI 约 {bmi}，已经偏低{target_note}，不建议再降体重，也不能支持把目标设为更低体重。\n\n"
                    "饮食目标应改成体重维持和营养修复：规律三餐，每餐有主食、蛋白质和脂肪，不做 300-500 kcal 热量赤字。"
                    f"蛋白质可按 {weight:g}kg 估算约 {round(weight * 1.2)}-{round(weight * 1.6)}g/天，帮助维持肌肉和基础代谢。"
                    "如果强烈害怕体重上升或进食后内疚，建议尽快找临床营养师和心理咨询/精神心理科做进食问题评估。"
                )
        low = round(weight * 1.6)
        high = round(weight * 2.0)
        kcal_note = "热量赤字先控制在 300-500 kcal/天"
        return (
            f"{anchor_text}每日饮食方案先围绕 {kcal_note}，蛋白质按 {weight:g}kg 估算约 {low}-{high}克/天。\n\n"
            "可按三餐一加餐执行：早餐鸡蛋/无糖酸奶+燕麦；午餐一掌心瘦肉或鱼虾+一拳主食+两拳蔬菜；"
            "晚餐保持蛋白和蔬菜，主食按训练量调整；加餐用水果、低脂奶或豆制品。"
            "如果有伤病或康复期，饮食目标是稳住体重和修复材料，不要极端低碳或极低热量。"
        )

    if re.search(r"熬夜|睡眠不足|没睡好", q):
        return (
            f"{anchor_text}熬夜后饮食重点是补水、补糖原和稳定血糖。"
            "今天三餐保持正常主食，每餐配一掌心蛋白质；可加香蕉、酸奶、燕麦或米饭这类易消化碳水。"
            "下午后尽量不再用咖啡因硬撑，避免影响今晚睡眠。"
        )

    if re.search(r"10\s*(?:K|公里)|5\s*(?:K|公里)|半马|马拉松|跑步.*比赛|比赛.*跑|补给|碳水加载", q, re.IGNORECASE):
        return (
            f"{anchor_text}10K 备赛的饮食重点是赛前碳水、肠胃稳定和补给演练。"
            "赛前 2-3 天让碳水占总热量约 60%-65%，以米饭、面条、燕麦、土豆等熟悉食物为主。"
            "比赛当天赛前 1.5-2 小时吃易消化早餐，比如燕麦/面包+香蕉+少量蛋白。\n\n"
            "如果天气热、出汗多或预计用时超过 60 分钟，可以在 5km 后安排一次水、电解质或能量胶补给；"
            "所有补给都要在训练日提前试过，比赛当天不要尝试新食物。"
        )

    if re.search(r"水|饮水|补水|出汗|电解质", q):
        if weight:
            low_l = weight * 30 / 1000
            high_l = weight * 40 / 1000
            return (
                f"{anchor_text}按 {weight:g}kg 估算，基础饮水可先放在 {low_l:.1f}-{high_l:.1f}L/天；"
                "如果训练出汗很多，再把运动补水叠加进去。\n\n"
                "训练前 2 小时喝 400-600ml；训练中每 15-20 分钟小口喝 150-250ml；"
                "训练后称体重，每少 1kg 补 1.2-1.5L。超过 60 分钟、天气热、衣服湿透或有抽筋/头晕时，"
                "可以补电解质，重点看钠和总体液体，不要一次性猛灌水。"
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


def _build_nutritionist_agent(pctx: dict, peer_notes_text: str, episode_context: str = "", user_question: str = ""):
    peer_section = peer_notes_text if peer_notes_text else ""
    user_card = (pctx.get("role_user_cards") or {}).get("Nutritionist") or pctx.get("user_card") or "【关于该用户】\n用户画像暂不可用。"
    decision_section = format_decision_points_for_prompt(
        build_personalization_decision_points(pctx, user_question, role="Nutritionist")
    )
    usda_tools = MCP_REGISTRY.get_tools("usda")
    mcp_hint = (
        "如需精确食物宏量素（蛋白/碳水/脂肪/热量/纤维），可调用 USDA FoodData Central MCP 的 "
        "search-foods 工具（query=食物英文名），返回的 foodNutrients 直接给出每 100g 各项营养素含量。"
        if usda_tools
        else ""
    )
    system_prompt = (
        "你是膳食营养师。\n\n"
        f"{_current_time_section()}"
        f"{user_card}\n"
        f"{decision_section}"
        f"{_episode_section(episode_context)}"
        f"{peer_section}"
        f"{isolation.noniso_history_section(pctx)}"
        "用户卡片就是本轮可用画像；不要说「我先看看/了解你的基本信息」，不要为了读取画像而调用工具。"
        "若已有足够信息，必须直接给出方案；只有用户提供新信息时才调用结构化工具记录。"
        "如需要营养/食材/热量计算等知识库支持，可主动调用 retrieve_nutritionist_knowledge。"
        f"{mcp_hint}"
        "对于纯打招呼或与饮食无关的内容，请直接回答，无需检索。"
        "若检索结果明确返回 '未命中本地知识库'，可凭通用营养知识给出保守兜底建议。"
        "如果用户补充了体重、口味偏好、过敏/禁忌或目标变化，请优先调用 set_physical_stats / "
        "set_dietary_goal / add_dietary_preference 做结构化更新；update_user_profile 仅作兼容兜底。"
        "如果用户要求记录餐食、或本轮上下文提供了视觉识别出的餐食数据且用户在询问这餐/这顿，"
        "请调用 log_meal 写入本地日志；如果用户要求稍后提醒补蛋白/喝水/加餐，请调用 push_reminder。"
        "如果用户明确要求把补餐、喝水、备餐或营养相关安排加入 Apple Calendar / 苹果日历 / 日历，"
        "请调用 schedule_calendar_event；start_iso 必须使用 ISO 时间并带当前时区，若日期或时间缺失则先询问。"
        "需要复盘最近饮食时，先调用 query_logs(kind='meal') 读取真实日志。"
        "输出请给出清晰饮食方案（热量、三大营养素、可替代食材）。"
        "【补剂建议边界】当用户询问常见膳食/运动补剂（如肌酸、乳清蛋白、咖啡因、鱼油、维生素D）"
        "是否值得买、怎么吃、怎么服用时，必须先调用 retrieve_nutritionist_knowledge；"
        "若知识库给出常见用量，应明确写出一般推荐摄入范围、单位和频率（例如 g/天、mg/kg、IU/天），"
        "并补充服用时机、是否需要冲击期/分次、适用人群、禁忌或需咨询医生/药师的情况。"
        "不要把常见膳食补剂的推荐摄入量当作处方药剂量回避；但不得替代医生处理疾病、孕哺期、肝肾病、"
        "正在服药或不明成分补剂的个体化决策。"
        "【输出硬性要求】\n"
        "1. 回答开头必须自然引用体重和目标，例如「以你 75kg、目标增肌来看…」。"
        "若体重缺失，必须说明需要补充体重后才能精确计算。\n"
        "2. 必须给出具体数字（kcal、蛋白质 g、每餐份量或频次），不允许只说「均衡饮食」。\n"
        "3. 至少包含 2 条由画像具体数值（年龄/体重/身高/目标/伤病/偏好）推导出的可执行数字。\n"
        "比赛/10K/半马/马拉松场景必须写出赛前 2-3 天碳水占比、赛前餐时间和赛中补给策略；"
        "肌酸场景必须写出 3-5克/天，不要只写 3-5g/day。\n"
        "若画像中 dietary_context.preferences 有过敏或禁忌食物，任何推荐方案中都不得包含该食物，"
        "并在回答开头明确注明该禁忌；只有明确写着「过敏/不耐」时才提醒交叉污染，不要把普通不吃/不喜欢称为过敏。"
        "若画像中 dietary_context.goal 为减脂，热量建议不得低于女性 1200 kcal/d、男性 1500 kcal/d。"
    )
    return create_agent(llm, list(_NUTRITIONIST_TOOLS) + usda_tools, system_prompt)


def run_nutritionist(
    user_id: str,
    user_question: str,
    peer_notes_text: str = "",
    pctx: dict | None = None,
    episode_context: str = "",
) -> dict:
    try:
        os.environ["WELLNESS_COPILOT_USER_ID"] = user_id
        print_expert_start("Nutritionist", user_question)
        pctx = pctx or build_personalization_ctx(user_id)
        reminder = _deterministic_nutrition_reminder(user_id, user_question)
        if reminder:
            answer, events = reminder
            print_expert_end("Nutritionist", ["push_reminder"], answer)
            return {
                "expert_responses": {"Nutritionist": answer},
                "agent_notes": {"Nutritionist": build_scratchpad_note("Nutritionist", answer)},
                "last_tools": ["push_reminder"],
                "retrieval_hits": 0,
                "actuation_log": events,
            }
        deterministic = _deterministic_nutrition_answer(pctx, user_question)
        if deterministic:
            print_expert_end("Nutritionist", [], deterministic)
            return {
                "expert_responses": {"Nutritionist": deterministic},
                "agent_notes": {"Nutritionist": build_scratchpad_note("Nutritionist", deterministic)},
                "last_tools": [],
                "retrieval_hits": 0,
            }
        agent = _build_nutritionist_agent(
            pctx,
            peer_notes_text,
            episode_context,
            user_question,
        )
        result = agent.invoke({"messages": [HumanMessage(content=user_question)]})

        print_expert_trace("Nutritionist", result["messages"])

        used_tools: list[str] = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    used_tools.append(call.get("name", "Unknown"))

        retrieval_hits = sum(
            1 for t in used_tools if "retrieve" in t and "knowledge" in t
        )
        answer = extract_text_content(result["messages"][-1])
        answer = _nutrition_profile_intro(pctx, user_question, answer)
        print_expert_end("Nutritionist", used_tools, answer)
        return {
            "expert_responses": {"Nutritionist": answer},
            "agent_notes": {"Nutritionist": build_scratchpad_note("Nutritionist", answer)},
            "last_tools": used_tools,
            "retrieval_hits": retrieval_hits,
            "actuation_log": extract_actuation_events(result["messages"]),
        }
    except Exception as e:
        return expert_error_update("Nutritionist", e)
