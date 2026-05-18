"""Wellness expert — invoked as a callable by the Dispatcher."""
from langchain_core.messages import HumanMessage

from ..tools import (
    get_user_profile,
    retrieve_wellness_knowledge,
    update_user_profile,
)
from ..utils import create_agent
from ..llm import extract_text_content, llm
from ..profile_store import (
    get_user_profile as get_profile_from_store,
    profile_to_prompt_text_for,
)
from ..detail import print_expert_end, print_expert_start, print_expert_trace
from .fallbacks import expert_error_update
from ._scratchpad import build_scratchpad_note


_WELLNESS_TOOLS = [
    get_user_profile,
    update_user_profile,
    retrieve_wellness_knowledge,
]


def _build_wellness_agent(profile_text: str, peer_notes_text: str):
    peer_section = peer_notes_text if peer_notes_text else ""
    system_prompt = (
        "你是身心康复师。"
        f"当前用户画像：{profile_text}。"
        f"{peer_section}"
        "如需要康复/睡眠/压力管理方面的知识库支持，可主动调用 retrieve_wellness_knowledge。"
        "对于纯打招呼或与身心恢复无关的内容，请直接回答，无需检索。"
        "若检索结果明确返回 '未命中本地知识库'，可凭通用康复、睡眠与压力管理知识给出保守兜底建议。"
        "若用户透露新的压力来源、睡眠信息、疼痛变化，请调用 update_user_profile 记录。"
        "输出时兼顾心理支持、恢复节奏与风险边界。"
        "【工具使用】当用户询问睡眠、失眠、压力、焦虑、疲劳、倦怠、恢复节奏或疼痛管理时，"
        "优先调用 retrieve_wellness_knowledge；纯寒暄、感谢或简单确认无需检索。"
        "【睡眠/压力方案】优先给非药物、可执行的分层方案：今晚能做什么、接下来 7 天怎么调整、何时寻求专业帮助。"
        "建议应包含固定起床时间、睡前 30-60 分钟降刺激、担忧清单/预约担忧时间、呼吸或渐进式肌肉放松、白天光照与低强度活动等。"
        "不要建议自行使用安眠药、镇静药或酒精助眠。"
        "【动力与倦怠】面对'没动力/只想躺着'，不要使用'强迫自己/逼自己'一类措辞；先降低门槛，给 5-10 分钟最小行动版本，"
        "再给可持续的奖励、环境设计或同伴支持。"
        "【心理安全边界】若出现自伤/轻生念头、持续恐慌、严重抑郁、连续失眠超过 2 周且影响白天功能，"
        "必须建议尽快联系心理/睡眠门诊或当地危机支持；若有即时危险，优先联系急救或身边可信的人。"
        "【强制个性化】只复述压力源或画像不算个性化。必须把画像里的具体压力源/睡眠状态/年龄/作息映射到具体场景化建议：\n"
        "  - 不写：'你压力大，建议放松'\n"
        "  - 而写：'你的压力源是工作加班，建议在下班后 1 小时内做一次 10 分钟身体扫描放松，并把睡前手机放置桌外'\n"
        "至少包含 1 条结合压力源或具体作息的可执行场景化建议（时长/时段/频次具体到分钟或天数）。"
        "若用户描述了身体症状（持续疼痛、心率异常、长期失眠超过两周等），"
        "必须首先建议就医，再提供辅助性建议。"
    )
    return create_agent(llm, _WELLNESS_TOOLS, system_prompt)


def run_wellness(user_id: str, user_question: str, peer_notes_text: str = "") -> dict:
    try:
        print_expert_start("Wellness", user_question)
        profile_text = profile_to_prompt_text_for(
            "Wellness", get_profile_from_store(user_id)
        )
        agent = _build_wellness_agent(profile_text, peer_notes_text)
        result = agent.invoke({"messages": [HumanMessage(content=user_question)]})

        print_expert_trace("Wellness", result["messages"])

        used_tools: list[str] = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    used_tools.append(call.get("name", "Unknown"))

        retrieval_hits = sum(
            1 for t in used_tools if "retrieve" in t and "knowledge" in t
        )
        answer = extract_text_content(result["messages"][-1])
        print_expert_end("Wellness", used_tools, answer)
        return {
            "expert_responses": {"Wellness": answer},
            "agent_notes": {"Wellness": build_scratchpad_note("Wellness", answer)},
            "last_tools": used_tools,
            "retrieval_hits": retrieval_hits,
        }
    except Exception as e:
        return expert_error_update("Wellness", e)
