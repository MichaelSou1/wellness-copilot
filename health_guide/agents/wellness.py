"""Wellness expert — invoked as a callable by the Dispatcher."""
import os

from langchain_core.messages import HumanMessage

from ..tools import (
    add_stress_source,
    retrieve_wellness_knowledge,
    set_response_style,
    update_user_profile,
)
from ..utils import create_agent
from ..llm import extract_text_content, llm
from ..personalization import build_personalization_ctx
from ..detail import print_expert_end, print_expert_start, print_expert_trace
from .fallbacks import expert_error_update
from ._scratchpad import build_scratchpad_note


_WELLNESS_TOOLS = [
    add_stress_source,
    set_response_style,
    update_user_profile,
    retrieve_wellness_knowledge,
]


def _episode_section(episode_context: str) -> str:
    if not episode_context:
        return ""
    return (
        "\n【近期/相关对话记录】\n"
        f"{episode_context}\n"
        "说明：[最近] 表示按时间最近的对话，[相关] 表示因话题相似而召回的更早对话；引用时请区分使用。\n"
    )


def _build_wellness_agent(pctx: dict, peer_notes_text: str, episode_context: str = ""):
    peer_section = peer_notes_text if peer_notes_text else ""
    user_card = pctx.get("user_card") or "【关于该用户】\n用户画像暂不可用。"
    system_prompt = (
        "你是身心康复师。\n\n"
        f"{user_card}\n"
        f"{_episode_section(episode_context)}"
        f"{peer_section}"
        "用户卡片就是本轮可用画像；不要说「我先看看/了解你的基本信息」，不要为了读取画像而调用工具。"
        "若已有足够信息，必须直接给出方案；只有用户提供新信息时才调用结构化工具记录。"
        "如需要康复/睡眠/压力管理方面的知识库支持，可主动调用 retrieve_wellness_knowledge。"
        "对于纯打招呼或与身心恢复无关的内容，请直接回答，无需检索。"
        "若检索结果明确返回 '未命中本地知识库'，可凭通用康复、睡眠与压力管理知识给出保守兜底建议。"
        "若用户透露新的压力来源、睡眠信息、疼痛变化或回答风格偏好，请优先调用 add_stress_source / "
        "set_response_style 记录；update_user_profile 仅作兼容兜底。"
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
        "【输出硬性要求】\n"
        "1. 若用户卡片有压力源，回答开头必须点名压力源；若有年龄/伤病，也要自然衔接。\n"
        "2. 必须把压力源/睡眠状态/年龄/作息映射到具体场景化建议，不允许只说「放松」。\n"
        "3. 至少包含 1 条结合压力源或具体作息的可执行建议，时长/时段/频次具体到分钟或天数。"
        "若用户描述了身体症状（持续疼痛、心率异常、长期失眠超过两周等），"
        "必须首先建议就医，再提供辅助性建议。"
    )
    return create_agent(llm, _WELLNESS_TOOLS, system_prompt)


def run_wellness(
    user_id: str,
    user_question: str,
    peer_notes_text: str = "",
    pctx: dict | None = None,
    episode_context: str = "",
) -> dict:
    try:
        os.environ["HEALTH_GUIDE_USER_ID"] = user_id
        print_expert_start("Wellness", user_question)
        agent = _build_wellness_agent(
            pctx or build_personalization_ctx(user_id),
            peer_notes_text,
            episode_context,
        )
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
