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
        "【个性化要求】回答必须结合画像中已知的压力来源（mental_state.stress_sources）、"
        "睡眠状态和身份背景，给出有针对性的建议而非通用清单。"
        "例如：若画像显示压力来源是'工作加班'，建议应结合该场景（如碎片化放松、下班边界管理），"
        "而不是泛泛说'建议减少压力'。"
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
