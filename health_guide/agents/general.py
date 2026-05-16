"""General assistant expert — invoked as a callable by the Dispatcher."""
from langchain_core.messages import HumanMessage

from ..tools import (
    get_user_profile,
    retrieve_general_knowledge,
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


_GENERAL_TOOLS = [
    retrieve_general_knowledge,
    get_user_profile,
    update_user_profile,
]


def _build_general_agent(profile_text: str, peer_notes_text: str):
    peer_section = peer_notes_text if peer_notes_text else ""
    system_prompt = (
        "你是健康管理团队的贴心助理。"
        f"当前用户画像：{profile_text}。"
        f"{peer_section}"
        "负责寒暄、通用问题与多轮澄清。"
        "若用户询问健康常识，可调用 retrieve_general_knowledge 补充知识；"
        "对于纯打招呼/告别，请直接回答，无需检索。"
        "若用户提供新的个人偏好、作息、目标变化，请调用 update_user_profile 进行记录。"
        "回答保持自然、简洁。"
        "优先短回复，不主动给健康方案；只有用户明确询问时再进入健康建议。"
    )
    return create_agent(llm, _GENERAL_TOOLS, system_prompt)


def run_general(user_id: str, user_question: str, peer_notes_text: str = "") -> dict:
    try:
        print_expert_start("General", user_question)
        profile_text = profile_to_prompt_text_for(
            "General", get_profile_from_store(user_id)
        )
        agent = _build_general_agent(profile_text, peer_notes_text)
        result = agent.invoke({"messages": [HumanMessage(content=user_question)]})

        print_expert_trace("General", result["messages"])

        used_tools: list[str] = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    used_tools.append(call.get("name", "Unknown"))

        retrieval_hits = sum(
            1 for t in used_tools if "retrieve" in t and "knowledge" in t
        )
        answer = extract_text_content(result["messages"][-1])
        print_expert_end("General", used_tools, answer)
        return {
            "expert_responses": {"General": answer},
            "agent_notes": {"General": build_scratchpad_note("General", answer)},
            "last_tools": used_tools,
            "retrieval_hits": retrieval_hits,
        }
    except Exception as e:
        return expert_error_update("General", e)
