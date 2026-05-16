"""Trainer expert — invoked as a callable by the Dispatcher (no longer a graph node).

Receives an isolated, scoped input from the parent agent:
  - SystemMessage: role profile (cropped) + optional peer scratchpad
  - HumanMessage: contextualized user question

RAG is *on-demand*: `retrieve_trainer_knowledge` lives in the tool list so the
expert's ReAct loop decides whether to call it (skip greetings / pure personal
chat, fire for actual training questions).
"""
from langchain_core.messages import HumanMessage

from ..mcp_client import MCP_REGISTRY
from ..tools import (
    calculate_tdee,
    get_user_profile,
    retrieve_trainer_knowledge,
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


_TRAINER_TOOLS = [
    calculate_tdee,
    get_user_profile,
    update_user_profile,
    retrieve_trainer_knowledge,
]


def _build_trainer_agent(profile_text: str, peer_notes_text: str):
    peer_section = peer_notes_text if peer_notes_text else ""
    wger_tools = MCP_REGISTRY.get_tools("wger")
    mcp_hint = (
        "如需查询具体动作百科（标准动作要领、目标肌群、所需器械），可调用 wger MCP 工具："
        "search_exercises / get_exercise_details / list_muscles / list_equipment / list_categories。"
        if wger_tools
        else ""
    )
    system_prompt = (
        "你是力量训练教练。"
        f"当前用户画像：{profile_text}。"
        f"{peer_section}"
        "如需要训练/动作/恢复方面的知识库支持，可主动调用 retrieve_trainer_knowledge。"
        f"{mcp_hint}"
        "对于纯打招呼或与训练无关的内容，请直接回答，无需检索。"
        "若检索结果明确返回 '未命中本地知识库'，可凭通用训练知识给出保守兜底建议。"
        "对动作安全与伤病风险进行约束。"
        "如果用户提供了新的身体信息，请调用 update_user_profile 做结构化更新。"
        "回答尽量给出可执行计划（频次/组数/强度/恢复）。"
        "【个性化要求】回答必须代入画像中的具体数值，给出量化建议而非通用说明。"
        "例如：不写'建议适量有氧'，而写'以你目前体重 Xkg、年龄 Y 岁，建议每周 Z 次有氧'。"
        "若画像中有伤病记录（injuries 不为空），必须在建议开头明确点出该伤病的限制条件，"
        "且不得推荐任何与该伤病部位相关的负重动作，除非明确注明须在理疗师许可和监督下进行。"
    )
    return create_agent(llm, list(_TRAINER_TOOLS) + wger_tools, system_prompt)


def run_trainer(user_id: str, user_question: str, peer_notes_text: str = "") -> dict:
    """Execute the Trainer expert and return a state update dict."""
    try:
        print_expert_start("Trainer", user_question)
        profile_text = profile_to_prompt_text_for(
            "Trainer", get_profile_from_store(user_id)
        )
        agent = _build_trainer_agent(profile_text, peer_notes_text)
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
        print_expert_end("Trainer", used_tools, answer)
        return {
            "expert_responses": {"Trainer": answer},
            "agent_notes": {"Trainer": build_scratchpad_note("Trainer", answer)},
            "last_tools": used_tools,
            "retrieval_hits": retrieval_hits,
        }
    except Exception as e:
        return expert_error_update("Trainer", e)
