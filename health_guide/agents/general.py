from ..tools import get_user_profile, update_user_profile, retrieve_general_knowledge
from ..utils import create_agent
from ..llm import extract_text_content, llm
from ..profile_store import get_user_profile as get_profile_from_store, profile_to_prompt_text
from ._scratchpad import format_peer_notes, build_scratchpad_note
from .query_rewriter import get_user_question


def _build_general_agent(user_id: str, peer_notes_text: str, rag_context: str):
    profile_text = profile_to_prompt_text(get_profile_from_store(user_id))
    if rag_context and "未命中" not in rag_context:
        rag_section = (
            "【已为你预先检索到相关知识，如需引用请直接基于以下内容，无需再调用 retrieve_general_knowledge】\n"
            f"{rag_context}\n"
        )
    else:
        rag_section = ""
    system_prompt = (
        "你是健康管理团队的贴心助理。"
        f"当前用户画像：{profile_text}。"
        f"{peer_notes_text}"
        f"{rag_section}"
        "负责寒暄、通用问题与多轮澄清。"
        "若 rag_section 为空且用户询问健康常识，可调用 retrieve_general_knowledge 补充知识。"
        "若用户提供新的个人偏好、作息、目标变化，请调用 update_user_profile 进行记录。"
        "回答保持自然、简洁。"
        "优先短回复，不主动给健康方案；只有用户明确询问时再进入健康建议。"
    )
    return create_agent(llm, [retrieve_general_knowledge, get_user_profile, update_user_profile], system_prompt)


def general_node(state):
    user_id = state.get("profile_user_id", "default_user")
    user_question = get_user_question(state)
    rag_context = retrieve_general_knowledge.invoke({"query": user_question}) if user_question else ""

    peer_notes_text = format_peer_notes(state.get("agent_notes") or {}, self_role="General")
    general_agent = _build_general_agent(user_id, peer_notes_text, rag_context)
    result = general_agent.invoke({"messages": state["messages"]})
    used_tools = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                used_tools.append(call.get("name", "Unknown"))

    rag_hit = rag_context and "未命中" not in rag_context
    retrieval_hits = (1 if rag_hit else 0) + sum(
        1 for t in used_tools if "retrieve" in t and "knowledge" in t
    )
    answer = extract_text_content(result["messages"][-1])
    return {
        "expert_responses": {"General": answer},
        "agent_notes": {"General": build_scratchpad_note("General", answer)},
        "last_tools": used_tools,
        "retrieval_hits": retrieval_hits,
    }
