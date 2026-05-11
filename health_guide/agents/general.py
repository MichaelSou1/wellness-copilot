from ..tools import get_user_profile, update_user_profile, retrieve_general_knowledge
from ..utils import create_agent
from ..llm import extract_text_content, llm
from ..profile_store import get_user_profile as get_profile_from_store, profile_to_prompt_text
from ._scratchpad import format_peer_notes, build_scratchpad_note

def _build_general_agent(user_id: str, peer_notes_text: str):
    profile_text = profile_to_prompt_text(get_profile_from_store(user_id))
    system_prompt = (
        "你是健康管理团队的贴心助理。"
        f"当前用户画像：{profile_text}。"
        f"{peer_notes_text}"
        "负责寒暄、通用问题与多轮澄清。"
        "若用户询问健康常识并需要给出建议，先调用一次 retrieve_general_knowledge 再回答。"
        "若 retrieve_general_knowledge 返回了知识片段（结果包含'命中以下知识片段'），"
        "必须直接基于这些片段作答，不得仅凭模型内部知识回答。"
        "若 retrieve_general_knowledge 明确返回'未命中本地知识库'，"
        "可以直接基于你的通用健康常识给出简洁、保守的兜底回答。"
        "若用户提供新的个人偏好、作息、目标变化，请调用 update_user_profile 进行记录。"
        "回答保持自然、简洁。"
        "优先短回复，不主动给健康方案；只有用户明确询问时再进入健康建议。"
    )
    return create_agent(llm, [retrieve_general_knowledge, get_user_profile, update_user_profile], system_prompt)

def general_node(state):
    user_id = state.get("profile_user_id", "default_user")
    peer_notes_text = format_peer_notes(state.get("agent_notes") or {}, self_role="General")
    general_agent = _build_general_agent(user_id, peer_notes_text)
    result = general_agent.invoke({"messages": state["messages"]})
    used_tools = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                used_tools.append(call.get("name", "Unknown"))

    retrieval_hits = sum(1 for t in used_tools if "retrieve" in t and "knowledge" in t)
    answer = extract_text_content(result["messages"][-1])
    return {
        "expert_responses": {"General": answer},
        "agent_notes": {"General": build_scratchpad_note("General", answer)},
        "last_tools": used_tools,
        "retrieval_hits": retrieval_hits,
    }
