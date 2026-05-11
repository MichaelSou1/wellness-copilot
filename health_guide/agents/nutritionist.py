from ..tools import retrieve_nutritionist_knowledge, get_user_profile, update_user_profile
from ..utils import create_agent
from ..llm import extract_text_content, llm
from ..profile_store import get_user_profile as get_profile_from_store, profile_to_prompt_text
from ._scratchpad import format_peer_notes, build_scratchpad_note

def _build_nutritionist_agent(user_id: str, peer_notes_text: str):
    profile_text = profile_to_prompt_text(get_profile_from_store(user_id))
    system_prompt = (
        "你是膳食营养师。"
        f"当前用户画像：{profile_text}。"
        f"{peer_notes_text}"
        "只要用户在问饮食/营养建议，必须先调用一次 retrieve_nutritionist_knowledge 再回答。"
        "若 retrieve_nutritionist_knowledge 返回了知识片段（结果包含'命中以下知识片段'），"
        "必须直接基于这些片段作答，不得仅凭模型内部知识回答。"
        "仅当 retrieve_nutritionist_knowledge 明确返回'未命中本地知识库'时，"
        "才可不再依赖知识库片段，直接用你的通用营养知识给出保守、清晰的兜底建议。"
        "如果用户补充了口味偏好/禁忌/目标变化，请调用 update_user_profile 更新画像。"
        "输出请给出清晰饮食方案（热量、三大营养素、可替代食材）。"
    )
    return create_agent(
        llm,
        [retrieve_nutritionist_knowledge, get_user_profile, update_user_profile],
        system_prompt,
    )

def nutritionist_node(state):
    user_id = state.get("profile_user_id", "default_user")
    peer_notes_text = format_peer_notes(state.get("agent_notes") or {}, self_role="Nutritionist")
    nutritionist_agent = _build_nutritionist_agent(user_id, peer_notes_text)
    result = nutritionist_agent.invoke({"messages": state["messages"]})
    used_tools = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                used_tools.append(call.get("name", "Unknown"))

    retrieval_hits = sum(1 for t in used_tools if "retrieve" in t and "knowledge" in t)
    answer = extract_text_content(result["messages"][-1])
    return {
        "expert_responses": {"Nutritionist": answer},
        "agent_notes": {"Nutritionist": build_scratchpad_note("Nutritionist", answer)},
        "last_tools": used_tools,
        "retrieval_hits": retrieval_hits,
    }
