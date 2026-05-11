from ..tools import calculate_tdee, retrieve_trainer_knowledge, get_user_profile, update_user_profile
from ..utils import create_agent
from ..llm import extract_text_content, llm
from ..profile_store import get_user_profile as get_profile_from_store, profile_to_prompt_text
from ._scratchpad import format_peer_notes, build_scratchpad_note

def _build_trainer_agent(user_id: str, peer_notes_text: str):
    profile_text = profile_to_prompt_text(get_profile_from_store(user_id))
    system_prompt = (
        "你是力量训练教练。"
        f"当前用户画像：{profile_text}。"
        f"{peer_notes_text}"
        "先读取画像，再按需调用工具。"
        "只要用户在问训练/恢复/伤痛相关建议，就必须先调用一次 retrieve_trainer_knowledge 再回答。"
        "若 retrieve_trainer_knowledge 返回了知识片段（结果包含'命中以下知识片段'），"
        "必须直接基于这些片段作答，不得仅凭模型内部知识回答。"
        "若 retrieve_trainer_knowledge 明确返回'未命中本地知识库'，"
        "可以直接基于你的通用训练知识给出更保守的兜底建议。"
        "对动作安全与伤病风险进行约束；涉及训练知识时优先调用 retrieve_trainer_knowledge。"
        "如果用户提供了新的身体信息，请调用 update_user_profile 做结构化更新。"
        "回答尽量给出可执行计划（频次/组数/强度/恢复）。"
    )
    return create_agent(
        llm,
        [calculate_tdee, retrieve_trainer_knowledge, get_user_profile, update_user_profile],
        system_prompt,
    )

def trainer_node(state):
    user_id = state.get("profile_user_id", "default_user")
    peer_notes_text = format_peer_notes(state.get("agent_notes") or {}, self_role="Trainer")
    trainer_agent = _build_trainer_agent(user_id, peer_notes_text)
    result = trainer_agent.invoke({"messages": state["messages"]})
    used_tools = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                used_tools.append(call.get("name", "Unknown"))

    retrieval_hits = sum(1 for t in used_tools if "retrieve" in t and "knowledge" in t)
    answer = extract_text_content(result["messages"][-1])
    return {
        "expert_responses": {"Trainer": answer},
        "agent_notes": {"Trainer": build_scratchpad_note("Trainer", answer)},
        "last_tools": used_tools,
        "retrieval_hits": retrieval_hits,
    }
