from ..tools import retrieve_nutritionist_knowledge, get_user_profile, update_user_profile
from ..utils import create_agent
from ..llm import extract_text_content, llm
from ..profile_store import get_user_profile as get_profile_from_store, profile_to_prompt_text
from ._scratchpad import format_peer_notes, build_scratchpad_note
from .query_rewriter import get_user_question


def _build_nutritionist_agent(user_id: str, peer_notes_text: str, rag_context: str):
    profile_text = profile_to_prompt_text(get_profile_from_store(user_id))
    if rag_context:
        rag_section = (
            "【已为你预先检索到相关营养知识，请直接基于以下内容作答，无需再调用 retrieve_nutritionist_knowledge】\n"
            f"{rag_context}\n"
        )
    else:
        rag_section = "（本次未能预先检索，如需知识库支持请调用 retrieve_nutritionist_knowledge）"
    system_prompt = (
        "你是膳食营养师。"
        f"当前用户画像：{profile_text}。"
        f"{peer_notes_text}"
        f"{rag_section}"
        "若知识片段明确返回'未命中本地知识库'，可用通用营养知识给出保守、清晰的兜底建议。"
        "如果用户补充了口味偏好/禁忌/目标变化，请调用 update_user_profile 更新画像。"
        "输出请给出清晰饮食方案（热量、三大营养素、可替代食材）。"
        "【个性化要求】回答必须代入画像中的具体数值，给出量化建议而非通用说明。"
        "例如：不写'建议摄入足够蛋白质'，而写'以你体重 Xkg，建议每天摄入 Y–Z 克蛋白质'。"
        "若画像中 dietary_context.preferences 有过敏或禁忌食物，任何推荐方案中都不得包含该食物，"
        "并在回答开头明确注明该禁忌。"
        "若画像中 dietary_context.goal 为减脂，热量建议不得低于女性 1200 kcal/d、男性 1500 kcal/d。"
    )
    return create_agent(
        llm,
        [get_user_profile, update_user_profile],
        system_prompt,
    )


def nutritionist_node(state):
    user_id = state.get("profile_user_id", "default_user")
    user_question = get_user_question(state)
    rag_context = retrieve_nutritionist_knowledge.invoke({"query": user_question}) if user_question else ""

    peer_notes_text = format_peer_notes(state.get("agent_notes") or {}, self_role="Nutritionist")
    nutritionist_agent = _build_nutritionist_agent(user_id, peer_notes_text, rag_context)
    result = nutritionist_agent.invoke({"messages": state["messages"]})
    used_tools = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                used_tools.append(call.get("name", "Unknown"))

    retrieval_hits = (1 if rag_context else 0) + sum(
        1 for t in used_tools if "retrieve" in t and "knowledge" in t
    )
    answer = extract_text_content(result["messages"][-1])
    return {
        "expert_responses": {"Nutritionist": answer},
        "agent_notes": {"Nutritionist": build_scratchpad_note("Nutritionist", answer)},
        "last_tools": used_tools,
        "retrieval_hits": retrieval_hits,
    }
