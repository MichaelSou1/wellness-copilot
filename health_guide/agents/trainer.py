from ..tools import calculate_tdee, retrieve_trainer_knowledge, get_user_profile, update_user_profile
from ..utils import create_agent
from ..llm import extract_text_content, llm
from ..profile_store import get_user_profile as get_profile_from_store, profile_to_prompt_text
from ._scratchpad import format_peer_notes, build_scratchpad_note
from .query_rewriter import get_user_question


def _build_trainer_agent(user_id: str, peer_notes_text: str, rag_context: str):
    profile_text = profile_to_prompt_text(get_profile_from_store(user_id))
    if rag_context:
        rag_section = (
            "【已为你预先检索到相关训练知识，请直接基于以下内容作答，无需再调用 retrieve_trainer_knowledge】\n"
            f"{rag_context}\n"
        )
    else:
        rag_section = "（本次未能预先检索，如需知识库支持请调用 retrieve_trainer_knowledge）"
    system_prompt = (
        "你是力量训练教练。"
        f"当前用户画像：{profile_text}。"
        f"{peer_notes_text}"
        f"{rag_section}"
        "若知识片段明确返回'未命中本地知识库'，可凭通用训练知识给出保守兜底建议。"
        "对动作安全与伤病风险进行约束。"
        "如果用户提供了新的身体信息，请调用 update_user_profile 做结构化更新。"
        "回答尽量给出可执行计划（频次/组数/强度/恢复）。"
        "【个性化要求】回答必须代入画像中的具体数值，给出量化建议而非通用说明。"
        "例如：不写'建议适量有氧'，而写'以你目前体重 Xkg、年龄 Y 岁，建议每周 Z 次有氧'。"
        "若画像中有伤病记录（injuries 不为空），必须在建议开头明确点出该伤病的限制条件，"
        "且不得推荐任何与该伤病部位相关的负重动作，除非明确注明须在理疗师许可和监督下进行。"
    )
    return create_agent(
        llm,
        [calculate_tdee, get_user_profile, update_user_profile],
        system_prompt,
    )


def trainer_node(state):
    user_id = state.get("profile_user_id", "default_user")
    user_question = get_user_question(state)
    rag_context = retrieve_trainer_knowledge.invoke({"query": user_question}) if user_question else ""

    peer_notes_text = format_peer_notes(state.get("agent_notes") or {}, self_role="Trainer")
    trainer_agent = _build_trainer_agent(user_id, peer_notes_text, rag_context)
    result = trainer_agent.invoke({"messages": state["messages"]})
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
        "expert_responses": {"Trainer": answer},
        "agent_notes": {"Trainer": build_scratchpad_note("Trainer", answer)},
        "last_tools": used_tools,
        "retrieval_hits": retrieval_hits,
    }
