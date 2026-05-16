"""Nutritionist expert — invoked as a callable by the Dispatcher."""
from langchain_core.messages import HumanMessage

from ..mcp_client import MCP_REGISTRY
from ..tools import (
    get_user_profile,
    retrieve_nutritionist_knowledge,
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


_NUTRITIONIST_TOOLS = [
    get_user_profile,
    update_user_profile,
    retrieve_nutritionist_knowledge,
]


def _build_nutritionist_agent(profile_text: str, peer_notes_text: str):
    peer_section = peer_notes_text if peer_notes_text else ""
    usda_tools = MCP_REGISTRY.get_tools("usda")
    mcp_hint = (
        "如需精确食物宏量素（蛋白/碳水/脂肪/热量/纤维），可调用 USDA FoodData Central MCP 的 "
        "search-foods 工具（query=食物英文名），返回的 foodNutrients 直接给出每 100g 各项营养素含量。"
        if usda_tools
        else ""
    )
    system_prompt = (
        "你是膳食营养师。"
        f"当前用户画像：{profile_text}。"
        f"{peer_section}"
        "如需要营养/食材/热量计算等知识库支持，可主动调用 retrieve_nutritionist_knowledge。"
        f"{mcp_hint}"
        "对于纯打招呼或与饮食无关的内容，请直接回答，无需检索。"
        "若检索结果明确返回 '未命中本地知识库'，可凭通用营养知识给出保守兜底建议。"
        "如果用户补充了口味偏好/禁忌/目标变化，请调用 update_user_profile 更新画像。"
        "输出请给出清晰饮食方案（热量、三大营养素、可替代食材）。"
        "【个性化要求】回答必须代入画像中的具体数值，给出量化建议而非通用说明。"
        "例如：不写'建议摄入足够蛋白质'，而写'以你体重 Xkg，建议每天摄入 Y–Z 克蛋白质'。"
        "若画像中 dietary_context.preferences 有过敏或禁忌食物，任何推荐方案中都不得包含该食物，"
        "并在回答开头明确注明该禁忌。"
        "若画像中 dietary_context.goal 为减脂，热量建议不得低于女性 1200 kcal/d、男性 1500 kcal/d。"
    )
    return create_agent(llm, list(_NUTRITIONIST_TOOLS) + usda_tools, system_prompt)


def run_nutritionist(user_id: str, user_question: str, peer_notes_text: str = "") -> dict:
    try:
        print_expert_start("Nutritionist", user_question)
        profile_text = profile_to_prompt_text_for(
            "Nutritionist", get_profile_from_store(user_id)
        )
        agent = _build_nutritionist_agent(profile_text, peer_notes_text)
        result = agent.invoke({"messages": [HumanMessage(content=user_question)]})

        print_expert_trace("Nutritionist", result["messages"])

        used_tools: list[str] = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    used_tools.append(call.get("name", "Unknown"))

        retrieval_hits = sum(
            1 for t in used_tools if "retrieve" in t and "knowledge" in t
        )
        answer = extract_text_content(result["messages"][-1])
        print_expert_end("Nutritionist", used_tools, answer)
        return {
            "expert_responses": {"Nutritionist": answer},
            "agent_notes": {"Nutritionist": build_scratchpad_note("Nutritionist", answer)},
            "last_tools": used_tools,
            "retrieval_hits": retrieval_hits,
        }
    except Exception as e:
        return expert_error_update("Nutritionist", e)
