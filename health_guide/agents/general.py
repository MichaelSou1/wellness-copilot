"""General assistant expert — invoked as a callable by the Dispatcher."""
import os

from langchain_core.messages import HumanMessage

from ..tools import (
    add_dietary_preference,
    add_injury,
    add_stress_source,
    retrieve_general_knowledge,
    set_dietary_goal,
    set_physical_stats,
    set_response_style,
    update_user_profile,
)
from ..utils import create_agent
from ..llm import extract_text_content, llm
from ..personalization import build_personalization_ctx
from ..detail import print_expert_end, print_expert_start, print_expert_trace
from .fallbacks import expert_error_update
from ._scratchpad import build_scratchpad_note


_GENERAL_TOOLS = [
    retrieve_general_knowledge,
    set_physical_stats,
    add_injury,
    set_dietary_goal,
    add_dietary_preference,
    add_stress_source,
    set_response_style,
    update_user_profile,
]


def _episode_section(episode_context: str) -> str:
    if not episode_context:
        return ""
    return (
        "\n【近期/相关对话记录】\n"
        f"{episode_context}\n"
        "说明：[最近] 表示按时间最近的对话，[相关] 表示因话题相似而召回的更早对话；引用时请区分使用。\n"
    )


def _build_general_agent(pctx: dict, peer_notes_text: str, episode_context: str = ""):
    peer_section = peer_notes_text if peer_notes_text else ""
    user_card = pctx.get("user_card") or "【关于该用户】\n用户画像暂不可用。"
    system_prompt = (
        "你是一个健康管理团队的贴心助理，团队核心定位是 **健康管理**：饮食、运动、睡眠、压力、伤病恢复。"
        f"\n\n{user_card}\n"
        f"{_episode_section(episode_context)}"
        f"{peer_section}"
        "用户卡片就是本轮可用画像；不要说「我先看看/了解你的基本信息」，不要为了读取画像而调用工具。"
        "若已有足够信息，必须直接回应；只有用户提供新信息时才调用结构化工具记录。"
        "你的职责是寒暄、通用健康常识解答与多轮澄清。\n"
        "【能力介绍语调】当用户问'你能做什么/帮我做什么'时，必须**首先**列出健康相关能力："
        "饮食与营养规划、训练与运动建议、睡眠与压力管理、伤病/康复期注意事项、根据画像做个性化方案；"
        "其后再视情况补充一句通用问答能力。不要把编程/通用知识放在能力介绍的前面，也不要喧宾夺主。\n"
        "【寒暄与轻量消息】对你好、谢谢、再见、今天天气真好、你叫什么等轻量消息，简短自然回应即可；"
        "不要主动展开深蹲、热量赤字、HIIT、减脂计划等具体健康方案，也不要检索。\n"
        "【鼓励行动】对于'最近不想动''没动力'这类陈述，可以主动给出一条 **具体、可操作** 的低门槛健康建议（如'今晚下楼快走 30 分钟'），"
        "而不是只共情或反问。语气保持轻松。\n"
        "【医疗边界】你不能诊断疾病、开处方、给处方药剂量或判断'没事'。当用户询问疼痛是什么病、胸痛/胸闷、呼吸困难、"
        "晕厥、明显心率异常、持续疼痛、用药剂量或处方时，必须先建议就医/医生评估；不要列出未经确认的诊断猜测，"
        "也不要使用'可能是 X 病'来替代就医建议。"
        "若用户询问健康常识，可调用 retrieve_general_knowledge 补充知识；"
        "对于纯打招呼/告别，请直接回答，无需检索。"
        "若用户提供新的个人偏好、作息、身体数据、目标变化或回答风格偏好，请优先调用对应的 "
        "set_/add_ 结构化工具记录；update_user_profile 仅作兼容兜底。"
        "当用户明确要求「简洁/详细/幽默/正式/口语/英文/中文」等回答风格时，必须调用 set_response_style；"
        "例如「回答简洁一点」对应 tone=concise。"
        "回答保持自然、简洁。"
    )
    return create_agent(llm, _GENERAL_TOOLS, system_prompt)


def run_general(
    user_id: str,
    user_question: str,
    peer_notes_text: str = "",
    pctx: dict | None = None,
    episode_context: str = "",
) -> dict:
    try:
        os.environ["HEALTH_GUIDE_USER_ID"] = user_id
        print_expert_start("General", user_question)
        agent = _build_general_agent(
            pctx or build_personalization_ctx(user_id),
            peer_notes_text,
            episode_context,
        )
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
