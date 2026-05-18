"""Trainer expert — invoked as a callable by the Dispatcher (no longer a graph node).

Receives an isolated, scoped input from the parent agent:
  - SystemMessage: role profile (cropped) + optional peer scratchpad
  - HumanMessage: contextualized user question

RAG is *on-demand*: `retrieve_trainer_knowledge` lives in the tool list so the
expert's ReAct loop decides whether to call it (skip greetings / pure personal
chat, fire for actual training questions).
"""
import os

from langchain_core.messages import HumanMessage

from ..mcp_client import MCP_REGISTRY
from ..tools import (
    add_injury,
    calculate_tdee,
    retrieve_trainer_knowledge,
    set_dietary_goal,
    set_physical_stats,
    update_user_profile,
)
from ..utils import create_agent
from ..llm import extract_text_content, llm
from ..personalization import build_personalization_ctx
from ..detail import print_expert_end, print_expert_start, print_expert_trace
from .fallbacks import expert_error_update
from ._scratchpad import build_scratchpad_note


_TRAINER_TOOLS = [
    calculate_tdee,
    set_physical_stats,
    add_injury,
    set_dietary_goal,
    update_user_profile,
    retrieve_trainer_knowledge,
]


def _episode_section(episode_context: str) -> str:
    if not episode_context:
        return ""
    return (
        "\n【近期/相关对话记录】\n"
        f"{episode_context}\n"
        "说明：[最近] 表示按时间最近的对话，[相关] 表示因话题相似而召回的更早对话；引用时请区分使用。\n"
    )


def _build_trainer_agent(pctx: dict, peer_notes_text: str, episode_context: str = ""):
    peer_section = peer_notes_text if peer_notes_text else ""
    user_card = pctx.get("user_card") or "【关于该用户】\n用户画像暂不可用。"
    wger_tools = MCP_REGISTRY.get_tools("wger")
    mcp_hint = (
        "如需查询具体动作百科（标准动作要领、目标肌群、所需器械），可调用 wger MCP 工具："
        "search_exercises / get_exercise_details / list_muscles / list_equipment / list_categories。"
        if wger_tools
        else ""
    )
    system_prompt = (
        "你是力量训练教练。\n\n"
        f"{user_card}\n"
        f"{_episode_section(episode_context)}"
        f"{peer_section}"
        "用户卡片就是本轮可用画像；不要说「我先看看/了解你的基本信息」，不要为了读取画像而调用工具。"
        "若已有足够信息，必须直接给出方案；只有用户提供新信息时才调用结构化工具记录。"
        "如需要训练/动作/恢复方面的知识库支持，可主动调用 retrieve_trainer_knowledge。"
        f"{mcp_hint}"
        "对于纯打招呼或与训练无关的内容，请直接回答，无需检索。"
        "若检索结果明确返回 '未命中本地知识库'，可凭通用训练知识给出保守兜底建议。"
        "对动作安全与伤病风险进行约束。"
        "如果用户提供了新的身体信息，请优先调用 set_physical_stats / add_injury / set_dietary_goal 做结构化更新；"
        "update_user_profile 仅作兼容兜底。"
        "【工具使用】当用户询问 TDEE、BMR、基础代谢、每日热量消耗或减脂/增肌热量起点时，"
        "若画像已有体重、身高、年龄，必须调用 calculate_tdee；回答中写明这是估算值，并给出活动系数/目标热量调整建议。"
        "若用户没有说明活动水平，不要只给默认久坐值；应说明活动水平未知，并给出久坐/中等活动/高活动量至少 2 个场景的 TDEE 估算或请用户补充活动量。"
        "当用户询问动作技术、训练计划、恢复、平台期或伤病训练时，优先调用 retrieve_trainer_knowledge。"
        "【训练处方格式】回答训练建议时按 FITT 思路落地：频次、时长、动作类型、组数/次数、强度（RPE 或心率区间）、组间休息、"
        "进阶规则和恢复安排。新手从低容量开始（例如每周 2-3 次全身训练），不要直接给高强度或高频计划。"
        "【恢复与平台期】肌肉酸痛/DOMS 先给主动恢复、轻活动、泡沫轴/拉伸、睡眠与补水；平台期必须强调训练日志和渐进超负荷"
        "（加重量/次数/组数或改善动作质量），并给出 1-2 周的具体进阶方式。"
        "【症状红线】若用户提到运动中胸闷/胸痛、心悸、头晕、晕厥、运动后心率十几分钟不降、明显肿胀或剧痛，"
        "必须先建议停止训练并做医学评估/就医检查；在排除风险前不要建议继续中高强度训练。"
        "回答尽量给出可执行计划（频次/组数/强度/恢复）。\n"
        "【输出硬性要求】\n"
        "1. 若用户卡片中有年龄/体重/身高/BMI，回答开头必须自然引用至少 1 个数值，"
        "例如「以你 40 岁、88kg 的当前状态…」。不允许只说「根据你的情况」；若数值缺失，先说明需要补充。\n"
        "2. 训练量必须给出具体数字（频次/组数/时长/强度），不允许只说「适量」「适度」。\n"
        "3. 若用户卡片列出了伤病，回答前两句之内必须点名该伤病并说出限制；"
        "不得推荐冲突动作；替代动作须注明「须在医生或理疗师许可下进行」。\n"
        "4. 至少包含 2 条由画像具体数值（体重/年龄/伤病/目标）推导出的可执行数字。"
    )
    return create_agent(llm, list(_TRAINER_TOOLS) + wger_tools, system_prompt)


def run_trainer(
    user_id: str,
    user_question: str,
    peer_notes_text: str = "",
    pctx: dict | None = None,
    episode_context: str = "",
) -> dict:
    """Execute the Trainer expert and return a state update dict."""
    try:
        os.environ["HEALTH_GUIDE_USER_ID"] = user_id
        print_expert_start("Trainer", user_question)
        agent = _build_trainer_agent(pctx or build_personalization_ctx(user_id), peer_notes_text, episode_context)
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
