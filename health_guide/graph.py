import sqlite3
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .config import SQLITE_DB_PATH
from .state import AgentState
from .agents.turn_start import turn_start_node
from .agents.query_rewriter import query_rewriter_node
from .agents.multimodal_preprocessor import multimodal_preprocessor_node
from .agents.orchestrator import orchestrator_node
from .agents.aggregator import aggregator_node
from .agents.critic import critic_node

_checkpoint_path = Path(SQLITE_DB_PATH)
if _checkpoint_path.parent and str(_checkpoint_path.parent) not in {"", "."}:
    _checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(str(_checkpoint_path), check_same_thread=False)
memory = SqliteSaver(conn)


def _route_after_orchestrator(state: AgentState):
    """The parent agent either answered directly or called child agents."""
    if state.get("messages") and state.get("orchestrator_decision") == "DIRECT":
        return END
    if state.get("draft_answer"):
        return "Critic"
    executed = [role for role in (state.get("executed") or []) if role != "Orchestrator"]
    if executed:
        return "Aggregator"
    return END


def _route_after_critic(state: AgentState):
    """Critic can demand a fresh replan when it detects a missing-expert gap.

    When it sets `replan_context`, route back to Orchestrator so the parent can
    call one more child agent. Otherwise the answer is final.
    """
    if state.get("replan_context"):
        return "Orchestrator"
    return END


workflow = StateGraph(AgentState)

workflow.add_node("TurnStart", turn_start_node)
workflow.add_node("QueryRewriter", query_rewriter_node)
workflow.add_node("MultiModalPreprocessor", multimodal_preprocessor_node)
workflow.add_node("Orchestrator", orchestrator_node)
workflow.add_node("Aggregator", aggregator_node)
workflow.add_node("Critic", critic_node)

# 入口先经 TurnStart 做轮边界清理 + 长历史摘要压缩，
# 再经 QueryRewriter 解决多轮指代，最后交给 Orchestrator。
# 注意：replan 路径直接 Critic → Orchestrator，不会再触发 TurnStart/QueryRewriter，
# 因为同一轮内 contextualized_query 已经稳定，且不能重复清理本轮状态。
workflow.set_entry_point("TurnStart")
workflow.add_edge("TurnStart", "QueryRewriter")
workflow.add_edge("QueryRewriter", "MultiModalPreprocessor")
workflow.add_edge("MultiModalPreprocessor", "Orchestrator")
workflow.add_conditional_edges(
    "Orchestrator",
    _route_after_orchestrator,
    ["Aggregator", "Critic", END],
)

workflow.add_edge("Aggregator", "Critic")
workflow.add_conditional_edges(
    "Critic",
    _route_after_critic,
    ["Orchestrator", END],
)

graph = workflow.compile(checkpointer=memory)
