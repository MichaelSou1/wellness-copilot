import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import AgentState
from .agents.turn_start import turn_start_node
from .agents.query_rewriter import query_rewriter_node
from .agents.orchestrator import orchestrator_node
from .agents.dispatcher import dispatcher_node
from .agents.aggregator import aggregator_node
from .agents.critic import critic_node
from .agents.replan_judge import replan_judge_node

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)


def _route_after_dispatch(state: AgentState):
    """After Dispatcher, decide whether to replan, judge, or finish.

    - `next == ["__REPLAN__"]` → back to Orchestrator
    - executed anything this turn → ReplanJudge (which decides if more is needed)
    - otherwise (FINISH path or empty plan) → END
    """
    nxt = state.get("next") or []
    if nxt and nxt[0] == "__REPLAN__":
        return "Orchestrator"
    executed = [role for role in (state.get("executed") or []) if role != "Orchestrator"]
    if executed:
        return "ReplanJudge"
    return END


def _route_after_orchestrator(state: AgentState):
    """The parent agent either answered directly or emitted a specialist plan."""
    if state.get("plan"):
        return "Dispatcher"
    if state.get("messages") and state.get("orchestrator_decision") == "DIRECT":
        return END
    executed = [role for role in (state.get("executed") or []) if role != "Orchestrator"]
    if executed:
        return "Aggregator"
    return END


def _route_after_judge(state: AgentState):
    """ReplanJudge either sets `replan_request` (→ Dispatcher → Orchestrator)
    or stays silent (→ Aggregator)."""
    if state.get("replan_request"):
        return "Dispatcher"
    return "Aggregator"


def _route_after_critic(state: AgentState):
    """Critic can demand a fresh replan when it detects a missing-expert gap.

    When it sets `replan_context`, route back to Orchestrator (which will produce
    an append-only plan for the new expert). Otherwise the answer is final.
    """
    if state.get("replan_context"):
        return "Orchestrator"
    return END


workflow = StateGraph(AgentState)

workflow.add_node("TurnStart", turn_start_node)
workflow.add_node("QueryRewriter", query_rewriter_node)
workflow.add_node("Orchestrator", orchestrator_node)
workflow.add_node("Dispatcher", dispatcher_node)
workflow.add_node("ReplanJudge", replan_judge_node)
workflow.add_node("Aggregator", aggregator_node)
workflow.add_node("Critic", critic_node)

# 入口先经 TurnStart 做轮边界清理 + 长历史摘要压缩，
# 再经 QueryRewriter 解决多轮指代，最后交给 Orchestrator。
# 注意：replan 路径直接 Dispatcher/Critic → Orchestrator，不会再触发 TurnStart/QueryRewriter，
# 因为同一轮内 contextualized_query 已经稳定，且不能重复清理本轮状态。
workflow.set_entry_point("TurnStart")
workflow.add_edge("TurnStart", "QueryRewriter")
workflow.add_edge("QueryRewriter", "Orchestrator")
workflow.add_conditional_edges(
    "Orchestrator",
    _route_after_orchestrator,
    ["Dispatcher", "Aggregator", END],
)

workflow.add_conditional_edges(
    "Dispatcher",
    _route_after_dispatch,
    ["Orchestrator", "ReplanJudge", END],
)
workflow.add_conditional_edges(
    "ReplanJudge",
    _route_after_judge,
    ["Dispatcher", "Aggregator"],
)

workflow.add_edge("Aggregator", "Critic")
workflow.add_conditional_edges(
    "Critic",
    _route_after_critic,
    ["Orchestrator", END],
)

graph = workflow.compile(checkpointer=memory)
