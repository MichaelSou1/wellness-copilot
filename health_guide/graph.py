import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import AgentState
from .agents.turn_start import turn_start_node
from .agents.query_rewriter import query_rewriter_node
from .agents.planner import planner_node
from .agents.dispatcher import dispatcher_node
from .agents.aggregator import aggregator_node
from .agents.critic import critic_node
from .agents.replan_judge import replan_judge_node

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)


def _route_after_dispatch(state: AgentState):
    """After Dispatcher, decide whether to replan, judge, or finish.

    - `next == ["__REPLAN__"]` → back to Planner
    - executed anything this turn → ReplanJudge (which decides if more is needed)
    - otherwise (FINISH path or empty plan) → END
    """
    nxt = state.get("next") or []
    if nxt and nxt[0] == "__REPLAN__":
        return "Planner"
    if state.get("executed"):
        return "ReplanJudge"
    return END


def _route_after_judge(state: AgentState):
    """ReplanJudge either sets `replan_request` (→ Dispatcher → Planner)
    or stays silent (→ Aggregator)."""
    if state.get("replan_request"):
        return "Dispatcher"
    return "Aggregator"


workflow = StateGraph(AgentState)

workflow.add_node("TurnStart", turn_start_node)
workflow.add_node("QueryRewriter", query_rewriter_node)
workflow.add_node("Planner", planner_node)
workflow.add_node("Dispatcher", dispatcher_node)
workflow.add_node("ReplanJudge", replan_judge_node)
workflow.add_node("Aggregator", aggregator_node)
workflow.add_node("Critic", critic_node)

# 入口先经 TurnStart 做轮边界清理 + 长历史摘要压缩，
# 再经 QueryRewriter 解决多轮指代，最后交给 Planner。
# 注意：replan 路径直接 Dispatcher → Planner，不会再触发 TurnStart/QueryRewriter，
# 因为同一轮内 contextualized_query 已经稳定，且不能重复清理本轮状态。
workflow.set_entry_point("TurnStart")
workflow.add_edge("TurnStart", "QueryRewriter")
workflow.add_edge("QueryRewriter", "Planner")
workflow.add_edge("Planner", "Dispatcher")

workflow.add_conditional_edges(
    "Dispatcher",
    _route_after_dispatch,
    ["Planner", "ReplanJudge", END],
)
workflow.add_conditional_edges(
    "ReplanJudge",
    _route_after_judge,
    ["Dispatcher", "Aggregator"],
)

workflow.add_edge("Aggregator", "Critic")
workflow.add_edge("Critic", END)

graph = workflow.compile(checkpointer=memory)
