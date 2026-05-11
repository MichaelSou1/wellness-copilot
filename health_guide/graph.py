import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import AgentState
from .agents.query_rewriter import query_rewriter_node
from .agents.planner import planner_node
from .agents.dispatcher import dispatcher_node
from .agents.trainer import trainer_node
from .agents.nutritionist import nutritionist_node
from .agents.wellness import wellness_node
from .agents.general import general_node
from .agents.aggregator import aggregator_node
from .agents.critic import critic_node
from .agents.replan_judge import replan_judge_node

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

_VALID_EXPERTS = {"Trainer", "Nutritionist", "Wellness", "General"}


def _route_after_dispatch(state: AgentState):
    """Pick next destination based on Dispatcher's output.

    Order of checks:
      1. `next == ["__REPLAN__"]` → hand control back to Planner
      2. `next == [<expert>]` → execute that expert next
      3. empty next + at least one expert executed → Aggregator
      4. nothing executed (FINISH path) → END
    """
    nxt = state.get("next") or []
    if nxt and nxt[0] == "__REPLAN__":
        return "Planner"
    if nxt and nxt[0] in _VALID_EXPERTS:
        return nxt[0]
    if state.get("executed"):
        return "Aggregator"
    return END


workflow = StateGraph(AgentState)

workflow.add_node("QueryRewriter", query_rewriter_node)
workflow.add_node("Planner", planner_node)
workflow.add_node("Dispatcher", dispatcher_node)
workflow.add_node("Trainer", trainer_node)
workflow.add_node("Nutritionist", nutritionist_node)
workflow.add_node("Wellness", wellness_node)
workflow.add_node("General", general_node)
workflow.add_node("ReplanJudge", replan_judge_node)
workflow.add_node("Aggregator", aggregator_node)
workflow.add_node("Critic", critic_node)

# 入口先经 QueryRewriter 解决多轮指代，再交给 Planner。
# 注意：replan 路径直接 Dispatcher → Planner，不会再触发 QueryRewriter，
# 因为同一轮内 contextualized_query 已经稳定。
workflow.set_entry_point("QueryRewriter")
workflow.add_edge("QueryRewriter", "Planner")
workflow.add_edge("Planner", "Dispatcher")

workflow.add_conditional_edges(
    "Dispatcher",
    _route_after_dispatch,
    ["Planner", "Trainer", "Nutritionist", "Wellness", "General", "Aggregator", END],
)

# 每个专家执行完先过 ReplanJudge，由判官决定是否设置 replan_request，再到 Dispatcher
for expert in ["Trainer", "Nutritionist", "Wellness", "General"]:
    workflow.add_edge(expert, "ReplanJudge")
workflow.add_edge("ReplanJudge", "Dispatcher")

workflow.add_edge("Aggregator", "Critic")
workflow.add_edge("Critic", END)

graph = workflow.compile(checkpointer=memory)
