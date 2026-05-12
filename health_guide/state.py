from typing import TypedDict, Annotated, List, Dict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


RESET_SENTINEL = "__RESET__"


def _turn_dict(a: Dict[str, str], b: Dict[str, str]) -> Dict[str, str]:
    """Merge dict, with a reset sentinel.

    When the incoming dict contains key ``__RESET__`` (any value), the
    accumulated state is dropped and replaced by the remaining keys of the
    incoming dict. Used by TurnStart to clear stale per-turn entries.
    """
    if not isinstance(b, dict):
        return a or {}
    if RESET_SENTINEL in b:
        return {k: v for k, v in b.items() if k != RESET_SENTINEL}
    return {**(a or {}), **b}


def _turn_list(a: List[str], b: List[str]) -> List[str]:
    """Append list, with a reset sentinel as the first element."""
    if isinstance(b, list) and b and b[0] == RESET_SENTINEL:
        return list(b[1:])
    return (a or []) + (b or [])


def _turn_int(a: int, b) -> int:
    """Add ints, except a tuple ``(__RESET__, value)`` resets to ``value``."""
    if isinstance(b, tuple) and b and b[0] == RESET_SENTINEL:
        return int(b[1]) if len(b) > 1 else 0
    return int(a or 0) + int(b or 0)


def _take_last_str(a: str, b: str) -> str:
    return b


class AgentState(TypedDict, total=False):
    # add_messages dedupes by id and honors RemoveMessage — required by TurnStart
    # when it summarizes / drops old messages from long sessions.
    messages: Annotated[List[AnyMessage], add_messages]
    # 本轮路由的专家列表（支持多专家并行）
    next: List[str]
    profile_user_id: str
    # 并行执行时各专家写入；reset 触发于 TurnStart
    last_tools: Annotated[List[str], _turn_list]
    retrieval_hits: Annotated[int, _turn_int]
    # 各专家本轮回答，key=专家名，value=回答文本
    expert_responses: Annotated[Dict[str, str], _turn_dict]
    # 共享 scratchpad：turn-scoped（TurnStart 每轮清空，避免上轮残留污染 Critic）
    agent_notes: Annotated[Dict[str, str], _turn_dict]
    # Aggregator 产出的草稿；Critic 审核后才落到 messages
    draft_answer: Annotated[str, _take_last_str]
    # Critic 审核记录（用于可观测性）
    critic_verdict: Annotated[str, _take_last_str]
    # plan-and-execute：Planner 给出的待执行专家队列（按顺序消费）
    plan: List[str]
    # 已经执行过的专家（顺序保留，Aggregator/Critic 读这个而非 next）
    executed: List[str]
    # 动态 replan：专家通过 [REPLAN_REQUEST: <理由>] 标记请求重新规划
    replan_request: Annotated[str, _take_last_str]
    # Dispatcher 把 replan 理由传给 Planner 的载体
    replan_context: Annotated[str, _take_last_str]
    # 同一轮内的 replan 次数，避免无限循环
    replan_count: int
    # QueryRewriter 改写后的独立问题；解决多轮指代消解
    # （Planner/Judge/Aggregator/Critic 优先用这个而非最新 HumanMessage）
    contextualized_query: Annotated[str, _take_last_str]
    # 长历史压缩后的摘要（TurnStart 写入，作为 SystemMessage 注入 messages）
    history_summary: Annotated[str, _take_last_str]
    # 跨 thread 情节记忆（TurnStart 从 episode_store 读取，供 Planner 路由参考）
    episode_context: Annotated[str, _take_last_str]
