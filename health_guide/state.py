from typing import TypedDict, Annotated, List, Dict
import operator
from langchain_core.messages import AnyMessage


def _merge_dict(a: Dict[str, str], b: Dict[str, str]) -> Dict[str, str]:
    return {**a, **b}


def _take_last_str(a: str, b: str) -> str:
    # Always take the latest write, including explicit empty-string clears.
    # (Old semantics of "keep old when b is falsy" prevented clearing
    # replan_request/replan_context after consumption.)
    return b


class AgentState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], operator.add]
    # 本轮路由的专家列表（支持多专家并行）
    next: List[str]
    profile_user_id: str
    # 并行执行时各专家写入，用 operator.add 合并
    last_tools: Annotated[List[str], operator.add]
    retrieval_hits: Annotated[int, operator.add]
    # 各专家本轮回答，key=专家名，value=回答文本
    expert_responses: Annotated[Dict[str, str], _merge_dict]
    # 共享 scratchpad：各专家给协作伙伴的精简要点（跨轮持久化）
    agent_notes: Annotated[Dict[str, str], _merge_dict]
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
