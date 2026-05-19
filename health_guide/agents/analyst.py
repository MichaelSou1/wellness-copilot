"""Analyst expert — reads structured local logs and reports trends."""
from __future__ import annotations

import json
import os
import re

from langchain_core.messages import HumanMessage

from ..detail import print_expert_end, print_expert_start, print_expert_trace
from ..llm import extract_text_content, llm
from ..personalization import build_personalization_ctx
from ..tools import get_user_profile, query_logs
from ..utils import create_agent
from ._scratchpad import build_scratchpad_note
from .fallbacks import expert_error_update


_ANALYST_TOOLS = [query_logs, get_user_profile]


def _load_logs(user_id: str, days_back: int = 30) -> dict:
    raw = query_logs.invoke({"kind": "all", "days_back": days_back, "user_id": user_id})
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed.get("data") or {}


def _avg(nums: list[float]) -> float:
    return round(sum(nums) / len(nums), 1) if nums else 0


def _deterministic_analysis(user_id: str, user_question: str) -> str:
    if not re.search(r"最近|这周|本周|上周|进展|趋势|复盘|怎么样|达成|日志", user_question or ""):
        return ""
    data = _load_logs(user_id, days_back=30)
    meals = data.get("meals") or []
    workouts = data.get("workouts") or []
    wellness = data.get("wellness") or []
    if not meals and not workouts and not wellness:
        return "我现在还没有读到你的结构化饮食、训练或恢复日志；先记录一餐、一次训练或一次睡眠/情绪 check-in 后，我就能按真实数据给你做复盘。"

    lines = []
    if meals:
        kcal = [float(row.get("kcal") or 0) for row in meals if float(row.get("kcal") or 0) > 0]
        protein = [float(row.get("protein_g") or 0) for row in meals if float(row.get("protein_g") or 0) > 0]
        lines.append(
            f"饮食日志有 {len(meals)} 条；已记录餐次平均约 {_avg(kcal)} kcal、蛋白 {_avg(protein)}g。"
        )
    if workouts:
        done = sum(1 for row in workouts if str(row.get("status") or "").lower() in {"done", "completed", "finished"})
        planned = sum(1 for row in workouts if str(row.get("status") or "").lower() == "planned")
        lines.append(f"训练日志有 {len(workouts)} 条，其中完成 {done} 条、计划中 {planned} 条。")
    if wellness:
        sleep = [float(row.get("sleep_h") or 0) for row in wellness if float(row.get("sleep_h") or 0) > 0]
        lines.append(f"恢复日志有 {len(wellness)} 条；有睡眠记录的均值约 {_avg(sleep)} 小时。")
    lines.append("这些数字只代表已记录的部分，漏记会让均值偏低；下一步建议先提高记录完整度，再看趋势。")
    return "\n".join(lines)


def _build_analyst_agent(user_question: str = ""):
    system_prompt = (
        "你是 Health Guide 的数据复盘分析师，只基于结构化日志和用户画像做趋势总结，不开训练或医学处方。\n"
        "必须优先调用 query_logs 读取真实日志；如果没有日志，要明确说明数据不足。\n"
        "输出真实数字、均值、完成次数、缺口和不确定性，不要虚构未记录的数据。"
    )
    return create_agent(llm, _ANALYST_TOOLS, system_prompt)


def run_analyst(
    user_id: str,
    user_question: str,
    peer_notes_text: str = "",
    pctx: dict | None = None,
    episode_context: str = "",
) -> dict:
    try:
        os.environ["HEALTH_GUIDE_USER_ID"] = user_id
        print_expert_start("Analyst", user_question)
        deterministic = _deterministic_analysis(user_id, user_question)
        if deterministic:
            print_expert_end("Analyst", [], deterministic)
            return {
                "expert_responses": {"Analyst": deterministic},
                "agent_notes": {"Analyst": build_scratchpad_note("Analyst", deterministic)},
                "last_tools": ["query_logs"],
                "retrieval_hits": 0,
            }
        _ = pctx or build_personalization_ctx(user_id)
        agent = _build_analyst_agent(user_question)
        result = agent.invoke({"messages": [HumanMessage(content=user_question)]})
        print_expert_trace("Analyst", result["messages"])
        used_tools = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                used_tools.extend(call.get("name", "Unknown") for call in msg.tool_calls)
        answer = extract_text_content(result["messages"][-1])
        print_expert_end("Analyst", used_tools, answer)
        return {
            "expert_responses": {"Analyst": answer},
            "agent_notes": {"Analyst": build_scratchpad_note("Analyst", answer)},
            "last_tools": used_tools,
            "retrieval_hits": 0,
        }
    except Exception as e:
        return expert_error_update("Analyst", e)
