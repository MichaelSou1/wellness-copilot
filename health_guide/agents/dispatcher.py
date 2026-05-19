"""Dispatcher — invokes specialist child agents as callables.

Architecture:
1. Receives the specialist plan from Orchestrator.
2. If any expert raised a replan request and the cap isn't hit, hands control
   back to Orchestrator with the reason.
3. Otherwise consumes the whole plan in one shot:
   - Single expert  → call inline.
   - Multiple experts → run them in parallel via ThreadPoolExecutor.
4. Each expert gets an *isolated* input (its own SystemMessage with a cropped
   profile + a HumanMessage with the contextualized question) — it never sees
   the parent's full message history or peer experts' tool traces.
5. Empties `plan` and bubbles outputs up via state reducers.

After the dispatcher returns, the graph routes to ReplanJudge (which decides
whether to re-enter Orchestrator) or directly to Aggregator. Per-expert
ReplanJudge calls are gone — judgment runs exactly once after the whole plan
finishes.
"""
from __future__ import annotations

import concurrent.futures
from typing import Callable, Dict, List

from ._scratchpad import format_peer_notes
from .analyst import run_analyst
from .doctor import run_doctor
from .fallbacks import expert_error_update
from .nutritionist import run_nutritionist
from .query_rewriter import get_user_question
from .trainer import run_trainer
from .psychologist import run_psychologist


REPLAN_CAP = 2

EXPERT_RUNNERS: Dict[str, Callable[..., dict]] = {
    "Analyst": run_analyst,
    "Trainer": run_trainer,
    "Nutritionist": run_nutritionist,
    "Psychologist": run_psychologist,
    "Doctor": run_doctor,
}


def _merge_result(acc: dict, result: dict) -> None:
    if not result:
        return
    acc["expert_responses"].update(result.get("expert_responses") or {})
    acc["agent_notes"].update(result.get("agent_notes") or {})
    acc["last_tools"].extend(result.get("last_tools") or [])
    acc.setdefault("actuation_log", []).extend(result.get("actuation_log") or [])
    acc["retrieval_hits"] += int(result.get("retrieval_hits") or 0)


def _run_plan(
    plan: List[str],
    user_id: str,
    user_question: str,
    prior_notes: dict,
    pctx: dict,
    episode_context: str,
) -> dict:
    """Execute experts in the plan and return merged state update."""
    runners = [(role, EXPERT_RUNNERS[role]) for role in plan if role in EXPERT_RUNNERS]
    acc = {
        "expert_responses": {},
        "agent_notes": {},
        "last_tools": [],
        "actuation_log": [],
        "retrieval_hits": 0,
    }
    if not runners:
        return acc

    def _safe_call(role: str, fn, peer_text: str) -> dict:
        try:
            return fn(user_id, user_question, peer_text, pctx, episode_context)
        except Exception as e:
            # run_* already wraps; this is a belt-and-braces guard so a
            # rogue runner can never abort the dispatch batch.
            return expert_error_update(role, e)

    # In a single-expert plan, threadpool overhead isn't worth it.
    if len(runners) == 1:
        role, fn = runners[0]
        peer_text = format_peer_notes(prior_notes, self_role=role)
        _merge_result(acc, _safe_call(role, fn, peer_text))
        return acc

    # Parallel fan-out. Experts in the *same plan batch* don't see each
    # other's scratchpad — Aggregator does the cross-domain integration.
    # Notes from *prior* batches (e.g., replan rounds) are still injected.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(runners)) as pool:
        futures = [
            pool.submit(_safe_call, role, fn, format_peer_notes(prior_notes, self_role=role))
            for role, fn in runners
        ]
        for fut in concurrent.futures.as_completed(futures):
            _merge_result(acc, fut.result())
    return acc


def dispatcher_node(state):
    replan_req = state.get("replan_request", "") or ""
    replan_count = int(state.get("replan_count", 0) or 0)

    # ---- Replan path: hand control back to Orchestrator ----
    if replan_req and replan_count < REPLAN_CAP:
        return {
            "replan_request": "",
            "replan_context": replan_req,
            "replan_count": replan_count + 1,
            "next": ["__REPLAN__"],
        }
    # Replan request after the cap is silently dropped.
    extra: Dict[str, str] = {}
    if replan_req:
        extra["replan_request"] = ""

    plan = list(state.get("plan", []) or [])
    if not plan:
        # Nothing to do — graph router will fall through to ReplanJudge/END
        # based on whether anything was executed earlier this turn.
        return {"next": [], **extra}

    user_id = state.get("profile_user_id", "default_user")
    user_question = get_user_question(state)
    prior_notes = dict(state.get("agent_notes") or {})
    pctx = dict(state.get("personalization_ctx") or {})
    episode_context = state.get("episode_context") or ""

    batch = _run_plan(plan, user_id, user_question, prior_notes, pctx, episode_context)

    executed = [role for role in (state.get("executed", []) or []) if role != "Orchestrator"] + [
        role for role in plan if role in EXPERT_RUNNERS
    ]
    return {
        "expert_responses": batch["expert_responses"],
        "agent_notes": batch["agent_notes"],
        "last_tools": batch["last_tools"],
        "actuation_log": batch.get("actuation_log") or [],
        "retrieval_hits": batch["retrieval_hits"],
        "executed": executed,
        "plan": [],
        "next": [],
        **extra,
    }
