"""Dispatcher — sequential plan executor with dynamic replan support.

Priority:
1. If any expert has requested replan (`replan_request` non-empty) and we
   haven't hit the replan cap, hand control back to Planner with the reason.
2. Otherwise, pop the head of `plan` and route to that expert.
3. If `plan` is empty, downstream routing falls through to Aggregator (or END
   if nothing was executed).
"""

REPLAN_CAP = 2


def dispatcher_node(state):
    replan_req = state.get("replan_request", "") or ""
    replan_count = int(state.get("replan_count", 0) or 0)

    update = {}

    if replan_req:
        # Always clear the request so we don't loop on the same signal.
        update["replan_request"] = ""
        if replan_count < REPLAN_CAP:
            update["replan_context"] = replan_req
            update["replan_count"] = replan_count + 1
            update["next"] = ["__REPLAN__"]
            return update
        # cap hit: drop the request and fall through to normal plan consumption.

    plan = list(state.get("plan", []) or [])
    if not plan:
        update["next"] = []
        return update

    head = plan[0]
    rest = plan[1:]
    executed = list(state.get("executed", []) or []) + [head]
    update["plan"] = rest
    update["executed"] = executed
    update["next"] = [head]
    return update
