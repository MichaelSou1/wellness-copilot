"""Backward-compatible alias for the Orchestrator node.

The old Planner role has been promoted into the parent Orchestrator: it now
answers ordinary turns directly and delegates only specialist work to child
agents. Imports of ``planner_node`` are kept so older smoke scripts and docs
continue to run while they are updated.
"""

from .orchestrator import orchestrator_node, planner_node

__all__ = ["orchestrator_node", "planner_node"]
