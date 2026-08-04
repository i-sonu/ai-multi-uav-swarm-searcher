"""Planner lookup by name, so the planner is selectable from config.

All planners share the signature
    planner(grid, start, goal, treat_unknown_as_free=True) -> PlanResult
"""

from __future__ import annotations

from src.planning.astar import astar
from src.planning.baselines import bfs, dfs, ucs

PLANNERS = {
    "astar": astar,
    "bfs": bfs,
    "dfs": dfs,
    "ucs": ucs,
}


def get_planner(name: str):
    """Return the planner function registered under ``name``."""
    try:
        return PLANNERS[name]
    except KeyError:
        raise ValueError(f"Unknown planner {name!r}; choose from {sorted(PLANNERS)}")
