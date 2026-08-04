"""Single-agent autonomous exploration loop (CLAUDE.md Phase 2.5).

The loop, each round:
    1. sense from the agent into the shared occupancy grid,
    2. detect frontiers and cluster them into candidate goals,
    3. pick the nearest cluster (by straight-line distance),
    4. plan a path to it with the configured planner,
    5. move the agent along that path, sensing at every step,
    6. repeat until no frontiers remain (map explored) or a step limit is hit.

Unreachable frontiers are blacklisted after ``blacklist_after_failures`` failed
plans so the agent does not loop forever chasing a goal it cannot plan to.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.constants import GT_FREE, OCCUPIED
from src.frontier.clustering import cluster_frontiers
from src.frontier.detection import find_frontiers
from src.planning.common import neighbors


@dataclass
class ExplorationResult:
    done: bool                 # True if terminated because no frontiers remained
    steps: int                 # number of simulation steps taken
    reached_goals: int         # how many frontier goals the agent drove to
    blacklisted: int           # frontier goals abandoned as unreachable
    reason: str                # "explored" | "step_limit" | "stuck"


def reachable_free_mask(ground_truth: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Boolean mask of ground-truth free cells reachable from ``start``.

    Flood fill using the *same* 8-connected, no-corner-cutting movement model as
    the planner, so "reachable" means "the agent could actually get there". This
    defines the coverage denominator (CLAUDE.md §5 excludes unreachable cells).
    """
    mask = np.zeros_like(ground_truth, dtype=bool)
    if ground_truth[start] != GT_FREE:
        return mask
    mask[start] = True
    stack = [start]
    while stack:
        r, c = stack.pop()
        # treat_unknown_as_free=True on a 0/1 GT map means "free cells only".
        for nr, nc, _step in neighbors(ground_truth, r, c, treat_unknown_as_free=True):
            if not mask[nr, nc]:
                mask[nr, nc] = True
                stack.append((nr, nc))
    return mask


def count_reachable_free(ground_truth: np.ndarray, start: tuple[int, int]) -> int:
    """Number of ground-truth free cells reachable from ``start``."""
    return int(reachable_free_mask(ground_truth, start).sum())


def explore(
    ground_truth: np.ndarray,
    grid,
    agent,
    planner: Callable,
    *,
    n_beams: int = 360,
    max_range: float = 12.0,
    min_cluster_size: int = 3,
    max_steps: int = 5000,
    blacklist_after_failures: int = 3,
    treat_unknown_as_free: bool = True,
    on_step: Callable[[int], None] | None = None,
) -> ExplorationResult:
    """Run autonomous exploration for a single agent. Mutates ``grid``/``agent``."""

    def _sense() -> None:
        obs = agent.sense(ground_truth, n_beams=n_beams, max_range=max_range)
        grid.apply_observations(obs, agent_id=agent.id)

    _sense()  # reveal the surroundings of the start pose

    step = 0
    reached = 0
    failures: dict[tuple[int, int], int] = defaultdict(int)
    blacklist: set[tuple[int, int]] = set()

    while step < max_steps:
        mask = find_frontiers(grid.grid)
        clusters = cluster_frontiers(mask, min_cluster_size)
        goals = [c.representative for c in clusters if c.representative not in blacklist]

        # Fallback for thin-corridor maps (e.g. mazes): width-1 corridors produce
        # 1-2 cell frontier clusters that the min_cluster_size noise filter would
        # discard, prematurely ending exploration. If size-filtering removed every
        # goal but raw frontiers still exist, retry without the filter so we never
        # dead-end while reachable unknown space remains.
        if not goals:
            clusters = cluster_frontiers(mask, min_cluster_size=1)
            goals = [c.representative for c in clusters if c.representative not in blacklist]

        if not goals:
            return ExplorationResult(True, step, reached, len(blacklist), "explored")

        start = grid.world_to_grid(agent.x, agent.y)
        # Nearest cluster first (straight-line). Planning cost would be more
        # accurate but far more expensive; nearest-centroid is the Phase 2 rule.
        goals.sort(key=lambda g: (g[0] - start[0]) ** 2 + (g[1] - start[1]) ** 2)

        moved = False
        for goal in goals:
            result = planner(grid.grid, start, goal, treat_unknown_as_free)
            if result.path is None or len(result.path) < 2:
                failures[goal] += 1
                if failures[goal] >= blacklist_after_failures:
                    blacklist.add(goal)
                continue

            # Execute the plan, re-sensing each step. Stop early if a freshly
            # discovered wall now blocks the next cell (then we replan).
            agent.set_path(result.path)
            agent.path_idx = 1  # path[0] is the current cell
            while agent.has_path() and step < max_steps:
                nr, nc = agent.path[agent.path_idx]
                if grid.grid[nr, nc] == OCCUPIED:
                    break
                agent.step()
                _sense()
                step += 1
                if on_step is not None:
                    on_step(step)
            reached += 1
            moved = True
            break

        if not moved:
            # Every candidate failed to plan this round. They accrue failures and
            # will be blacklisted; once all are, the next round terminates. If
            # somehow none get blacklisted we would spin, so guard against it.
            if all(g in blacklist for g in goals) or not failures:
                return ExplorationResult(False, step, reached, len(blacklist), "stuck")

    return ExplorationResult(False, step, reached, len(blacklist), "step_limit")
