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
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.constants import FREE, GT_FREE, OCCUPIED
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
    # --- optional instrumentation (populated only when collect_metrics=True) ---
    coverage_series: list = field(default_factory=list)      # coverage % per step
    plan_nodes: list = field(default_factory=list)           # nodes expanded per planner call
    plan_wallclock: list = field(default_factory=list)       # seconds per planner call


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


def center_free_cell(ground_truth: np.ndarray) -> tuple[int, int]:
    """Pick the ground-truth free cell nearest the map centre as a start pose.

    A fixed, deterministic choice so runs are reproducible from the seed alone.
    """
    free = np.argwhere(ground_truth == GT_FREE)
    centre = np.array(ground_truth.shape) / 2.0
    r, c = free[int(((free - centre) ** 2).sum(axis=1).argmin())]
    return int(r), int(c)


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
    collect_metrics: bool = False,
) -> ExplorationResult:
    """Run autonomous exploration for a single agent. Mutates ``grid``/``agent``.

    When ``collect_metrics`` is True, the returned result carries a per-step
    coverage series and per-planner-call node/wall-clock lists (used by the
    experiment runner). It is off by default to keep unit tests fast.
    """
    result = ExplorationResult(False, 0, 0, 0, "step_limit")

    # Reachable-free mask defines the coverage denominator; compute once.
    reachable = reachable_free_mask(ground_truth, grid.world_to_grid(agent.x, agent.y))
    reachable_count = int(reachable.sum())

    def _sense() -> None:
        obs = agent.sense(ground_truth, n_beams=n_beams, max_range=max_range)
        grid.apply_observations(obs, agent_id=agent.id)

    def _log_step() -> None:
        if not collect_metrics:
            return
        known = int(((grid.grid == FREE) & reachable).sum())
        result.coverage_series.append(100.0 * known / reachable_count if reachable_count else 0.0)

    _sense()  # reveal the surroundings of the start pose

    step = 0
    reached = 0
    failures: dict[tuple[int, int], int] = defaultdict(int)
    blacklist: set[tuple[int, int]] = set()

    def _finish(reason: str) -> ExplorationResult:
        result.steps = step
        result.reached_goals = reached
        result.blacklisted = len(blacklist)
        result.reason = reason
        result.done = reason == "explored"
        return result

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
            return _finish("explored")

        start = grid.world_to_grid(agent.x, agent.y)
        # Nearest cluster first (straight-line). Planning cost would be more
        # accurate but far more expensive; nearest-centroid is the Phase 2 rule.
        goals.sort(key=lambda g: (g[0] - start[0]) ** 2 + (g[1] - start[1]) ** 2)

        moved = False
        for goal in goals:
            plan = planner(grid.grid, start, goal, treat_unknown_as_free)
            if collect_metrics:
                result.plan_nodes.append(plan.nodes_expanded)
                result.plan_wallclock.append(plan.wall_clock)
            if plan.path is None or len(plan.path) < 2:
                failures[goal] += 1
                if failures[goal] >= blacklist_after_failures:
                    blacklist.add(goal)
                continue

            # Execute the plan, re-sensing each step. Stop early if a freshly
            # discovered wall now blocks the next cell (then we replan).
            agent.set_path(plan.path)
            agent.path_idx = 1  # path[0] is the current cell
            while agent.has_path() and step < max_steps:
                nr, nc = agent.path[agent.path_idx]
                if grid.grid[nr, nc] == OCCUPIED:
                    break
                agent.step()
                _sense()
                step += 1
                _log_step()
                if on_step is not None:
                    on_step(step)
            reached += 1
            moved = True
            break

        if not moved:
            # No candidate frontier could be planned to this round. Once every
            # current frontier is blacklisted, the agent is genuinely trapped and
            # we stop ("stuck"). This is a real terminal state, not a safeguard:
            # LiDAR marks cells FREE that it can *see*, but movement is
            # 8-connected with no corner cutting, so in width-1 corridors rays
            # slip diagonally past walls and create frontiers in pockets the
            # agent cannot actually drive into. When all reachable frontiers are
            # exhausted and only such isolated pockets remain, exploration ends
            # here with < 100% coverage (notably common under DFS, whose long
            # detours strand the agent in a corner).
            if all(g in blacklist for g in goals) or not failures:
                return _finish("stuck")

    return _finish("step_limit")
