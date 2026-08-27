"""Single-agent and Multi-agent autonomous exploration loop (CLAUDE.md Phase 2 & 4).

The loop, each round:
    1. sense from all agents into the shared occupancy grid,
    2. detect frontiers and cluster them into candidate goals,
    3. allocate frontiers to agents using allocation strategy (none, greedy, hungarian),
    4. plan paths to assigned frontiers with the configured planner,
    5. move agents along their paths, sensing at every step,
    6. repeat until no frontiers remain (map explored) or a step limit is hit.

Unreachable frontiers are blacklisted after ``blacklist_after_failures`` failed
plans so agents do not loop forever chasing goals they cannot plan to.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.constants import FREE, GT_FREE, OCCUPIED
from src.frontier.clustering import cluster_frontiers
from src.frontier.detection import find_frontiers
from src.planning.allocation import allocate, build_cost_matrix, build_utility_matrix
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
    # --- target registration metrics (Phase 5) ---
    time_to_first_detect: int | None = None
    target_recall: float = 0.0
    loc_error_mean: float = 0.0


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
        for nr, nc, _step in neighbors(ground_truth, r, c, treat_unknown_as_free=True):
            if not mask[nr, nc]:
                mask[nr, nc] = True
                stack.append((nr, nc))
    return mask


def count_reachable_free(ground_truth: np.ndarray, start: tuple[int, int]) -> int:
    """Number of ground-truth free cells reachable from ``start``."""
    return int(reachable_free_mask(ground_truth, start).sum())


def center_free_cell(ground_truth: np.ndarray) -> tuple[int, int]:
    """Pick the ground-truth free cell nearest the map centre as a start pose."""
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
    """Run autonomous exploration for a single agent. Mutates ``grid``/``agent``."""
    return explore_multi(
        ground_truth,
        grid,
        [agent],
        planner,
        allocation_method="none",
        n_beams=n_beams,
        max_range=max_range,
        min_cluster_size=min_cluster_size,
        max_steps=max_steps,
        blacklist_after_failures=blacklist_after_failures,
        treat_unknown_as_free=treat_unknown_as_free,
        on_step=on_step,
        collect_metrics=collect_metrics,
    )


def explore_multi(
    ground_truth: np.ndarray,
    grid,
    agents: list,
    planner: Callable,
    *,
    allocation_method: str = "hungarian",
    lambda_val: float = 0.5,
    n_beams: int = 360,
    max_range: float = 12.0,
    min_cluster_size: int = 3,
    max_steps: int = 5000,
    blacklist_after_failures: int = 3,
    treat_unknown_as_free: bool = True,
    on_step: Callable[[int], None] | None = None,
    collect_metrics: bool = False,
    # --- Target perception params (Phase 5) ---
    targets: list = None,
    camera_sensor: object = None,
    target_register: object = None,
) -> ExplorationResult:
    """Run multi-agent autonomous exploration for N agents. Mutates ``grid`` and ``agents``."""
    result = ExplorationResult(False, 0, 0, 0, "step_limit")

    if not agents:
        return result

    # Coverage denominator from first agent's starting reachable mask
    start_grid = grid.world_to_grid(agents[0].x, agents[0].y)
    reachable = reachable_free_mask(ground_truth, start_grid)
    reachable_count = int(reachable.sum())

    def _sense_all() -> None:
        for a in agents:
            obs = a.sense(ground_truth, n_beams=n_beams, max_range=max_range)
            grid.apply_observations(obs, agent_id=a.id)

    def _log_step() -> None:
        if not collect_metrics:
            return
        known = int(((grid.grid == FREE) & reachable).sum())
        result.coverage_series.append(100.0 * known / reachable_count if reachable_count else 0.0)

    if targets is None:
        targets = []

    # Initialize perception sensor and register if needed
    if targets and (camera_sensor is None or target_register is None):
        from src.perception.registration import CameraSensor, TargetRegister
        if camera_sensor is None:
            camera_sensor = CameraSensor()
        if target_register is None:
            target_register = TargetRegister()

    def _sense_targets_all(current_step: int) -> None:
        if not targets or camera_sensor is None or target_register is None:
            return
        for a in agents:
            # Deterministic seed based on step index and agent ID
            detections = camera_sensor.sense_targets(a.pose, a.id, targets, seed=current_step * 100 + a.id)
            for det in detections:
                target_register.register_detection(det, current_step)

    _sense_all()
    _sense_targets_all(0)

    step = 0
    reached = 0
    failures: dict[tuple[int, int], int] = defaultdict(int)
    blacklist: set[tuple[int, int]] = set()

    import math

    def _finish(reason: str) -> ExplorationResult:
        result.steps = step
        result.reached_goals = reached
        result.blacklisted = len(blacklist)
        result.reason = reason
        result.done = reason == "explored"

        # Calculate target registration metrics (Phase 5)
        if targets and target_register is not None:
            registered = target_register.get_registered_targets()
            first_detect = None
            for r in registered:
                if first_detect is None or r["last_updated"] < first_detect:
                    first_detect = r["last_updated"]
            result.time_to_first_detect = first_detect

            localized_count = 0
            error_sum = 0.0
            for t in targets:
                best_dist = float('inf')
                for r in registered:
                    if r["class_id"] == t.class_id:
                        dist = math.hypot(r["x"] - t.x, r["y"] - t.y)
                        if dist < best_dist:
                            best_dist = dist
                if best_dist <= 2.0:  # 2.0m tolerance
                    localized_count += 1
                    error_sum += best_dist

            result.target_recall = 100.0 * localized_count / len(targets) if targets else 0.0
            result.loc_error_mean = error_sum / localized_count if localized_count > 0 else 0.0

        return result

    while step < max_steps:
        mask = find_frontiers(grid.grid)
        clusters = cluster_frontiers(mask, min_cluster_size)
        goals = [c.representative for c in clusters if c.representative not in blacklist]

        if not goals:
            clusters = cluster_frontiers(mask, min_cluster_size=1)
            goals = [c.representative for c in clusters if c.representative not in blacklist]

        if not goals:
            return _finish("explored")

        # Build Cost and Utility matrices for active goals
        cost_matrix = build_cost_matrix(agents, goals, grid, planner_fn=planner)
        utility_vector = build_utility_matrix(grid, goals, sensor_range=max_range)

        # Allocate goals to agents
        assignments = allocate(cost_matrix, utility_vector, method=allocation_method, lambda_val=lambda_val)

        assigned_any = False
        for i, a in enumerate(agents):
            goal_idx = assignments[i]
            if goal_idx != -1 and goal_idx < len(goals):
                goal = goals[goal_idx]
                start = grid.world_to_grid(a.x, a.y)
                plan = planner(grid.grid, start, goal, treat_unknown_as_free)
                if collect_metrics:
                    result.plan_nodes.append(plan.nodes_expanded)
                    result.plan_wallclock.append(plan.wall_clock)
                if plan.path is not None and len(plan.path) >= 2:
                    a.set_path(plan.path)
                    a.path_idx = 1
                    assigned_any = True
                else:
                    failures[goal] += 1
                    if failures[goal] >= blacklist_after_failures:
                        blacklist.add(goal)

        if not assigned_any:
            if all(g in blacklist for g in goals) or not failures:
                return _finish("stuck")
            # If no agent could plan to their current goal, continue to next step iteration
            step += 1
            _sense_all()
            _sense_targets_all(step)
            _log_step()
            continue

        # Execute paths step by step across all active agents
        steps_in_round = 0
        max_round_steps = max(len(a.path) - 1 for a in agents if a.has_path()) if any(a.has_path() for a in agents) else 0

        while any(a.has_path() for a in agents) and step < max_steps and steps_in_round < max_round_steps:
            for a in agents:
                if a.has_path():
                    nr, nc = a.path[a.path_idx]
                    if grid.grid[nr, nc] == OCCUPIED:
                        a.clear_path()
                    else:
                        a.step()
                        if not a.has_path():
                            reached += 1
            _sense_all()
            _sense_targets_all(step)
            step += 1
            steps_in_round += 1
            _log_step()
            if on_step is not None:
                on_step(step)

    return _finish("step_limit")
