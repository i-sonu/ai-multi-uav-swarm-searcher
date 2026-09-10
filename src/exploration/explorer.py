"""Autonomous exploration loops (CLAUDE.md Phase 2.5 and Phase 4).

``explore`` — single agent (Phase 2/3): detect frontiers, cluster, pick the
nearest, plan, move+sense, repeat until explored or a step limit.

``explore_team`` — N agents sharing one occupancy grid (Phase 4): each planning
round it detects frontiers, allocates *distinct* goals to agents via the chosen
method (none / greedy / hungarian), plans a path per agent, then advances all
agents in lock-step (sensing every step) until an agent needs a new goal or the
re-plan interval elapses. Reducing to N=1 with method ``none`` reproduces the
single-agent behaviour.

Unreachable frontiers are blacklisted after ``blacklist_after_failures`` failed
plans so agents do not loop forever chasing a goal they cannot plan to.
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


def team_start_cells(
    ground_truth: np.ndarray, n: int, reference: tuple[int, int] | None = None, layout: str = "base"
) -> list[tuple[int, int]]:
    """Pick ``n`` distinct free start cells for a team, all in one connected region.

    Two layouts:

    ``base`` (default) — the ``n`` reachable-free cells closest to ``reference``
    (the map centre by default). This models a *shared deployment base*: all
    drones launch from roughly the same spot. It is the realistic UAV scenario
    and also the one where coordination matters most — uncoordinated agents that
    start together fight over the same nearby frontiers, so any reduction in
    redundant coverage is attributable to the allocator, not to lucky spacing.

    ``spread`` — farthest-point sampling, so the starts are pushed apart. Useful
    as a sanity contrast (coordination helps less when agents already start in
    different regions).

    ``n == 1`` returns ``[reference]`` under either layout, so a single-agent team
    run starts exactly where the Phase 2 single agent does (regression check).
    """
    if reference is None:
        reference = center_free_cell(ground_truth)
    if n <= 0:
        return []
    if n == 1:
        return [reference]

    reachable = reachable_free_mask(ground_truth, reference)
    cells = np.argwhere(reachable)  # (M, 2) reachable-free cells
    ref = np.asarray(reference)

    if layout == "base":
        # The n reachable-free cells nearest the reference (compact cluster).
        d = ((cells - ref) ** 2).sum(axis=1)
        order = np.argsort(d, kind="stable")  # stable -> deterministic ties
        return [(int(cells[i, 0]), int(cells[i, 1])) for i in order[:n]]

    if layout == "spread":
        # Farthest-point sampling seeded at the reference.
        chosen = [reference]
        d = ((cells - ref) ** 2).sum(axis=1).astype(float)
        for _ in range(n - 1):
            idx = int(d.argmax())
            pick = (int(cells[idx, 0]), int(cells[idx, 1]))
            chosen.append(pick)
            d = np.minimum(d, ((cells - np.asarray(pick)) ** 2).sum(axis=1))
        return chosen

    raise ValueError(f"unknown start layout {layout!r}; use 'base' or 'spread'")


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


# --------------------------------------------------------------------------- #
# Multi-agent coordinated exploration (Phase 4)
# --------------------------------------------------------------------------- #
def explore_team(
    ground_truth: np.ndarray,
    grid,
    agents,
    planner: Callable,
    *,
    method: str = "hungarian",
    lambda_cost: float = 1.0,
    n_beams: int = 360,
    max_range: float = 12.0,
    min_cluster_size: int = 3,
    max_steps: int = 5000,
    blacklist_after_failures: int = 3,
    replan_every: int = 10,
    max_candidates: int = 12,
    treat_unknown_as_free: bool = True,
    on_step: Callable[[int], None] | None = None,
    collect_metrics: bool = False,
    detector=None,
    register=None,
) -> ExplorationResult:
    """Coordinated exploration for ``len(agents)`` agents on one shared grid.

    ``method`` selects the allocation strategy (see ``planning.allocation``):
    ``none`` (uncoordinated baseline B2), ``greedy``, or ``hungarian``.
    ``lambda_cost`` weights travel cost against information gain.

    One simulation step = every agent advances one cell and senses. Allocation
    is recomputed when an agent finishes/loses its goal or every ``replan_every``
    steps (CLAUDE.md Phase 4.5).

    Detection (Phase 5, half B) is optional and fully decoupled: if a ``detector``
    and shared ``register`` are supplied, each agent runs the detector on its
    viewpoint every step and fused detections are written to the register. This
    never influences frontier detection, allocation, or planning — the register
    is a write-only observer of the same run.
    """
    result = ExplorationResult(False, 0, 0, 0, "step_limit")
    sensor_range_cells = max_range / grid.resolution

    # Coverage denominator: cells reachable from ANY agent's start (they share
    # one connected free region in practice; union is the safe general choice).
    reachable = np.zeros_like(ground_truth, dtype=bool)
    for a in agents:
        reachable |= reachable_free_mask(ground_truth, grid.world_to_grid(a.x, a.y))
    reachable_count = int(reachable.sum())

    def _sense_all() -> None:
        for a in agents:
            grid.apply_observations(
                a.sense(ground_truth, n_beams=n_beams, max_range=max_range), agent_id=a.id
            )

    def _detect_all(step: int) -> None:
        # Perception layer (Phase 5): runs alongside sensing, writes to the shared
        # register, and has no effect on the exploration decisions below.
        if detector is None or register is None:
            return
        for a in agents:
            for det in detector.detect(a.pose, a.id, step):
                register.add(det)

    def _log_step() -> None:
        if not collect_metrics:
            return
        known = int(((grid.grid == FREE) & reachable).sum())
        result.coverage_series.append(100.0 * known / reachable_count if reachable_count else 0.0)

    def _finish(reason: str) -> ExplorationResult:
        result.steps = step
        result.reached_goals = reached
        result.blacklisted = len(blacklist)
        result.reason = reason
        result.done = reason == "explored"
        return result

    _sense_all()
    _detect_all(0)

    step = 0
    reached = 0
    failures: dict[tuple[int, int], int] = defaultdict(int)
    blacklist: set[tuple[int, int]] = set()

    while step < max_steps:
        mask = find_frontiers(grid.grid)
        clusters = cluster_frontiers(mask, min_cluster_size)
        goals = [c.representative for c in clusters if c.representative not in blacklist]
        if not goals:  # thin-corridor fallback (see explore())
            clusters = cluster_frontiers(mask, min_cluster_size=1)
            goals = [c.representative for c in clusters if c.representative not in blacklist]
        if not goals:
            return _finish("explored")

        agent_cells = [grid.world_to_grid(a.x, a.y) for a in agents]

        # Keep only the nearest ``max_candidates`` frontiers to the team. The
        # cost matrix costs one A* per (agent, frontier) pair, so capping the
        # candidate set bounds per-round planning without hurting decisions —
        # distant frontiers are never the best pick while nearer ones remain.
        if len(goals) > max_candidates:
            team_centre = np.mean(agent_cells, axis=0)
            goals.sort(key=lambda g: (g[0] - team_centre[0]) ** 2 + (g[1] - team_centre[1]) ** 2)
            goals = goals[:max_candidates]

        cost = build_cost_matrix(agent_cells, goals, grid.grid, planner, treat_unknown_as_free)

        if method == "none":
            assignment = allocate(cost, None, method="none")
        else:
            utility = build_utility_matrix(grid.grid, goals, sensor_range_cells)
            assignment = allocate(cost, utility, method=method, lambda_cost=lambda_cost)

        # Plan the assigned path for each agent (re-plan to recover the actual
        # path; the cost matrix only kept costs).
        any_path = False
        for i, a in enumerate(agents):
            j = int(assignment[i])
            if j < 0 or not np.isfinite(cost[i, j]):
                a.set_path([])
                continue
            plan = planner(grid.grid, agent_cells[i], goals[j], treat_unknown_as_free)
            if collect_metrics:
                result.plan_nodes.append(plan.nodes_expanded)
                result.plan_wallclock.append(plan.wall_clock)
            if plan.path is None or len(plan.path) < 2:
                failures[goals[j]] += 1
                if failures[goals[j]] >= blacklist_after_failures:
                    blacklist.add(goals[j])
                a.set_path([])
            else:
                a.set_path(plan.path)
                a.path_idx = 1
                any_path = True

        if not any_path:
            # No agent could be routed to any frontier this round. If the goals
            # that remain are all unreachable/blacklisted, the team is stuck.
            if all(g in blacklist for g in goals) or not failures:
                return _finish("stuck")
            continue

        # Advance all agents in lock-step until an agent needs a new goal or the
        # re-plan interval elapses.
        executed = 0
        while step < max_steps and executed < replan_every:
            moved_any = False
            for a in agents:
                if not a.has_path():
                    continue
                nr, nc = a.path[a.path_idx]
                if grid.grid[nr, nc] == OCCUPIED:  # freshly discovered wall
                    a.set_path([])
                    continue
                a.step()
                moved_any = True
                if not a.has_path():
                    reached += 1
            _sense_all()
            step += 1
            executed += 1
            _detect_all(step)
            _log_step()
            if on_step is not None:
                on_step(step)
            if not moved_any:
                break
            # Reallocate as soon as any agent has run out of path.
            if any(not a.has_path() for a in agents):
                break

    return _finish("step_limit")
