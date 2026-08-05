"""Multi-Agent Task Allocation (MATA) for frontier-based exploration.

Supports three allocation methods:
- 'none' (Baseline B2): Uncoordinated nearest-frontier selection for each agent.
- 'greedy': Sequential highest-score matching.
- 'hungarian': Optimal bipartite matching via scipy.optimize.linear_sum_assignment.
"""

from __future__ import annotations

import importlib
import numpy as np

# Dynamic import to support execution in environments where scipy is loaded in venv
linear_sum_assignment = None
try:
    _scipy_opt = importlib.import_module("scipy.optimize")
    linear_sum_assignment = getattr(_scipy_opt, "linear_sum_assignment", None)
except Exception:
    linear_sum_assignment = None

from src.planning.astar import astar


def build_cost_matrix(
    agents: list,
    frontiers: list[tuple[int, int]],
    grid: object,
    planner_fn=astar,
) -> np.ndarray:
    """Compute an (N x K) cost matrix representing path costs from N agents to K frontiers.

    If a frontier is unreachable from an agent, assigns infinity (np.inf).
    """
    n_agents = len(agents)
    n_frontiers = len(frontiers)
    cost_matrix = np.full((n_agents, n_frontiers), np.inf, dtype=float)

    grid_array = grid.grid if hasattr(grid, "grid") else grid

    for i, agent in enumerate(agents):
        start = (int(round(agent.pose[1] / agent.resolution)), int(round(agent.pose[0] / agent.resolution)))
        for j, goal in enumerate(frontiers):
            if start == goal:
                cost_matrix[i, j] = 0.0
                continue
            plan = planner_fn(grid_array, start, goal)
            path = plan.path if hasattr(plan, "path") else plan[0]
            if path is not None:
                cost_matrix[i, j] = float(len(path) - 1)

    return cost_matrix


def build_utility_matrix(
    grid: object,
    frontiers: list[tuple[int, int]],
    sensor_range: float = 10.0,
) -> np.ndarray:
    """Compute a (K,) utility array representing information gain.

    Utility is measured as the count of UNKNOWN cells within sensor range of each frontier.
    """
    n_frontiers = len(frontiers)
    utility = np.zeros(n_frontiers, dtype=float)
    grid_array = grid.grid if hasattr(grid, "grid") else grid
    rows, cols = grid_array.shape
    r_int = int(np.ceil(sensor_range))

    for j, (fx, fy) in enumerate(frontiers):
        unknown_count = 0
        min_x, max_x = max(0, fx - r_int), min(rows, fx + r_int + 1)
        min_y, max_y = max(0, fy - r_int), min(cols, fy + r_int + 1)

        for rx in range(min_x, max_x):
            for ry in range(min_y, max_y):
                if (rx - fx) ** 2 + (ry - fy) ** 2 <= sensor_range**2:
                    if grid_array[rx, ry] == -1:  # UNKNOWN
                        unknown_count += 1
        utility[j] = float(unknown_count)

    return utility


def allocate(
    cost_matrix: np.ndarray,
    utility_vector: np.ndarray,
    method: str = "hungarian",
    lambda_val: float = 0.5,
) -> np.ndarray:
    """Allocate N agents to K frontiers given cost matrix (N x K) and utility vector (K,).

    Returns an array assignment of shape (N,) containing assigned frontier indices for each agent.
    If K < N or no valid path exists, unassigned agents are assigned -1.
    """
    n_agents, n_frontiers = cost_matrix.shape
    assignment = np.full(n_agents, -1, dtype=int)

    if n_frontiers == 0:
        return assignment

    if method == "none":
        # Baseline B2: Uncoordinated nearest frontier (lowest cost independently)
        for i in range(n_agents):
            valid_costs = cost_matrix[i]
            min_idx = np.argmin(valid_costs)
            if not np.isinf(valid_costs[min_idx]):
                assignment[i] = min_idx
        return assignment

    # Construct Score matrix: S = Utility - lambda * Cost
    score_matrix = np.full((n_agents, n_frontiers), -np.inf, dtype=float)
    for i in range(n_agents):
        for j in range(n_frontiers):
            if not np.isinf(cost_matrix[i, j]):
                score_matrix[i, j] = utility_vector[j] - lambda_val * cost_matrix[i, j]

    if method == "greedy" or linear_sum_assignment is None:
        # Greedily match highest global score pairs, enforcing distinct frontiers
        assigned_frontiers = set()
        flat_indices = np.argsort(-score_matrix.ravel())
        for idx in flat_indices:
            i, j = np.unravel_index(idx, score_matrix.shape)
            if np.isneginf(score_matrix[i, j]):
                break
            if assignment[i] == -1 and j not in assigned_frontiers:
                assignment[i] = j
                assigned_frontiers.add(j)
                if len(assigned_frontiers) == min(n_agents, n_frontiers):
                    break
        return assignment

    elif method == "hungarian":
        # Hungarian bipartite assignment via linear_sum_assignment
        cost_for_hungarian = -score_matrix.copy()
        max_val = 1e9
        cost_for_hungarian[np.isneginf(score_matrix)] = max_val

        row_ind, col_ind = linear_sum_assignment(cost_for_hungarian)
        for i, j in zip(row_ind, col_ind):
            if not np.isneginf(score_matrix[i, j]):
                assignment[i] = j
        return assignment

    else:
        raise ValueError(f"Unknown allocation method: {method}")
