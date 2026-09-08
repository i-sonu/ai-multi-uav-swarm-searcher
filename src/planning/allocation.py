"""Multi-agent frontier task allocation — the project's primary contribution.

Given several agents and several candidate frontier goals, decide *which agent
goes to which frontier*. Good allocation stops the agents from crowding the same
region (measured by the redundant-coverage ratio, §5).

Pipeline (CLAUDE.md Phase 4.4):
    build_cost_matrix     (n_agents, K)  A* path cost agent -> frontier
    build_utility_matrix  (n_agents, K)  information gain of each frontier
    allocate              -> assignment (n_agents,)   frontier index per agent, or -1

Score for the coordinated methods:  ``score = utility - lambda * cost``
(``lambda`` trades information gain against travel effort; swept in E4).

Methods:
    none       uncoordinated baseline B2 — each agent independently takes its
               nearest (lowest-cost) frontier; duplicates allowed.
    greedy     repeatedly take the best (agent, frontier) score, remove both.
    hungarian  globally optimal one-to-one assignment (scipy).
"""

from __future__ import annotations

import numpy as np

from src.constants import UNKNOWN

# Finite stand-in for an unreachable pair, so the Hungarian solver (which cannot
# handle inf) still runs; any pair actually chosen at this value is filtered out.
_BIG = 1e9


def build_cost_matrix(agent_cells, frontiers, grid, planner=None, treat_unknown_as_free=True):
    """Optimal 8-connected path cost from each agent to each frontier.

    Returns an ``(n_agents, K)`` float array; unreachable pairs are ``inf``.

    Uses one single-source Dijkstra per agent (``dijkstra_costs``) to get all K
    frontier costs in a single sweep — K× cheaper than K separate A* searches
    and identical in value (both give the optimal 8-connected cost). The
    ``planner`` argument is accepted for interface compatibility but unused; the
    cost is planner-independent (it is the true shortest-path cost).
    """
    from src.planning.common import dijkstra_costs

    n, K = len(agent_cells), len(frontiers)
    cost = np.full((n, K), np.inf, dtype=float)
    for i, start in enumerate(agent_cells):
        costs = dijkstra_costs(grid, start, frontiers, treat_unknown_as_free)
        for j, goal in enumerate(frontiers):
            cost[i, j] = costs[goal]
    return cost


def build_utility_matrix(grid, frontiers, sensor_range_cells):
    """Information gain of each frontier = # UNKNOWN cells within sensor range.

    A frontier that opens onto a lot of unexplored space is worth more. Computed
    with a summed-area table (integral image) of the UNKNOWN mask so each
    frontier's square window is an O(1) lookup — cheap even for many frontiers.

    The value depends only on the frontier, so it is returned as a single row
    ``(1, K)`` that broadcasts against the ``(n_agents, K)`` cost matrix — every
    agent sees the same per-frontier utility.
    """
    unknown = (grid == UNKNOWN).astype(np.int64)
    # Padded integral image so window sums never index out of bounds.
    integral = np.zeros((unknown.shape[0] + 1, unknown.shape[1] + 1), dtype=np.int64)
    integral[1:, 1:] = unknown.cumsum(axis=0).cumsum(axis=1)

    h, w = grid.shape
    rad = int(sensor_range_cells)
    util = np.empty(len(frontiers), dtype=float)
    for j, (r, c) in enumerate(frontiers):
        r0, r1 = max(0, r - rad), min(h, r + rad + 1)
        c0, c1 = max(0, c - rad), min(w, c + rad + 1)
        # Sum over window [r0:r1, c0:c1] via the four integral-image corners.
        util[j] = (integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0])

    # Shape (1, K); broadcasts against the (n_agents, K) cost matrix.
    return util[None, :]


def allocate(cost, utility, method="hungarian", lambda_cost=1.0):
    """Assign at most one distinct frontier to each agent.

    Returns an int array of length ``n_agents``; entry ``i`` is the frontier
    index assigned to agent ``i``, or ``-1`` if it is left idle (e.g. more agents
    than reachable frontiers).
    """
    n, K = cost.shape
    assignment = np.full(n, -1, dtype=int)
    if K == 0:
        return assignment

    if method == "none":
        # Uncoordinated: each agent independently grabs its nearest reachable
        # frontier. Duplicates are allowed on purpose — this is the baseline that
        # *should* waste effort, which coordination must beat.
        for i in range(n):
            reachable = np.where(np.isfinite(cost[i]))[0]
            if len(reachable):
                assignment[i] = reachable[int(np.argmin(cost[i, reachable]))]
        return assignment

    # Coordinated methods maximise total score = utility - lambda * cost.
    score = utility - lambda_cost * cost  # (n, K); -inf where unreachable

    if method == "greedy":
        # Repeatedly take the best remaining (agent, goal) score, then knock out
        # that agent's row and that goal's column so both are used at most once.
        s = score.copy()
        s[~np.isfinite(s)] = -np.inf
        for _ in range(min(n, K)):
            i, j = np.unravel_index(int(np.argmax(s)), s.shape)
            if not np.isfinite(s[i, j]):
                break  # only unreachable pairs remain
            assignment[i] = j
            s[i, :] = -np.inf
            s[:, j] = -np.inf
        return assignment

    if method == "hungarian":
        from scipy.optimize import linear_sum_assignment

        # Maximise score == minimise (-score); inf-safe via _BIG.
        cost_like = -score
        cost_like[~np.isfinite(cost_like)] = _BIG
        rows, cols = linear_sum_assignment(cost_like)
        for i, j in zip(rows, cols):
            if np.isfinite(score[i, j]):  # skip pairs that were only reachable via _BIG
                assignment[i] = j
        return assignment

    raise ValueError(f"unknown allocation method {method!r}")
