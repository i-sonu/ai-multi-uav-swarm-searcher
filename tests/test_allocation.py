"""Unit tests for multi-agent task allocation (CLAUDE.md Phase 4.4).

These pin down the *decision* logic in isolation (hand-built cost/utility
matrices), separate from the exploration loop, so a regression in the allocator
is caught directly.
"""

import numpy as np

from src.constants import GT_FREE, UNKNOWN
from src.planning.allocation import (
    allocate,
    build_cost_matrix,
    build_utility_matrix,
)
from src.planning.common import dijkstra_costs


# --------------------------------------------------------------------------- #
# allocate()
# --------------------------------------------------------------------------- #
def test_none_allows_duplicate_frontiers():
    """Uncoordinated baseline: both agents grab their own nearest frontier, and
    nothing stops them picking the *same* one."""
    # Frontier 0 is nearest for both agents; frontier 1 is far for both.
    cost = np.array([[1.0, 9.0], [2.0, 8.0]])
    assignment = allocate(cost, None, method="none")
    assert assignment[0] == 0 and assignment[1] == 0  # duplicate, on purpose


def test_greedy_and_hungarian_are_distinct():
    """Coordinated methods must never assign one frontier to two agents."""
    cost = np.array([[1.0, 2.0], [1.5, 1.0]])
    utility = np.array([[10.0, 10.0]])  # equal utility -> decided by cost
    for method in ("greedy", "hungarian"):
        a = allocate(cost, utility, method=method, lambda_cost=1.0)
        assert set(a.tolist()) == {0, 1}, method  # both frontiers used exactly once


def test_hungarian_beats_greedy_when_greedy_is_myopic():
    """A classic case where the locally-best first pick is globally suboptimal.

    Greedy grabs the single highest score (agent0->f0) and is then forced into a
    bad leftover; Hungarian optimises the total, so its total score is >= greedy's.
    """
    # score = utility - cost. Build so the greedy first pick strands agent1.
    utility = np.array([[0.0, 0.0]])
    cost = np.array([[0.0, 1.0], [1.0, 100.0]])
    # scores: agent0: f0=0, f1=-1 ; agent1: f0=-1, f1=-100
    g = allocate(cost, utility, method="greedy")
    h = allocate(cost, utility, method="hungarian")

    def total(a):
        return sum((utility - cost)[i, a[i]] for i in range(len(a)) if a[i] >= 0)

    # greedy: picks (0,f0)=0 first, leaves agent1 with f1=-100 -> total -100.
    # hungarian: (0,f1)=-1 + (1,f0)=-1 = -2 -> strictly better.
    assert total(h) > total(g)


def test_more_agents_than_frontiers_leaves_some_idle():
    """K < n_agents: distinct assignment means at least one agent gets -1."""
    cost = np.array([[1.0], [2.0], [3.0]])  # 3 agents, 1 frontier
    utility = np.array([[5.0]])
    for method in ("greedy", "hungarian"):
        a = allocate(cost, utility, method=method)
        assigned = [j for j in a if j >= 0]
        assert len(assigned) == 1 and len(set(assigned)) == 1  # only one agent placed


def test_unreachable_pairs_never_chosen():
    """inf-cost (unreachable) pairs must not be assigned; a fully-blocked agent
    is left idle (-1) rather than sent somewhere it cannot reach."""
    cost = np.array([[np.inf, np.inf], [1.0, 2.0]])
    utility = np.array([[5.0, 5.0]])
    for method in ("none", "greedy", "hungarian"):
        a = allocate(cost, utility if method != "none" else None, method=method)
        assert a[0] == -1  # agent 0 can reach nothing
        assert a[1] in (0, 1)


def test_empty_frontier_set():
    """No frontiers -> every agent idle, no crash."""
    cost = np.empty((2, 0))
    a = allocate(cost, None, method="none")
    assert a.tolist() == [-1, -1]


# --------------------------------------------------------------------------- #
# build_utility_matrix()
# --------------------------------------------------------------------------- #
def test_utility_counts_unknown_cells_in_window():
    """Information gain = number of UNKNOWN cells within the sensor window."""
    grid = np.full((5, 5), UNKNOWN, dtype=np.int8)
    grid[2, 2] = 0  # one FREE cell in a sea of UNKNOWN
    # radius 1 window around (2,2) is the 3x3 block: 9 cells, 8 UNKNOWN (centre is FREE).
    util = build_utility_matrix(grid, [(2, 2)], sensor_range_cells=1)
    assert util.shape == (1, 1)
    assert util[0, 0] == 8


def test_utility_window_clips_at_border():
    """A window at the grid corner must clip, not wrap or overflow."""
    grid = np.full((4, 4), UNKNOWN, dtype=np.int8)
    # corner (0,0), radius 1 -> valid window is rows/cols [0:2] = 4 cells, all UNKNOWN.
    util = build_utility_matrix(grid, [(0, 0)], sensor_range_cells=1)
    assert util[0, 0] == 4


# --------------------------------------------------------------------------- #
# build_cost_matrix() / dijkstra_costs()
# --------------------------------------------------------------------------- #
def test_cost_matrix_matches_hand_computed_distance():
    """On an open grid, cost equals the true 8-connected shortest-path cost."""
    grid = np.zeros((5, 5), dtype=np.int8)  # all FREE
    agents = [(0, 0)]
    frontiers = [(0, 3), (3, 3)]
    cost = build_cost_matrix(agents, frontiers, grid)
    # (0,0)->(0,3): 3 orthogonal steps = 3.0
    assert np.isclose(cost[0, 0], 3.0)
    # (0,0)->(3,3): 3 diagonal steps = 3*sqrt(2)
    assert np.isclose(cost[0, 1], 3 * np.sqrt(2))


def test_cost_matrix_unreachable_is_inf():
    """A frontier walled off from the agent has infinite cost."""
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[:, 2] = 1  # solid vertical wall splits the grid
    cost = build_cost_matrix([(0, 0)], [(0, 4)], grid, treat_unknown_as_free=False)
    assert not np.isfinite(cost[0, 0])


def test_dijkstra_matches_individual_search():
    """One Dijkstra sweep gives the same optimal costs as per-target searches."""
    grid = np.zeros((6, 6), dtype=np.int8)
    grid[1:5, 3] = 1  # partial wall
    targets = [(0, 5), (5, 5), (5, 0)]
    costs = dijkstra_costs(grid, (0, 0), targets)
    # Symmetry: distance is the same measured from the target back to the start.
    for t in targets:
        back = dijkstra_costs(grid, t, [(0, 0)])[(0, 0)]
        assert np.isclose(costs[t], back)
