"""Tests for the hand-written planners (A*, BFS, DFS, UCS)."""

import math

import numpy as np

from src.constants import FREE, OCCUPIED
from src.planning.astar import astar
from src.planning.baselines import bfs, dfs, ucs


def _free_grid(h, w):
    return np.full((h, w), FREE, dtype=np.int8)


def test_astar_optimal_open_grid():
    """On an open grid the optimal path is straight diagonal: 2*sqrt(2)."""
    grid = _free_grid(3, 3)
    res = astar(grid, (0, 0), (2, 2))
    assert res.path is not None
    assert res.path[0] == (0, 0) and res.path[-1] == (2, 2)
    assert math.isclose(res.cost, 2 * math.sqrt(2), rel_tol=1e-9)


def test_astar_optimal_with_obstacle():
    """A wall forces a detour with a hand-checkable optimal cost."""
    # 3-wide column of wall at col 2, rows 0..1 (a gap at row 2 to pass under).
    grid = _free_grid(5, 5)
    grid[0:4, 2] = OCCUPIED  # wall blocking cols, gap at row 4
    res = astar(grid, (0, 0), (0, 4))
    assert res.path is not None
    # A* must equal UCS (both optimal) on the same problem.
    assert math.isclose(res.cost, ucs(grid, (0, 0), (0, 4)).cost, rel_tol=1e-9)


def test_astar_expands_no_more_than_ucs():
    """Consistent heuristic => A* expands a subset of UCS's nodes."""
    grid = _free_grid(25, 25)
    grid[5:20, 12] = OCCUPIED  # a wall to make the search non-trivial
    grid[12, 12] = FREE        # leave a gap
    a = astar(grid, (0, 0), (24, 24))
    u = ucs(grid, (0, 0), (24, 24))
    assert a.path is not None and u.path is not None
    assert math.isclose(a.cost, u.cost, rel_tol=1e-9)  # both optimal
    assert a.nodes_expanded <= u.nodes_expanded


def test_no_path_returns_none():
    """A goal walled off entirely yields no path."""
    grid = _free_grid(7, 7)
    grid[:, 3] = OCCUPIED  # full vertical wall splits the grid
    res = astar(grid, (0, 0), (0, 6))
    assert res.path is None
    assert math.isinf(res.cost)


def test_all_planners_same_cost_unique_corridor():
    """In a width-1 corridor there is one path, so all four agree on cost."""
    grid = np.full((5, 10), OCCUPIED, dtype=np.int8)
    grid[2, 1:9] = FREE  # single horizontal corridor on row 2
    start, goal = (2, 1), (2, 8)
    costs = [p(grid, start, goal).cost for p in (astar, bfs, dfs, ucs)]
    for c in costs:
        assert math.isclose(c, 7.0, rel_tol=1e-9)  # 7 orthogonal steps


def test_no_corner_cutting():
    """A diagonal move between two touching walls is forbidden."""
    grid = _free_grid(3, 3)
    grid[0, 1] = OCCUPIED  # north shoulder of the (1,0)->(0,1) diagonal
    grid[1, 0] = OCCUPIED  # ... wait, that blocks start neighbourhood
    # Reset: block the two shoulders of the (1,1)->(0,0) diagonal instead.
    grid = _free_grid(3, 3)
    grid[0, 1] = OCCUPIED
    grid[1, 0] = OCCUPIED
    # From (1,1) the diagonal to (0,0) needs both (0,1) and (1,0) open — they are
    # not, so the only route to (0,0) is blocked; goal is isolated.
    res = astar(grid, (1, 1), (0, 0))
    assert res.path is None
