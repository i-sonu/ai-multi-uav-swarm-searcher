"""A* search on the occupancy grid — hand-written (CLAUDE.md §3 hard rule).

A* finds a least-cost path from ``start`` to ``goal`` by always expanding the
open node with the smallest ``f = g + h``, where

    g(n) = best known cost from start to n
    h(n) = Euclidean distance from n to goal  (admissible & consistent for an
           8-connected grid with diagonal cost sqrt(2), so A* is optimal)

Why Euclidean and not, say, Manhattan? Manhattan overestimates when diagonal
moves are allowed (it would forbid the cheaper diagonal shortcut), making it
inadmissible here — an inadmissible heuristic can return a non-optimal path.
Euclidean never overestimates the true 8-connected cost, so optimality holds.

Because h is consistent, every node A* expands is also expanded by uniform-cost
search (Dijkstra); this is why A* expands no more nodes than UCS on the same
problem — verified in the tests.
"""

from __future__ import annotations

import heapq
import math
import time

import numpy as np

from src.planning.common import PlanResult, neighbors, reconstruct_path


def _heuristic(r: int, c: int, goal_r: int, goal_c: int) -> float:
    """Euclidean straight-line distance (admissible for 8-connected moves)."""
    return math.hypot(r - goal_r, c - goal_c)


def astar(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    treat_unknown_as_free: bool = True,
) -> PlanResult:
    """Plan a least-cost 8-connected path from ``start`` to ``goal``."""
    t0 = time.perf_counter()
    gr, gc = goal

    # g_score[n]: best cost from start to n found so far.
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}

    # Priority queue of (f, tie, node). The monotonically increasing tie counter
    # makes pop order deterministic when f values are equal.
    tie = 0
    open_heap = [(_heuristic(*start, gr, gc), tie, start)]
    closed: set[tuple[int, int]] = set()
    nodes_expanded = 0

    while open_heap:
        _f, _t, current = heapq.heappop(open_heap)
        if current in closed:
            continue  # a stale duplicate left in the heap — skip
        closed.add(current)
        nodes_expanded += 1

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            return PlanResult(path, g_score[goal], nodes_expanded, time.perf_counter() - t0)

        cr, cc = current
        for nr, nc, step in neighbors(grid, cr, cc, treat_unknown_as_free):
            neighbor = (nr, nc)
            if neighbor in closed:
                continue
            tentative_g = g_score[current] + step
            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + _heuristic(nr, nc, gr, gc)
                tie += 1
                heapq.heappush(open_heap, (f, tie, neighbor))

    # Open set exhausted without reaching the goal.
    return PlanResult(None, math.inf, nodes_expanded, time.perf_counter() - t0)
