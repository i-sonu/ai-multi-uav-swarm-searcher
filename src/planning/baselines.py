"""Uninformed search baselines: BFS, DFS, UCS — hand-written (CLAUDE.md §3).

Same signature and ``PlanResult`` return type as ``astar`` so planners are
swappable via config. These exist to (a) benchmark A* against classic search
(Experiment E1) and (b) give the student implementations to defend in the viva.

Cost model is shared with A* via ``planning.common`` (8-connected, diagonal
sqrt(2), no corner cutting). Note the differing optimality guarantees:

    UCS  — optimal (expands by least path cost g; = Dijkstra).
    BFS  — optimal in *number of edges*, NOT in path cost when diagonals are
           cheaper per unit distance. Returned cost is the true cost of the
           fewest-edge path it finds.
    DFS  — NOT optimal; returns the first path it stumbles upon. Included as a
           deliberately weak baseline.

On a graph with a single unique route (e.g. a width-1 corridor) all four return
the identical path and cost — used as a cross-check in the tests.
"""

from __future__ import annotations

import heapq
import math
import time
from collections import deque

import numpy as np

from src.planning.common import PlanResult, neighbors, path_cost, reconstruct_path


def bfs(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    treat_unknown_as_free: bool = True,
) -> PlanResult:
    """Breadth-first search: fewest-edges path (ignores step costs while searching)."""
    t0 = time.perf_counter()
    if start == goal:
        return PlanResult([start], 0.0, 0, time.perf_counter() - t0)

    frontier = deque([start])
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    visited = {start}
    nodes_expanded = 0

    while frontier:
        current = frontier.popleft()
        nodes_expanded += 1
        cr, cc = current
        for nr, nc, _step in neighbors(grid, cr, cc, treat_unknown_as_free):
            neighbor = (nr, nc)
            if neighbor in visited:
                continue
            visited.add(neighbor)
            came_from[neighbor] = current
            if neighbor == goal:
                path = reconstruct_path(came_from, start, goal)
                return PlanResult(path, path_cost(path), nodes_expanded, time.perf_counter() - t0)
            frontier.append(neighbor)

    return PlanResult(None, math.inf, nodes_expanded, time.perf_counter() - t0)


def dfs(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    treat_unknown_as_free: bool = True,
) -> PlanResult:
    """Depth-first search: returns the first path found (not optimal)."""
    t0 = time.perf_counter()
    if start == goal:
        return PlanResult([start], 0.0, 0, time.perf_counter() - t0)

    stack = [start]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    visited = {start}
    nodes_expanded = 0

    while stack:
        current = stack.pop()
        nodes_expanded += 1
        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            return PlanResult(path, path_cost(path), nodes_expanded, time.perf_counter() - t0)
        cr, cc = current
        for nr, nc, _step in neighbors(grid, cr, cc, treat_unknown_as_free):
            neighbor = (nr, nc)
            if neighbor in visited:
                continue
            visited.add(neighbor)
            came_from[neighbor] = current
            stack.append(neighbor)

    return PlanResult(None, math.inf, nodes_expanded, time.perf_counter() - t0)


def ucs(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    treat_unknown_as_free: bool = True,
) -> PlanResult:
    """Uniform-cost search (Dijkstra): optimal least-cost path.

    Structurally identical to A* with h = 0 — which is exactly why A* (with an
    admissible, consistent heuristic) never expands more nodes than this.
    """
    t0 = time.perf_counter()

    g_score: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    tie = 0
    open_heap = [(0.0, tie, start)]
    closed: set[tuple[int, int]] = set()
    nodes_expanded = 0

    while open_heap:
        _g, _t, current = heapq.heappop(open_heap)
        if current in closed:
            continue
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
                tie += 1
                heapq.heappush(open_heap, (tentative_g, tie, neighbor))

    return PlanResult(None, math.inf, nodes_expanded, time.perf_counter() - t0)
