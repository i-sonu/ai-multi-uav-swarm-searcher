"""Shared primitives for all grid path planners.

All four planners (A*, BFS, DFS, UCS) share the same movement model and the same
result type so they are swappable via config (CLAUDE.md Phase 2.4). The search
algorithms themselves are hand-written from scratch in ``astar.py`` /
``baselines.py`` — nothing here does the searching.

Movement model
--------------
- **8-connected**: orthogonal step cost 1.0, diagonal step cost sqrt(2).
- **No corner cutting**: a diagonal move is only allowed when both orthogonally
  adjacent cells are also traversable. This prevents the agent from "slipping"
  between two touching walls, which would be physically impossible for a drone
  with any footprint and would let paths clip through obstacle corners.
- **UNKNOWN is optimistically traversable** during exploration (CLAUDE.md
  Phase 2.3): the planner may route through unexplored space, because that is
  exactly where we want the agent to go. Discovered walls (OCCUPIED) block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.constants import OCCUPIED

# (dr, dc, step_cost). Orthogonals first, then diagonals — a fixed order so
# search is deterministic (CLAUDE.md §3: same seed -> identical results).
ORTHOGONAL = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
DIAGONAL = [
    (-1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (1, 1, math.sqrt(2)),
]
MOVES = ORTHOGONAL + DIAGONAL


@dataclass
class PlanResult:
    """Uniform return type for every planner.

    Attributes:
        path: list of ``(row, col)`` cells from start to goal inclusive, or
            ``None`` if no path exists.
        cost: total path cost (sum of step costs); ``inf`` if no path.
        nodes_expanded: number of nodes popped-and-processed from the frontier
            (the "nodes expanded" metric, CLAUDE.md §5).
        wall_clock: seconds spent inside the planner call.
    """

    path: list[tuple[int, int]] | None
    cost: float
    nodes_expanded: int
    wall_clock: float


def is_traversable(grid: np.ndarray, r: int, c: int, treat_unknown_as_free: bool) -> bool:
    """Whether cell ``(r, c)`` may be entered."""
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if treat_unknown_as_free:
        return grid[r, c] != OCCUPIED  # FREE or UNKNOWN
    return grid[r, c] == 0  # FREE only


def neighbors(grid: np.ndarray, r: int, c: int, treat_unknown_as_free: bool):
    """Yield ``(nr, nc, step_cost)`` for each legal move from ``(r, c)``.

    Applies the no-corner-cutting rule for diagonal moves.
    """
    for dr, dc, cost in MOVES:
        nr, nc = r + dr, c + dc
        if not is_traversable(grid, nr, nc, treat_unknown_as_free):
            continue
        if dr != 0 and dc != 0:
            # Diagonal: require both orthogonal "shoulders" to be open.
            if not is_traversable(grid, r, nc, treat_unknown_as_free):
                continue
            if not is_traversable(grid, nr, c, treat_unknown_as_free):
                continue
        yield nr, nc, cost


def dijkstra_costs(grid, start, targets, treat_unknown_as_free=True):
    """Least-cost distance from ``start`` to each cell in ``targets``.

    A single-source Dijkstra (uniform-cost search) that returns the optimal
    8-connected path cost to every target in one sweep — used to build the
    multi-agent cost matrix cheaply. Computing K individual A* searches would
    repeat almost all the work; one Dijkstra per agent gives all K costs at once
    and early-exits once every target has been settled.

    Returns a dict ``{target: cost}``; unreachable targets get ``inf``.
    """
    import heapq

    target_set = set(targets)
    remaining = set(target_set)
    out = {t: math.inf for t in target_set}

    g = {start: 0.0}
    tie = 0
    heap = [(0.0, tie, start)]
    closed = set()
    while heap and remaining:
        d, _t, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        if cur in remaining:
            out[cur] = d
            remaining.discard(cur)
        cr, cc = cur
        for nr, nc, step in neighbors(grid, cr, cc, treat_unknown_as_free):
            nxt = (nr, nc)
            if nxt in closed:
                continue
            nd = d + step
            if nd < g.get(nxt, math.inf):
                g[nxt] = nd
                tie += 1
                heapq.heappush(heap, (nd, tie, nxt))
    return out


def reconstruct_path(came_from: dict, start, goal) -> list[tuple[int, int]]:
    """Walk parent pointers back from ``goal`` to ``start``; return start->goal."""
    path = [goal]
    node = goal
    while node != start:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path


def path_cost(path: list[tuple[int, int]]) -> float:
    """Total movement cost of a cell path under the 8-connected cost model."""
    total = 0.0
    for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
        total += math.sqrt(2) if (r0 != r1 and c0 != c1) else 1.0
    return total
