"""Frontier detection.

A **frontier** cell is the boundary between known-free space and the unknown:
a ``FREE`` cell that has at least one ``UNKNOWN`` cell among its 4-neighbours
(CLAUDE.md Phase 2.1). Frontiers are the reachable edges of current knowledge —
driving the agent to them is what expands the map.

This runs every exploration step, so it is fully vectorised with numpy (no
Python per-cell loop).
"""

from __future__ import annotations

import numpy as np

from src.constants import FREE, UNKNOWN


def find_frontiers(grid: np.ndarray) -> np.ndarray:
    """Return a boolean mask of frontier cells.

    A cell is a frontier iff it is FREE and at least one of its 4-connected
    neighbours is UNKNOWN. Cells on the array edge simply have fewer neighbours
    (the map border is walls anyway, so true frontiers never sit on the edge).
    """
    free = grid == FREE
    unknown = grid == UNKNOWN

    # For each of the 4 directions, mark FREE cells whose neighbour in that
    # direction is UNKNOWN. Shifting the UNKNOWN mask toward a cell is equivalent
    # to asking "is my neighbour on that side unknown?". Edges are padded False.
    has_unknown_neighbor = np.zeros_like(free, dtype=bool)

    # Up: neighbour at (r-1, c) is unknown -> shift unknown down into this cell.
    has_unknown_neighbor[1:, :] |= unknown[:-1, :]
    # Down.
    has_unknown_neighbor[:-1, :] |= unknown[1:, :]
    # Left.
    has_unknown_neighbor[:, 1:] |= unknown[:, :-1]
    # Right.
    has_unknown_neighbor[:, :-1] |= unknown[:, 1:]

    return free & has_unknown_neighbor
