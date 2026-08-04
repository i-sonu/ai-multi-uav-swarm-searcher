"""Shared occupancy grid — the drones' collective world model.

This is the *belief* map (as opposed to the ground truth in ``world/``). Every
cell starts ``UNKNOWN``; sensing fills cells in as ``FREE`` or ``OCCUPIED``.
In the multi-agent phases a single ``OccupancyGrid`` instance is shared by all
agents — that shared write target is what makes coordination meaningful.

Conventions:
    - Array shape is ``(H, W)``; indexed ``[row, col]`` == ``[y, x]`` in cell space.
    - Cell values use ``UNKNOWN / FREE / OCCUPIED`` from ``src.constants``.
    - World coordinates are metres; ``resolution`` metres per cell. The world
      origin (0, 0) is the *centre of cell (row=0, col=0)*.
"""

from __future__ import annotations

import numpy as np

from src.constants import FREE, OCCUPIED, UNKNOWN


class OccupancyGrid:
    def __init__(self, height: int, width: int, resolution: float = 0.25):
        """Create an all-``UNKNOWN`` grid.

        Args:
            height: number of rows (cells).
            width: number of columns (cells).
            resolution: metres per cell.
        """
        self.height = int(height)
        self.width = int(width)
        self.resolution = float(resolution)

        # Belief map, int8 for memory (values are -1/0/1).
        self.grid = np.full((self.height, self.width), UNKNOWN, dtype=np.int8)

        # Per-cell observation count across ALL agents. Needed later for the
        # redundant-coverage metric (cells observed by more than one agent).
        self.obs_count = np.zeros((self.height, self.width), dtype=np.int32)

        # Per-cell set of agent ids that have observed it, packed as a bitmask
        # (agent i sets bit i). Lets us count *distinct* agents per cell without
        # storing a Python set per cell. Supports up to 64 agents (ample).
        self.obs_agents = np.zeros((self.height, self.width), dtype=np.int64)

    # --------------------------------------------------------------------- #
    # Cell updates
    # --------------------------------------------------------------------- #
    def update_cell(self, r: int, c: int, value: int, agent_id: int = 0) -> None:
        """Set cell ``(r, c)`` to ``value`` and record the observation.

        The observation bookkeeping (count + which agent) is updated on every
        sensing hit, even when the cell's value does not change — re-observing a
        known cell still counts as coverage effort by that agent.
        """
        if not (0 <= r < self.height and 0 <= c < self.width):
            return
        self.grid[r, c] = value
        self.obs_count[r, c] += 1
        self.obs_agents[r, c] |= np.int64(1) << np.int64(agent_id)

    def apply_observations(self, cells: list[tuple[int, int, int]], agent_id: int = 0) -> None:
        """Apply a batch of ``(row, col, value)`` observations from one agent.

        This is the update path the sensor feeds: ``world/sensor.py`` returns the
        cells a ray sweep touched, and the mapping module writes them here.
        """
        for r, c, value in cells:
            self.update_cell(r, c, value, agent_id=agent_id)

    # --------------------------------------------------------------------- #
    # Coordinate transforms
    # --------------------------------------------------------------------- #
    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert world metres ``(x, y)`` to grid indices ``(row, col)``.

        x maps to column, y maps to row. Rounds to nearest cell centre.
        """
        c = int(round(x / self.resolution))
        r = int(round(y / self.resolution))
        return r, c

    def grid_to_world(self, r: int, c: int) -> tuple[float, float]:
        """Convert grid indices ``(row, col)`` to world metres ``(x, y)``."""
        x = c * self.resolution
        y = r * self.resolution
        return x, y

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.height and 0 <= c < self.width

    # --------------------------------------------------------------------- #
    # Coverage
    # --------------------------------------------------------------------- #
    def known_mask(self) -> np.ndarray:
        """Boolean mask of cells that are no longer ``UNKNOWN``."""
        return self.grid != UNKNOWN

    def get_coverage(self, traversable_count: int | None = None) -> float:
        """Fraction of the map that is known, as a percentage in [0, 100].

        Args:
            traversable_count: denominator = number of ground-truth traversable
                cells reachable from the start (CLAUDE.md §5). If ``None``, falls
                back to total cell count — only meaningful for rough progress,
                not for reported metrics.
        """
        known = int(np.count_nonzero(self.known_mask()))
        denom = traversable_count if traversable_count is not None else self.grid.size
        if denom <= 0:
            return 0.0
        return 100.0 * known / denom

    def redundant_coverage_ratio(self) -> float:
        """(cells observed by >1 agent) / (total observed cells).

        The key coordination metric (CLAUDE.md §5). A cell counts as
        multiply-observed when two or more distinct agent bits are set.
        """
        observed = self.obs_count > 0
        total_observed = int(np.count_nonzero(observed))
        if total_observed == 0:
            return 0.0
        # popcount of the agent bitmask per cell.
        distinct_agents = _popcount64(self.obs_agents)
        multi = int(np.count_nonzero((distinct_agents > 1) & observed))
        return multi / total_observed


def _popcount64(a: np.ndarray) -> np.ndarray:
    """Vectorised population count for an int64 array (number of set bits)."""
    # View as unsigned bytes and use numpy's bit_count where available.
    if hasattr(np, "bitwise_count"):  # numpy >= 2.0
        return np.bitwise_count(a.astype(np.uint64))
    # Fallback: sum set bits across the 8 constituent bytes.
    b = a.astype(np.uint64).view(np.uint8).reshape(*a.shape, 8)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
    return table[b].sum(axis=-1)
