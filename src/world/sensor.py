"""Simulated 2-D LiDAR via grid ray casting.

Casts ``n_beams`` rays evenly around the agent, each marching outward until it
hits a wall in the ground-truth map or reaches ``max_range``. Cells the ray
passes through are reported ``FREE``; the wall cell it stops on is ``OCCUPIED``.

The grid update itself lives in the mapping module (CLAUDE.md Phase 1.3): this
function is pure — it reads the ground truth and returns a list of
``(row, col, value)`` observations for the caller to apply to the belief grid.

Coordinate convention (shared with ``OccupancyGrid``): cell ``(r, c)`` is centred
at world ``(c*res, r*res)``. We ray-march in "u-space" — world metres divided by
resolution and shifted by +0.5 — so that cell index == ``floor(u)`` and each cell
is the unit square ``[c, c+1) x [r, r+1)``. This keeps the traversal's floor()
indexing consistent with ``OccupancyGrid.world_to_grid``'s round().
"""

from __future__ import annotations

import math

import numpy as np

from src.constants import FREE, GT_WALL, OCCUPIED


def cast_rays(
    ground_truth: np.ndarray,
    pose: tuple[float, float, float],
    n_beams: int = 360,
    max_range: float = 12.0,
    resolution: float = 0.25,
) -> list[tuple[int, int, int]]:
    """Cast ``n_beams`` LiDAR rays from ``pose`` into ``ground_truth``.

    Args:
        ground_truth: ``(H, W)`` array of GT_FREE/GT_WALL.
        pose: ``(x, y, theta)`` in world metres / radians. ``theta`` is unused
            for a 360-degree sweep but kept for interface symmetry.
        n_beams: number of rays, spread evenly over 2*pi.
        max_range: maximum sensing distance in metres.
        resolution: metres per cell.

    Returns:
        List of ``(row, col, value)`` observations. A cell may appear more than
        once across beams; applying them is idempotent for value but each apply
        counts as an observation (intended — that is sensing effort).
    """
    height, width = ground_truth.shape
    x, y, _theta = pose
    max_range_cells = max_range / resolution

    # Agent position in u-space (see module docstring).
    ux = x / resolution + 0.5
    uy = y / resolution + 0.5

    observations: list[tuple[int, int, int]] = []

    # Always report the agent's own cell as free (it is standing on free space).
    r0, c0 = int(math.floor(uy)), int(math.floor(ux))
    if 0 <= r0 < height and 0 <= c0 < width:
        observations.append((r0, c0, FREE))

    for i in range(n_beams):
        angle = 2.0 * math.pi * i / n_beams
        dx = math.cos(angle)
        dy = math.sin(angle)
        _march_ray(
            ground_truth, ux, uy, dx, dy, max_range_cells, height, width, observations
        )

    return observations


def _march_ray(
    gt: np.ndarray,
    ux: float,
    uy: float,
    dx: float,
    dy: float,
    max_range_cells: float,
    height: int,
    width: int,
    out: list[tuple[int, int, int]],
) -> None:
    """Amanatides & Woo grid traversal along one ray, appending observations.

    Marks every cell the ray enters as FREE until it either (a) enters a wall
    cell — marked OCCUPIED, traversal stops (the wall blocks line of sight), or
    (b) exceeds ``max_range_cells`` — traversal stops with nothing marked beyond
    the range (this is what the "never marks beyond max range" test checks).
    """
    # Current cell.
    cx = int(math.floor(ux))
    cy = int(math.floor(uy))

    # Step direction along each axis (+1 / -1 / 0).
    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)

    # t at which the ray crosses the next vertical / horizontal grid line, and
    # the t increment for a full cell crossing. t is measured in cell units,
    # which equals ray length because (dx, dy) is a unit vector.
    if dx != 0:
        t_delta_x = abs(1.0 / dx)
        next_x = (cx + 1) if step_x > 0 else cx
        t_max_x = (next_x - ux) / dx
    else:
        t_delta_x = math.inf
        t_max_x = math.inf

    if dy != 0:
        t_delta_y = abs(1.0 / dy)
        next_y = (cy + 1) if step_y > 0 else cy
        t_max_y = (next_y - uy) / dy
    else:
        t_delta_y = math.inf
        t_max_y = math.inf

    t = 0.0
    while t <= max_range_cells:
        # Step into the next cell along whichever axis boundary is nearer.
        if t_max_x < t_max_y:
            t = t_max_x
            t_max_x += t_delta_x
            cx += step_x
        else:
            t = t_max_y
            t_max_y += t_delta_y
            cy += step_y

        if t > max_range_cells:
            break  # would exceed range — do not mark this cell
        if not (0 <= cx < width and 0 <= cy < height):
            break  # left the map

        if gt[cy, cx] == GT_WALL:
            out.append((cy, cx, OCCUPIED))
            break  # opaque: nothing beyond is visible
        out.append((cy, cx, FREE))
