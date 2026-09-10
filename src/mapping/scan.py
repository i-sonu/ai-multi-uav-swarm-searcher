"""Integrate a real LiDAR scan (ranges + pose) into an occupancy grid.

Phase 1's ``sensor.py`` ray-casts against the *ground-truth* map (it knows where
the walls are). In the ROS 2 / Gazebo port (Phase 6) the drone instead receives a
``LaserScan`` — a list of measured ranges — with no knowledge of the true map.
This module closes that gap: given the drone pose and the measured ranges, it
marks cells ``FREE`` along each beam and ``OCCUPIED`` at the hit point, exactly
like the sim sensor but driven by measured data instead of the ground truth.

Kept as pure NumPy/Python (no ROS import) so it is unit-testable under the normal
test suite; the ROS node in ``ros2_ws`` just feeds it messages.
"""

from __future__ import annotations

import math

from src.constants import FREE, OCCUPIED


def _bresenham(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Integer grid cells on the line from ``(r0,c0)`` to ``(r1,c1)``, inclusive."""
    cells = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return cells


def integrate_scan(
    grid,
    pose: tuple[float, float, float],
    ranges,
    angle_min: float,
    angle_increment: float,
    range_max: float,
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    agent_id: int = 0,
) -> int:
    """Fold one LaserScan into ``grid``; return the number of cells updated.

    Args:
        grid: an ``OccupancyGrid`` (its ``resolution`` and ``update_cell`` are used).
        pose: drone ``(x, y, theta)`` in world metres/radians (ground truth from
            the simulator — no SLAM, per project scope).
        ranges: measured range per beam (metres). A non-finite value or a value
            ``>= range_max`` is a "no return" — the beam is free out to ``range_max``
            with no occupied endpoint.
        angle_min, angle_increment: beam 0 angle and spacing (LaserScan fields).
        range_max: sensor max range (beyond this, no obstacle is inferred).
        origin: world coordinate of grid cell ``(0, 0)``'s centre — lets the grid
            cover negative world coordinates (the Gazebo world spans the origin).
        agent_id: which drone observed this (for per-cell observation bookkeeping).
    """
    res = grid.resolution
    ox, oy = origin
    rx, ry, rtheta = pose
    rr = int(round((ry - oy) / res))
    rc = int(round((rx - ox) / res))

    updated = 0
    for i, rng in enumerate(ranges):
        angle = rtheta + angle_min + i * angle_increment
        hit = (rng is not None) and math.isfinite(rng) and (rng < range_max)
        d = rng if hit else range_max
        ex = rx + d * math.cos(angle)
        ey = ry + d * math.sin(angle)
        er = int(round((ey - oy) / res))
        ec = int(round((ex - ox) / res))

        cells = _bresenham(rr, rc, er, ec)
        for k, (cr, cc) in enumerate(cells):
            last = k == len(cells) - 1
            value = OCCUPIED if (last and hit) else FREE
            grid.update_cell(cr, cc, value, agent_id=agent_id)
            updated += 1
    return updated
