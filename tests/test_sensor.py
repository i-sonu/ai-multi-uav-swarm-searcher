"""Tests for LiDAR ray casting (src/world/sensor.py)."""

import math

import numpy as np

from src.constants import FREE, GT_FREE, GT_WALL, OCCUPIED
from src.world.sensor import cast_rays


def _empty_map(size):
    """All-free interior with a wall border (like a real map, minus obstacles)."""
    g = np.full((size, size), GT_FREE, dtype=np.int8)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = GT_WALL
    return g


def test_never_marks_beyond_max_range():
    """No observation may lie further than max_range from the agent."""
    size = 200
    gt = _empty_map(size)  # large enough that the border never limits the range
    res = 0.25
    max_range = 12.0
    # Agent at the exact centre of a cell.
    cx = cy = size // 2
    pose = (cx * res, cy * res, 0.0)

    obs = cast_rays(gt, pose, n_beams=360, max_range=max_range, resolution=res)

    # Tolerance: a marked cell's centre can sit up to ~one cell diagonal beyond
    # the ray-entry point that passed the range test.
    tol = max_range + res * math.sqrt(2)
    ax, ay = pose[0], pose[1]
    for r, c, _v in obs:
        d = math.hypot(c * res - ax, r * res - ay)
        assert d <= tol, f"cell ({r},{c}) at {d:.3f} m exceeds range {max_range} m"


def test_symmetric_point_reflection():
    """On an all-free map, marked cells are symmetric about the agent.

    Because beams are cast at angle i and i+180 deg (n_beams even) and the agent
    sits at a cell centre, the set of touched cells must be invariant under
    180-degree point reflection through the agent's cell.
    """
    size = 81
    gt = _empty_map(size)
    res = 0.25
    r0 = c0 = size // 2
    pose = (c0 * res, r0 * res, 0.0)

    obs = cast_rays(gt, pose, n_beams=360, max_range=6.0, resolution=res)
    marked = {(r, c) for r, c, _ in obs}
    mirrored = {(2 * r0 - r, 2 * c0 - c) for (r, c) in marked}
    assert marked == mirrored


def test_wall_marked_occupied_and_blocks():
    """A ray hitting a wall marks it OCCUPIED and does not see past it."""
    size = 40
    gt = _empty_map(size)
    # Vertical wall column to the right of the agent.
    wall_col = 25
    gt[:, wall_col] = GT_WALL
    res = 0.25
    r0 = size // 2
    c0 = 10
    pose = (c0 * res, r0 * res, 0.0)

    obs = cast_rays(gt, pose, n_beams=360, max_range=20.0, resolution=res)
    # The wall cell straight ahead (same row) must be observed OCCUPIED.
    hits = {(r, c): v for r, c, v in obs}
    assert hits.get((r0, wall_col)) == OCCUPIED
    # Nothing on the agent's row beyond the wall should be marked at all.
    for (r, c), _v in hits.items():
        if r == r0 and c > wall_col:
            raise AssertionError(f"saw past wall at ({r},{c})")


def test_agent_cell_marked_free():
    size = 40
    gt = _empty_map(size)
    res = 0.25
    r0 = c0 = 20
    pose = (c0 * res, r0 * res, 0.0)
    obs = cast_rays(gt, pose, n_beams=36, max_range=5.0, resolution=res)
    assert (r0, c0, FREE) in obs
