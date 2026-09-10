"""Tests for LaserScan -> occupancy-grid integration (Phase 6 mapping glue)."""

import math

import numpy as np

from src.constants import FREE, OCCUPIED, UNKNOWN
from src.mapping.occupancy_grid import OccupancyGrid
from src.mapping.scan import integrate_scan


def test_single_beam_marks_free_then_occupied():
    grid = OccupancyGrid(20, 20, resolution=0.25)
    # Drone at cell (10,10) -> world (2.5, 2.5). One beam east (+x), hit at 1.0 m.
    integrate_scan(
        grid, pose=(2.5, 2.5, 0.0), ranges=[1.0],
        angle_min=0.0, angle_increment=0.0, range_max=12.0,
    )
    # Hit point world (3.5, 2.5) -> cell (10, 14) OCCUPIED.
    assert grid.grid[10, 14] == OCCUPIED
    # Cells between are FREE, and nothing beyond the hit is touched.
    assert grid.grid[10, 12] == FREE
    assert grid.grid[10, 16] == UNKNOWN


def test_no_return_beam_marks_free_to_max_no_occupied():
    grid = OccupancyGrid(60, 60, resolution=0.25)
    # A beam with infinite range: free out to range_max, never occupied.
    integrate_scan(
        grid, pose=(5.0, 5.0, 0.0), ranges=[math.inf],
        angle_min=0.0, angle_increment=0.0, range_max=3.0,
    )
    assert not np.any(grid.grid == OCCUPIED)
    assert np.any(grid.grid == FREE)


def test_origin_offset_supports_negative_world_coords():
    grid = OccupancyGrid(40, 40, resolution=0.25)
    # World origin shifted so cell (0,0) is at world (-5, -5): a drone at the
    # world origin sits in the middle of the grid.
    origin = (-5.0, -5.0)
    integrate_scan(
        grid, pose=(0.0, 0.0, 0.0), ranges=[1.0],
        angle_min=0.0, angle_increment=0.0, range_max=12.0, origin=origin,
    )
    # Drone cell = round((0 - -5)/0.25) = 20. Hit at world (1,0) -> col 24.
    assert grid.grid[20, 24] == OCCUPIED
    assert grid.grid[20, 22] == FREE


def test_heading_rotates_the_beam():
    grid = OccupancyGrid(20, 20, resolution=0.25)
    # Facing +y (theta = pi/2), a "forward" beam (angle_min 0) should hit north.
    integrate_scan(
        grid, pose=(2.5, 2.5, math.pi / 2), ranges=[1.0],
        angle_min=0.0, angle_increment=0.0, range_max=12.0,
    )
    # Hit world (2.5, 3.5) -> cell (row 14, col 10).
    assert grid.grid[14, 10] == OCCUPIED
