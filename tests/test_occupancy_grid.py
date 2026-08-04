"""Tests for the shared occupancy grid, including coverage monotonicity."""

import numpy as np

from src.constants import FREE, GT_FREE, GT_WALL, OCCUPIED, UNKNOWN
from src.mapping.occupancy_grid import OccupancyGrid
from src.world.sensor import cast_rays


def test_starts_all_unknown():
    g = OccupancyGrid(20, 30)
    assert (g.grid == UNKNOWN).all()
    assert g.grid.shape == (20, 30)


def test_coordinate_roundtrip():
    g = OccupancyGrid(50, 50, resolution=0.25)
    for (r, c) in [(0, 0), (10, 20), (49, 49)]:
        x, y = g.grid_to_world(r, c)
        assert g.world_to_grid(x, y) == (r, c)


def test_update_and_obs_count():
    g = OccupancyGrid(10, 10)
    g.update_cell(3, 4, OCCUPIED, agent_id=0)
    g.update_cell(3, 4, OCCUPIED, agent_id=1)
    assert g.grid[3, 4] == OCCUPIED
    assert g.obs_count[3, 4] == 2


def test_coverage_increases_monotonically():
    """Sensing along a walk must never decrease coverage."""
    size = 120
    gt = np.full((size, size), GT_FREE, dtype=np.int8)
    gt[0, :] = gt[-1, :] = gt[:, 0] = gt[:, -1] = GT_WALL
    res = 0.25
    grid = OccupancyGrid(size, size, resolution=res)
    trav = int((gt == GT_FREE).sum())

    coverages = []
    for c in range(10, 110, 5):  # walk across the map in x
        pose = (c * res, (size // 2) * res, 0.0)
        obs = cast_rays(gt, pose, n_beams=180, max_range=8.0, resolution=res)
        grid.apply_observations(obs, agent_id=0)
        coverages.append(grid.get_coverage(trav))

    for prev, nxt in zip(coverages, coverages[1:]):
        assert nxt >= prev - 1e-9, f"coverage decreased: {prev} -> {nxt}"
    assert coverages[-1] > coverages[0]  # we did reveal new area


def test_redundant_coverage_ratio():
    """A cell seen by two agents counts as redundant; one agent does not."""
    g = OccupancyGrid(10, 10)
    # Cell A seen by agents 0 and 1 (redundant); cell B by agent 0 only.
    g.update_cell(1, 1, FREE, agent_id=0)
    g.update_cell(1, 1, FREE, agent_id=1)
    g.update_cell(2, 2, FREE, agent_id=0)
    # 1 of 2 observed cells is multiply-observed -> 0.5.
    assert abs(g.redundant_coverage_ratio() - 0.5) < 1e-9


def test_coverage_denominator_reachable():
    """Coverage uses the provided traversable-cell denominator."""
    g = OccupancyGrid(10, 10)
    g.update_cell(0, 0, FREE, agent_id=0)
    # 1 known cell out of a claimed 4 traversable -> 25%.
    assert abs(g.get_coverage(traversable_count=4) - 25.0) < 1e-9
