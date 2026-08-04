"""Tests for frontier detection and clustering."""

import numpy as np

from src.constants import FREE, OCCUPIED, UNKNOWN
from src.frontier.clustering import cluster_frontiers
from src.frontier.detection import find_frontiers


def test_frontier_ring_around_free_block():
    """A free block in unknown space: its border cells are frontiers, not the core."""
    grid = np.full((5, 5), UNKNOWN, dtype=np.int8)
    grid[1:4, 1:4] = FREE  # central 3x3 free region
    mask = find_frontiers(grid)
    # The 8 border cells of the 3x3 touch UNKNOWN; the centre (2,2) does not.
    assert mask.sum() == 8
    assert not mask[2, 2]
    assert mask[1, 1] and mask[3, 3]


def test_no_frontier_when_fully_known():
    """If nothing is UNKNOWN, there are no frontiers."""
    grid = np.full((4, 4), FREE, dtype=np.int8)
    grid[0, 0] = OCCUPIED
    assert find_frontiers(grid).sum() == 0


def test_occupied_cells_are_never_frontiers():
    grid = np.full((3, 3), UNKNOWN, dtype=np.int8)
    grid[1, 1] = OCCUPIED
    # An occupied cell adjacent to unknown must not be a frontier (only FREE can).
    assert not find_frontiers(grid)[1, 1]


def test_clustering_discards_small():
    """Clusters below min_cluster_size are dropped as noise."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True  # isolated single cell -> size 1
    mask[5:8, 5:8] = True  # a 3x3 block -> size 9
    clusters = cluster_frontiers(mask, min_cluster_size=3)
    assert len(clusters) == 1
    assert clusters[0].size == 9


def test_clustering_representative_is_a_frontier_cell():
    """The representative point must be an actual frontier cell in the cluster."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    clusters = cluster_frontiers(mask, min_cluster_size=3)
    assert len(clusters) == 1
    r, c = clusters[0].representative
    assert mask[r, c]


def test_clustering_separates_components():
    """Two disjoint blocks form two clusters."""
    mask = np.zeros((12, 12), dtype=bool)
    mask[0:3, 0:3] = True
    mask[8:11, 8:11] = True
    clusters = cluster_frontiers(mask, min_cluster_size=3)
    assert len(clusters) == 2
