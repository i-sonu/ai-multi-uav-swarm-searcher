"""Tests for ground-truth map generation."""

import numpy as np

from src.constants import GT_FREE, GT_WALL
from src.world.map_generator import MAP_KINDS, generate_map


def test_determinism_same_seed():
    """Same (kind, size, seed) must produce byte-identical maps."""
    for kind in MAP_KINDS:
        a = generate_map(kind, 60, seed=7)
        b = generate_map(kind, 60, seed=7)
        assert np.array_equal(a, b), f"{kind} not deterministic"


def test_different_seeds_differ():
    """Different seeds should (almost surely) produce different maps."""
    a = generate_map("cluttered", 60, seed=1)
    b = generate_map("cluttered", 60, seed=2)
    assert not np.array_equal(a, b)


def test_border_wall_present():
    """The outer ring must always be wall."""
    for kind in MAP_KINDS:
        g = generate_map(kind, 50, seed=3)
        assert (g[0, :] == GT_WALL).all()
        assert (g[-1, :] == GT_WALL).all()
        assert (g[:, 0] == GT_WALL).all()
        assert (g[:, -1] == GT_WALL).all()


def test_values_are_binary():
    for kind in MAP_KINDS:
        g = generate_map(kind, 40, seed=0)
        assert set(np.unique(g)).issubset({GT_FREE, GT_WALL})


def test_has_free_space():
    """A generated map must contain traversable cells to explore."""
    for kind in MAP_KINDS:
        g = generate_map(kind, 60, seed=5)
        assert (g == GT_FREE).any(), f"{kind} has no free space"
