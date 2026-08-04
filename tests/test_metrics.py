"""Tests for the §5 metric implementations and the experiment runner."""

import numpy as np

from src.constants import FREE, UNKNOWN
from src.metrics.metrics import (
    coverage_percentage,
    nodes_expanded_stats,
    planning_wallclock_stats,
    time_to_coverage,
    total_path_length,
)


def test_coverage_percentage_bounded():
    reachable = np.zeros((5, 5), dtype=bool)
    reachable[1:4, 1:4] = True  # 9 reachable-free cells
    grid = np.full((5, 5), UNKNOWN, dtype=np.int8)
    grid[1:3, 1:3] = FREE  # 4 of them observed
    assert coverage_percentage(grid, reachable) == 100.0 * 4 / 9


def test_coverage_never_exceeds_100():
    """Even if walls are 'known', coverage stays bounded (the whole point)."""
    reachable = np.zeros((3, 3), dtype=bool)
    reachable[1, 1] = True
    grid = np.array([[1, 1, 1], [1, FREE, 1], [1, 1, 1]], dtype=np.int8)  # walls all known
    assert coverage_percentage(grid, reachable) == 100.0


def test_time_to_coverage_and_censoring():
    series = [10.0, 50.0, 89.9, 90.0, 95.0]
    assert time_to_coverage(series, 90.0) == 4  # first step reaching 90
    assert time_to_coverage([10.0, 20.0], 90.0) is None  # never reached -> censored


def test_total_path_length():
    class A:
        def __init__(self, d):
            self.distance_travelled = d

    assert total_path_length([A(3.0), A(4.5)]) == 7.5


def test_nodes_and_wallclock_stats():
    ns = nodes_expanded_stats([10, 20, 30])
    assert ns["total"] == 60 and ns["mean"] == 20.0 and ns["max"] == 30 and ns["n_calls"] == 3
    wc = planning_wallclock_stats([0.001, 0.002, 0.003, 0.004])
    assert wc["n_calls"] == 4 and wc["mean_s"] > 0 and wc["p95_s"] >= wc["mean_s"]
    # Empty inputs must not crash.
    assert nodes_expanded_stats([])["total"] == 0
    assert planning_wallclock_stats([])["p95_s"] == 0.0


def test_runner_determinism():
    """Same config must reproduce identical metrics (CLAUDE.md §3)."""
    from src.experiments.runner import RunConfig, run_single

    cfg = RunConfig(planner="astar", map_kind="open_field", seed=0, size=40, max_steps=2000)
    r1, s1 = run_single(cfg)
    r2, s2 = run_single(cfg)
    # Compare everything except the (noisy) wall-clock fields.
    assert r1.steps == r2.steps
    assert r1.final_coverage == r2.final_coverage
    assert r1.path_length_m == r2.path_length_m
    assert r1.nodes_total == r2.nodes_total
    assert s1 == s2
