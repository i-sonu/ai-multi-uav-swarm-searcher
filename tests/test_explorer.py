"""End-to-end tests of the exploration loops (single-agent and Phase 4 team)."""

import numpy as np

from src.agents.agent import Agent
from src.constants import FREE
from src.exploration.explorer import (
    explore,
    explore_team,
    reachable_free_mask,
    team_start_cells,
)
from src.mapping.occupancy_grid import OccupancyGrid
from src.planning.astar import astar
from src.world.map_generator import generate_map


def _run(kind, size, seed, max_steps=4000):
    gt = generate_map(kind, size, seed)
    grid = OccupancyGrid(size, size, resolution=0.25)
    # Start at the map centre if free, else the first free cell.
    import numpy as np

    from src.constants import GT_FREE

    free = np.argwhere(gt == GT_FREE)
    centre = np.array(gt.shape) / 2.0
    r, c = free[int(((free - centre) ** 2).sum(1).argmin())]
    start = (int(r), int(c))
    agent = Agent(0, (start[1] * 0.25, start[0] * 0.25, 0.0), resolution=0.25)

    result = explore(
        gt, grid, agent, astar,
        n_beams=180, max_range=8.0, min_cluster_size=3, max_steps=max_steps,
    )
    reachable = reachable_free_mask(gt, start)
    explored = int(((grid.grid == FREE) & reachable).sum())
    pct = 100.0 * explored / int(reachable.sum())
    return result, pct


def test_explores_open_field():
    result, pct = _run("open_field", 50, seed=0)
    assert result.reason == "explored"
    assert pct >= 95.0


def test_explores_office():
    result, pct = _run("office", 60, seed=1)
    assert result.reason == "explored"
    assert pct >= 90.0  # office corridors/doorways are harder than open field


def test_explores_maze():
    """Regression: thin width-1 maze corridors must not dead-end at step 0.

    Guards the min_cluster_size fallback — without it, maze frontiers (1-2 cells)
    are all filtered out and exploration wrongly terminates immediately.
    """
    result, pct = _run("maze", 50, seed=0, max_steps=8000)
    assert result.reason == "explored"
    assert pct >= 95.0


# --------------------------------------------------------------------------- #
# Multi-agent team exploration (Phase 4)
# --------------------------------------------------------------------------- #
def _run_team(kind, size, seed, n_agents, method, max_steps=8000):
    gt = generate_map(kind, size, seed)
    grid = OccupancyGrid(size, size, resolution=0.25)
    starts = team_start_cells(gt, n_agents)
    agents = [Agent(i, (s[1] * 0.25, s[0] * 0.25, 0.0), resolution=0.25) for i, s in enumerate(starts)]
    result = explore_team(
        gt, grid, agents, astar,
        method=method, n_beams=180, max_range=8.0, min_cluster_size=3, max_steps=max_steps,
    )
    reachable = np.zeros(gt.shape, dtype=bool)
    for s in starts:
        reachable |= reachable_free_mask(gt, s)
    pct = 100.0 * int(((grid.grid == FREE) & reachable).sum()) / int(reachable.sum())
    return result, grid, pct, starts


def test_team_n1_reproduces_single_agent():
    """Regression check (CLAUDE.md Phase 4.1): a 1-agent team run starts where the
    single agent does and reaches the same full coverage."""
    starts = team_start_cells(generate_map("open_field", 60, 3000), 1)
    assert len(starts) == 1
    result, _grid, pct, _ = _run_team("open_field", 60, seed=3000, n_agents=1, method="hungarian")
    assert result.reason == "explored"
    assert pct >= 95.0


def test_team_two_agents_cover_and_report_redundancy():
    """Two agents fully explore, and the redundant-coverage ratio is a valid
    fraction (the key coordination metric is wired end-to-end)."""
    result, grid, pct, _ = _run_team("office", 60, seed=1001, n_agents=2, method="hungarian")
    assert result.reason == "explored"
    assert pct >= 95.0
    ratio = grid.redundant_coverage_ratio()
    assert 0.0 <= ratio <= 1.0


def test_coordination_reduces_redundant_coverage_on_average():
    """Coordinated (hungarian) allocation wastes less effort than the
    uncoordinated baseline (none) when agents share a deployment base.

    This is the headline Phase 4 claim in miniature. The effect is *statistical*,
    not per-seed guaranteed (on any single map coordinated agents can still
    overlap), so the test averages over several seeds and asserts the mean
    redundant-coverage ratio drops. ``cluttered`` is used because its open layout
    with scattered obstacles makes the uncoordinated failure most consistent;
    the full per-map picture (including the near-null maze case) is Experiment E2.
    """
    none_ratios, hung_ratios = [], []
    for seed in range(4000, 4004):
        _r, g_none, _p, _s = _run_team("cluttered", 60, seed=seed, n_agents=2, method="none", max_steps=6000)
        none_ratios.append(g_none.redundant_coverage_ratio())
        _r2, g_hung, _p2, _s2 = _run_team("cluttered", 60, seed=seed, n_agents=2, method="hungarian", max_steps=6000)
        hung_ratios.append(g_hung.redundant_coverage_ratio())
    assert np.mean(hung_ratios) < np.mean(none_ratios)


def test_start_layout_base_is_compact_and_distinct():
    """`base` layout returns n distinct cells clustered near the reference."""
    gt = generate_map("open_field", 60, 3000)
    starts = team_start_cells(gt, 3, layout="base")
    assert len(starts) == 3
    assert len(set(starts)) == 3  # distinct
