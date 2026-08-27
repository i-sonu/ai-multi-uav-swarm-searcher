"""End-to-end test of the single-agent exploration loop."""

from src.agents.agent import Agent
from src.constants import FREE
from src.exploration.explorer import explore, reachable_free_mask
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


def _run_multi(kind, size, seed, method, max_steps=4000):
    from src.exploration.explorer import explore_multi
    gt = generate_map(kind, size, seed)
    grid = OccupancyGrid(size, size, resolution=0.25)
    import numpy as np
    from src.constants import GT_FREE

    free = np.argwhere(gt == GT_FREE)
    centre = np.array(gt.shape) / 2.0
    r, c = free[int(((free - centre) ** 2).sum(1).argmin())]
    start = (int(r), int(c))

    # Spawn 2 agents
    agents = [
        Agent(0, (c * 0.25, r * 0.25, 0.0), resolution=0.25),
        Agent(1, ((c + 1) * 0.25, r * 0.25, 0.0), resolution=0.25)
    ]

    result = explore_multi(
        gt, grid, agents, astar,
        allocation_method=method,
        n_beams=180, max_range=8.0, min_cluster_size=3, max_steps=max_steps,
    )
    reachable = reachable_free_mask(gt, start)
    explored = int(((grid.grid == FREE) & reachable).sum())
    pct = 100.0 * explored / int(reachable.sum())
    return result, pct, grid


def test_explore_multi_greedy():
    result, pct, grid = _run_multi("office", 50, seed=1, method="greedy")
    assert result.reason == "explored"
    assert pct >= 90.0
    assert 0.0 <= grid.redundant_coverage_ratio() <= 1.0


def test_explore_multi_hungarian():
    result, pct, grid = _run_multi("office", 50, seed=1, method="hungarian")
    assert result.reason == "explored"
    assert pct >= 90.0
    assert 0.0 <= grid.redundant_coverage_ratio() <= 1.0
