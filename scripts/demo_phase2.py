"""Phase 2 demo: one drone autonomously explores a full map.

Unlike Phase 1 (fixed patrol), the agent now chooses where to go: detect
frontiers -> cluster -> plan with A* (or a baseline) -> move -> sense -> repeat,
until the map is fully explored.

Run (live window):
    python -m scripts.demo_phase2 --map office

Headless (just print the coverage achieved):
    python -m scripts.demo_phase2 --map maze --no-show

Options: --planner {astar,bfs,dfs,ucs}  --seed N  --size CELLS  --render-every K
"""

from __future__ import annotations

import argparse

import numpy as np

from src.agents.agent import Agent
from src.config import load_config
from src.constants import FREE, GT_FREE
from src.exploration.explorer import explore, reachable_free_mask
from src.mapping.occupancy_grid import OccupancyGrid
from src.planning.registry import get_planner
from src.world.map_generator import generate_map


def pick_start(gt: np.ndarray) -> tuple[int, int]:
    """Choose a free start cell nearest the map centre."""
    free = np.argwhere(gt == GT_FREE)
    centre = np.array(gt.shape) / 2.0
    d2 = ((free - centre) ** 2).sum(axis=1)
    r, c = free[int(np.argmin(d2))]
    return int(r), int(c)


def coverage_report(grid: OccupancyGrid, gt: np.ndarray, start: tuple[int, int]) -> tuple[float, int, int]:
    """Return (explored_free_%, explored_free_cells, reachable_free_cells)."""
    reachable = reachable_free_mask(gt, start)
    denom = int(reachable.sum())
    explored = int(((grid.grid == FREE) & reachable).sum())
    pct = 100.0 * explored / denom if denom else 0.0
    return pct, explored, denom


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Phase 2 autonomous exploration demo")
    ap.add_argument("--map", default="office", choices=("office", "maze", "open_field", "cluttered"))
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    ap.add_argument("--size", type=int, default=cfg["grid"]["cells_per_side"])
    ap.add_argument("--planner", default=cfg["planning"]["planner"], choices=("astar", "bfs", "dfs", "ucs"))
    ap.add_argument("--max-steps", type=int, default=None, help="override step limit")
    ap.add_argument("--render-every", type=int, default=15, help="redraw every K steps (live mode)")
    ap.add_argument("--no-show", action="store_true", help="run headless, print coverage only")
    args = ap.parse_args()

    res = cfg["grid"]["resolution_m"]
    n_beams = cfg["lidar"]["n_beams"]
    max_range = cfg["lidar"]["max_range_m"]
    min_cluster = cfg["frontier"]["min_cluster_size"]
    max_steps = args.max_steps if args.max_steps is not None else cfg["exploration"]["max_steps"]
    blacklist_after = cfg["exploration"]["blacklist_after_failures"]

    gt = generate_map(args.map, args.size, args.seed)
    grid = OccupancyGrid(args.size, args.size, resolution=res)
    start = pick_start(gt)
    agent = Agent(0, (start[1] * res, start[0] * res, 0.0), resolution=res)
    planner = get_planner(args.planner)

    print(f"map={args.map} seed={args.seed} size={args.size} planner={args.planner} start={start}")

    if args.no_show:
        result = explore(
            gt, grid, agent, planner,
            n_beams=n_beams, max_range=max_range, min_cluster_size=min_cluster,
            max_steps=max_steps, blacklist_after_failures=blacklist_after,
        )
    else:
        import matplotlib.pyplot as plt

        from src.viz.render import Renderer

        renderer = Renderer(resolution=res)
        plt.ion()

        def on_step(step: int) -> None:
            if step % args.render_every == 0:
                renderer.draw(grid.grid, [agent], title=f"step {step} ({args.planner})")
                plt.pause(0.001)

        result = explore(
            gt, grid, agent, planner,
            n_beams=n_beams, max_range=max_range, min_cluster_size=min_cluster,
            max_steps=max_steps, blacklist_after_failures=blacklist_after,
            on_step=on_step,
        )
        renderer.draw(grid.grid, [agent], title=f"done: step {result.steps}")
        plt.ioff()
        plt.show()

    pct, explored, denom = coverage_report(grid, gt, start)
    print(
        f"terminated: {result.reason} | steps={result.steps} "
        f"goals_reached={result.reached_goals} blacklisted={result.blacklisted}"
    )
    print(f"coverage: {pct:.1f}% of reachable free cells ({explored}/{denom})")
    print(f"path length: {agent.distance_travelled:.1f} m")


if __name__ == "__main__":
    main()
