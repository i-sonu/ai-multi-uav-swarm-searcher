"""Phase 4 demo: multiple drones cooperatively explore a map using Hungarian or Greedy allocation.

Run (live window):
    python -m scripts.demo_phase4 --map office --n-agents 2 --method hungarian

Headless (just print the metrics):
    python -m scripts.demo_phase4 --map office --n-agents 2 --method hungarian --no-show
"""

from __future__ import annotations

import argparse

import numpy as np

from src.agents.agent import Agent
from src.config import load_config
from src.constants import FREE
from src.exploration.explorer import center_free_cell, explore_multi, reachable_free_mask
from src.mapping.occupancy_grid import OccupancyGrid
from src.planning.registry import get_planner
from src.world.map_generator import generate_map


def coverage_report(grid: OccupancyGrid, gt: np.ndarray, start: tuple[int, int]) -> tuple[float, int, int]:
    """Return (explored_free_%, explored_free_cells, reachable_free_cells)."""
    reachable = reachable_free_mask(gt, start)
    denom = int(reachable.sum())
    explored = int(((grid.grid == FREE) & reachable).sum())
    pct = 100.0 * explored / denom if denom else 0.0
    return pct, explored, denom


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Phase 4 Multi-agent cooperative exploration demo")
    ap.add_argument("--map", default="office", choices=("office", "maze", "open_field", "cluttered"))
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    ap.add_argument("--size", type=int, default=cfg["grid"]["cells_per_side"])
    ap.add_argument("--planner", default=cfg["planning"]["planner"], choices=("astar", "bfs", "dfs", "ucs"))
    ap.add_argument("--n-agents", type=int, default=2, help="number of exploration drones")
    ap.add_argument("--method", default="hungarian", choices=("none", "greedy", "hungarian"), help="allocation strategy")
    ap.add_argument("--lambda-val", type=float, default=0.5, help="distance cost discount factor")
    ap.add_argument("--max-steps", type=int, default=None, help="override step limit")
    ap.add_argument("--render-every", type=int, default=10, help="redraw every K steps (live mode)")
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
    start_r, start_c = center_free_cell(gt)

    # Spawn N agents near the start center cell
    agents = []
    for i in range(args.n_agents):
        # Shift initial placement slightly so agents do not occupy exact same spot
        dr = i % 2
        dc = i // 2
        ar, ac = start_r + dr, start_c + dc
        if not (0 <= ar < args.size and 0 <= ac < args.size and gt[ar, ac] == 0):
            ar, ac = start_r, start_c
        agents.append(Agent(i, (ac * res, ar * res, 0.0), resolution=res))

    planner = get_planner(args.planner)

    print(f"map={args.map} seed={args.seed} size={args.size} planner={args.planner}")
    print(f"n_agents={args.n_agents} allocation_method={args.method} lambda={args.lambda_val}")
    print(f"starts near center cell {start_r, start_c}")

    if args.no_show:
        result = explore_multi(
            gt, grid, agents, planner,
            allocation_method=args.method, lambda_val=args.lambda_val,
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
                renderer.draw(grid.grid, agents, title=f"step {step} ({args.planner}) - method: {args.method}")
                plt.pause(0.001)

        result = explore_multi(
            gt, grid, agents, planner,
            allocation_method=args.method, lambda_val=args.lambda_val,
            n_beams=n_beams, max_range=max_range, min_cluster_size=min_cluster,
            max_steps=max_steps, blacklist_after_failures=blacklist_after,
            on_step=on_step,
        )
        renderer.draw(grid.grid, agents, title=f"done: step {result.steps} - method: {args.method}")
        plt.ioff()
        plt.show()

    pct, explored, denom = coverage_report(grid, gt, (start_r, start_c))
    redundancy = grid.redundant_coverage_ratio()
    total_dist = sum(a.distance_travelled for a in agents)

    print("-" * 50)
    print(
        f"terminated: {result.reason} | steps={result.steps} "
        f"goals_reached={result.reached_goals} blacklisted={result.blacklisted}"
    )
    print(f"coverage: {pct:.2f}% of reachable free cells ({explored}/{denom})")
    print(f"redundant coverage ratio: {redundancy:.4f}")
    print(f"total distance travelled: {total_dist:.2f} m")


if __name__ == "__main__":
    main()
