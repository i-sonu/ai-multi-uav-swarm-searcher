"""Phase 4 demo: a *coordinated team* of drones explores a map together.

N agents share one occupancy grid and are assigned distinct frontiers each
planning round by the chosen allocation method. Compare, on the same map/seed:

    python -m scripts.demo_phase4 --method none        # uncoordinated (agents overlap)
    python -m scripts.demo_phase4 --method hungarian   # coordinated (agents split up)

Headless (print coverage + the redundant-coverage ratio, the key metric):

    python -m scripts.demo_phase4 --method hungarian --no-show

Options: --method {none,greedy,hungarian}  --n-agents N  --map KIND  --seed N
         --size CELLS  --lambda-cost F  --render-every K
"""

from __future__ import annotations

import argparse

import numpy as np

from src.agents.agent import Agent
from src.config import load_config
from src.constants import FREE
from src.exploration.explorer import explore_team, reachable_free_mask, team_start_cells
from src.mapping.occupancy_grid import OccupancyGrid
from src.planning.registry import get_planner
from src.world.map_generator import generate_map


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Phase 4 coordinated multi-agent exploration demo")
    ap.add_argument("--map", default="office", choices=("office", "maze", "open_field", "cluttered"))
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    ap.add_argument("--size", type=int, default=cfg["grid"]["cells_per_side"])
    ap.add_argument("--n-agents", type=int, default=cfg["agents"]["n_agents"])
    ap.add_argument("--method", default=cfg["allocation"]["method"], choices=("none", "greedy", "hungarian"))
    ap.add_argument("--lambda-cost", type=float, default=cfg["allocation"]["lambda_cost"])
    ap.add_argument("--planner", default=cfg["planning"]["planner"], choices=("astar", "bfs", "dfs", "ucs"))
    ap.add_argument("--layout", default="base", choices=("base", "spread"))
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--render-every", type=int, default=15)
    ap.add_argument("--no-show", action="store_true", help="run headless, print metrics only")
    args = ap.parse_args()

    res = cfg["grid"]["resolution_m"]
    n_beams = cfg["lidar"]["n_beams"]
    max_range = cfg["lidar"]["max_range_m"]
    min_cluster = cfg["frontier"]["min_cluster_size"]
    max_steps = args.max_steps if args.max_steps is not None else cfg["exploration"]["max_steps"]
    replan_every = cfg["exploration"]["replan_every"]

    gt = generate_map(args.map, args.size, args.seed)
    grid = OccupancyGrid(args.size, args.size, resolution=res)
    starts = team_start_cells(gt, args.n_agents, layout=args.layout)
    agents = [Agent(i, (s[1] * res, s[0] * res, 0.0), resolution=res) for i, s in enumerate(starts)]
    planner = get_planner(args.planner)

    print(f"map={args.map} seed={args.seed} size={args.size} N={args.n_agents} "
          f"method={args.method} lambda={args.lambda_cost} starts={starts}")

    common = dict(
        method=args.method, lambda_cost=args.lambda_cost,
        n_beams=n_beams, max_range=max_range, min_cluster_size=min_cluster,
        max_steps=max_steps, replan_every=replan_every,
    )

    if args.no_show:
        result = explore_team(gt, grid, agents, planner, **common)
    else:
        import matplotlib.pyplot as plt

        from src.viz.render import Renderer

        renderer = Renderer(resolution=res)
        plt.ion()

        def on_step(step: int) -> None:
            if step % args.render_every == 0:
                renderer.draw(grid.grid, agents, title=f"step {step} ({args.method}, N={args.n_agents})")
                plt.pause(0.001)

        result = explore_team(gt, grid, agents, planner, on_step=on_step, **common)
        renderer.draw(grid.grid, agents, title=f"done: step {result.steps} ({args.method})")
        plt.ioff()
        plt.show()

    reachable = np.zeros(gt.shape, dtype=bool)
    for s in starts:
        reachable |= reachable_free_mask(gt, s)
    denom = int(reachable.sum())
    explored = int(((grid.grid == FREE) & reachable).sum())
    pct = 100.0 * explored / denom if denom else 0.0
    path_len = sum(a.distance_travelled for a in agents)

    print(f"terminated: {result.reason} | steps={result.steps} goals_reached={result.reached_goals}")
    print(f"coverage: {pct:.1f}% of reachable free cells ({explored}/{denom})")
    print(f"total path length: {path_len:.1f} m (summed over {args.n_agents} agents)")
    print(f"redundant-coverage ratio: {grid.redundant_coverage_ratio():.3f}  (lower = better coordination)")


if __name__ == "__main__":
    main()
