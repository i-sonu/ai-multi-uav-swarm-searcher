"""Phase 5 demo: a coordinated team explores *and* finds targets.

Same coordinated exploration as Phase 4, now with M hidden targets and the
simulated detector attached. Ground-truth targets are drawn as green squares;
the system's registered estimates as red x marks — the gap between them is the
localisation error.

Run (live window):
    python -m scripts.demo_phase5 --map office --n-targets 8

Headless (print detection metrics only):
    python -m scripts.demo_phase5 --map cluttered --no-show

Options: --method {none,greedy,hungarian}  --n-agents N  --n-targets M
         --map KIND  --seed N  --size CELLS  --recall F  --render-every K
"""

from __future__ import annotations

import argparse

import numpy as np

from src.agents.agent import Agent
from src.config import load_config
from src.constants import FREE
from src.exploration.explorer import explore_team, reachable_free_mask, team_start_cells
from src.mapping.occupancy_grid import OccupancyGrid
from src.metrics.metrics import detection_metrics
from src.perception.detector import SimulatedDetector
from src.perception.registration import TargetRegister
from src.perception.targets import place_targets
from src.planning.registry import get_planner
from src.world.map_generator import generate_map


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Phase 5 exploration + detection demo")
    ap.add_argument("--map", default="office", choices=("office", "maze", "open_field", "cluttered"))
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    ap.add_argument("--size", type=int, default=cfg["grid"]["cells_per_side"])
    ap.add_argument("--n-agents", type=int, default=cfg["agents"]["n_agents"])
    ap.add_argument("--method", default=cfg["allocation"]["method"], choices=("none", "greedy", "hungarian"))
    ap.add_argument("--planner", default=cfg["planning"]["planner"], choices=("astar", "bfs", "dfs", "ucs"))
    ap.add_argument("--n-targets", type=int, default=cfg["perception"]["n_targets"])
    ap.add_argument("--recall", type=float, default=cfg["perception"]["detector"]["recall"])
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--render-every", type=int, default=15)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    res = cfg["grid"]["resolution_m"]
    n_beams = cfg["lidar"]["n_beams"]
    max_range = cfg["lidar"]["max_range_m"]
    min_cluster = cfg["frontier"]["min_cluster_size"]
    max_steps = args.max_steps if args.max_steps is not None else cfg["exploration"]["max_steps"]
    replan_every = cfg["exploration"]["replan_every"]
    cam = cfg["perception"]["camera"]
    dparams = cfg["perception"]["detector"]
    rparams = cfg["perception"]["registration"]

    gt = generate_map(args.map, args.size, args.seed)
    grid = OccupancyGrid(args.size, args.size, resolution=res)
    starts = team_start_cells(gt, args.n_agents)
    agents = [Agent(i, (s[1] * res, s[0] * res, 0.0), resolution=res) for i, s in enumerate(starts)]
    planner = get_planner(args.planner)

    reachable = np.zeros(gt.shape, dtype=bool)
    for s in starts:
        reachable |= reachable_free_mask(gt, s)
    targets = place_targets(args.n_targets, args.seed, reachable_mask=reachable, resolution=res)
    detector = SimulatedDetector(
        gt, targets, seed=args.seed, resolution=res,
        range_m=cam["range_m"], fov_deg=cam["fov_deg"],
        recall=args.recall, localisation_noise_m=dparams["localisation_noise_m"],
        false_positive_rate=dparams["false_positive_rate"],
    )
    register = TargetRegister(dedup_radius_m=rparams["dedup_radius_m"])

    print(f"map={args.map} seed={args.seed} N={args.n_agents} method={args.method} "
          f"targets={len(targets)} recall={args.recall}")

    common = dict(
        method=args.method, n_beams=n_beams, max_range=max_range,
        min_cluster_size=min_cluster, max_steps=max_steps, replan_every=replan_every,
        detector=detector, register=register,
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
                renderer.draw(grid.grid, agents, targets=targets, register=register,
                              title=f"step {step} ({args.method}) — {len(register)} registered")
                plt.pause(0.001)

        result = explore_team(gt, grid, agents, planner, on_step=on_step, **common)
        renderer.draw(grid.grid, agents, targets=targets, register=register,
                      title=f"done: step {result.steps} — {len(register)} registered")
        plt.ioff()
        plt.show()

    explored = int(((grid.grid == FREE) & reachable).sum())
    pct = 100.0 * explored / int(reachable.sum())
    m = detection_metrics(register, targets, resolution=res, tolerance_m=rparams["localisation_tolerance_m"])
    print(f"terminated: {result.reason} | steps={result.steps} | coverage {pct:.1f}%")
    print(f"targets: {m['true_positives']}/{m['n_targets']} localised "
          f"({m['fraction_localised']*100:.0f}%), {m['false_positives']} false positive(s)")
    print(f"time-to-first-detection: {m['time_to_first_detection']} steps")
    if m["mean_localisation_error_m"] is not None:
        print(f"localisation error: mean {m['mean_localisation_error_m']:.2f} m, "
              f"p95 {m['p95_localisation_error_m']:.2f} m")


if __name__ == "__main__":
    main()
