"""Experiment E1 — planner comparison: A* vs BFS vs DFS vs UCS.

Sweeps 4 planners x 4 map kinds x 30 held-out seeds (480 runs) and writes tidy
CSVs to results/raw/. Single-agent (Phase 3).

Scale note: runs on a 96x96 (24 m) grid so the 480-run sweep is tractable — a
full 240x240 maze needs ~90k steps/run, which is infeasible x480. The
planner-comparison conclusions (nodes expanded, path length, planning time) are
scale-invariant; headline single-agent coverage at 240 is shown by demo_phase2.

    python -m scripts.run_e1                 # full sweep (parallel)
    python -m scripts.run_e1 --seeds 3       # quick smoke: first 3 seeds/kind
"""

from __future__ import annotations

import argparse
import time

from src.experiments.runner import RunConfig, load_heldout_seeds, run_batch

PLANNERS = ("astar", "bfs", "dfs", "ucs")
MAPS = ("office", "maze", "open_field", "cluttered")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Experiment E1 (planner comparison)")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--max-steps", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=None, help="use only first N seeds/kind (smoke test)")
    ap.add_argument("--out", default="results/raw/e1_summary.csv")
    ap.add_argument("--steps-out", default="results/raw/e1_steps.csv")
    args = ap.parse_args()

    heldout = load_heldout_seeds()
    configs: list[RunConfig] = []
    for map_kind in MAPS:
        seeds = heldout[map_kind]
        if args.seeds is not None:
            seeds = seeds[: args.seeds]
        for seed in seeds:
            for planner in PLANNERS:
                configs.append(
                    RunConfig(
                        planner=planner, map_kind=map_kind, seed=seed,
                        n_agents=1, size=args.size, max_steps=args.max_steps,
                    )
                )

    print(f"E1: {len(configs)} runs | size={args.size} max_steps={args.max_steps} workers={args.workers}")
    t0 = time.perf_counter()
    run_batch(configs, args.out, steps_csv=args.steps_out, n_workers=args.workers, progress=True)
    print(f"E1 done in {time.perf_counter() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
