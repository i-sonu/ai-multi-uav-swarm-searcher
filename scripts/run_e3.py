"""Experiment E3 — team size (CLAUDE.md Phase 4.7).

Runs the coordinated (hungarian) explorer with N = 1, 2, 3 agents over 4 map
kinds x 30 held-out seeds. The comparison of interest is at *matched total flight
time*: N agents each taking S steps spend N*S agent-steps, so speed-up is only
real if coverage-vs-(N*step) improves. The plotting script rescales the x-axis by
N to make that comparison; this runner just records per-run metrics + curves.

    python -m scripts.run_e3
    python -m scripts.run_e3 --seeds 3
"""

from __future__ import annotations

import argparse
import time

from src.experiments.runner import RunConfig, load_heldout_seeds, run_batch

TEAM_SIZES = (1, 2, 3)
MAPS = ("office", "maze", "open_field", "cluttered")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Experiment E3 (team size)")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--max-steps", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=None, help="use only first N seeds/kind (smoke test)")
    ap.add_argument("--out", default="results/raw/e3_summary.csv")
    ap.add_argument("--steps-out", default="results/raw/e3_steps.csv")
    args = ap.parse_args()

    heldout = load_heldout_seeds()
    configs: list[RunConfig] = []
    for map_kind in MAPS:
        seeds = heldout[map_kind]
        if args.seeds is not None:
            seeds = seeds[: args.seeds]
        for seed in seeds:
            for n in TEAM_SIZES:
                configs.append(
                    RunConfig(
                        planner="astar", map_kind=map_kind, seed=seed,
                        n_agents=n, size=args.size, max_steps=args.max_steps,
                        method="hungarian",
                    )
                )

    print(f"E3: {len(configs)} runs | N in {TEAM_SIZES} size={args.size} "
          f"max_steps={args.max_steps} workers={args.workers}")
    t0 = time.perf_counter()
    run_batch(configs, args.out, steps_csv=args.steps_out, n_workers=args.workers, progress=True)
    print(f"E3 done in {time.perf_counter() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
