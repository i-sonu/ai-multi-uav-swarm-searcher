"""Experiment E5 — does exploration strategy change detection outcomes?

Same 2-agent runs as E2 (none / greedy / hungarian allocation), but now with M
targets placed and the simulated detector attached. Question: does *how* the team
explores affect *what it finds* — time-to-first-detection, fraction of targets
localised, and localisation error?

Detection is decoupled from planning, so any difference comes purely from where
each strategy sends the agents and how fast it covers ground. Held-out seeds,
4 map kinds x 30 seeds.

    python -m scripts.run_e5
    python -m scripts.run_e5 --seeds 3
"""

from __future__ import annotations

import argparse
import time

from src.experiments.runner import RunConfig, load_heldout_seeds, run_batch

METHODS = ("none", "greedy", "hungarian")
MAPS = ("office", "maze", "open_field", "cluttered")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Experiment E5 (strategy vs detection)")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--max-steps", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-agents", type=int, default=2)
    ap.add_argument("--n-targets", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=None, help="use only first N seeds/kind (smoke test)")
    ap.add_argument("--out", default="results/raw/e5_summary.csv")
    ap.add_argument("--steps-out", default="results/raw/e5_steps.csv")
    args = ap.parse_args()

    heldout = load_heldout_seeds()
    configs: list[RunConfig] = []
    for map_kind in MAPS:
        seeds = heldout[map_kind]
        if args.seeds is not None:
            seeds = seeds[: args.seeds]
        for seed in seeds:
            for method in METHODS:
                configs.append(
                    RunConfig(
                        planner="astar", map_kind=map_kind, seed=seed,
                        n_agents=args.n_agents, size=args.size, max_steps=args.max_steps,
                        method=method, with_targets=True, n_targets=args.n_targets,
                    )
                )

    print(f"E5: {len(configs)} runs | N={args.n_agents} targets={args.n_targets} "
          f"size={args.size} workers={args.workers}")
    t0 = time.perf_counter()
    run_batch(configs, args.out, steps_csv=args.steps_out, n_workers=args.workers, progress=True)
    print(f"E5 done in {time.perf_counter() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
