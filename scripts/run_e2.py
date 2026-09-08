"""Experiment E2 — allocation ablation (CLAUDE.md Phase 4.6).

Compares the three task-allocation strategies for a 2-agent team:
    none       uncoordinated baseline B2 (each agent takes its nearest frontier)
    greedy     greedy score maximisation
    hungarian  globally optimal one-to-one assignment

over 4 map kinds x 30 held-out seeds. Headline metrics: time-to-90%-coverage and
redundant-coverage ratio (the key coordination metric, §5).

    python -m scripts.run_e2                 # full sweep (parallel)
    python -m scripts.run_e2 --seeds 3       # quick smoke: first 3 seeds/kind
"""

from __future__ import annotations

import argparse
import time

from src.experiments.runner import RunConfig, load_heldout_seeds, run_batch

METHODS = ("none", "greedy", "hungarian")
MAPS = ("office", "maze", "open_field", "cluttered")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Experiment E2 (allocation ablation)")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--max-steps", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-agents", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=None, help="use only first N seeds/kind (smoke test)")
    ap.add_argument("--out", default="results/raw/e2_summary.csv")
    ap.add_argument("--steps-out", default="results/raw/e2_steps.csv")
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
                        method=method,
                    )
                )

    print(f"E2: {len(configs)} runs | N={args.n_agents} size={args.size} "
          f"max_steps={args.max_steps} workers={args.workers}")
    t0 = time.perf_counter()
    run_batch(configs, args.out, steps_csv=args.steps_out, n_workers=args.workers, progress=True)
    print(f"E2 done in {time.perf_counter() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
