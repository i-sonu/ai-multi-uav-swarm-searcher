"""Experiment E4 — lambda sensitivity (CLAUDE.md Phase 4.8).

The coordinated score is ``utility - lambda * cost``. lambda trades expected
information gain against travel effort. This sweep runs the hungarian allocator
at several lambda values (2 agents) over 4 map kinds x 30 held-out seeds to see
how sensitive time-to-90% and redundant coverage are to the choice.

    python -m scripts.run_e4
    python -m scripts.run_e4 --seeds 3
"""

from __future__ import annotations

import argparse
import time

from src.experiments.runner import RunConfig, load_heldout_seeds, run_batch

LAMBDAS = (0.0, 0.5, 1.0, 2.0, 4.0)
MAPS = ("office", "maze", "open_field", "cluttered")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Experiment E4 (lambda sweep)")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--max-steps", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-agents", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=None, help="use only first N seeds/kind (smoke test)")
    ap.add_argument("--out", default="results/raw/e4_summary.csv")
    ap.add_argument("--steps-out", default="results/raw/e4_steps.csv")
    args = ap.parse_args()

    heldout = load_heldout_seeds()
    configs: list[RunConfig] = []
    for map_kind in MAPS:
        seeds = heldout[map_kind]
        if args.seeds is not None:
            seeds = seeds[: args.seeds]
        for seed in seeds:
            for lam in LAMBDAS:
                configs.append(
                    RunConfig(
                        planner="astar", map_kind=map_kind, seed=seed,
                        n_agents=args.n_agents, size=args.size, max_steps=args.max_steps,
                        method="hungarian", lambda_cost=lam,
                    )
                )

    print(f"E4: {len(configs)} runs | lambdas={LAMBDAS} N={args.n_agents} "
          f"size={args.size} workers={args.workers}")
    t0 = time.perf_counter()
    run_batch(configs, args.out, steps_csv=args.steps_out, n_workers=args.workers, progress=True)
    print(f"E4 done in {time.perf_counter() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
