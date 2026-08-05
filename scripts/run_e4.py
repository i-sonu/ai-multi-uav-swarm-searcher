"""Experiment E4: Cost-Utility Lambda Weight Sensitivity Sweep (Phase 4).

Sweeps lambda in [0.0, 0.1, 0.5, 1.0, 2.0] for N=2 agents with Hungarian allocation.
Outputs results to results/raw/e4_lambda_sweep.csv.
"""

import multiprocessing
import time
from pathlib import Path

from src.config import REPO_ROOT
from src.experiments.runner import RunConfig, load_heldout_seeds, run_batch


def main():
    print("=" * 70)
    print("Experiment E4: Cost-Utility Lambda Weight Sensitivity Sweep (N=2 Agents)")
    print("=" * 70)

    seeds_dict = load_heldout_seeds()
    map_kinds = ["office", "maze", "open_field", "cluttered"]
    lambda_vals = [0.0, 0.1, 0.5, 1.0, 2.0]

    n_seeds = 2

    configs = []
    for map_kind in map_kinds:
        seeds = seeds_dict[map_kind][:n_seeds]
        for seed in seeds:
            for l_val in lambda_vals:
                cfg = RunConfig(
                    planner="astar",
                    map_kind=map_kind,
                    seed=seed,
                    n_agents=2,
                    allocation_method="hungarian",
                    lambda_val=l_val,
                    size=60,
                    max_steps=500,
                )
                configs.append(cfg)

    print(f"Total E4 runs: {len(configs)}")
    out_dir = REPO_ROOT / "results" / "raw"
    summary_csv = out_dir / "e4_lambda_sweep.csv"
    steps_csv = out_dir / "e4_coverage_over_time.csv"

    n_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Running on {n_workers} parallel workers...")

    t0 = time.perf_counter()
    records = run_batch(configs, summary_csv, steps_csv=steps_csv, n_workers=n_workers, progress=True)
    dt = time.perf_counter() - t0

    print("-" * 70)
    print(f"E4 complete in {dt:.1f}s. Results written to:")
    print(f"  - {summary_csv}")
    print(f"  - {steps_csv}")


if __name__ == "__main__":
    main()
