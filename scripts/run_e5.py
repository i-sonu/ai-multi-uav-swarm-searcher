"""Experiment E5: Target Detection and Localisation vs. Exploration Strategy (Phase 5).

Compares Target Recall, Mean Localisation Error, and Time-to-First-Detection
across Uncoordinated Baseline (none), Greedy, and Hungarian allocation strategies.
Outputs results to results/raw/e5_target_registration.csv.
"""

import multiprocessing
import time
from pathlib import Path

from src.config import REPO_ROOT
from src.experiments.runner import RunConfig, load_heldout_seeds, run_batch


def main():
    print("=" * 70)
    print("Experiment E5: Target Registration vs. Exploration Strategy (N=2 Agents)")
    print("=" * 70)

    seeds_dict = load_heldout_seeds()
    map_kinds = ["office", "maze", "open_field", "cluttered"]
    methods = ["none", "greedy", "hungarian"]

    # Fast benchmark config matching E2/E3: 2 seeds per map, size 60, 10 targets
    n_seeds = 2
    n_targets = 10

    configs = []
    for map_kind in map_kinds:
        seeds = seeds_dict[map_kind][:n_seeds]
        for seed in seeds:
            for method in methods:
                cfg = RunConfig(
                    planner="astar",
                    map_kind=map_kind,
                    seed=seed,
                    n_agents=2,
                    allocation_method=method,
                    lambda_val=0.5,
                    size=60,
                    max_steps=500,
                    n_targets=n_targets,
                )
                configs.append(cfg)

    print(f"Total E5 runs: {len(configs)}")
    out_dir = REPO_ROOT / "results" / "raw"
    summary_csv = out_dir / "e5_target_registration.csv"

    n_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Running on {n_workers} parallel workers...")

    t0 = time.perf_counter()
    records = run_batch(configs, summary_csv, n_workers=n_workers, progress=True)
    dt = time.perf_counter() - t0

    print("-" * 70)
    print(f"E5 complete in {dt:.1f}s. Results written to:")
    print(f"  - {summary_csv}")


if __name__ == "__main__":
    main()
