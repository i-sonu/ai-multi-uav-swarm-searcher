"""Experiment E2: Task Allocation Strategy Ablation (Phase 4).

Compares Uncoordinated Baseline (none/B2) vs Greedy vs Hungarian allocation for N=2 agents.
Outputs results to results/raw/e2_allocation_ablation.csv.
"""

import multiprocessing
import time
from pathlib import Path

from src.config import REPO_ROOT
from src.experiments.runner import RunConfig, load_heldout_seeds, run_batch


def main():
    print("=" * 70)
    print("Experiment E2: Task Allocation Strategy Ablation (N=2 Agents)")
    print("=" * 70)

    seeds_dict = load_heldout_seeds()
    map_kinds = ["office", "maze", "open_field", "cluttered"]
    methods = ["none", "greedy", "hungarian"]

    # Fast benchmark config: 2 seeds per map type, 60x60 grid
    n_seeds = 2

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
                )
                configs.append(cfg)

    print(f"Total E2 runs: {len(configs)}")
    out_dir = REPO_ROOT / "results" / "raw"
    summary_csv = out_dir / "e2_allocation_ablation.csv"
    steps_csv = out_dir / "e2_coverage_over_time.csv"

    n_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Running on {n_workers} parallel workers...")

    t0 = time.perf_counter()
    records = run_batch(configs, summary_csv, steps_csv=steps_csv, n_workers=n_workers, progress=True)
    dt = time.perf_counter() - t0

    print("-" * 70)
    print(f"E2 complete in {dt:.1f}s. Results written to:")
    print(f"  - {summary_csv}")
    print(f"  - {steps_csv}")


if __name__ == "__main__":
    main()
