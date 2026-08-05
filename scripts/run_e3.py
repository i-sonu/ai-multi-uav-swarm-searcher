"""Experiment E3: Swarm Team Size Scaling (Phase 4).

Benchmarks N=1, N=2, and N=3 UAV agents using Hungarian allocation.
Outputs results to results/raw/e3_team_scaling.csv.
"""

import multiprocessing
import time
from pathlib import Path

from src.config import REPO_ROOT
from src.experiments.runner import RunConfig, load_heldout_seeds, run_batch


def main():
    print("=" * 70)
    print("Experiment E3: Swarm Team Size Scaling (N=1, N=2, N=3 Agents)")
    print("=" * 70)

    seeds_dict = load_heldout_seeds()
    map_kinds = ["office", "maze", "open_field", "cluttered"]
    team_sizes = [1, 2, 3]

    n_seeds = 2

    configs = []
    for map_kind in map_kinds:
        seeds = seeds_dict[map_kind][:n_seeds]
        for seed in seeds:
            for n_agents in team_sizes:
                cfg = RunConfig(
                    planner="astar",
                    map_kind=map_kind,
                    seed=seed,
                    n_agents=n_agents,
                    allocation_method="hungarian",
                    lambda_val=0.5,
                    size=60,
                    max_steps=500,
                )
                configs.append(cfg)

    print(f"Total E3 runs: {len(configs)}")
    out_dir = REPO_ROOT / "results" / "raw"
    summary_csv = out_dir / "e3_team_scaling.csv"
    steps_csv = out_dir / "e3_coverage_over_time.csv"

    n_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Running on {n_workers} parallel workers...")

    t0 = time.perf_counter()
    records = run_batch(configs, summary_csv, steps_csv=steps_csv, n_workers=n_workers, progress=True)
    dt = time.perf_counter() - t0

    print("-" * 70)
    print(f"E3 complete in {dt:.1f}s. Results written to:")
    print(f"  - {summary_csv}")
    print(f"  - {steps_csv}")


if __name__ == "__main__":
    main()
