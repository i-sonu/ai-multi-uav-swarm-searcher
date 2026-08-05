"""Headless experiment runner (CLAUDE.md Phase 3 & Phase 4).

Runs any ``(planner, map_kind, seed, n_agents, allocation_method, lambda_val)``
combination with no rendering, collects §5 metrics, and writes tidy CSVs to
``results/raw/``. Supports parallel execution across runs.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.agents.agent import Agent
from src.config import REPO_ROOT
from src.exploration.explorer import center_free_cell, explore_multi, reachable_free_mask
from src.mapping.occupancy_grid import OccupancyGrid
from src.metrics.metrics import (
    coverage_percentage,
    nodes_expanded_stats,
    planning_wallclock_stats,
    time_to_coverage,
    total_path_length,
)
from src.planning.registry import get_planner
from src.world.map_generator import generate_map


@dataclass
class RunConfig:
    planner: str
    map_kind: str
    seed: int
    n_agents: int = 1
    allocation_method: str = "hungarian"
    lambda_val: float = 0.5
    size: int = 240
    resolution: float = 0.25
    n_beams: int = 360
    max_range: float = 12.0
    min_cluster_size: int = 3
    max_steps: int = 20000


@dataclass
class RunRecord:
    planner: str
    map_kind: str
    seed: int
    n_agents: int
    allocation_method: str
    lambda_val: float
    size: int
    reason: str
    steps: int
    final_coverage: float
    time_to_90: object          # int step count, or None if censored
    path_length_m: float
    redundant_ratio: float
    nodes_total: int
    nodes_mean: float
    n_plans: int
    plan_mean_ms: float
    plan_p95_ms: float
    wall_time_s: float


def run_single(cfg: RunConfig) -> tuple[RunRecord, list[float]]:
    """Execute one exploration run; return its summary record and coverage series."""
    t0 = time.perf_counter()
    gt = generate_map(cfg.map_kind, cfg.size, cfg.seed)
    grid = OccupancyGrid(cfg.size, cfg.size, resolution=cfg.resolution)
    start_r, start_c = center_free_cell(gt)

    # Spawn N agents near the start center cell
    agents = []
    for i in range(cfg.n_agents):
        # Shift initial placement slightly so agents do not occupy exact same spot
        dr = i % 2
        dc = i // 2
        ar, ac = start_r + dr, start_c + dc
        if not (0 <= ar < cfg.size and 0 <= ac < cfg.size and gt[ar, ac] == 0):
            ar, ac = start_r, start_c
        agents.append(Agent(i, (ac * cfg.resolution, ar * cfg.resolution, 0.0), resolution=cfg.resolution))

    planner = get_planner(cfg.planner)

    result = explore_multi(
        gt, grid, agents, planner,
        allocation_method=cfg.allocation_method,
        lambda_val=cfg.lambda_val,
        n_beams=cfg.n_beams, max_range=cfg.max_range,
        min_cluster_size=cfg.min_cluster_size, max_steps=cfg.max_steps,
        collect_metrics=True,
    )
    wall = time.perf_counter() - t0

    reachable = reachable_free_mask(gt, (start_r, start_c))
    final_cov = coverage_percentage(grid.grid, reachable)
    node_stats = nodes_expanded_stats(result.plan_nodes)
    wc_stats = planning_wallclock_stats(result.plan_wallclock)

    record = RunRecord(
        planner=cfg.planner,
        map_kind=cfg.map_kind,
        seed=cfg.seed,
        n_agents=cfg.n_agents,
        allocation_method=cfg.allocation_method,
        lambda_val=cfg.lambda_val,
        size=cfg.size,
        reason=result.reason,
        steps=result.steps,
        final_coverage=round(final_cov, 3),
        time_to_90=time_to_coverage(result.coverage_series, 90.0),
        path_length_m=round(total_path_length(agents), 3),
        redundant_ratio=round(grid.redundant_coverage_ratio(), 4),
        nodes_total=node_stats["total"],
        nodes_mean=round(node_stats["mean"], 2),
        n_plans=node_stats["n_calls"],
        plan_mean_ms=round(wc_stats["mean_s"] * 1e3, 4),
        plan_p95_ms=round(wc_stats["p95_s"] * 1e3, 4),
        wall_time_s=round(wall, 2),
    )
    return record, result.coverage_series


# --------------------------------------------------------------------------- #
# Batch execution
# --------------------------------------------------------------------------- #
def _worker(cfg: RunConfig):
    record, series = run_single(cfg)
    return record, series


def run_batch(
    configs: list[RunConfig],
    summary_csv: str | Path,
    steps_csv: str | Path | None = None,
    n_workers: int = 1,
    step_stride: int = 10,
    progress: bool = True,
):
    """Run many configs (optionally in parallel) and write CSV output."""
    summary_csv = Path(summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    results: list[tuple[RunRecord, list[float]]] = []
    if n_workers > 1:
        with Pool(n_workers) as pool:
            for i, out in enumerate(pool.imap_unordered(_worker, configs), 1):
                results.append(out)
                if progress:
                    print(f"  [{i}/{len(configs)}] {out[0].map_kind} {out[0].planner} method={out[0].allocation_method} seed={out[0].seed}")
    else:
        for i, cfg in enumerate(configs, 1):
            out = run_single(cfg)
            results.append(out)
            if progress:
                print(f"  [{i}/{len(configs)}] {out[0].map_kind} {out[0].planner} method={out[0].allocation_method} seed={out[0].seed} "
                      f"cov={out[0].final_coverage:.1f}% steps={out[0].steps}")

    # Summary CSV
    fields = list(asdict(results[0][0]).keys())
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for record, _ in results:
            w.writerow(asdict(record))

    # Per-step coverage CSV
    if steps_csv is not None:
        steps_csv = Path(steps_csv)
        with open(steps_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["planner", "map_kind", "seed", "allocation_method", "step", "coverage"])
            for record, series in results:
                for i in range(0, len(series), step_stride):
                    w.writerow([record.planner, record.map_kind, record.seed, record.allocation_method, i + 1, round(series[i], 3)])
                if series and (len(series) - 1) % step_stride != 0:
                    w.writerow([record.planner, record.map_kind, record.seed, record.allocation_method, len(series), round(series[-1], 3)])

    return [r for r, _ in results]


def load_heldout_seeds() -> dict:
    """Load the frozen held-out seed set."""
    import yaml

    path = REPO_ROOT / "configs" / "experiments" / "heldout_seeds.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
