"""Headless experiment runner (CLAUDE.md Phase 3.3).

Runs any ``(planner, map_kind, seed, n_agents)`` combination with no rendering,
collects the §5 metrics, and writes tidy CSVs to ``results/raw/``. Supports
parallel execution across runs.

Determinism: every run is fully determined by its ``(map_kind, seed, size)`` —
the map is seeded, the start pose is a fixed function of the map, and planning is
deterministic. Parallelism therefore never changes results.

Phase 3 is single-agent (``n_agents=1``); the signature already carries
``n_agents`` so Phase 4 can extend it without changing callers.
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
from src.exploration.explorer import center_free_cell, explore, reachable_free_mask
from src.mapping.occupancy_grid import OccupancyGrid
from src.metrics.metrics import (
    coverage_percentage,
    nodes_expanded_stats,
    planning_wallclock_stats,
    time_to_coverage,
)
from src.planning.registry import get_planner
from src.world.map_generator import generate_map


@dataclass
class RunConfig:
    planner: str
    map_kind: str
    seed: int
    n_agents: int = 1
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
    if cfg.n_agents != 1:
        raise NotImplementedError("multi-agent runs arrive in Phase 4")

    t0 = time.perf_counter()
    gt = generate_map(cfg.map_kind, cfg.size, cfg.seed)
    grid = OccupancyGrid(cfg.size, cfg.size, resolution=cfg.resolution)
    start = center_free_cell(gt)
    agent = Agent(0, (start[1] * cfg.resolution, start[0] * cfg.resolution, 0.0), resolution=cfg.resolution)
    planner = get_planner(cfg.planner)

    result = explore(
        gt, grid, agent, planner,
        n_beams=cfg.n_beams, max_range=cfg.max_range,
        min_cluster_size=cfg.min_cluster_size, max_steps=cfg.max_steps,
        collect_metrics=True,
    )
    wall = time.perf_counter() - t0

    reachable = reachable_free_mask(gt, start)
    final_cov = coverage_percentage(grid.grid, reachable)
    node_stats = nodes_expanded_stats(result.plan_nodes)
    wc_stats = planning_wallclock_stats(result.plan_wallclock)

    record = RunRecord(
        planner=cfg.planner,
        map_kind=cfg.map_kind,
        seed=cfg.seed,
        n_agents=cfg.n_agents,
        size=cfg.size,
        reason=result.reason,
        steps=result.steps,
        final_coverage=round(final_cov, 3),
        time_to_90=time_to_coverage(result.coverage_series, 90.0),
        path_length_m=round(agent.distance_travelled, 3),
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
    """Run many configs (optionally in parallel) and write CSV output.

    Args:
        summary_csv: one row per run (the §5 metrics).
        steps_csv: optional coverage-over-time, downsampled every ``step_stride``
            steps (for the coverage curves). Skipped if None.
        n_workers: parallel processes (1 = serial, easiest to debug).
        step_stride: downsample factor for the per-step coverage CSV.
    """
    summary_csv = Path(summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    results: list[tuple[RunRecord, list[float]]] = []
    if n_workers > 1:
        with Pool(n_workers) as pool:
            for i, out in enumerate(pool.imap_unordered(_worker, configs), 1):
                results.append(out)
                if progress:
                    print(f"  [{i}/{len(configs)}] {out[0].map_kind} {out[0].planner} seed={out[0].seed}")
    else:
        for i, cfg in enumerate(configs, 1):
            out = run_single(cfg)
            results.append(out)
            if progress:
                print(f"  [{i}/{len(configs)}] {out[0].map_kind} {out[0].planner} seed={out[0].seed} "
                      f"cov={out[0].final_coverage:.1f}% steps={out[0].steps}")

    # Summary CSV.
    fields = list(asdict(results[0][0]).keys())
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for record, _ in results:
            w.writerow(asdict(record))

    # Per-step coverage CSV (downsampled).
    if steps_csv is not None:
        steps_csv = Path(steps_csv)
        with open(steps_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["planner", "map_kind", "seed", "step", "coverage"])
            for record, series in results:
                for i in range(0, len(series), step_stride):
                    w.writerow([record.planner, record.map_kind, record.seed, i + 1, round(series[i], 3)])
                # Always include the final point so curves end at the true value.
                if series and (len(series) - 1) % step_stride != 0:
                    w.writerow([record.planner, record.map_kind, record.seed, len(series), round(series[-1], 3)])

    return [r for r, _ in results]


def load_heldout_seeds() -> dict:
    """Load the frozen held-out seed set."""
    import yaml

    path = REPO_ROOT / "configs" / "experiments" / "heldout_seeds.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
