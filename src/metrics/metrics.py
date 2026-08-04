"""Metric implementations (CLAUDE.md §5).

This module covers the **exploration** metrics used from Phase 3 onward. The
**detection** metrics (mAP@50, time-to-first-detection, localisation error) need
detector output and are added in Phase 5.

Coverage — interpretation note
------------------------------
§5 defines Coverage % as ``(cells != UNKNOWN) / (traversable reachable cells)``.
Taken literally the numerator counts *all* known cells including discovered
walls, so it can exceed 100% (a wall is "known" but is not in the traversable
denominator). That is meaningless to report, so we use the bounded reading:

    coverage = (reachable-free cells that are now known) / (reachable-free cells)

Both share the same reachable-free domain, so coverage is always in [0, 100] and
means exactly "what fraction of the space the agent could reach has been mapped".
A reachable-free cell, once observed, is always marked FREE (it is free in
ground truth), so "known reachable-free" == ``(grid == FREE) & reachable``.
"""

from __future__ import annotations

import numpy as np

from src.constants import FREE


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #
def coverage_percentage(grid: np.ndarray, reachable_free_mask: np.ndarray) -> float:
    """Percentage of reachable-free cells that have been observed. In [0, 100]."""
    denom = int(reachable_free_mask.sum())
    if denom == 0:
        return 0.0
    known_free = int(((grid == FREE) & reachable_free_mask).sum())
    return 100.0 * known_free / denom


def time_to_coverage(coverage_series: list[float], threshold: float = 90.0):
    """First step index at which coverage reaches ``threshold``.

    Returns the 1-based step count, or ``None`` if never reached. Per §5 this is
    **censored** (None), not recorded as a large sentinel — callers must handle
    None explicitly so censored runs don't pollute means.
    """
    for i, cov in enumerate(coverage_series):
        if cov >= threshold:
            return i + 1  # coverage_series[i] is the state after step i+1
    return None


# --------------------------------------------------------------------------- #
# Path length
# --------------------------------------------------------------------------- #
def total_path_length(agents) -> float:
    """Sum of Euclidean distance travelled across agents, in metres."""
    return float(sum(a.distance_travelled for a in agents))


# --------------------------------------------------------------------------- #
# Coordination
# --------------------------------------------------------------------------- #
def redundant_coverage_ratio(occupancy_grid) -> float:
    """(cells observed by >1 agent) / (total observed cells).

    Thin wrapper over ``OccupancyGrid.redundant_coverage_ratio`` so all metric
    definitions live in one place. The key coordination metric (Phase 4).
    """
    return occupancy_grid.redundant_coverage_ratio()


# --------------------------------------------------------------------------- #
# Planner efficiency
# --------------------------------------------------------------------------- #
def nodes_expanded_stats(per_call_nodes: list[int]) -> dict:
    """Summarise nodes-expanded across planner calls: total / mean / max."""
    if not per_call_nodes:
        return {"total": 0, "mean": 0.0, "max": 0, "n_calls": 0}
    arr = np.asarray(per_call_nodes, dtype=float)
    return {
        "total": int(arr.sum()),
        "mean": float(arr.mean()),
        "max": int(arr.max()),
        "n_calls": len(per_call_nodes),
    }


def planning_wallclock_stats(per_call_seconds: list[float]) -> dict:
    """Summarise planner wall-clock: mean and p95 (seconds), per §5."""
    if not per_call_seconds:
        return {"mean_s": 0.0, "p95_s": 0.0, "n_calls": 0}
    arr = np.asarray(per_call_seconds, dtype=float)
    return {
        "mean_s": float(arr.mean()),
        # p95 with linear interpolation; deterministic given the same inputs.
        "p95_s": float(np.percentile(arr, 95)),
        "n_calls": len(per_call_seconds),
    }
