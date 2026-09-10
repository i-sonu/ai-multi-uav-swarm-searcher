"""Metric implementations (CLAUDE.md §5).

This module covers the **exploration** metrics (Phase 3 onward) and the
simulator-side **detection/registration** metrics (Phase 5, half B):
time-to-first-detection, fraction localised within tolerance, and localisation
error. The learned-detector metrics (mAP@50, etc.) are computed by the training
code in ``perception/`` against a real dataset, not here.

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


# --------------------------------------------------------------------------- #
# Detection / registration (Phase 5, half B)
# --------------------------------------------------------------------------- #
def match_registrations(register, targets, resolution: float, tolerance_m: float):
    """Greedily match register entries to true targets by nearest distance.

    Each true target and each register entry is matched at most once, closest
    pairs first, and only if within ``tolerance_m``. Same-class only.

    Returns ``(matches, unmatched_entries, unmatched_targets)`` where ``matches``
    is a list of ``(entry, target, distance_m)`` — the true positives.
    """
    pairs = []
    for e in register.entries():
        for t in targets:
            if t.cls != e.cls:
                continue
            tx, ty = t.world_xy(resolution)
            d = float(np.hypot(e.x - tx, e.y - ty))
            if d <= tolerance_m:
                pairs.append((d, e, t))
    pairs.sort(key=lambda p: p[0])  # closest first

    matches, used_e, used_t = [], set(), set()
    for d, e, t in pairs:
        if id(e) in used_e or t.id in used_t:
            continue
        used_e.add(id(e)); used_t.add(t.id)
        matches.append((e, t, d))

    unmatched_entries = [e for e in register.entries() if id(e) not in used_e]
    unmatched_targets = [t for t in targets if t.id not in used_t]
    return matches, unmatched_entries, unmatched_targets


def detection_metrics(register, targets, resolution: float, tolerance_m: float = 1.0) -> dict:
    """§5 detection metrics from a finished run's register + ground-truth targets.

    - ``time_to_first_detection``: step of the earliest *true-positive* entry, or
      ``None`` if no target was ever correctly localised (censored, not a large
      sentinel — same convention as time-to-coverage).
    - ``fraction_localised``: true positives / total true targets.
    - ``mean_localisation_error_m`` / ``p95_localisation_error_m``: over true
      positives only (metres between the entry and its matched target).
    - ``precision`` / ``recall`` of the register: TP / (TP+FP), TP / (TP+FN).
    """
    matches, unmatched_entries, unmatched_targets = match_registrations(
        register, targets, resolution, tolerance_m
    )
    tp = len(matches)
    fp = len(unmatched_entries)
    fn = len(unmatched_targets)
    errors = [d for _e, _t, d in matches]
    ttfd = min((e.first_step for e, _t, _d in matches), default=None)

    return {
        "n_targets": len(targets),
        "n_registered": len(register.entries()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "fraction_localised": (tp / len(targets)) if targets else 0.0,
        "time_to_first_detection": ttfd,
        "mean_localisation_error_m": float(np.mean(errors)) if errors else None,
        "p95_localisation_error_m": float(np.percentile(errors, 95)) if errors else None,
        "precision": (tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": (tp / (tp + fn)) if (tp + fn) else 0.0,
    }
