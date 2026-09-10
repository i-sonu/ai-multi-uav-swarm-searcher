"""Ground-truth targets to be found during exploration (Phase 5, half B).

A ``Target`` is a person/object the drones are searching for. Targets are placed
at random *reachable free* cells (so the search can actually reach and see them)
and are part of the simulated world's ground truth — the drones do not know where
they are; the detector must find them and ``registration`` must record them.

Placement is deterministic given a seed (CLAUDE.md §3: every experiment is
reproducible from its seed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Target:
    """A ground-truth target at grid cell ``(r, c)`` (its world position is the
    cell centre). ``cls`` is the object class the detector reports."""

    id: int
    r: int
    c: int
    cls: str = "person"

    def world_xy(self, resolution: float) -> tuple[float, float]:
        """World ``(x, y)`` metres of this target (cell centre; x=col, y=row)."""
        return (self.c * resolution, self.r * resolution)


def place_targets(
    n: int,
    seed: int,
    *,
    reachable_mask: np.ndarray,
    resolution: float = 0.25,
    min_separation_m: float = 2.0,
    cls: str = "person",
) -> list[Target]:
    """Place ``n`` targets at distinct reachable free cells, ≥ ``min_separation_m`` apart.

    Args:
        n: number of targets to place.
        seed: RNG seed — same seed places the same targets every run.
        reachable_mask: boolean ``(H, W)`` mask of cells a drone could reach
            (typically the exploration reachable-free mask). Targets are only
            placed on ``True`` cells so every target is in principle findable.
        resolution: metres per cell (for the separation constraint).
        min_separation_m: reject a candidate closer than this to an already-placed
            target, so targets don't clump into one blob the detector sees at once.

    Returns:
        A list of ``n`` targets (fewer only if the reachable area genuinely cannot
        fit ``n`` at the requested separation, in which case the constraint is
        relaxed as a last resort so we always return ``n`` when at all possible).
    """
    rng = np.random.default_rng(seed)
    candidates = np.argwhere(reachable_mask)
    if len(candidates) == 0:
        return []

    min_sep_cells = min_separation_m / resolution
    # Shuffle candidate order deterministically, then greedily accept spaced ones.
    order = rng.permutation(len(candidates))

    def _pick(min_sep: float) -> list[Target]:
        chosen: list[tuple[int, int]] = []
        for idx in order:
            r, c = int(candidates[idx, 0]), int(candidates[idx, 1])
            if all((r - pr) ** 2 + (c - pc) ** 2 >= min_sep ** 2 for pr, pc in chosen):
                chosen.append((r, c))
                if len(chosen) == n:
                    break
        return [Target(i, r, c, cls) for i, (r, c) in enumerate(chosen)]

    targets = _pick(min_sep_cells)
    if len(targets) < n:  # area too tight for the separation — relax it to fit n
        targets = _pick(0.0)
    return targets
