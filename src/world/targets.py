"""Synthetic target generator for search-and-rescue simulation (Task 5.2).

Placess static targets (people and vehicles) in free, reachable areas of the map.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Target:
    id: int
    x: float            # world coordinate x (meters)
    y: float            # world coordinate y (meters)
    class_id: int       # 1 for pedestrian/person, 4 for car/vehicle (VisDrone indices)
    class_name: str     # "person" or "vehicle"


def generate_targets(
    gt: np.ndarray,
    n_targets: int = 10,
    seed: int = 0,
    resolution: float = 0.25,
    reachable_mask: np.ndarray | None = None,
) -> list[Target]:
    """Generate static targets randomly distributed across reachable free cells.

    Args:
        gt: ground-truth binary map (0 = free, 1 = wall).
        n_targets: number of targets to spawn.
        seed: random seed for reproducibility.
        resolution: meters per cell.
        reachable_mask: optional boolean mask of cells reachable by the agents.
    """
    state = np.random.RandomState(seed)

    # Use reachable mask to filter candidate cells, falling back to all free cells
    if reachable_mask is not None:
        candidate_cells = np.argwhere((gt == 0) & reachable_mask)
    else:
        candidate_cells = np.argwhere(gt == 0)

    if len(candidate_cells) == 0:
        return []

    # Choose unique positions for targets
    n_spawn = min(n_targets, len(candidate_cells))
    chosen_indices = state.choice(len(candidate_cells), size=n_spawn, replace=False)

    targets = []
    # VisDrone targets of interest: 1 (pedestrian) and 4 (car)
    classes = [1, 4]
    class_names = {1: "person", 4: "vehicle"}

    for i, idx in enumerate(chosen_indices):
        r, c = candidate_cells[idx]
        # Project cell index to world coordinates (centered in cell)
        x = c * resolution
        y = r * resolution
        class_id = int(state.choice(classes))
        class_name = class_names[class_id]

        targets.append(
            Target(
                id=i,
                x=x,
                y=y,
                class_id=class_id,
                class_name=class_name,
            )
        )

    return targets
