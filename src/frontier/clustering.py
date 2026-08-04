"""Frontier clustering.

Raw frontier cells come in connected strips along the boundary of known space.
Treating every cell as its own goal would be wasteful and noisy, so we group
adjacent frontier cells into clusters (connected components) and reduce each to a
single candidate goal (CLAUDE.md Phase 2.2).

Clusters smaller than ``min_cluster_size`` are discarded as sensor noise.

Each cluster yields two points:
    centroid        — mean (row, col) of its cells (the spec's ``(K, 2)`` output).
    representative  — the actual frontier cell nearest the centroid. This is what
                      the planner targets, because the centroid itself can land
                      on a non-frontier or non-traversable cell (e.g. the mean of
                      a C-shaped strip falls in the hollow).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Cluster:
    centroid: tuple[float, float]
    representative: tuple[int, int]
    size: int


# 8-connectivity so diagonally-touching frontier cells join the same cluster.
_STRUCT = np.ones((3, 3), dtype=int)


def cluster_frontiers(mask: np.ndarray, min_cluster_size: int = 3) -> list[Cluster]:
    """Cluster a boolean frontier ``mask`` into a list of ``Cluster`` objects."""
    labels, n = ndimage.label(mask, structure=_STRUCT)
    clusters: list[Cluster] = []
    for label in range(1, n + 1):
        cells = np.argwhere(labels == label)  # (m, 2) rows of (row, col)
        if len(cells) < min_cluster_size:
            continue
        centroid = cells.mean(axis=0)  # (row, col) float
        # Representative = member cell closest to the centroid.
        d2 = ((cells - centroid) ** 2).sum(axis=1)
        rep = cells[int(np.argmin(d2))]
        clusters.append(
            Cluster(
                centroid=(float(centroid[0]), float(centroid[1])),
                representative=(int(rep[0]), int(rep[1])),
                size=int(len(cells)),
            )
        )
    return clusters


def cluster_centroids(mask: np.ndarray, min_cluster_size: int = 3) -> np.ndarray:
    """Convenience: return just the centroids as a ``(K, 2)`` float array.

    Matches the spec's stated return type for Phase 2.2.
    """
    clusters = cluster_frontiers(mask, min_cluster_size)
    if not clusters:
        return np.empty((0, 2), dtype=float)
    return np.array([c.centroid for c in clusters], dtype=float)
