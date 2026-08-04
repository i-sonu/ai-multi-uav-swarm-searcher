"""Ground-truth environment generation.

Produces the *true* map the drones do not know in advance. Everything here is
deterministic given a seed (CLAUDE.md §3: same seed -> identical results). The
output is a binary array:

    0 = free / traversable   (GT_FREE)
    1 = wall / obstacle      (GT_WALL)

Four environment kinds are supported, each exercising the exploration algorithms
differently:

    office      rooms connected by doorways and corridors
    maze        perfect maze (single connected path, many dead ends)
    open_field  mostly-open space with a few sparse obstacles
    cluttered   open space densely peppered with small obstacles

A solid one-cell border wall is always added so rays terminate and agents cannot
leave the world.
"""

from __future__ import annotations

import numpy as np

from src.constants import GT_FREE, GT_WALL

MAP_KINDS = ("office", "maze", "open_field", "cluttered")


def generate_map(kind: str, size: int, seed: int) -> np.ndarray:
    """Generate a ground-truth map.

    Args:
        kind: one of ``MAP_KINDS``.
        size: side length in cells (square map, ``size`` x ``size``).
        seed: RNG seed for reproducibility.

    Returns:
        ``(size, size)`` int8 array of ``GT_FREE`` (0) / ``GT_WALL`` (1).
    """
    if kind not in MAP_KINDS:
        raise ValueError(f"Unknown map kind {kind!r}; expected one of {MAP_KINDS}")
    if size < 5:
        raise ValueError(f"size must be >= 5, got {size}")

    rng = np.random.default_rng(seed)

    if kind == "office":
        grid = _generate_office(size, rng)
    elif kind == "maze":
        grid = _generate_maze(size, rng)
    elif kind == "open_field":
        grid = _generate_open_field(size, rng)
    else:  # cluttered
        grid = _generate_cluttered(size, rng)

    _add_border(grid)
    return grid.astype(np.int8)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _add_border(grid: np.ndarray) -> None:
    """Set the outermost ring of cells to wall, in place."""
    grid[0, :] = GT_WALL
    grid[-1, :] = GT_WALL
    grid[:, 0] = GT_WALL
    grid[:, -1] = GT_WALL


def _generate_open_field(size: int, rng: np.random.Generator) -> np.ndarray:
    """Mostly free space with a handful of sparse rectangular obstacles."""
    grid = np.full((size, size), GT_FREE, dtype=np.int8)

    # Number of obstacles scales with area but stays sparse (~0.05%).
    n_obstacles = max(3, int(size * size * 0.0005))
    for _ in range(n_obstacles):
        h = int(rng.integers(2, max(3, size // 12)))
        w = int(rng.integers(2, max(3, size // 12)))
        r = int(rng.integers(1, size - h - 1))
        c = int(rng.integers(1, size - w - 1))
        grid[r : r + h, c : c + w] = GT_WALL
    return grid


def _generate_cluttered(size: int, rng: np.random.Generator) -> np.ndarray:
    """Open space densely peppered with small 1-2 cell obstacles.

    Kept below the percolation threshold so the free space stays connected in
    practice; obstacles are placed as isolated blobs rather than a random field.
    """
    grid = np.full((size, size), GT_FREE, dtype=np.int8)

    n_obstacles = max(10, int(size * size * 0.01))
    for _ in range(n_obstacles):
        blob = int(rng.integers(1, 3))  # 1x1 or 2x2
        r = int(rng.integers(1, size - blob - 1))
        c = int(rng.integers(1, size - blob - 1))
        grid[r : r + blob, c : c + blob] = GT_WALL
    return grid


def _generate_office(size: int, rng: np.random.Generator) -> np.ndarray:
    """Rooms separated by walls, connected by doorways.

    Approach: overlay a regular grid of wall lines partitioning the interior
    into rooms, then knock a doorway through every internal wall segment between
    adjacent rooms. Knocking a door through *every* shared wall guarantees the
    whole interior is connected (any room reaches its neighbours), which matters
    because coverage is measured only over reachable cells.
    """
    grid = np.full((size, size), GT_FREE, dtype=np.int8)

    # Choose a room size that yields a few rooms per side.
    room = max(6, size // 6)
    # Interior wall line positions (skip the border, added later).
    wall_rows = list(range(room, size - 1, room))
    wall_cols = list(range(room, size - 1, room))

    for r in wall_rows:
        grid[r, :] = GT_WALL
    for c in wall_cols:
        grid[:, c] = GT_WALL

    door = 2  # doorway width in cells

    # Doorways through horizontal walls (connect the room above/below).
    col_bounds = [0] + wall_cols + [size - 1]
    for r in wall_rows:
        for a, b in zip(col_bounds[:-1], col_bounds[1:]):
            lo, hi = a + 1, b - 1
            if hi - lo < door:
                continue
            d = int(rng.integers(lo, hi - door + 1))
            grid[r, d : d + door] = GT_FREE

    # Doorways through vertical walls (connect the room left/right).
    row_bounds = [0] + wall_rows + [size - 1]
    for c in wall_cols:
        for a, b in zip(row_bounds[:-1], row_bounds[1:]):
            lo, hi = a + 1, b - 1
            if hi - lo < door:
                continue
            d = int(rng.integers(lo, hi - door + 1))
            grid[d : d + door, c] = GT_FREE

    return grid


def _generate_maze(size: int, rng: np.random.Generator) -> np.ndarray:
    """Perfect maze via randomised depth-first search (recursive backtracker).

    The maze is carved on a coarse lattice where cells sit on even coordinates
    and walls on odd coordinates. A "perfect" maze is fully connected with
    exactly one path between any two cells and no loops — a hard case for
    exploration (many dead-end frontiers).
    """
    grid = np.full((size, size), GT_WALL, dtype=np.int8)

    # Carve on the odd-index lattice so walls remain between carved cells.
    # Iterative stack (not recursion) to avoid Python recursion limits on
    # large grids: 240x240 would carve ~14k cells deep.
    start = (1, 1)
    stack = [start]
    grid[start] = GT_FREE
    while stack:
        r, c = stack[-1]
        dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        rng.shuffle(dirs)
        advanced = False
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 1 <= nr < size - 1 and 1 <= nc < size - 1 and grid[nr, nc] == GT_WALL:
                grid[r + dr // 2, c + dc // 2] = GT_FREE
                grid[nr, nc] = GT_FREE
                stack.append((nr, nc))
                advanced = True
                break
        if not advanced:
            stack.pop()

    return grid
