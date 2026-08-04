"""Agent (drone) model.

Deliberately minimal (CLAUDE.md Phase 1.4): no dynamics, no controller. An agent
holds a pose, follows a precomputed path one cell per step, and senses its
surroundings into the shared occupancy grid. Pose is ground truth from the
simulator — there is no SLAM (a deliberate scoping decision, §3).
"""

from __future__ import annotations

import math

import numpy as np

from src.world.sensor import cast_rays


class Agent:
    def __init__(
        self,
        agent_id: int,
        pose: tuple[float, float, float],
        resolution: float = 0.25,
    ):
        """
        Args:
            agent_id: unique small integer id (used for per-cell observation bits).
            pose: initial ``(x, y, theta)`` in world metres / radians.
            resolution: metres per cell (kept so the agent can map its own pose
                to grid cells consistently with the sensor and grid).
        """
        self.id = int(agent_id)
        self.x, self.y, self.theta = pose
        self.resolution = float(resolution)

        # Current planned path as a list of (row, col) grid cells, and an index
        # into it. Empty / exhausted path means the agent stays put.
        self.path: list[tuple[int, int]] = []
        self.path_idx = 0

        # Total Euclidean distance travelled, in metres (for the path-length
        # metric). Accumulated across steps.
        self.distance_travelled = 0.0

    # --------------------------------------------------------------------- #
    @property
    def pose(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.theta)

    def set_path(self, path: list[tuple[int, int]]) -> None:
        """Assign a new path of grid cells and reset progress along it."""
        self.path = list(path)
        self.path_idx = 0

    def has_path(self) -> bool:
        """True if there are still waypoints left to move to."""
        return self.path_idx < len(self.path)

    # --------------------------------------------------------------------- #
    def step(self) -> None:
        """Advance one waypoint along the current path.

        Moves directly to the next cell centre (no interpolation — one grid cell
        per simulation step). Updates pose, heading, and accumulated distance.
        """
        if not self.has_path():
            return
        r, c = self.path[self.path_idx]
        nx = c * self.resolution
        ny = r * self.resolution

        # Accumulate travelled distance and face the direction of travel.
        d = math.hypot(nx - self.x, ny - self.y)
        if d > 0:
            self.theta = math.atan2(ny - self.y, nx - self.x)
        self.distance_travelled += d

        self.x, self.y = nx, ny
        self.path_idx += 1

    def sense(
        self,
        ground_truth: np.ndarray,
        n_beams: int = 360,
        max_range: float = 12.0,
    ) -> list[tuple[int, int, int]]:
        """Cast LiDAR from the current pose; return observations to apply.

        The caller applies the returned ``(row, col, value)`` list to the shared
        occupancy grid (tagged with this agent's id), keeping sensing (here) and
        mapping (the grid) cleanly separated.
        """
        return cast_rays(
            ground_truth,
            self.pose,
            n_beams=n_beams,
            max_range=max_range,
            resolution=self.resolution,
        )
