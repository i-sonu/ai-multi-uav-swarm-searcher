"""Object detector interface + a simulated stand-in (Phase 5, half B).

The exploration loop calls ``Detector.detect(pose, agent_id, step)`` once per
agent per step and hands the returned detections to ``registration``. This is the
*only* seam between search and perception: a real trained model (plugged in
later) implements the same ``detect`` on real camera frames, and nothing
downstream changes.

``SimulatedDetector`` models the camera as a **forward-facing cone**: a target is
visible if it is within ``range_m``, within ``fov_deg`` of the agent's heading,
and has clear line-of-sight (no wall between agent and target). A visible target
is reported with probability ``recall``, at its true position plus Gaussian
localisation noise. Optional Poisson false positives model detector error. All
randomness flows from one seeded RNG, so runs are reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from src.constants import GT_WALL
from src.perception.targets import Target


@dataclass(frozen=True)
class Detection:
    """One detector output: a reported world position, class, and confidence,
    tagged with the observing agent and simulation step."""

    x: float
    y: float
    cls: str
    confidence: float
    agent_id: int
    step: int


@runtime_checkable
class Detector(Protocol):
    """A detector maps an agent's viewpoint to zero or more detections.

    ``pose`` is ``(x, y, theta)`` in world metres/radians. Implementations may
    ignore any argument they do not need (a real model would take a frame too,
    supplied via its own constructor/state)."""

    def detect(self, pose: tuple[float, float, float], agent_id: int, step: int) -> list[Detection]:
        ...


def _has_line_of_sight(gt: np.ndarray, ax: float, ay: float, tx: float, ty: float, resolution: float) -> bool:
    """True if no wall cell lies strictly between agent ``(ax,ay)`` and target
    ``(tx,ty)`` (world metres). Samples the segment at sub-cell steps — simple and
    ample for detection LOS (the LiDAR uses exact DDA; here we don't need it)."""
    dx, dy = tx - ax, ty - ay
    dist = math.hypot(dx, dy)
    if dist == 0:
        return True
    steps = max(1, int(dist / (resolution * 0.5)))  # ~2 samples per cell
    h, w = gt.shape
    for i in range(1, steps):  # exclude both endpoints (agent cell + target cell)
        x = ax + dx * i / steps
        y = ay + dy * i / steps
        c = int(round(x / resolution))
        r = int(round(y / resolution))
        if 0 <= r < h and 0 <= c < w and gt[r, c] == GT_WALL:
            return False
    return True


class SimulatedDetector:
    """Ground-truth-driven stand-in for a trained detector.

    Deterministic given its ``seed``. Placeholder for the real model, which will
    implement the same ``detect`` signature.
    """

    def __init__(
        self,
        ground_truth: np.ndarray,
        targets: list[Target],
        *,
        seed: int,
        resolution: float = 0.25,
        range_m: float = 6.0,
        fov_deg: float = 90.0,
        recall: float = 0.9,
        localisation_noise_m: float = 0.3,
        false_positive_rate: float = 0.0,
    ):
        self.gt = ground_truth
        self.targets = targets
        self.resolution = float(resolution)
        self.range_m = float(range_m)
        self.half_fov = math.radians(fov_deg) / 2.0
        self.recall = float(recall)
        self.noise = float(localisation_noise_m)
        self.fp_rate = float(false_positive_rate)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    def in_view(self, pose: tuple[float, float, float], target: Target) -> bool:
        """Geometry-only visibility: within range, within the FOV cone, and with
        clear line-of-sight. (Separate from the probabilistic ``recall`` roll so
        it can be unit-tested directly.)"""
        ax, ay, theta = pose
        tx, ty = target.world_xy(self.resolution)
        dx, dy = tx - ax, ty - ay
        dist = math.hypot(dx, dy)
        if dist > self.range_m or dist == 0:
            return dist == 0  # a target on the agent is trivially "in view"
        # Angular difference between heading and the target bearing, wrapped to [-pi, pi].
        bearing = math.atan2(dy, dx)
        diff = (bearing - theta + math.pi) % (2 * math.pi) - math.pi
        if abs(diff) > self.half_fov:
            return False
        return _has_line_of_sight(self.gt, ax, ay, tx, ty, self.resolution)

    def detect(self, pose: tuple[float, float, float], agent_id: int, step: int) -> list[Detection]:
        """Return detections for one agent viewpoint at one step."""
        out: list[Detection] = []
        for t in self.targets:
            if not self.in_view(pose, t):
                continue
            if self.rng.random() > self.recall:
                continue  # missed detection this frame
            tx, ty = t.world_xy(self.resolution)
            nx = tx + self.rng.normal(0.0, self.noise)
            ny = ty + self.rng.normal(0.0, self.noise)
            conf = float(np.clip(self.rng.normal(0.85, 0.08), 0.0, 1.0))
            out.append(Detection(nx, ny, t.cls, conf, agent_id, step))

        # Optional false positives near the agent (Poisson count per frame).
        if self.fp_rate > 0:
            for _ in range(int(self.rng.poisson(self.fp_rate))):
                ang = self.rng.uniform(-self.half_fov, self.half_fov) + pose[2]
                d = self.rng.uniform(0.0, self.range_m)
                fx = pose[0] + d * math.cos(ang)
                fy = pose[1] + d * math.sin(ang)
                conf = float(np.clip(self.rng.normal(0.5, 0.1), 0.0, 1.0))
                out.append(Detection(fx, fy, "person", conf, agent_id, step))
        return out
