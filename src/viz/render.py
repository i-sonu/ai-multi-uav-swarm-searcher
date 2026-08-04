"""Visualisation of the occupancy grid and agents.

Renders the belief map (grey = unknown, white = free, black = occupied) with
agent markers and their current paths overlaid. Supports both a live
``matplotlib`` animation and saving frames to disk for headless runs / reports.

Kept intentionally dependency-light: matplotlib only, imported lazily so that
``import src`` and headless experiment code never pull in a GUI backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.constants import FREE, OCCUPIED, UNKNOWN

# Distinct colours for up to a handful of agents.
_AGENT_COLORS = ["#e6194B", "#4363d8", "#3cb44b", "#f58231", "#911eb4", "#42d4f4"]


def grid_to_rgb(grid: np.ndarray) -> np.ndarray:
    """Map an occupancy grid to an ``(H, W, 3)`` float RGB image in [0, 1]."""
    rgb = np.empty((*grid.shape, 3), dtype=float)
    rgb[grid == UNKNOWN] = (0.50, 0.50, 0.50)  # grey
    rgb[grid == FREE] = (1.00, 1.00, 1.00)  # white
    rgb[grid == OCCUPIED] = (0.00, 0.00, 0.00)  # black
    return rgb


class Renderer:
    """Draws frames of an exploration run onto a matplotlib axis."""

    def __init__(self, resolution: float = 0.25, figsize: tuple[float, float] = (7, 7)):
        import matplotlib.pyplot as plt

        self.resolution = float(resolution)
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self._im = None

    # --------------------------------------------------------------------- #
    def draw(self, grid: np.ndarray, agents: Sequence = (), title: str | None = None) -> None:
        """Draw one frame: the grid, each agent, and its remaining path.

        ``agents`` is any sequence of objects exposing ``.x``, ``.y``, ``.id``
        and (optionally) ``.path`` / ``.path_idx`` — i.e. ``Agent`` instances.
        """
        rgb = grid_to_rgb(grid)
        if self._im is None:
            # origin='upper' so row 0 is at the top; extent maps cells->metres.
            h, w = grid.shape
            self._im = self.ax.imshow(
                rgb, origin="upper", extent=(0, w * self.resolution, h * self.resolution, 0)
            )
        else:
            self._im.set_data(rgb)

        # Clear previous agent/path overlays (keep the base image).
        for artist in list(self.ax.lines) + list(self.ax.collections):
            artist.remove()

        for a in agents:
            color = _AGENT_COLORS[a.id % len(_AGENT_COLORS)]
            # Remaining path as a thin line in world coordinates.
            path = getattr(a, "path", None)
            idx = getattr(a, "path_idx", 0)
            if path:
                remaining = path[idx:]
                if remaining:
                    ys = [r * self.resolution for r, _ in remaining]
                    xs = [c * self.resolution for _, c in remaining]
                    self.ax.plot(xs, ys, "-", color=color, linewidth=1.0, alpha=0.7)
            # Agent marker.
            self.ax.plot(a.x, a.y, "o", color=color, markersize=8, markeredgecolor="k")

        if title:
            self.ax.set_title(title)

    # --------------------------------------------------------------------- #
    def animate(
        self,
        update: Callable[[int], tuple[np.ndarray, Sequence]],
        frames: int,
        interval_ms: int = 100,
        save_path: str | Path | None = None,
    ):
        """Run a live animation.

        ``update(frame_index)`` must advance the simulation by one step and
        return ``(grid, agents)`` for that frame. If ``save_path`` is given, the
        animation is also written to that file (``.mp4`` or ``.gif``).
        """
        import matplotlib.animation as animation

        def _step(i):
            grid, agents = update(i)
            self.draw(grid, agents, title=f"step {i}")
            return []

        anim = animation.FuncAnimation(
            self.fig, _step, frames=frames, interval=interval_ms, blit=False, repeat=False
        )
        if save_path is not None:
            save_path = Path(save_path)
            writer = "pillow" if save_path.suffix == ".gif" else "ffmpeg"
            anim.save(str(save_path), writer=writer)
        return anim

    def save_frame(self, path: str | Path) -> None:
        """Save the current figure to ``path`` (creates parent dirs)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, bbox_inches="tight", dpi=100)

    def show(self) -> None:
        import matplotlib.pyplot as plt

        plt.show()
