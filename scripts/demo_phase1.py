"""Phase 1 demo: one drone follows a hardcoded waypoint path; the map fills in.

This is the Review-1 visual. There is NO autonomy yet (no frontiers, no
planning) — the agent walks a fixed rectangular patrol while its LiDAR reveals
the environment on the shared occupancy grid.

Run (live window):
    python -m scripts.demo_phase1

Save a GIF instead of / as well as showing:
    python -m scripts.demo_phase1 --save results/raw/phase1_demo.gif --no-show

Options: --map {office,maze,open_field,cluttered}  --seed N  --size CELLS
"""

from __future__ import annotations

import argparse

from src.agents.agent import Agent
from src.config import load_config
from src.mapping.occupancy_grid import OccupancyGrid
from src.world.map_generator import generate_map


def bresenham(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Integer Bresenham line from (r0,c0) to (r1,c1), inclusive of both ends.

    Used here only to densify the hardcoded waypoints into a cell-by-cell path
    so the agent moves one cell per step (this is line *drawing*, not
    pathfinding — A* arrives in Phase 2)."""
    cells: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return cells


def build_patrol_path(size: int) -> list[tuple[int, int]]:
    """A rectangular patrol loop inset from the border, densified to cells."""
    m = max(6, size // 5)  # inset margin
    corners = [
        (m, m),
        (m, size - 1 - m),
        (size - 1 - m, size - 1 - m),
        (size - 1 - m, m),
        (m, m),  # close the loop
    ]
    path: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(corners[:-1], corners[1:]):
        seg = bresenham(r0, c0, r1, c1)
        if path:
            seg = seg[1:]  # avoid duplicating the shared corner
        path.extend(seg)
    return path


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Phase 1 single-agent demo")
    ap.add_argument("--map", default="open_field", choices=("office", "maze", "open_field", "cluttered"))
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    ap.add_argument("--size", type=int, default=160, help="cells per side (smaller = faster demo)")
    ap.add_argument("--save", default=None, help="path to save an animation (.gif or .mp4)")
    ap.add_argument("--no-show", action="store_true", help="do not open a live window")
    args = ap.parse_args()

    res = cfg["grid"]["resolution_m"]
    n_beams = cfg["lidar"]["n_beams"]
    max_range = cfg["lidar"]["max_range_m"]

    # Ground truth (unknown to the agent) and the belief grid it fills in.
    gt = generate_map(args.map, args.size, args.seed)
    grid = OccupancyGrid(args.size, args.size, resolution=res)

    # Agent starts at the first patrol waypoint.
    path = build_patrol_path(args.size)
    r0, c0 = path[0]
    agent = Agent(0, (c0 * res, r0 * res, 0.0), resolution=res)
    agent.set_path(path)

    total_frames = len(path)
    n_free = int((gt == 0).sum())  # ground-truth traversable cells

    def update(_i: int):
        agent.step()
        obs = agent.sense(gt, n_beams=n_beams, max_range=max_range)
        grid.apply_observations(obs, agent_id=agent.id)
        return grid.grid, [agent]

    # Sense once from the start pose before any movement.
    grid.apply_observations(agent.sense(gt, n_beams=n_beams, max_range=max_range), agent_id=0)

    print(f"map={args.map} seed={args.seed} size={args.size} frames={total_frames}")

    want_visual = (not args.no_show) or (args.save is not None)
    if want_visual:
        from src.viz.render import Renderer

        renderer = Renderer(resolution=res)
        renderer.draw(grid.grid, [agent], title="step 0")
        anim = renderer.animate(update, frames=total_frames, interval_ms=40, save_path=args.save)
        if not args.no_show:
            renderer.show()
        del anim  # keep the reference alive until show()/save() consumed it
    else:
        # Headless with no output file: just run the loop for the readout.
        for i in range(total_frames):
            update(i)
    # Bounded progress readout: fraction of ground-truth free cells now seen as
    # FREE. (The official §5 coverage metric — known/traversable, which can
    # exceed 100% because discovered walls count — arrives in Phase 3.)
    explored_free = int(((grid.grid == 0) & (gt == 0)).sum())
    print(f"explored free area: {100.0 * explored_free / n_free:.1f}% of GT-free cells")


if __name__ == "__main__":
    main()
