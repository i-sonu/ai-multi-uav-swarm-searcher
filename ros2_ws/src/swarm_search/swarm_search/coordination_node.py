"""Milestone 6d node: coordinated multi-drone exploration in Gazebo.

This is the ROS port of ``src/exploration/explorer.explore_team`` — the Phase 4
primary contribution — with **no algorithm change**. One "brain" node reads the
shared ``/map`` and every drone's odometry, and on a timer:

    frontiers -> cluster -> cost matrix (A*) + utility matrix
              -> allocate(method) DISTINCT goal per drone -> A* path -> cmd_vel

so the drones split the space instead of crowding it. The allocation code
(``src/planning/allocation``: build_cost_matrix / build_utility_matrix /
allocate) and the frontier + A* code are imported unchanged from ``src/``; only
the I/O (map in, cmd_vel out, per-drone poses) is ROS.

Goal persistence carries over from the single-drone node: each drone commits to
its assigned frontier until reached / timed-out / unreachable, and allocation is
recomputed only for drones that have freed up (so a busy drone is not yanked off
its goal — that both matches the Phase 4 re-plan policy and avoids oscillation).
"""

from __future__ import annotations

import math
import os
import sys
import time


def _add_repo_to_path() -> None:
    bases = [os.environ.get("AISWARM_REPO"), os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
    for base in bases:
        p = base
        while p and p != os.path.dirname(p):
            if os.path.isdir(os.path.join(p, "src", "planning")) and os.path.isfile(os.path.join(p, "CLAUDE.md")):
                if p not in sys.path:
                    sys.path.insert(0, p)
                return
            p = os.path.dirname(p)


_add_repo_to_path()

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSDurabilityPolicy, QoSProfile  # noqa: E402

from src.constants import FREE, OCCUPIED, UNKNOWN  # noqa: E402
from src.frontier.clustering import cluster_frontiers  # noqa: E402
from src.frontier.detection import find_frontiers  # noqa: E402
from src.planning.allocation import allocate, build_cost_matrix, build_utility_matrix  # noqa: E402
from src.planning.astar import astar  # noqa: E402


def _yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _msg_to_grid(msg: OccupancyGridMsg) -> np.ndarray:
    arr = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
    g = np.full(arr.shape, UNKNOWN, dtype=np.int8)
    g[arr == 0] = FREE
    g[arr == 100] = OCCUPIED
    return g


class Drone:
    """Per-drone control state (pose in, cmd_vel out, current committed goal)."""

    def __init__(self, node: Node, name: str):
        self.name = name
        self.pose: tuple[float, float, float] | None = None
        self.goal: tuple[int, int] | None = None
        self.path: list[tuple[int, int]] = []
        self.path_idx = 0
        self.goal_since = 0.0
        self.last_path_plan = 0.0
        self.cmd_pub = node.create_publisher(Twist, f"/model/{name}/cmd_vel", 10)
        node.create_subscription(Odometry, f"/model/{name}/odometry", self._on_odom, 10)

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, _yaw_from_quat(p.orientation))


class CoordinationNode(Node):
    def __init__(self) -> None:
        super().__init__("coordination_node")
        self.declare_parameter("drones", "drone1,drone2")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("method", "hungarian")     # none | greedy | hungarian
        self.declare_parameter("lambda_cost", 1.0)
        self.declare_parameter("resolution", 0.25)
        self.declare_parameter("origin_x", -30.0)
        self.declare_parameter("origin_y", -30.0)
        self.declare_parameter("min_cluster_size", 3)
        self.declare_parameter("max_candidates", 12)
        self.declare_parameter("sensor_range_m", 12.0)
        self.declare_parameter("max_speed", 0.7)
        self.declare_parameter("gain", 1.5)
        self.declare_parameter("waypoint_tol", 0.25)
        self.declare_parameter("replan_period", 1.0)
        self.declare_parameter("blacklist_after", 3)
        self.declare_parameter("goal_reach_radius", 0.6)
        self.declare_parameter("goal_timeout", 20.0)
        self.declare_parameter("snap_radius_cells", 4)
        self.declare_parameter("control_period", 0.1)

        self.res = float(self.get_parameter("resolution").value)
        self.origin = (float(self.get_parameter("origin_x").value),
                       float(self.get_parameter("origin_y").value))
        self.method = self.get_parameter("method").value
        self.lambda_cost = float(self.get_parameter("lambda_cost").value)
        self.min_cluster = int(self.get_parameter("min_cluster_size").value)
        self.max_candidates = int(self.get_parameter("max_candidates").value)
        self.sensor_cells = float(self.get_parameter("sensor_range_m").value) / self.res
        self.vmax = float(self.get_parameter("max_speed").value)
        self.gain = float(self.get_parameter("gain").value)
        self.wp_tol = float(self.get_parameter("waypoint_tol").value)
        self.replan_period = float(self.get_parameter("replan_period").value)
        self.blacklist_after = int(self.get_parameter("blacklist_after").value)
        self.reach_radius = float(self.get_parameter("goal_reach_radius").value)
        self.goal_timeout = float(self.get_parameter("goal_timeout").value)
        self.snap_radius = int(self.get_parameter("snap_radius_cells").value)

        names = [d.strip() for d in self.get_parameter("drones").value.split(",") if d.strip()]
        self.drones = [Drone(self, n) for n in names]

        self.grid: np.ndarray | None = None
        self.plan_grid: np.ndarray | None = None
        self.blacklist: set[tuple[int, int]] = set()
        self.failures: dict[tuple[int, int], int] = {}
        self.done = False

        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGridMsg, self.get_parameter("map_topic").value, self._on_map, latched)
        self.create_timer(float(self.get_parameter("control_period").value), self._control)
        self.get_logger().info(
            f"coordination_node up: {len(self.drones)} drones, method={self.method}")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _inflate(g: np.ndarray) -> np.ndarray:
        occ = g == OCCUPIED
        d = occ.copy()
        d[:-1, :] |= occ[1:, :]; d[1:, :] |= occ[:-1, :]
        d[:, :-1] |= occ[:, 1:]; d[:, 1:] |= occ[:, :-1]
        d[:-1, :-1] |= occ[1:, 1:]; d[1:, 1:] |= occ[:-1, :-1]
        d[:-1, 1:] |= occ[1:, :-1]; d[1:, :-1] |= occ[:-1, 1:]
        out = g.copy()
        out[d] = OCCUPIED
        return out

    def _on_map(self, msg: OccupancyGridMsg) -> None:
        self.grid = _msg_to_grid(msg)
        # Inflated grid is the PREFERRED planning grid (keeps the drone off walls,
        # so grazing views don't leak the map). But inflation can seal a drone in
        # a pocket with no reachable frontier; when that happens the planner falls
        # back to the raw grid for that drone (see _plan_to / _allocate).
        self.plan_grid = self._inflate(self.grid)

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return int(round((y - self.origin[1]) / self.res)), int(round((x - self.origin[0]) / self.res))

    def _world(self, r: int, c: int) -> tuple[float, float]:
        return self.origin[0] + c * self.res, self.origin[1] + r * self.res

    def _blacklist_area(self, cell: tuple[int, int], radius: int = 2) -> None:
        """Blacklist a small neighbourhood around a dead/phantom frontier.

        Frontier representatives (and their snapped forms) drift by a cell or two
        between scans, so blacklisting a single cell lets the same pocket keep
        re-appearing one cell over. Blacklisting a 5x5 area retires the whole
        pocket at once, guaranteeing convergence to 'exploration complete'."""
        r, c = cell
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                self.blacklist.add((r + dr, c + dc))

    def _fail(self, goal: tuple[int, int]) -> None:
        self.failures[goal] = self.failures.get(goal, 0) + 1
        if self.failures[goal] >= self.blacklist_after:
            self._blacklist_area(goal)

    def _snap(self, goal: tuple[int, int]) -> tuple[int, int] | None:
        """Rescue a frontier swallowed by wall inflation -> nearest free cell."""
        if self.plan_grid[goal] != OCCUPIED:
            return goal
        gr, gc = goal
        h, w = self.plan_grid.shape
        for rad in range(1, self.snap_radius + 1):
            best, best_d = None, None
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    r, c = gr + dr, gc + dc
                    if 0 <= r < h and 0 <= c < w and self.plan_grid[r, c] == FREE:
                        d = dr * dr + dc * dc
                        if best_d is None or d < best_d:
                            best, best_d = (r, c), d
            if best is not None:
                return best
        return None

    def _plan_grid_with_starts(self, cells: list[tuple[int, int]]) -> np.ndarray:
        """Inflated grid with each drone's own cell forced traversable."""
        pg = self.plan_grid.copy()
        for rc in cells:
            if pg[rc] == OCCUPIED:
                pg[rc] = FREE
        return pg

    def _raw_grid_with_starts(self, cells: list[tuple[int, int]]) -> np.ndarray:
        """Un-inflated grid (solid walls only) — the fallback when inflation traps."""
        pg = self.grid.copy()
        for rc in cells:
            if pg[rc] == OCCUPIED:
                pg[rc] = FREE
        return pg

    # ------------------------------------------------------------------ #
    def _candidate_frontiers(self, busy_goals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Frontier goals available to free drones: snapped, de-blacklisted, and
        not already claimed by a busy drone."""
        mask = find_frontiers(self.grid)
        clusters = cluster_frontiers(mask, self.min_cluster)
        cand = [c.representative for c in clusters if c.representative not in self.blacklist]
        if not cand:  # thin-corridor fallback
            clusters = cluster_frontiers(mask, min_cluster_size=1)
            cand = [c.representative for c in clusters if c.representative not in self.blacklist]

        out = []
        for g in cand:
            s = self._snap(g)
            if s is None or s in self.blacklist:  # filter the SNAPPED cell too
                if s is None:
                    self._fail(g)
                continue
            # Skip frontiers a busy drone is already heading to (within reach radius).
            if any(math.hypot(s[0] - bg[0], s[1] - bg[1]) * self.res <= self.reach_radius
                   for bg in busy_goals):
                continue
            out.append(s)
        return out

    def _allocate(self, free: list[Drone]) -> None:
        """Assign distinct frontiers to the free drones via the Phase 4 allocator."""
        busy_goals = [d.goal for d in self.drones if d.goal is not None]
        cand = self._candidate_frontiers(busy_goals)
        if not cand:
            # Only conclude "complete" once the map is actually populated — at
            # startup the first /map can be all-UNKNOWN (scans not integrated
            # yet), which has no frontiers but is NOT the finished state.
            known_free = int((self.grid == FREE).sum())
            if not busy_goals and not self.done and known_free > 50:
                self.get_logger().info(f"exploration complete: no frontiers remain ({known_free} free cells)")
                self.done = True
            return

        # Cap candidates to the nearest to the free team (bounds A* cost work).
        cells = [self._cell(d.pose[0], d.pose[1]) for d in free]
        if len(cand) > self.max_candidates:
            centre = np.mean(cells, axis=0)
            cand.sort(key=lambda g: (g[0] - centre[0]) ** 2 + (g[1] - centre[1]) ** 2)
            cand = cand[:self.max_candidates]

        # Reachability/cost is judged on the RAW grid (physical truth): inflation
        # is only a path *preference*, never a reason a frontier looks unreachable
        # (that was the 6d trap). UNKNOWN is traversable (optimistic exploration).
        pg_raw = self._raw_grid_with_starts(cells)
        cost = build_cost_matrix(cells, cand, pg_raw, None, True)  # (n_free, K)

        # A candidate unreachable by EVERY free drone (even through unknown) sits
        # behind walls — blacklist it so exploration converges to "complete".
        for j in range(len(cand)):
            if not np.isfinite(cost[:, j]).any():
                self._blacklist_area(cand[j])

        if self.method == "none":
            assignment = allocate(cost, None, method="none")
        else:
            utility = build_utility_matrix(self.grid, cand, self.sensor_cells)
            assignment = allocate(cost, utility, method=self.method, lambda_cost=self.lambda_cost)

        for i, d in enumerate(free):
            j = int(assignment[i])
            if j < 0 or not np.isfinite(cost[i, j]):
                continue
            path = self._plan_to(cells[i], cand[j])
            if path is not None:
                d.goal, d.path, d.path_idx = cand[j], path, 1
                d.goal_since = d.last_path_plan = time.monotonic()
                gx, gy = self._world(*cand[j])
                self.get_logger().info(f"{d.name} -> frontier ({gx:.1f}, {gy:.1f})")
            else:
                self._fail(cand[j])

    def _plan_to(self, start: tuple[int, int], goal: tuple[int, int]) -> list | None:
        """A* path start->goal, preferring the wall-margin (inflated) grid and
        falling back to the raw grid if inflation blocks it. None if unreachable."""
        for pg in (self._plan_grid_with_starts([start]), self._raw_grid_with_starts([start])):
            if pg[goal] == OCCUPIED:
                continue
            plan = astar(pg, start, goal, True)
            if plan.path is not None and len(plan.path) >= 2:
                return plan.path
        return None

    def _replan_path(self, d: Drone) -> None:
        start = self._cell(d.pose[0], d.pose[1])
        path = self._plan_to(start, d.goal)
        if path is not None:
            d.path, d.path_idx = path, 1
        else:
            self._fail(d.goal)
            d.goal = None

    def _drive(self, d: Drone) -> None:
        if d.goal is None or not d.path or d.path_idx >= len(d.path):
            d.cmd_pub.publish(Twist())
            return
        nr, nc = d.path[d.path_idx]
        if self.grid[nr, nc] == OCCUPIED:  # only a REAL wall stops the drive
            d.path = []
            d.cmd_pub.publish(Twist())
            return
        tx, ty = self._world(nr, nc)
        ex, ey = tx - d.pose[0], ty - d.pose[1]
        if math.hypot(ex, ey) <= self.wp_tol:
            d.path_idx += 1
            return
        cmd = Twist()
        cmd.linear.x = max(-self.vmax, min(self.vmax, self.gain * ex))
        cmd.linear.y = max(-self.vmax, min(self.vmax, self.gain * ey))
        d.cmd_pub.publish(cmd)

    # ------------------------------------------------------------------ #
    def _control(self) -> None:
        try:
            self._control_impl()
        except Exception as exc:  # never let a timer exception silently kill spin
            import traceback
            self.get_logger().error(f"_control error: {exc}\n{traceback.format_exc()}")

    def _control_impl(self) -> None:
        if self.grid is None or self.plan_grid is None:
            return
        if any(d.pose is None for d in self.drones):
            return
        now = time.monotonic()

        # 1) Retire goals that are reached / timed out / now inside a wall.
        for d in self.drones:
            if d.goal is None:
                continue
            gx, gy = self._world(*d.goal)
            if math.hypot(gx - d.pose[0], gy - d.pose[1]) <= self.reach_radius:
                # Reached it. If it survives as a frontier next scan it is an
                # unreachable pocket (unknown behind a wall) — blacklist so we
                # don't loop back onto it. Cleared frontiers vanish anyway, so
                # this only ever removes phantoms.
                self._blacklist_area(d.goal)
                d.goal = None
            elif (now - d.goal_since) > self.goal_timeout:
                self._fail(d.goal); d.goal = None
            elif self.plan_grid[d.goal] == OCCUPIED:
                d.goal = None

        # 2) Allocate frontiers to any drones that freed up (distinct assignment).
        free = [d for d in self.drones if d.goal is None]
        if free and not self.done:
            self._allocate(free)

        # 3) Refresh the path of each committed drone as the map sharpens.
        for d in self.drones:
            if d.goal is not None and (not d.path or d.path_idx >= len(d.path)
                                       or (now - d.last_path_plan) > self.replan_period):
                d.last_path_plan = now
                self._replan_path(d)

        # 4) Drive everyone.
        for d in self.drones:
            self._drive(d)


def main() -> None:
    rclpy.init()
    node = CoordinationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
