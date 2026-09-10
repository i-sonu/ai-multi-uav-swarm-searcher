"""Milestone 6c node: autonomous single-drone exploration in Gazebo.

This is the Phase 2 exploration loop, ported to ROS with **no algorithm change**.
It subscribes to the shared ``/map`` (built by ``mapping_node`` from real LiDAR)
and the drone odometry, then on a timer:

    frontiers -> cluster -> nearest centroid -> A* -> follow path via cmd_vel

exactly as ``src/exploration/explorer.explore`` does in the pure-Python sim. The
only differences are I/O: the grid arrives as a ``nav_msgs/OccupancyGrid`` instead
of being sensed in-process, and motion is commanded over ``cmd_vel`` instead of
teleporting along the path. Frontier detection, clustering and A* are imported
unchanged from ``src/``.
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
from src.planning.astar import astar  # noqa: E402


def _yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _msg_to_grid(msg: OccupancyGridMsg) -> np.ndarray:
    """nav_msgs/OccupancyGrid (-1/0/100) -> our int8 grid (UNKNOWN/FREE/OCCUPIED)."""
    arr = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
    g = np.full(arr.shape, UNKNOWN, dtype=np.int8)
    g[arr == 0] = FREE
    g[arr == 100] = OCCUPIED
    return g


class ExplorationNode(Node):
    def __init__(self) -> None:
        super().__init__("exploration_node")
        self.declare_parameter("cmd_topic", "/model/drone1/cmd_vel")
        self.declare_parameter("odom_topic", "/model/drone1/odometry")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("resolution", 0.25)
        self.declare_parameter("origin_x", -30.0)
        self.declare_parameter("origin_y", -30.0)
        self.declare_parameter("min_cluster_size", 3)
        self.declare_parameter("max_speed", 0.7)
        self.declare_parameter("gain", 1.5)
        self.declare_parameter("waypoint_tol", 0.25)     # m: advance to next path cell
        self.declare_parameter("replan_period", 1.0)     # s: refresh the PATH to the current goal
        self.declare_parameter("control_period", 0.1)    # s: cmd_vel rate
        self.declare_parameter("blacklist_after", 3)
        self.declare_parameter("goal_reach_radius", 0.6)  # m: current goal counts as reached
        self.declare_parameter("goal_timeout", 20.0)     # s: give up on a goal we can't reach
        self.declare_parameter("snap_radius_cells", 4)   # search radius to rescue a wall-orphaned goal

        self.res = float(self.get_parameter("resolution").value)
        self.origin = (float(self.get_parameter("origin_x").value),
                       float(self.get_parameter("origin_y").value))
        self.min_cluster = int(self.get_parameter("min_cluster_size").value)
        self.vmax = float(self.get_parameter("max_speed").value)
        self.gain = float(self.get_parameter("gain").value)
        self.wp_tol = float(self.get_parameter("waypoint_tol").value)
        self.replan_period = float(self.get_parameter("replan_period").value)
        self.blacklist_after = int(self.get_parameter("blacklist_after").value)
        self.reach_radius = float(self.get_parameter("goal_reach_radius").value)
        self.goal_timeout = float(self.get_parameter("goal_timeout").value)
        self.snap_radius = int(self.get_parameter("snap_radius_cells").value)

        self.grid: np.ndarray | None = None        # raw belief grid (for frontiers)
        self.plan_grid: np.ndarray | None = None    # obstacle-inflated grid (for A*)
        self.pose: tuple[float, float, float] | None = None
        self.path: list[tuple[int, int]] = []
        self.path_idx = 0
        # Goal persistence: commit to one frontier goal until it is reached, times
        # out, or becomes unreachable — re-selecting the nearest every cycle makes
        # the drone dither between two near-equidistant frontiers (oscillation).
        self.goal: tuple[int, int] | None = None
        self.goal_since = 0.0
        self.last_path_plan = 0.0
        self.blacklist: set[tuple[int, int]] = set()
        self.failures: dict[tuple[int, int], int] = {}
        self.done = False

        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGridMsg, self.get_parameter("map_topic").value, self._on_map, latched)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._on_odom, 10)
        self.cmd_pub = self.create_publisher(Twist, self.get_parameter("cmd_topic").value, 10)
        self.create_timer(float(self.get_parameter("control_period").value), self._control)
        self.get_logger().info("exploration_node up: frontier-driven autonomous exploration")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _inflate(g: np.ndarray) -> np.ndarray:
        """Grow OCCUPIED by one cell (8-connected) so paths keep a wall margin.

        The Gazebo drone is a kinematic model with no collision response, so a
        path that hugs a wall can clip through it. Inflating obstacles by one cell
        (a minimal costmap inflation) keeps the planned path off the walls."""
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
        self.plan_grid = self._inflate(self.grid)

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, _yaw_from_quat(p.orientation))

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return int(round((y - self.origin[1]) / self.res)), int(round((x - self.origin[0]) / self.res))

    def _world(self, r: int, c: int) -> tuple[float, float]:
        return self.origin[0] + c * self.res, self.origin[1] + r * self.res

    def _stop(self) -> None:
        self.cmd_pub.publish(Twist())

    # ------------------------------------------------------------------ #
    def _planning_grid(self, start: tuple[int, int]) -> np.ndarray:
        """Inflated grid, but with the drone's own cell forced traversable (it may
        sit within one cell of a wall, which inflation would otherwise seal in)."""
        pg = self.plan_grid
        if pg[start] == OCCUPIED:
            pg = pg.copy()
            pg[start] = FREE
        return pg

    def _snap(self, goal: tuple[int, int], plan_grid: np.ndarray) -> tuple[int, int] | None:
        """Rescue a frontier that fell inside the wall-inflation margin.

        A frontier cell hard against a thin wall becomes OCCUPIED after inflation,
        so A* could never target it and the drone would loop back to it forever.
        Snap it to the nearest non-inflated cell within ``snap_radius`` (spiral
        search); return None if the whole neighbourhood is blocked (then it is
        blacklisted by the caller)."""
        if plan_grid[goal] != OCCUPIED:
            return goal
        gr, gc = goal
        h, w = plan_grid.shape
        for rad in range(1, self.snap_radius + 1):
            best = None
            best_d = None
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    r, c = gr + dr, gc + dc
                    if 0 <= r < h and 0 <= c < w and plan_grid[r, c] == FREE:
                        d = dr * dr + dc * dc
                        if best_d is None or d < best_d:
                            best, best_d = (r, c), d
            if best is not None:
                return best
        return None

    def _select_goal(self, start: tuple[int, int]) -> None:
        """Choose a NEW frontier goal (nearest reachable) and plan a path to it.

        Called only when there is no committed goal, or the current one was
        reached / timed out / became unreachable — this commitment is what stops
        the oscillation."""
        mask = find_frontiers(self.grid)
        clusters = cluster_frontiers(mask, self.min_cluster)
        cand = [c.representative for c in clusters if c.representative not in self.blacklist]
        if not cand:  # thin-corridor fallback, as in explore()
            clusters = cluster_frontiers(mask, min_cluster_size=1)
            cand = [c.representative for c in clusters if c.representative not in self.blacklist]
        if not cand:
            if not self.done:
                self.get_logger().info("exploration complete: no frontiers remain")
            self.done = True
            self.goal, self.path = None, []
            return

        plan_grid = self._planning_grid(start)
        cand.sort(key=lambda g: (g[0] - start[0]) ** 2 + (g[1] - start[1]) ** 2)
        for raw_goal in cand:
            goal = self._snap(raw_goal, plan_grid)
            if goal is None:
                self._fail(raw_goal)
                continue
            plan = astar(plan_grid, start, goal, True)
            if plan.path is not None and len(plan.path) >= 2:
                self.goal = goal
                self.path = plan.path
                self.path_idx = 1
                self.goal_since = time.monotonic()
                self.last_path_plan = time.monotonic()
                self.done = False
                return
            self._fail(raw_goal)
        self.goal, self.path = None, []  # nothing reachable this cycle; retry next

    def _fail(self, goal: tuple[int, int]) -> None:
        self.failures[goal] = self.failures.get(goal, 0) + 1
        if self.failures[goal] >= self.blacklist_after:
            self.blacklist.add(goal)

    def _replan_path(self, start: tuple[int, int]) -> None:
        """Refresh the path to the CURRENT goal as the map sharpens (goal kept)."""
        plan_grid = self._planning_grid(start)
        if plan_grid[self.goal] == OCCUPIED:      # goal swallowed by a wall now
            self.goal = None
            return
        plan = astar(plan_grid, start, self.goal, True)
        if plan.path is not None and len(plan.path) >= 2:
            self.path = plan.path
            self.path_idx = 1
        else:
            self._fail(self.goal)
            self.goal = None                       # can't reach it; pick another

    def _control(self) -> None:
        if self.grid is None or self.pose is None or self.plan_grid is None:
            return
        now = time.monotonic()
        start = self._cell(self.pose[0], self.pose[1])

        # 1) Retire the current goal if reached, timed out, or now inside a wall.
        if self.goal is not None:
            gx, gy = self._world(*self.goal)
            if math.hypot(gx - self.pose[0], gy - self.pose[1]) <= self.reach_radius:
                self.goal = None
            elif (now - self.goal_since) > self.goal_timeout:
                self._fail(self.goal)
                self.goal = None
            elif self.plan_grid[self.goal] == OCCUPIED:
                self.goal = None

        # 2) Pick a new goal, or periodically refresh the path to the current one.
        if self.goal is None:
            self._select_goal(start)
        elif (not self.path or self.path_idx >= len(self.path)
              or (now - self.last_path_plan) > self.replan_period):
            self.last_path_plan = now
            self._replan_path(start)

        if self.done or self.goal is None or not self.path or self.path_idx >= len(self.path):
            self._stop()
            return

        # 3) If the next cell just became a wall/margin, refresh the path.
        nr, nc = self.path[self.path_idx]
        if self.plan_grid[nr, nc] == OCCUPIED:
            self._replan_path(start)
            self._stop()
            return

        # 4) Drive toward the current path cell with a clamped P-controller.
        tx, ty = self._world(nr, nc)
        ex, ey = tx - self.pose[0], ty - self.pose[1]
        if math.hypot(ex, ey) <= self.wp_tol:
            self.path_idx += 1
            return
        cmd = Twist()
        cmd.linear.x = max(-self.vmax, min(self.vmax, self.gain * ex))
        cmd.linear.y = max(-self.vmax, min(self.vmax, self.gain * ey))
        self.cmd_pub.publish(cmd)


def main() -> None:
    rclpy.init()
    node = ExplorationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
