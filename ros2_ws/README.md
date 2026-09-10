# ros2_ws — Phase 6 ROS 2 / Gazebo port

ROS 2 **Jazzy** + **Gazebo Sim 8 (Harmonic)** glue for `ai-uav-swarm-searcher`.

**The algorithms do not live here.** Exploration, mapping, frontier, planning and
perception are the top-level `src/` package and are imported unchanged; this
workspace only wraps them as ROS 2 nodes (the I/O boundary, per CLAUDE.md §6).
The drone is a **simple kinematic model** (velocity-controlled, fixed altitude,
Gazebo pose taken as ground truth — no dynamics, no SLAM), and sensing comes from
Gazebo's simulated **LiDAR** (from milestone 6b onward).

## Build

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select swarm_search
source install/setup.bash
```

## Milestones

- **6a — one drone to a waypoint** *(done)*
  ```bash
  ros2 launch swarm_search one_drone.launch.py            # Gazebo GUI
  ros2 launch swarm_search one_drone.launch.py gui:=false # headless
  ```
  A single drone flies a square patrol of waypoints, driven by
  `waypoint_driver` over `cmd_vel`, using ground-truth odometry.
- **6b — Gazebo LiDAR builds the shared `/map`** *(done)*
  ```bash
  ros2 launch swarm_search mapping.launch.py       # + rviz2, Map on /map
  ```
- **6c — autonomous single-drone exploration** *(done)*
  ```bash
  ros2 launch swarm_search explore.launch.py
  ```
  Frontier + A* (from `src/`) drive the drone over `/map` until covered.
- **6d — two coordinated drones + one shared map** *(done)*
  ```bash
  ros2 launch swarm_search two.launch.py               # hungarian (default)
  ros2 launch swarm_search two.launch.py method:=none  # uncoordinated baseline
  ```
  One `mapping_node` fuses both LiDARs into `/map`; one `coordination_node`
  runs the Phase 4 allocator to hand each drone a distinct frontier so they
  split the room.
- 6e — record the demo video *(next)*

Build artifacts (`build/ install/ log/`) are git-ignored.
