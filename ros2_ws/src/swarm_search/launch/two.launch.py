"""Milestone 6d launch: two coordinated drones share one map.

Gazebo (room_two world) + ROS<->GZ bridge for BOTH drones (cmd_vel, odometry,
scan) + one mapping_node fusing both LiDARs into /map + one coordination_node
running the Phase 4 allocator to hand each drone a distinct frontier.

Usage:
  ros2 launch swarm_search two.launch.py                 # Gazebo GUI, hungarian
  ros2 launch swarm_search two.launch.py gui:=false      # headless
  ros2 launch swarm_search two.launch.py method:=none    # uncoordinated baseline

View:  rviz2 -> Map on /map (fixed frame "map").
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _find_repo_root() -> str:
    p = os.getcwd()
    while p and p != os.path.dirname(p):
        if os.path.isdir(os.path.join(p, "src", "planning")) and os.path.isfile(os.path.join(p, "CLAUDE.md")):
            return p
        p = os.path.dirname(p)
    return os.getcwd()


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("swarm_search")
    world = os.path.join(share, "worlds", "room_two.sdf")
    gui = LaunchConfiguration("gui")
    method = LaunchConfiguration("method")
    drones = "drone1,drone2"

    bridge_args = []
    for name in ("drone1", "drone2"):
        bridge_args += [
            f"/model/{name}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            f"/model/{name}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            f"/model/{name}/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
        ]

    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("method", default_value="hungarian",
                              description="allocation: none | greedy | hungarian"),
        SetEnvironmentVariable("AISWARM_REPO", _find_repo_root()),

        ExecuteProcess(cmd=["gz", "sim", "-r", world], output="screen", condition=IfCondition(gui)),
        ExecuteProcess(cmd=["gz", "sim", "-r", "-s", world], output="screen", condition=UnlessCondition(gui)),

        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
             arguments=bridge_args),
        Node(package="swarm_search", executable="mapping_node", output="screen",
             parameters=[{"drones": drones}]),
        Node(package="swarm_search", executable="coordination_node", output="screen",
             parameters=[{"drones": drones, "method": method}]),
    ])
