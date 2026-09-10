"""ament_python build for swarm_search (Phase 6 ROS 2 glue).

Only the ROS wrapper lives here; the algorithms are imported from the project's
top-level ``src/`` package, which is added to the Python path by the launch
files (kept unchanged per CLAUDE.md Phase 6.3).
"""

from glob import glob

from setuptools import find_packages, setup

package_name = "swarm_search"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/worlds", glob("worlds/*.sdf")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="i-sonu",
    maintainer_email="adarshh3000@gmail.com",
    description="ROS 2 / Gazebo glue for ai-uav-swarm-searcher (Phase 6).",
    license="MIT",
    entry_points={
        "console_scripts": [
            # milestone 6a: drive one drone to a waypoint
            "waypoint_driver = swarm_search.waypoint_driver:main",
            # milestone 6b: build the shared occupancy grid from LiDAR
            "mapping_node = swarm_search.mapping_node:main",
        ],
    },
)
