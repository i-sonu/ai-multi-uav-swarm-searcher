# Project Contribution Matrix & Work Breakdown Structure (WBS)

**Project Title:** Utility-Driven Coordinated Multi-UAV Exploration with Target Detection and Localization  
**Course:** BCSE306L - Artificial Intelligence (Digital Assignment 1)  
**Faculty:** VIJAYAPRABHAKARAN  
**Repository:** https://github.com/i-sonu/ai-multi-uav-swarm-searcher  
**Team Members:**  
- **SaiAmirthesh (24BYB):** Lead AI Systems & Search Architect  
- **Adarsh H Pillai (24BRS1082):** Swarm Systems, Benchmarking & Infrastructure Lead  

---

## 1. Overview & Role Descriptions

Both team members participate in all major design decisions. Responsibility effort is split **50/50**.

- **SaiAmirthesh — Lead AI Systems & Search Architect:** Core AI algorithm design, graph search implementations (A*, BFS, DFS, UCS), multi-agent dynamic task allocation (Hungarian and greedy scoring), computer vision detection pipeline, target spatial registration, and statistical experiment design.
- **Adarsh — Swarm Systems, Benchmarking & Infrastructure Lead:** Simulation environment generation, LiDAR sensor ray-casting mechanics, multi-agent observation tracking, headless parallel experiment execution framework, rendering and visualization tools, ROS 2 / Gazebo bridge, and automated testing suite.

---

## 2. Work Breakdown Structure (WBS) & Task Allocation Matrix

| WBS | Phase / Task Description | Primary Owner | Secondary Owner | Deliverables / Artifacts | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1.0** | **Environment & Single-Agent Foundation** | | | | **Completed** |
| 1.1 | Occupancy Map Generators (Office, Maze, Open Field, Obstacles) | Adarsh | SaiAmirthesh | `src/world/map_generator.py` | Completed |
| 1.2 | Ray-casting LiDAR Sensor Model & Noise Simulation | Adarsh | SaiAmirthesh | `src/world/sensor.py` | Completed |
| 1.3 | Shared Occupancy Grid World Model & State Tracking | SaiAmirthesh | Adarsh | `src/mapping/occupancy_grid.py` | Completed |
| 1.4 | Single-Agent Pose & Kinematic Motion Simulator | SaiAmirthesh | Adarsh | `src/agents/agent.py` | Completed |
| 1.5 | Phase 1 Fixed-Patrol Demo & Live Visualizer | Adarsh | SaiAmirthesh | `scripts/demo_phase1.py` | Completed |
| **2.0** | **Single-Agent Autonomous Exploration & Path Planning** | | | | **Completed** |
| 2.1 | Graph Search Algorithms (A*, BFS, DFS, UCS) | SaiAmirthesh | Adarsh | `src/planning/astar.py`, `baselines.py` | Completed |
| 2.2 | Frontier Boundary Detection Algorithm | SaiAmirthesh | Adarsh | `src/frontier/detection.py` | Completed |
| 2.3 | Frontier Clustering via Distance-Based DBSCAN | SaiAmirthesh | Adarsh | `src/frontier/clustering.py` | Completed |
| 2.4 | Single-Agent Autonomous Exploration Loop & Blacklisting | SaiAmirthesh | Adarsh | `src/exploration/explorer.py` | Completed |
| 2.5 | Phase 2 Autonomous Exploration Demo Script | Adarsh | SaiAmirthesh | `scripts/demo_phase2.py` | Completed |
| **3.0** | **Metrics Engine & Baseline Benchmarking (Experiment E1)** | | | | **Completed** |
| 3.1 | Core Metrics Module (Coverage %, Path Length, Time, Expansions) | SaiAmirthesh | Adarsh | `src/metrics/metrics.py` | Completed |
| 3.2 | Headless Parallel Experiment Runner | Adarsh | SaiAmirthesh | `src/experiments/runner.py` | Completed |
| 3.3 | Held-out Test Seeds Dataset Configuration | Adarsh | SaiAmirthesh | `configs/experiments/heldout_seeds.yaml` | Completed |
| 3.4 | Experiment E1 Execution (480 held-out runs across planners) | SaiAmirthesh | Adarsh | `scripts/run_e1.py` | Completed |
| 3.5 | Publication-Quality Figure Generation Scripts | Adarsh | SaiAmirthesh | `scripts/plot_e1.py` | Completed |
| 3.6 | Phase 3 Written Report & Empirical Findings | SaiAmirthesh | Adarsh | `docs/results_e1.md` | Completed |
| **4.0** | **Multi-UAV Swarm Coordination & Task Allocation** | | | | **Completed** |
| 4.1 | N-Agent Infrastructure & Shared Map Access Synchronization | Adarsh | SaiAmirthesh | `src/agents/agent.py`, `explorer.py` | Completed |
| 4.2 | Per-Agent Cell Observation & Redundant Coverage Tracker | Adarsh | SaiAmirthesh | `src/metrics/metrics.py` | Completed |
| 4.3 | Baseline B2 Implementation (Uncoordinated Nearest-Frontier) | Adarsh | SaiAmirthesh | `src/planning/allocation.py` | Completed |
| 4.4a | Cost-Utility Matrix Builder (A* Cost + Info Gain Utility) | SaiAmirthesh | Adarsh | `src/planning/allocation.py` | Completed |
| 4.4b | Greedy Multi-Agent Allocator | SaiAmirthesh | Adarsh | `src/planning/allocation.py` | Completed |
| 4.4c | Optimal Hungarian Assignment Allocator (`linear_sum_assignment`) | SaiAmirthesh | Adarsh | `src/planning/allocation.py` | Completed |
| 4.4d | Dynamic Re-planning & Goal Invalidated Re-assignment Policy | SaiAmirthesh | Adarsh | `src/planning/allocation.py` | Completed |
| 4.5 | Experiment E2 (Allocation Ablation: Uncoordinated vs Coordinated) | SaiAmirthesh | Adarsh | `scripts/run_e2.py`, `docs/results_e2.md` | Completed |
| 4.6 | Experiment E3 (Team Size Scaling: N=1, 2, 3 at matched flight time) | Adarsh | SaiAmirthesh | `scripts/run_e3.py`, `docs/results_e3.md` | Completed |
| 4.7 | Experiment E4 (Cost-Utility Weight $\lambda$ Sensitivity Sweep) | SaiAmirthesh | Adarsh | `scripts/run_e4.py`, `docs/results_e4.md` | Completed |
| 4.8 | Phase 4 Multi-Agent Benchmark Write-up | SaiAmirthesh | Adarsh | `docs/EXECUTIVE_FINDINGS_E1_E4.md` | Completed |
| **5.0** | **Perception, Aerial Target Detection & Spatial Registration** | | | | **In Progress** |
| 5.1 | Aerial Person Detection Dataset Ingestion & Preprocessing | Adarsh | SaiAmirthesh | `src/perception/data_loader.py` | Completed |
| 5.2 | Object Detector Fine-Tuning (YOLOv8 / MobileNet-SSD) | SaiAmirthesh | Adarsh | `src/perception/train.py` | Pending |
| 5.3 | Detector Evaluation (mAP@50, mAP@50-95, Small-Object Subset) | SaiAmirthesh | Adarsh | `src/perception/eval.py` | Pending |
| 5.4 | Target Placement & Sensor Camera Footprint Simulation | Adarsh | SaiAmirthesh | `src/world/sensor.py`, `agent.py` | Pending |
| 5.5 | Spatial Target Registration Engine (`TargetRegister` & Deduplication) | SaiAmirthesh | Adarsh | `src/perception/registration.py` | Pending |
| 5.6 | Experiment E5 (Exploration Strategy vs Target Detection Rate) | SaiAmirthesh | Adarsh | `scripts/run_e5.py` | Pending |
| 5.7 | Visual Target Pinning Overlay on Rendering Engine | Adarsh | SaiAmirthesh | `src/viz/render.py` | Pending |
| **6.0** | **ROS 2 & Gazebo High-Fidelity Simulation Bridge (Optional)** | | | | **Optional** |
| 6.1 | ROS 2 Package Architecture & Workspace Configuration | Adarsh | SaiAmirthesh | `ros2_ws/` setup | Optional |
| 6.2 | Dual-Drone Gazebo Model Spawning & Waypoint Control Nodes | Adarsh | SaiAmirthesh | `ros2_ws/src/uav_gazebo` | Optional |
| 6.3 | Multi-Agent Occupancy Map Aggregator & ROS 2 Services | SaiAmirthesh | Adarsh | `ros2_ws/src/uav_planner` | Optional |
| 6.4 | Gazebo Exploration Video Recording & Demonstration | Adarsh | SaiAmirthesh | `results/videos/gazebo_demo.mp4` | Optional |
| **7.0** | **Reproducibility, Verification & Master Documentation** | | | | **Final Stage** |
| 7.1 | Automated Test Suite & Regression Checks | Adarsh | SaiAmirthesh | `scripts/test.sh`, `tests/` | Completed |
| 7.2 | Master Experiment Results Consolidation | SaiAmirthesh | Adarsh | `docs/RESULTS.md` | Pending |
| 7.3 | Final AI Technical README | SaiAmirthesh | Adarsh | `docs/README.md` | Completed |
| 7.4 | Final Codebase Verification & Citations Updating | Adarsh | SaiAmirthesh | `CITATIONS.md`, `README.md` | Pending |
