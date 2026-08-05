# Project Contribution Matrix & Work Breakdown Structure (WBS)

**Project Name:** AI Multi-UAV Swarm Searcher (Coordinated Autonomous Exploration & Target Localisation)  
**Course:** BCSE306L - Artificial Intelligence  
**Contributors:** Adarsh (Phase 1–3 Completed Baseline Lead) & SaiAmirthesh (Phase 4–5 Swarm AI & Perception Lead)  

---

## 1. Overview & Work Division Summary

All foundational work completed up to **Phase 3** (Environment setup, LiDAR sensors, single-agent path planners A*/BFS/DFS/UCS, frontier detection & clustering, metrics engine, experiment runner, and Experiment E1) was executed by **Adarsh**.

The next upcoming major tasks (**Phase 4 & Phase 5**), encompassing multi-agent coordination algorithms, task allocation engines (Hungarian & Greedy), multi-drone benchmark sweeps (E2, E3, E4), object detection fine-tuning, and target spatial registration (E5), are allotted to **SaiAmirthesh**.

---

## 2. Work Breakdown Structure (WBS) & Task Ownership Matrix

| WBS Code | Phase / Task Description | Primary Owner | Secondary Owner | Deliverables / Artifacts | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1.0** | **Environment & Single-Agent Foundation** | | | | **Completed** |
| 1.1 | Occupancy Map Generators (Office, Maze, Open Field, Obstacles) | **Adarsh** | SaiAmirthesh | src/world/map_generator.py | Completed |
| 1.2 | Ray-casting LiDAR Sensor Model & Noise Simulation | **Adarsh** | SaiAmirthesh | src/world/sensor.py | Completed |
| 1.3 | Shared Occupancy Grid World Model & State Tracking | **Adarsh** | SaiAmirthesh | src/mapping/occupancy_grid.py | Completed |
| 1.4 | Single-Agent Pose & Kinematic Motion Simulator | **Adarsh** | SaiAmirthesh | src/agents/agent.py | Completed |
| 1.5 | Phase 1 Fixed-Patrol Demo & Live Visualizer | **Adarsh** | SaiAmirthesh | scripts/demo_phase1.py | Completed |
| **2.0** | **Single-Agent Autonomous Exploration & Path Planning** | | | | **Completed** |
| 2.1 | Graph Search Algorithms (A*, BFS, DFS, UCS) | **Adarsh** | SaiAmirthesh | src/planning/astar.py, aselines.py | Completed |
| 2.2 | Frontier Boundary Detection Algorithm | **Adarsh** | SaiAmirthesh | src/frontier/detection.py | Completed |
| 2.3 | Frontier Clustering via Distance-Based DBSCAN | **Adarsh** | SaiAmirthesh | src/frontier/clustering.py | Completed |
| 2.4 | Single-Agent Autonomous Exploration Loop & Blacklisting | **Adarsh** | SaiAmirthesh | src/exploration/explorer.py | Completed |
| 2.5 | Phase 2 Autonomous Exploration Demo Script | **Adarsh** | SaiAmirthesh | scripts/demo_phase2.py | Completed |
| **3.0** | **Metrics Engine & Baseline Benchmarking (Experiment E1)** | | | | **Completed** |
| 3.1 | Core Metrics Module (Coverage %, Path Length, Time, Expansions) | **Adarsh** | SaiAmirthesh | src/metrics/metrics.py | Completed |
| 3.2 | Headless Parallel Experiment Runner | **Adarsh** | SaiAmirthesh | src/experiments/runner.py | Completed |
| 3.3 | Held-out Test Seeds Dataset Configuration | **Adarsh** | SaiAmirthesh | configs/experiments/heldout_seeds.yaml | Completed |
| 3.4 | Experiment E1 Execution (480 held-out runs across planners) | **Adarsh** | SaiAmirthesh | scripts/run_e1.py | Completed |
| 3.5 | Publication-Quality Figure Generation Scripts | **Adarsh** | SaiAmirthesh | scripts/plot_e1.py | Completed |
| 3.6 | Phase 3 Written Report & Empirical Findings | **Adarsh** | SaiAmirthesh | docs/results_e1.md | Completed |
| **4.0** | **Multi-UAV Swarm Coordination & Task Allocation** | | | | **Assigned to SaiAmirthesh** |
| 4.1 | **[TASK 1]** Multi-Agent Exploration Engine & Baseline B2 (Uncoordinated Nearest Frontier) | **SaiAmirthesh** | Adarsh | src/agents/agent.py, src/planning/allocation.py | **NEXT TO DO** |
| 4.2 | **[TASK 2]** Cost-Utility Task Allocator (Greedy & Optimal Hungarian Bipartite Matcher) | **SaiAmirthesh** | Adarsh | src/planning/allocation.py | **NEXT TO DO** |
| 4.3 | **[TASK 3]** Multi-UAV Swarm Benchmark Experiments (E2 Ablation, E3 Team Scaling, E4 \(\lambda\) Sweep) | **SaiAmirthesh** | Adarsh | scripts/run_e2.py, un_e3.py, un_e4.py, docs/results_e2_e4.md | **NEXT TO DO** |
| **5.0** | **Perception, Aerial Target Detection & Spatial Registration** | | | | **Assigned to SaiAmirthesh** |
| 5.4 | **[TASK 4]** Aerial Object Detector Fine-Tuning & Evaluation Pipeline | **SaiAmirthesh** | Adarsh | src/perception/data_loader.py, 	rain.py, eval.py | **NEXT TO DO** |
| 5.5 | **[TASK 5]** Spatial Target Registration Engine & Multi-Agent Fusion (TargetRegister & E5 Experiment) | **SaiAmirthesh** | Adarsh | src/perception/registration.py, scripts/run_e5.py | **NEXT TO DO** |
| **6.0** | **ROS 2 & Gazebo Integration (Optional Upside)** | | | | **Optional** |
| 6.1 | Dual-Drone Spawning & ROS 2 Waypoint Navigation Bridge | **Adarsh** | SaiAmirthesh | os2_ws/ | Optional |
| **7.0** | **Final Documentation & Reproducibility** | | | | **Final Stage** |
| 7.1 | Master Experiment Results Document & Codebase Tagging | **SaiAmirthesh** | **Adarsh** | docs/RESULTS.md, CITATIONS.md | Pending |

---

## 3. Allocation Summary for Next Tasks (SaiAmirthesh)

1. **Task 1:** Implement Multi-Agent Exploration Loop Infrastructure & Uncoordinated Baseline (B2).
2. **Task 2:** Implement Cost-Utility Task Allocation Engine (src/planning/allocation.py) featuring Greedy and Hungarian (scipy.optimize.linear_sum_assignment) algorithms.
3. **Task 3:** Execute Multi-UAV Benchmark Sweeps (Experiment E2 Allocation Ablation, Experiment E3 Swarm Team Size Scaling N=1,2,3, and Experiment E4 \(\lambda\) Sensitivity) and write docs/results_e2_e4.md.
4. **Task 4:** Fine-tune Aerial Object Detector (YOLOv8 / MobileNet-SSD) & build evaluation metrics pipeline (mAP@50, mAP@50-95).
5. **Task 5:** Implement 2D Spatial Target Registration (TargetRegister), Camera FOV projection, multi-drone spatial deduplication, and execute Target Localisation Experiment E5.
