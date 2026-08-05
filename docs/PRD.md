# Product Requirements Document (PRD): AI Multi-UAV Swarm Searcher

**Project Name:** AI Multi-UAV Swarm Searcher  
**Course:** BCSE306L - Artificial Intelligence (Russell & Norvig Syllabus)  
**Authors:** SaiAmirthesh & Adarsh  
**Date:** August 2026  
**Status:** Active Development (Phase 4 & 5 Next)  

---

## 1. Executive Summary & Purpose

The **AI Multi-UAV Swarm Searcher** system is designed to simulate a fleet of Unmanned Aerial Vehicles (UAVs) performing **coordinated autonomous exploration** of unmapped 2D environments while concurrently carrying out **online target detection and spatial localization**.

The primary objective is to replace uncoordinated individual search strategies with an intelligent **Multi-Agent Task Allocation (MATA)** mechanism that minimizes redundant area coverage, reduces mission search time, and accurately logs target objects onto a unified global map.

---

## 2. Target Audience & System Boundaries

- **Target Audience:** AI Course Evaluators, Academic Researchers, Swarm Robotics Engineers.
- **Assumptions & Scope:**
  - 2D grid representation (occupancy grid).
  - Ground-truth agent poses (NO SLAM required; focus is on search allocation & object localization).
  - Synchronous / discrete-time step updates.

---

## 3. Functional Requirements (FR)

### FR-1: Environment Simulation & LiDAR Sensor Model
- **FR-1.1:** The system shall procedurally generate 2D occupancy grids across four distinct map topologies (office, maze, open_field, andom_obstacles).
- **FR-1.2:** Each cell in the occupancy grid must take one of three discrete states: FREE (0), OCCUPIED (1), or UNKNOWN (-1).
- **FR-1.3:** The system shall simulate a 360-degree 2D ray-casting LiDAR sensor mounted on each drone with configurable range (e.g., 10 units) and angular resolution.

### FR-2: Graph Path Planning & Frontier Detection
- **FR-2.1:** The system shall extract frontier cells defined as FREE cells adjacent to at least one UNKNOWN cell.
- **FR-2.2:** The system shall cluster raw frontier cells into spatial centroids using distance-based clustering (DBSCAN / connected components).
- **FR-2.3:** The system shall provide graph search planners (A\*, BFS, DFS, UCS) to compute collision-free trajectories from any agent position to any target frontier centroid.
- **FR-2.4:** The system shall maintain an unreachable frontier blacklist after \(N\) failed path planning attempts.

### FR-3: Multi-Agent Coordination & Dynamic Task Allocation (Primary Contribution)
- **FR-3.1:** The system shall support simultaneous exploration by \(N \ge 1\) UAV agents sharing a single global occupancy grid.
- **FR-3.2:** The system shall construct a Cost-Utility Matrix between \(N\) agents and \(K\) frontier clusters:
  - Cost \(C_{i,j}\): A\* path distance from agent \(i\) to frontier cluster \(j\).
  - Utility \(U_{i,j}\): Information gain (count of UNKNOWN cells within sensor range of cluster \(j\)).
  - Score \(S_{i,j} = U_{i,j} - \lambda \cdot C_{i,j}\).
- **FR-3.3:** The system shall implement **Greedy Task Allocation** (sequential highest-score matching).
- **FR-3.4:** The system shall implement **Optimal Hungarian Bipartite Assignment** using scipy.optimize.linear_sum_assignment to maximize \(\sum S_{i,j}\).
- **FR-3.5:** The system shall implement an **Uncoordinated Baseline (B2)** where each agent independently selects its nearest frontier without communication.
- **FR-3.6:** The system shall trigger dynamic re-planning when an agent reaches its goal, when its goal ceases to be a valid frontier, or after every \(M\) steps.

### FR-4: Computer Vision & Spatial Target Localisation
- **FR-4.1:** The system shall place \(M\) target objects (e.g., humans) at random free cells in the ground-truth environment.
- **FR-4.2:** Each agent shall possess a down-looking camera sensor footprint.
- **FR-4.3:** The system shall ingest camera frames and execute a fine-tuned deep neural network detector (YOLOv8 / MobileNet-SSD) returning target bounding boxes and confidence scores.
- **FR-4.4:** The system shall project 2D image detections into global map coordinates \((X_w, Y_w)\) based on agent pose.
- **FR-4.5:** The system shall maintain a centralized TargetRegister that performs spatial deduplication across agents and repeat observations within a distance threshold \(\Delta r\).

### FR-5: Metrics, Benchmarking & Experimentation Framework
- **FR-5.1:** The system shall record step-by-step metrics: Coverage %, Path Length, Planning Wall-Clock Time, Nodes Expanded, and Redundant Coverage Ratio.
- **FR-5.2:** The system shall provide a headless parallel experiment runner executing benchmark sweeps across held-out random seeds (configs/experiments/heldout_seeds.yaml).
- **FR-5.3:** The system shall generate publication-quality figures for:
  - Experiment E1 (Single-Agent Planner Comparison)
  - Experiment E2 (Allocation Strategy Ablation)
  - Experiment E3 (Swarm Team Size Scaling N=1, 2, 3)
  - Experiment E4 (\(\lambda\) Sensitivity Sweep)
  - Experiment E5 (Target Detection Rate vs Exploration Strategy)

---

## 4. Non-Functional Requirements (NFR)

### NFR-1: Performance & Real-Time Planning
- **NFR-1.1:** Single A\* path planning requests on 240x240 grids must execute in under 15ms.
- **NFR-1.2:** The Multi-Agent Task Allocation algorithm (Hungarian assignment for \(N=5, K=30\)) must execute in under 10ms per reallocation cycle.

### NFR-2: Reproducibility & Determinism
- **NFR-2.1:** All experiments must yield identical numerical metrics when executed with identical random seeds.
- **NFR-2.2:** Benchmark seed configurations must remain strictly held-out and isolated from hyperparameter tuning datasets.

### NFR-3: Code Quality, Modularity & Testability
- **NFR-3.1:** Source code must maintain complete separation between environment physics, planning logic, perception models, and visualization components.
- **NFR-3.2:** The automated unit test suite (./scripts/test.sh) must pass 100% of tests cleanly without ROS environment pollution.

---

## 5. Verification & Acceptance Criteria

1. **Phase 3 Acceptance:** Single agent reaches \(\ge 95\%\) coverage on office and maze maps; E1 benchmark completes reproducibly across 480 runs (**PASSED**).
2. **Phase 4 Acceptance:** Coordinated Hungarian allocation shows statistically significant reduction in redundant coverage ratio compared to uncoordinated Baseline B2 across held-out seeds.
3. **Phase 5 Acceptance:** Multi-UAV swarm achieves full map coverage while registering \(\ge 90\%\) of hidden targets with spatial localization error below 2 grid cells.
