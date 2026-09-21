# Product Requirements Document (PRD): AI Multi-UAV Swarm Searcher

**Project Name:** Utility-Driven Coordinated Multi-UAV Exploration with Target Detection and Localization  
**Course:** BCSE306L - Artificial Intelligence (Digital Assignment 1)  
**Faculty:** VIJAYAPRABHAKARAN  
**Authors:** SaiAmirthesh & Adarsh H Pillai  
**Date:** August 2026  
**Status:** Active Development (Phase 4 Benchmarks Complete | Phase 5 Next)  

---

## 1. Executive Summary & Purpose

The **AI Multi-UAV Swarm Searcher** system simulates a team of Unmanned Aerial Vehicles (UAVs) performing **coordinated autonomous exploration** of unmapped 2D environments while concurrently carrying out **online target detection and spatial localization**.

Designed for post-disaster search and rescue during the critical **72-hour golden period**, the system supports incident commanders by replacing uncoordinated search strategies with an intelligent **Multi-Agent Task Allocation (MATA)** mechanism that eliminates redundant overlapping search, accelerates coverage time, and registers detected targets onto a unified global map.

---

## 2. Target Audience & System Boundaries

- **Primary Stakeholder:** Incident Commander of search-and-rescue teams directing ground rescue units.
- **Secondary Audience:** Autonomy researchers comparing multi-robot coordination strategies.
- **System Scope:**
  - 2D grid representation (occupancy grid).
  - Ground-truth agent poses (NO SLAM required; focus is on search allocation & object localization).
  - Synchronous / discrete-time step updates across $N \ge 1$ UAVs.

---

## 3. Functional Requirements (FR)

### FR-1: Environment Simulation & LiDAR Sensor Model
- **FR-1.1:** Procedurally generate 2D occupancy grids across four map topologies (`office`, `maze`, `open_field`, `cluttered`) in a bounded $60\,\text{m} \times 60\,\text{m}$ search area.
- **FR-1.2:** Discretize the environment into a $240 \times 240$ occupancy grid at $0.25\,\text{m}$ resolution, where each cell takes one of three discrete states: `FREE (0)`, `OCCUPIED (1)`, or `UNKNOWN (-1)`.
- **FR-1.3:** Simulate a 360-beam, 2D ray-casting LiDAR sensor on each drone with $12\,\text{m}$ maximum range at $10\,\text{Hz}$.

### FR-2: Graph Path Planning & Frontier Detection
- **FR-2.1:** Extract boundary frontier cells defined as `FREE` cells adjacent to at least one `UNKNOWN` cell.
- **FR-2.2:** Cluster raw frontier cells into spatial centroids using distance-based DBSCAN clustering.
- **FR-2.3:** Provide graph search planners (A\*, BFS, DFS, UCS) to compute collision-free trajectories from agent position to target frontier centroids.
- **FR-2.4:** Maintain an unreachable frontier blacklist after $N$ failed path planning attempts.

### FR-3: Multi-Agent Coordination & Dynamic Task Allocation (Primary Contribution)
- **FR-3.1:** Support simultaneous exploration by $N \ge 1$ UAV agents sharing a single global `OccupancyGrid`.
- **FR-3.2:** Construct a Cost-Utility Matrix between $N$ agents and $K$ frontier clusters:
  - Cost $C_{i,j}$: A\* path distance from agent $i$ to frontier cluster $j$.
  - Utility $U_{i,j}$: Information gain (count of `UNKNOWN` cells within sensor range of cluster $j$).
  - Combined Score $S_{i,j} = U_{i,j} - \lambda \cdot C_{i,j}$.
- **FR-3.3:** Implement **Uncoordinated Baseline (B2)** where each agent greedily selects its nearest frontier independently.
- **FR-3.4:** Implement **Greedy Task Allocation** (sequential highest-score matching).
- **FR-3.5:** Implement **Optimal Hungarian Bipartite Assignment** using `scipy.optimize.linear_sum_assignment` to maximize $\sum S_{i,j}$.
- **FR-3.6:** Trigger dynamic re-planning when an agent reaches its goal, when its goal ceases to be a valid frontier, or every $M$ steps.

### FR-4: Computer Vision & Spatial Target Localisation
- **FR-4.1:** Place $M$ target objects (e.g., humans) at random free cells in the ground-truth environment.
- **FR-4.2:** Equip each agent with a downward-facing RGB camera stream ($640 \times 480$ resolution at $5\,\text{Hz}$).
- **FR-4.3:** Ingest camera frames and execute a fine-tuned deep neural network detector (YOLOv8 / MobileNet-SSD) returning target bounding boxes and confidence scores.
- **FR-4.4:** Project 2D image detections through agent pose $(x_a, y_a, \theta_a)$ into global map coordinates $(X_w, Y_w)$.
- **FR-4.5:** Maintain a centralized `TargetRegister` performing spatial deduplication across agents and repeat observations within a distance threshold $\Delta r$.

### FR-5: Metrics, Benchmarking & Experimentation Framework
- **FR-5.1:** Record step-by-step metrics: Coverage %, Path Length, Planning Wall-Clock Time, Nodes Expanded, and Redundant Coverage Ratio.
- **FR-5.2:** Provide a headless parallel experiment runner executing benchmark sweeps across held-out random seeds (`configs/experiments/heldout_seeds.yaml`).
- **FR-5.3:** Target at least a **35% reduction** in time-to-90%-coverage over the single-agent baseline B1, and a statistically significant reduction in redundant coverage over uncoordinated baseline B2.
- **FR-5.4:** Generate publication-quality figures for Experiments E1 (Planner Comparison), E2 (Allocation Ablation), E3 (Swarm Scaling), E4 ($\lambda$ Sensitivity), and E5 (Target Localisation).

---

## 4. Non-Functional Requirements (NFR)

### NFR-1: Performance & Real-Time Execution
- **NFR-1.1:** Single A\* path planning requests on 240x240 grids must execute in under $15\,\text{ms}$.
- **NFR-1.2:** The Multi-Agent Task Allocation algorithm (Hungarian assignment for $N=5, K=30$) must execute in under $10\,\text{ms}$ per reallocation cycle.

### NFR-2: Reproducibility & Determinism
- **NFR-2.1:** All experiments must yield identical numerical metrics when executed with identical random seeds.
- **NFR-2.2:** Benchmark seed configurations must remain strictly held-out and isolated from hyperparameter tuning datasets.

### NFR-3: Code Quality, Modularity & Testability
- **NFR-3.1:** Maintain complete separation between environment physics, planning logic, perception models, and visualization components.
- **NFR-3.2:** The automated unit test suite (`./scripts/test.sh`) must pass 100% of tests cleanly without ROS environment pollution.
EOF"
