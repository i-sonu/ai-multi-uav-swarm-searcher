# AI Architecture & Technical Specification: Multi-UAV Swarm Searcher

## 1. Executive Summary & AI Problem Formulation

The **AI Multi-UAV Swarm Searcher** addresses the challenge of **coordinated autonomous exploration** of unknown 2D environments coupled with **online target localisation** using a team of simulated Unmanned Aerial Vehicles (UAVs).

In search and rescue or reconnaissance missions, single-agent search is bottlenecked by battery runtime and limited sensor range. Multi-agent teams solve this, but without intelligent coordination, agents suffer from **redundant coverage** (searching areas already visited by teammates) and **goal conflict** (multiple agents navigating to the exact same frontier).

This project formulates the mission as a combined **Autonomous Exploration + Multi-Agent Task Allocation (MATA) + Online Target Registration** problem under uncertainty.

---

## 2. Core AI Modules & Algorithmic Design

### A. Graph Path Planning Algorithms (src/planning/)
Path planning enables agents to traverse unknown occupancy maps safely along obstacle-free trajectories. Four state-space graph search algorithms are implemented and benchmarked:

1. **A\* Search (src/planning/astar.py):**
   - Heuristic Function: Euclidean distance \(h(n) = \sqrt{(x_n - x_g)^2 + (y_n - y_g)^2}\).
   - Priority Queue Key: \(f(n) = g(n) + h(n)\), where \(g(n)\) is exact path cost.
   - Admissibility & Consistency: Euclidean metric guarantees optimal path discovery with minimum node expansion.
2. **Breadth-First Search (BFS):** Unweighted shortest-path exploration across grid cell neighbors.
3. **Depth-First Search (DFS):** Uninformed graph traversal (serves as an exploration baseline exhibiting high path cost and backtracking overhead).
4. **Uniform-Cost Search (UCS / Dijkstra):** \(f(n) = g(n)\) priority queue search to establish optimal ground-truth cost without heuristic guidance.

---

### B. Autonomous Frontier Detection & Spatial Clustering (src/frontier/)
Exploration is driven by discovering and reaching **frontiers**—the boundary cells separating known free space from unexplored unknown space.

1. **Frontier Extraction (src/frontier/detection.py):**
   - A cell \(c\) is a frontier if \(c \in \text{FREE}\) and at least one neighbor \(n(c) \in \text{UNKNOWN}\).
2. **Frontier Clustering (src/frontier/clustering.py):**
   - Raw frontier cell sets are aggregated into spatial clusters using distance-based DBSCAN or connected components.
   - Clustering reduces task allocation complexity from thousands of raw grid cells to \(K\) discrete frontier centroids.

---

### C. Multi-Agent Task Allocation (MATA) Engine (src/planning/allocation.py)
To prevent redundant coverage, task assignment dynamically pairs \(N\) UAV agents to \(K\) active frontier clusters.

1. **Cost Matrix Computation:**
   - \(C_{i,j} = \text{A* Path Cost}(\text{Agent}_i, \text{Frontier}_j)\)
2. **Information Gain Utility Matrix:**
   - \(U_{i,j} = \text{Count of UNKNOWN cells within sensor footprint at Frontier}_j\)
3. **Combined Objective Score:**
   - \(S_{i,j} = U_{i,j} - \lambda \cdot C_{i,j}\) (where \(\lambda\) balances exploration yield against flight energy/time).
4. **Allocation Algorithms:**
   - **Baseline B2 (Uncoordinated Nearest Frontier):** Each agent greedily targets its closest frontier independently, leading to heavy redundant coverage.
   - **Greedy Swarm Allocator:** Highest global score pair \((i, j)\) assigned sequentially, removing chosen frontier from pool.
   - **Optimal Hungarian Assignment:** Solves bipartite matching using scipy.optimize.linear_sum_assignment to optimize overall team objective \(\max \sum S_{i, j}\).

---

### D. Computer Vision & Target Localisation (src/perception/)
Agents carry down-looking camera sensors to detect targets (e.g., humans or vehicles) during flight.

1. **Aerial Object Detector (src/perception/train.py):**
   - Fine-tuned light-weight convolutional network (YOLOv8 / MobileNet-SSD) trained on aerial person detection datasets.
   - Outputs bounding boxes \((b_x, b_y, w, h)\) and classification confidence scores.
2. **Spatial Target Registration (src/perception/registration.py):**
   - Projects 2D image detections through current agent pose \((x_a, y_a, \theta_a)\) onto world map coordinates \((X_w, Y_w)\).
   - **Spatial Deduplication & Fusion:** Maintains a global TargetRegister. Duplicate detections across agents within a spatial threshold \(\Delta r\) are merged, updating confidence scores.

---

## 3. Empirical Benchmarks & Experiments

- **Experiment E1 (Completed):** Single-agent planner comparison (A\* vs BFS vs DFS vs UCS across 480 runs).
  - *Key Finding:* A\* expands ~8× fewer nodes and executes ~7× faster than UCS/BFS while achieving optimal path lengths.
- **Experiment E2 (Phase 4):** Allocation Ablation (Uncoordinated vs Greedy vs Hungarian Coordinated Task Allocation).
- **Experiment E3 (Phase 4):** Swarm Size Scaling (\(N = 1, 2, 3\) agents under equal cumulative flight time budgets).
- **Experiment E4 (Phase 4):** Cost-Utility Weight \(\lambda\) Hyperparameter Sensitivity Analysis.
- **Experiment E5 (Phase 5):** Target Localisation Precision & Detection Rate vs Exploration Strategy.

---

## 4. Execution & Verification Guide

### Single-Agent Baseline Demos
`ash
# Phase 1: Fixed Patrol LiDAR demo
python -m scripts.demo_phase1 --map office

# Phase 2: Single-Agent Autonomous Exploration with A*
python -m scripts.demo_phase2 --map office
`

### Reproducing Benchmark Experiments
`ash
# Run Experiment E1 (Single-agent planner sweep)
python -m scripts.run_e1

# Plot publication figures for E1
python -m scripts.plot_e1
`

### Running Unit Tests
`ash
./scripts/test.sh
`
