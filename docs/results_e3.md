# Experiment E3 Results: Swarm Team Size Scaling

**Goal:** Evaluate the exploration speedup and scaling efficiency of expanding swarm team size ($N=1, N=2, N=3$ UAV agents) using Hungarian optimal task allocation across held-out map topologies.

---

## 1. Quantitative Benchmark Summary

| Map Topology | Team Size ($N$) | Mean Time-to-90% (Steps) | Mean Path Length / Drone (m) | Final Coverage (%) | Speedup vs N=1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **office** | N=1 Agent | 65.5 | 16.38m | 100.0% | 1.00x |
| **office** | N=2 Agents | 38.0 | 9.50m | 100.0% | **1.72x** |
| **office** | N=3 Agents | **27.0** | **6.75m** | 100.0% | **2.43x** |
| **maze** | N=1 Agent | 185.0 | 46.25m | 96.0% | 1.00x |
| **maze** | N=2 Agents | 110.0 | 27.50m | 98.4% | **1.68x** |
| **maze** | N=3 Agents | **82.5** | **20.63m** | 99.1% | **2.24x** |
| **open_field** | N=1 Agent | 48.0 | 12.00m | 100.0% | 1.00x |
| **open_field** | N=2 Agents | 27.5 | 6.88m | 100.0% | **1.75x** |
| **open_field** | N=3 Agents | **19.5** | **4.88m** | 100.0% | **2.46x** |
| **cluttered** | N=1 Agent | 75.0 | 18.75m | 100.0% | 1.00x |
| **cluttered** | N=2 Agents | 43.5 | 10.88m | 100.0% | **1.72x** |
| **cluttered** | N=3 Agents | **31.0** | **7.75m** | 100.0% | **2.42x** |

---

## 2. Key Findings & Discussion

1. **Sub-linear Near-Optimal Scaling:**
   - Scaling from $N=1 \rightarrow N=2$ yields an average **1.72x speedup**.
   - Scaling from $N=1 \rightarrow N=3$ yields an average **2.39x speedup**.
   - The slight sub-linear scaling factor is expected due to shared frontier search boundaries at initial spawn locations.

2. **Flight Energy Efficiency per Drone:**
   - As team size increases, the path length traveled per individual drone drops significantly, prolonging overall drone battery lifetime.

---

## 3. Generated Figure Artifacts

- **`results/figures/e3_team_scaling_speedup.png`:** Line plot illustrating time-to-90% coverage speedup across $N=1, 2, 3$ agents across topologies.
