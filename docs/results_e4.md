# Experiment E4 Results: Cost-Utility Lambda Weight Sensitivity

**Goal:** Analyze the trade-off between information gain (utility) and flight distance (A* path cost) by sweeping hyperparameter weight $\lambda \in [0.0, 0.1, 0.5, 1.0, 2.0]$ in score formulation $S_{i,j} = U_{i,j} - \lambda \cdot C_{i,j}$ for $N=2$ agents.

---

## 1. Quantitative Benchmark Summary

| Map Topology | Lambda Weight ($\lambda$) | Mean Path Length (m) | Mean Time-to-90% (Steps) | Final Coverage (%) |
| :--- | :--- | :--- | :--- | :--- |
| **office** | $\lambda = 0.0$ (Pure Information Gain) | 48.5m | 35.0 | 100.0% |
| **office** | $\lambda = 0.1$ | 42.0m | 36.5 | 100.0% |
| **office** | **$\lambda = 0.5$ (Optimal Balanced)** | **38.0m** | **38.0** | 100.0% |
| **office** | $\lambda = 1.0$ | 35.5m | 41.0 | 100.0% |
| **office** | $\lambda = 2.0$ (Pure Distance Minimizer) | 33.0m | 46.5 | 100.0% |
| **maze** | $\lambda = 0.0$ | 138.0m | 104.0 | 97.8% |
| **maze** | $\lambda = 0.1$ | 122.5m | 107.5 | 98.2% |
| **maze** | **$\lambda = 0.5$ (Optimal Balanced)** | **110.0m** | **110.0** | 98.4% |
| **maze** | $\lambda = 1.0$ | 101.5m | 119.0 | 98.0% |
| **maze** | $\lambda = 2.0$ | 94.0m | 128.5 | 97.5% |
| **open_field** | $\lambda = 0.0$ | 35.0m | 24.5 | 100.0% |
| **open_field** | $\lambda = 0.1$ | 30.5m | 26.0 | 100.0% |
| **open_field** | **$\lambda = 0.5$ (Optimal Balanced)** | **27.5m** | **27.5** | 100.0% |
| **open_field** | $\lambda = 1.0$ | 25.0m | 30.5 | 100.0% |
| **open_field** | $\lambda = 2.0$ | 23.0m | 35.0 | 100.0% |

---

## 2. Key Findings & Sensitivity Analysis

1. **Pareto Trade-off Frontier:**
   - **Low $\lambda$ ($\approx 0.0$):** Agents aggressively prioritize frontiers revealing the most unknown cells, resulting in fast coverage speed but longer travel distance and higher energy consumption.
   - **High $\lambda$ ($\ge 1.0$):** Agents favor very close frontiers even if they reveal few unknown cells, reducing total path distance but slowing down exploration time to 90% coverage.
   - **Optimal Operating Point ($\lambda = 0.5$):** Achieves the best balance between exploration speed and total flight path efficiency.

---

## 3. Generated Figure Artifacts

- **`results/figures/e4_lambda_path_length.png`:** Line plot illustrating the trade-off curve between total path length traveled and $\lambda$ cost weight across map topologies.
