# Experiment E1 — Planner Comparison (A\* vs BFS vs DFS vs UCS)

**Question.** When the path planner inside the single-agent exploration loop is
swapped, how do search effort, planning time, path length, and final coverage
change?

## Setup

- **Planners:** A\*, BFS, DFS, UCS — all hand-written, sharing one 8-connected
  cost model (orthogonal 1.0, diagonal √2, no corner cutting; UNKNOWN treated as
  optimistically traversable).
- **Maps × seeds:** 4 kinds (`office`, `maze`, `open_field`, `cluttered`) × 30
  **held-out** seeds each = 120 runs per planner, **480 runs total**.
- **Scale:** 96×96 grid (24 m), `max_steps = 16000`. *Why not 240×240?* A full
  240 maze needs ~90k steps/run; ×480 that is infeasible. The planner-comparison
  conclusions (nodes expanded, planning time, path length) are scale-invariant;
  headline single-agent coverage at 240 is shown separately by `demo_phase2`.
- **Reproducibility:** every run is fully determined by `(map_kind, seed, size)`;
  re-running reproduces identical numbers (spot-checked across all four planners).

Raw data: `results/raw/e1_summary.csv` (one row/run) and `e1_steps.csv`
(coverage-over-time). Figures: `results/figures/e1_*.png`.

## Headline numbers (mean over all maps × 30 seeds)

| Planner | Coverage % | Nodes/call | Plan time (ms, mean) | Plan p95 (ms) | Path length (m) |
|---|---|---|---|---|---|
| **A\***  | **99.9** | **171**  | **2.1**  | **5.3**  | 855 |
| BFS      | 100.0 | 1446 | 12.9 | 27.8 | 835 |
| UCS      | 100.0 | 1390 | 15.5 | 32.2 | 843 |
| DFS      | 81.4  | 4056 | 36.5 | 70.0 | 2068 |

### Final coverage by map (%)

| Planner | office | open_field | cluttered | maze |
|---|---|---|---|---|
| A\*  | 100.0 | 100.0 | 100.0 | 99.5 |
| BFS  | 100.0 | 100.0 | 100.0 | 99.8 |
| UCS  | 100.0 | 100.0 | 100.0 | 99.8 |
| DFS  | 100.0 | 99.8  | 100.0 | **25.9** |

## Findings

1. **A\* is the clear efficiency winner.** It reaches the same coverage and a
   near-identical exploration path length as UCS/BFS while expanding **~8× fewer
   nodes** (171 vs ~1400) and running **~7× faster per call** (2.1 ms vs
   13–16 ms). This is exactly the payoff of an admissible, consistent heuristic:
   A\* expands a subset of the nodes UCS must expand. (See
   `e1_nodes_expanded.png`, `e1_planning_time.png`.)

2. **A\*, BFS and UCS produce near-identical trajectories** (835–855 m; within
   ~2%). A\* and UCS return minimum-*cost* paths; BFS returns minimum-*edge*
   paths — on these maps the resulting whole-run trajectories differ only
   marginally. Total exploration path length is an emergent property of the
   frontier-visiting order, so small differences between the three optimal-ish
   planners are expected and not meaningful.

3. **DFS is a poor exploration planner.** Its paths are **2.4× longer** (2068 m)
   because DFS returns the first path it finds, not a short one. On open maps it
   still eventually finishes, but on **mazes it collapses to 25.9% coverage**:
   its long, meandering detours strand the agent in a corner from which the
   remaining frontiers are unreachable.

4. **Why coverage is ~99.5–99.8% (not exactly 100%) even for A\*.** LiDAR marks
   cells FREE that it can *see*, but movement is 8-connected with no corner
   cutting. In width-1 maze corridors, rays slip diagonally past walls and create
   frontiers in pockets the agent cannot actually drive into. A tiny residue of
   such visible-but-unreachable cells remains. DFS is hit hard by the same effect
   (its trajectory maximises exposure to it); A\* is barely affected.

## Takeaway

A\* is the right default: same result as the uninformed optimal search at a
fraction of the compute. DFS is retained only as a deliberately weak baseline.
This motivates using A\* for the cost matrix in the Phase 4 multi-agent
allocation, where planner calls are far more frequent.
