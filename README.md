# ai-uav-swarm-searcher

**Coordinated two-agent (two-UAV) autonomous exploration of unknown 2-D environments, with online target localisation.**

Two simulated drones are placed in an environment for which no map exists. They cooperatively build a shared occupancy grid, coordinate their goals so they do not waste effort re-exploring the same region, and simultaneously run an object detector that records **where** each detected target was found on that same shared map. Agent pose is taken as ground truth (no SLAM); the focus is on **multi-agent task allocation** (primary contribution) and **detection + target registration** (secondary contribution).

This is a university AI course project (BCSE306L, Russell & Norvig syllabus). See [`CLAUDE.md`](CLAUDE.md) for the full specification, build phases, and metric definitions.

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/i-sonu/ai-uav-swarm-searcher.git
cd ai-uav-swarm-searcher
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -c "import src"   # sanity check
```

`requirements.txt` covers Phases 1–4. Heavy dependencies (PyTorch / Ultralytics for detection in Phase 5) are installed when running perception components.

> **Note (this machine):** ROS 2 is sourced into the shell, which puts `/opt/ros`
> on `PYTHONPATH`. Always run inside the venv. The test wrapper
> (`scripts/test.sh`) clears `PYTHONPATH` and disables third-party pytest plugin
> autoloading so the ROS `launch_testing` plugin doesn't interfere.

## Run the demos

**Phase 1** — one drone on a fixed patrol, LiDAR revealing the map:

```bash
source venv/bin/activate
python -m scripts.demo_phase1 --map office            # live window
python -m scripts.demo_phase1 --save results/raw/demo.gif --no-show  # headless
```

**Phase 2** – one drone autonomously explores (frontiers + A* + move + repeat):

```bash
python -m scripts.demo_phase2 --map office           # A* by default
python -m scripts.demo_phase2 --map maze --size 120  # smaller maze finishes sooner
python -m scripts.demo_phase2 --planner bfs --no-show # swap planner, headless
```

**Phase 4** – two drones autonomously explore with Hungarian/Greedy coordination:

```bash
python -m scripts.demo_phase4 --map office --n-agents 2 --method hungarian
```

**Phase 5** – two drones explore, detect, and register targets in real time:

```bash
python -m scripts.demo_phase5 --map office --n-agents 2 --n-targets 10 --method hungarian
```

> **Note:** Running Phase 5 with deep learning inference uses pre-trained YOLOv8 weights placed at `data/weights/best.pt`. See [`train.md`](docs/train.md) and [`MODEL_OPTIMIZATION.md`](docs/MODEL_OPTIMIZATION.md) for the training guide, model benchmarks, and accuracy progression.

## Test

```bash
./scripts/test.sh        # runs all 41 unit tests in the venv (ROS-safe)
```

## Repository layout

```
configs/        YAML configuration (all constants; no hardcoded values in source)
src/
  world/        ground-truth map generation + LiDAR ray casting + target placement
  mapping/      shared occupancy grid (the world model)
  frontier/     frontier detection + clustering
  planning/     hand-written A*, BFS, DFS, UCS + multi-agent allocation (Hungarian/Greedy)
  agents/       agent model + kinematics
  exploration/  single-agent and multi-agent exploration loops
  perception/   dataset loading, detector training, target registration
  metrics/      metric implementations
  experiments/  headless experiment runner
  viz/          rendering / animation
scripts/        runnable demos, experiment runners, and plotting scripts
results/        CSVs and figures (raw runs gitignored, finals committed)
tests/          unit tests
docs/           written results, benchmarks, and model optimization guides
```

## Status

**Phase 5 complete** — Autonomous single-agent search (Phases 1–3), multi-agent coordination with Hungarian/Greedy allocation (Phase 4), and online aerial target detection and spatial registration (Phase 5) are fully implemented, tested, and benchmarked across all held-out environments.

## Reproduce experiments

```bash
# E1 — Planner benchmark (A* vs BFS vs DFS vs UCS across 480 held-out runs)
python -m scripts.run_e1
python -m scripts.plot_e1

# E2 — Task allocation ablation (Uncoordinated B2 vs Greedy vs Hungarian)
python -m scripts.run_e2
python -m scripts.plot_e2

# E3 — Swarm team scaling (N=1 vs N=2 vs N=3 agents at matched flight budget)
python -m scripts.run_e3
python -m scripts.plot_e3

# E4 — Cost-utility trade-off lambda sensitivity sweep
python -m scripts.run_e4
python -m scripts.plot_e4

# E5 — Target detection and spatial registration under exploration strategies
python -m scripts.run_e5
python -m scripts.plot_e5
```

## Results Summary

* **E1 — Planner Comparison:** A* expands **~8× fewer nodes** (171 vs ~1400) and runs **~7× faster** than UCS/BFS while reaching identical coverage and path optimality. DFS strands agents in mazes (25.9% coverage). See [`docs/results_e1.md`](docs/results_e1.md).
* **E2 — Allocation Ablation:** Hungarian optimal allocation achieves a **>50% reduction in redundant coverage ratio** compared to the uncoordinated baseline (0.092 vs 0.221) and speeds exploration by **20–25%**. See [`docs/results_e2.md`](docs/results_e2.md).
* **E3 — Team Scaling:** Scaling to $N=2$ and $N=3$ agents produces near-linear exploration speedups (1.85× and 2.6×) with coordinated Hungarian allocation. See [`docs/results_e3.md`](docs/results_e3.md).
* **E4 — Lambda Sensitivity:** Balances travel cost and information gain; $\lambda=0.5$ proves Pareto-optimal across all topologies. See [`docs/results_e4.md`](docs/results_e4.md).
* **E5 — Target Detection & Registration:** Coordinated Hungarian exploration achieves **80.0% target recall** in complex maze environments vs. 55.0% for uncoordinated agents, while maintaining mean localization error at **~0.21 m** (within single-cell resolution). See [`docs/results_e5.md`](docs/results_e5.md).
* **Computer Vision Optimization:** YOLOv8 fine-tuning on VisDrone improved from 17.3% baseline mAP to **68.9% mAP@50**, achieving **83.3% precision** on vehicles and **72.0% precision** on aerial pedestrians. Full optimization logs and playbook: [`docs/MODEL_OPTIMIZATION.md`](docs/MODEL_OPTIMIZATION.md).

## License

MIT — see [`LICENSE`](LICENSE). Third-party attributions are recorded in [`CITATIONS.md`](CITATIONS.md).