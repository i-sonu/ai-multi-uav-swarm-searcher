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
python -c "import src"   # sanity check
```

`requirements.txt` covers Phases 1–4 only. Heavy dependencies (PyTorch for
detection in Phase 5; ROS 2 / Gazebo in Phase 6) are installed later, per phase.

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

**Phase 2** — one drone *autonomously* explores (frontiers → A* → move → repeat):

```bash
python -m scripts.demo_phase2 --map office           # A* by default
python -m scripts.demo_phase2 --map maze --size 120  # smaller maze finishes sooner
python -m scripts.demo_phase2 --planner bfs --no-show # swap planner, headless
```

> A 240×240 **maze** has ~28k reachable cells and width-1 corridors reveal little
> per step, so full coverage needs a large step budget (`--max-steps`, ~90k).
> Office/open_field maps at 240 finish in ~1.5–2k steps. Use a smaller `--size`
> for a quick live maze demo.

## Test

```bash
./scripts/test.sh        # runs pytest in the venv (ROS-safe)
pip install -r requirements-dev.txt   # if pytest is missing
```

## Repository layout

```
configs/        YAML configuration (all constants; no hardcoded values in source)
src/
  world/        ground-truth map generation + LiDAR ray casting
  mapping/      shared occupancy grid (the world model)
  frontier/     frontier detection + clustering
  planning/     hand-written A*, BFS, DFS, UCS + multi-agent allocation
  agents/       agent model + exploration loop
  perception/   dataset loading, detector training, target registration
  metrics/      metric implementations
  experiments/  headless experiment runner
  viz/          rendering / animation
scripts/        runnable demos and plotting scripts
results/        CSVs and figures (raw runs gitignored, finals committed)
tests/          unit tests
docs/           written results and analysis
```

## Status

**Phase 3 complete** — metrics, headless experiment runner, and the first
reportable result (Experiment E1: planner comparison over 480 held-out runs).
Subsequent phases (two-agent coordination, detection) are built and reviewed in
order.

## Reproduce experiments

```bash
python -m scripts.run_e1        # E1 planner sweep -> results/raw/e1_*.csv (parallel)
python -m scripts.plot_e1       # -> results/figures/e1_*.png
```

## Results

**E1 — planner comparison** (A\* vs BFS vs DFS vs UCS, 4 maps × 30 held-out
seeds, 480 runs). A\* reaches the same coverage and near-identical path length as
UCS/BFS while expanding **~8× fewer nodes** (171 vs ~1400 per call) and running
**~7× faster**. DFS is a poor exploration planner — 2.4× longer paths and only
**25.9%** coverage on mazes (it strands the agent). Full write-up:
[`docs/results_e1.md`](docs/results_e1.md); figures in `results/figures/`.

## License

MIT — see [`LICENSE`](LICENSE). Third-party attributions are recorded in
[`CITATIONS.md`](CITATIONS.md).
