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

## Run the Phase 1 demo

One drone follows a fixed patrol path while its LiDAR reveals the map:

```bash
source venv/bin/activate
python -m scripts.demo_phase1                       # live window
python -m scripts.demo_phase1 --map office --seed 1 # pick map / seed
python -m scripts.demo_phase1 --save results/raw/demo.gif --no-show  # headless
```

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

**Phase 1 complete** — 2-D simulator core: map generation, shared occupancy
grid, LiDAR ray casting, agent model, rendering, and a single-agent patrol demo.
Subsequent phases (autonomous exploration, benchmarking, two-agent coordination,
detection) are built and reviewed in order.

## Results

_To be populated from Phase 3 onward. Experiments E1–E5 and their figures will
be summarised here and in `docs/`._

## License

MIT — see [`LICENSE`](LICENSE). Third-party attributions are recorded in
[`CITATIONS.md`](CITATIONS.md).
