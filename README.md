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

**Phase 0 complete** — repository skeleton, config, and dataset-loader stub in
place. Subsequent phases (simulator core, autonomous exploration, benchmarking,
two-agent coordination, detection) are built and reviewed in order.

## Results

_To be populated from Phase 3 onward. Experiments E1–E5 and their figures will
be summarised here and in `docs/`._

## License

MIT — see [`LICENSE`](LICENSE). Third-party attributions are recorded in
[`CITATIONS.md`](CITATIONS.md).
