# CLAUDE.md — Project Instructions

> **Read this file fully before writing any code.** It defines the project, the build order, and the working protocol you must follow.

---

## 1. Project Overview

**Repository:** https://github.com/i-sonu/ai-uav-swarm-searcher

**What we are building:** A coordinated two-agent (two-UAV) autonomous exploration system for unknown environments, with online target localisation.

Two simulated drones are placed in an environment for which no map exists. They cooperatively build a shared occupancy grid map, coordinate so they do not waste effort exploring the same region, and simultaneously run an object detector that records **where** each detected target was found on that same shared map.

**Context:** This is a university course project for BCSE306L Artificial Intelligence (Russell & Norvig syllabus). It is assessed by written report + demo across three review milestones. Academic rigour matters as much as working code — every experiment must be reproducible and every result must be measured, not asserted.

**Team:** 2 students. Timeline: ~5 months, staged.

### The core loop (understand this before coding)

1. The world is discretised into a grid. Every cell starts `UNKNOWN`.
2. Each agent's simulated LiDAR ray-casts outward, marking cells `FREE` along each ray and `OCCUPIED` at the hit point.
3. **Frontiers** = cells that are `FREE` and adjacent to at least one `UNKNOWN` cell. These are the reachable boundaries of current knowledge.
4. Frontiers are clustered; each cluster centroid is a candidate goal.
5. **Task allocation** assigns *distinct* frontiers to the two agents based on travel cost and expected information gain. ← **primary contribution**
6. **A\*** plans a path over the grid to each agent's assigned frontier.
7. Agents move, sense, and the loop repeats until no frontiers remain.
8. In parallel, a trained detector processes camera frames; detections are projected to grid coordinates using agent pose and written to a shared **target register**. ← **secondary contribution**

### Syllabus mapping (do not lose these — they are why the project scores)

| Component | Course module |
|---|---|
| Agent-environment loop | M1 Introduction / Intelligent Agents |
| BFS, DFS, UCS, A\* | M2 Problem Solving by Searching |
| Local search / heuristic frontier selection | M3 Local Search |
| Multi-agent task allocation | M6 Planning (Multiagent planning) |
| Object detection | M7 Perception / Object Recognition |
| Detector training | Learning from data |

---

## 2. WORKING PROTOCOL — follow this strictly

This is the most important section. Violating it wastes the user's time.

### 2.1 Stop at every phase boundary

When you finish a phase:

1. Print a clear banner: `=== PHASE N COMPLETE ===`
2. List what was built (files created/changed, in one line each).
3. State the acceptance criteria for that phase and whether each is met.
4. **Explicitly tell the user the phase is done and stop.**
5. **Do not begin the next phase** until the user says to continue.

### 2.2 Ask the user to verify — do not self-certify

Whenever a phase produces something visual, interactive, or judgement-dependent, **ask the user to look at it and confirm** before proceeding. Examples of things you must ask about:

- Any matplotlib animation or rendered map ("please run `python -m src.viz.render_demo` and tell me if the map fills in correctly")
- Any result that looks surprising or too good
- Any design decision with more than one reasonable option
- Anything requiring credentials, dataset downloads, GPU access, or account setup
- Anything requiring installing heavy dependencies (ROS2, Gazebo, PyTorch/CUDA)

Phrase these as direct requests: *"Please run X and tell me what you see before I continue."*

### 2.3 Never skip ahead

Phases are ordered by dependency and by risk. Phases 1–3 alone constitute a complete, submittable project. Do not start Phase 4 work while Phase 3 is incomplete, even if it seems more interesting. If the user asks to skip ahead, flag the dependency risk once, then comply.

### 2.4 Commit discipline

- Commit at the end of each numbered task, not just each phase.
- Conventional commit messages: `feat(frontier): add connected-component clustering`
- Never commit datasets, model weights, `__pycache__`, or large result artifacts. Keep `.gitignore` current.

### 2.5 Ask before assuming

If a spec detail in this file is ambiguous or seems wrong, ask rather than guessing. A wrong assumption baked into Phase 2 costs days by Phase 5.

---

## 3. Technical Constraints

**Language:** Python 3.10+

**Phase 1–5 dependencies (keep minimal):**
`numpy`, `matplotlib`, `scipy`, `opencv-python`, `pandas`, `pyyaml`, `tqdm`

**Phase 5 adds:** `torch`, `torchvision`, `ultralytics` (or equivalent)

**Phase 6 adds:** ROS 2 (Humble or Jazzy), Gazebo — **do not install these before Phase 6, and ask the user first.**

### Hard rules

- **Write A\*, BFS, DFS and UCS yourself.** Do not import a pathfinding library. The user will be examined on these implementations in a viva. Comment them clearly.
- **Do not use SLAM.** Agent pose is taken as ground truth from the simulator. This is a deliberate scoping decision, not an oversight.
- **2-D only.** Fixed flight altitude. No 3-D volumetric mapping.
- **Exactly 2 agents** in the headline configuration, but write the code to take `n_agents` as a parameter so N=1 and N=3 can be run for ablations.
- **Determinism:** every experiment takes a seed. Same seed must reproduce identical results. This is non-negotiable for the report.
- **No hardcoded paths.** Config via `configs/*.yaml`.

### Grid conventions (use these exact values everywhere)

```python
UNKNOWN  = -1
FREE     =  0
OCCUPIED =  1
```

- Grid resolution: `0.25 m` per cell
- Default environment: `60 m x 60 m` → `240 x 240` cells
- LiDAR: 360 beams, 12 m max range
- Camera: 640 x 480, 5 Hz
- Connectivity for planning: 8-connected, diagonal cost `sqrt(2)`

---

## 4. Repository Structure

Create this in Phase 0.

```
ai-uav-swarm-searcher/
├── CLAUDE.md                  # this file
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── default.yaml
│   └── experiments/
├── src/
│   ├── __init__.py
│   ├── world/
│   │   ├── map_generator.py   # ground-truth environments
│   │   └── sensor.py          # LiDAR ray casting
│   ├── mapping/
│   │   └── occupancy_grid.py  # shared world model
│   ├── frontier/
│   │   ├── detection.py
│   │   └── clustering.py
│   ├── planning/
│   │   ├── astar.py
│   │   ├── baselines.py       # BFS, DFS, UCS
│   │   └── allocation.py      # ← primary contribution
│   ├── agents/
│   │   └── agent.py
│   ├── perception/
│   │   ├── data_loader.py
│   │   ├── train.py
│   │   └── registration.py    # ← secondary contribution
│   ├── metrics/
│   │   └── metrics.py
│   ├── experiments/
│   │   └── runner.py
│   └── viz/
│       └── render.py
├── scripts/
├── results/                   # CSVs and figures (gitignore raw, commit finals)
├── tests/
└── docs/
```

---

## 5. Metric Definitions (implement exactly as written)

| Metric | Definition |
|---|---|
| **Coverage %** | `(cells != UNKNOWN) / (traversable cells in ground truth)` × 100. Denominator excludes cells unreachable from the start pose. |
| **Time-to-90%-coverage** | Number of simulation steps until coverage ≥ 90%. If never reached, record as censored, not as a large number. |
| **Total path length** | Sum over agents of Euclidean distance travelled, in metres. |
| **Redundant-coverage ratio** | `(cells observed by >1 agent) / (total observed cells)`. **This is the key metric proving coordination works.** |
| **Nodes expanded** | Count of nodes popped from the open set, per planner call. |
| **Planning wall-clock** | Seconds per planner call. Report mean and p95. |
| **mAP@50** | Standard detection metric, on a held-out split, with small-object subset reported separately. |
| **Time-to-first-detection** | Steps until the first true-positive target is registered. |
| **Localisation error** | Euclidean distance between registered and true target position, in metres. |

---

## 6. BUILD PHASES

---

### PHASE 0 — Repository setup

**Goal:** a clean, runnable skeleton.

1. Initialise the directory structure in §4. Add `__init__.py` files.
2. Write `requirements.txt` (Phase 1–4 dependencies only).
3. Write `.gitignore` (Python, venv, `data/`, `*.pt`, `results/raw/`, IDE files).
4. Write `README.md`: one-paragraph problem statement, install instructions, repo layout, and a placeholder Results section.
5. Write `configs/default.yaml` with the constants from §3.
6. Write a stub `src/perception/data_loader.py` that can list and load images from a directory. *(Required by the DA1 submission checklist — the repo must contain dataset-loading code.)*
7. Add a `LICENSE` and a `CITATIONS.md` file. **Any third-party repo, paper, or dataset used must be recorded in `CITATIONS.md` as it is introduced.** This is an academic project; unattributed reuse is a serious problem.
8. Verify `python -c "import src"` works.

**Acceptance:** repo structure exists, imports work, config loads, first commit pushed.

**→ STOP. Report completion. Ask the user to confirm the repo looks right on GitHub.**

---

### PHASE 1 — 2-D simulator core

**Goal:** a drone on a fixed path with the map filling in live. This is the Review 1 demo.

1. `world/map_generator.py`:
   - `generate_map(kind, size, seed) -> np.ndarray` of 0/1 (free/wall).
   - Kinds: `office` (rooms + corridors), `maze`, `open_field` (sparse obstacles), `cluttered`.
   - Deterministic given seed. Always add a solid border wall.
2. `mapping/occupancy_grid.py`:
   - `OccupancyGrid` class wrapping a `(H, W)` int8 array initialised to `UNKNOWN`.
   - Methods: `update_cell`, `get_coverage`, `world_to_grid`, `grid_to_world`.
   - Track a per-cell observation count (needed later for redundant-coverage).
3. `world/sensor.py`:
   - `cast_rays(ground_truth, pose, n_beams=360, max_range=12.0)`.
   - Bresenham or DDA line marching. Mark `FREE` along the ray, `OCCUPIED` at the hit.
   - Return the list of updated cells; the grid update itself lives in the mapping module.
4. `agents/agent.py`:
   - `Agent` class: `id`, `pose (x, y, theta)`, `path`, `step()`, `sense()`.
   - Movement is a simple step along the current path — no dynamics, no controller.
5. `viz/render.py`:
   - Render the occupancy grid (grey unknown / white free / black occupied), agent positions, and current path.
   - `matplotlib.animation` for live view; also support saving frames to disk.
6. `scripts/demo_phase1.py`: one agent following a hardcoded waypoint list, map filling in.
7. Tests: sensor never marks cells beyond max range; ray casting is symmetric; coverage increases monotonically.

**Acceptance:** running the demo shows an agent moving with the map progressively revealed.

**→ STOP. Report completion. Ask the user to run `scripts/demo_phase1.py` and confirm the visualisation looks correct before proceeding.**

---

### PHASE 2 — Single-agent autonomous exploration

**Goal:** one drone explores a full map on its own.

1. `frontier/detection.py`:
   - `find_frontiers(grid) -> boolean mask`. A cell is a frontier if `FREE` and has ≥1 `UNKNOWN` 4-neighbour.
   - Vectorise with numpy — this runs every step.
2. `frontier/clustering.py`:
   - Group adjacent frontier cells via connected components (`scipy.ndimage.label`) or DBSCAN.
   - Discard clusters smaller than `min_cluster_size` (config, default 3 cells) as sensor noise.
   - Return centroids as `(K, 2)`.
3. `planning/astar.py`:
   - **Hand-written.** 8-connected grid, `heapq` priority queue, Euclidean heuristic, diagonal cost `sqrt(2)`.
   - Treat `UNKNOWN` as traversable (optimistic) — document this choice, it matters for exploration.
   - Return `(path, cost, nodes_expanded, wall_clock)`.
   - Comment the algorithm thoroughly. The user will be examined on it.
4. `planning/baselines.py`:
   - BFS, DFS, UCS behind the **same interface** as A\* so planners are swappable via config.
5. Exploration loop in `agents/` or a new `src/exploration/explorer.py`:
   - detect frontiers → cluster → select nearest centroid → plan → move → sense → repeat.
   - Terminate when no frontiers remain, or on a step limit.
   - Handle unreachable frontiers: blacklist after N failed plans.
6. `scripts/demo_phase2.py`: full autonomous single-agent exploration run.
7. Tests: A\* returns optimal paths on hand-checkable grids; A\* expands no more nodes than UCS on the same problem; all four planners return the same path *cost* on a uniform-cost grid.

**Acceptance:** one agent autonomously reaches ≥95% coverage on `office` and `maze` maps without intervention.

**→ STOP. Report completion and the coverage figures achieved. Ask the user to watch one full run before proceeding.**

---

### PHASE 3 — Metrics and benchmarking

**Goal:** the first real, reportable result.

1. `metrics/metrics.py`: implement every metric in §5 exactly as defined.
2. Instrument the exploration loop to log per-step: step index, coverage %, path length, nodes expanded, planner wall-clock.
3. `experiments/runner.py`:
   - Run any `(planner, map_kind, seed, n_agents)` combination **headlessly** (no rendering).
   - Write tidy CSV to `results/raw/`.
   - Support parallel execution across seeds.
4. Define and freeze a **held-out seed set**: 30 seeds per map kind, stored in `configs/experiments/heldout_seeds.yaml`. Never tune on these.
5. Run **Experiment E1**: A\* vs BFS vs DFS vs UCS, 4 map kinds × 30 seeds.
6. `scripts/plot_e1.py`: produce publication-quality figures — coverage-over-time curves, nodes-expanded bar chart, planning-time distribution.
7. Write `docs/results_e1.md` summarising findings in prose.

**Acceptance:** E1 runs end-to-end from a clean checkout and reproduces identical numbers on re-run.

**→ STOP. Report completion and show the user the E1 headline numbers and figures. Ask whether the results look plausible before proceeding.**

> **At this point the project is complete and submittable.** Everything beyond here is upside.

---

### PHASE 4 — Two-agent coordination ← PRIMARY CONTRIBUTION

**Goal:** coordinated exploration that measurably beats uncoordinated exploration.

1. Generalise the exploration loop to `n_agents`. One shared `OccupancyGrid` written by all agents. **Verify N=1 results are unchanged** — this is a regression check.
2. Track per-cell observation counts by agent so redundant-coverage can be computed.
3. Implement **Baseline B2 — uncoordinated**: each agent independently picks its nearest frontier. Measure it. You should observe agents duplicating work; confirm this before continuing.
4. `planning/allocation.py`:
   - `build_cost_matrix(agents, frontiers) -> (n_agents, K)` — A\* path cost from each agent to each frontier.
   - `build_utility_matrix(grid, frontiers) -> (n_agents, K)` — information gain = count of `UNKNOWN` cells within sensor range of the frontier.
   - `allocate(cost, utility, method) -> assignment (n_agents,)`.
   - Methods, implement in this order:
     - `greedy` — highest score first, remove that frontier, repeat.
     - `hungarian` — optimal assignment via `scipy.optimize.linear_sum_assignment`.
     - `auction` — iterative bidding (optional, good report material).
   - Score: `score = utility - lambda * cost`. `lambda` is a config parameter and gets its own sensitivity sweep.
   - **Must enforce distinct assignment.** Handle `K < n_agents` gracefully.
5. Re-plan policy: reassign when an agent reaches its goal, when its goal is no longer a frontier, or every `N` steps (config).
6. Run **Experiment E2 — allocation ablation**: `none (B2)` vs `greedy` vs `hungarian`, 4 map kinds × 30 seeds. Report time-to-90%-coverage and redundant-coverage ratio.
7. Run **Experiment E3 — team size**: N=1, 2, 3 at matched total flight time.
8. Run **Experiment E4 — lambda sweep**.
9. Write `docs/results_e2_e4.md`.

**Acceptance:** E2 shows a statistically meaningful reduction in redundant coverage for coordinated vs uncoordinated. **If it does not, report that honestly** — a well-analysed negative result is a legitimate finding and must not be hidden or tuned away on held-out seeds.

**→ STOP. Report completion with the E2/E3/E4 headline numbers. Ask the user to review the results before proceeding.**

---

### PHASE 5 — Detection and target registration

**Goal:** drones explore *and* record what they find.

1. **Ask the user first:** confirm GPU availability and which dataset they can access. Do not download large datasets without asking.
2. `perception/data_loader.py`: flesh out the Phase 0 stub into a real loader for the chosen aerial person-detection dataset. Record dataset name, source, licence and access date in `CITATIONS.md`.
3. `perception/train.py`: fine-tune a small pre-trained detector. **Do not train from scratch.** Log training curves.
4. Evaluate: mAP@50, mAP@50-95, precision/recall, with a **separate small-object subset** breakdown. Save to `results/`.
5. Add targets to the simulator: place `M` targets at random free cells; give each agent a camera footprint (a cone or rectangle ahead of/below it).
6. `perception/registration.py`:
   - Given a detection and agent pose, compute the target's grid coordinate.
   - Write to a shared `TargetRegister`: `(grid_x, grid_y, class, confidence, timestamp, agent_id)`.
   - **Deduplicate** across agents and across repeat sightings (distance threshold + confidence merge).
7. New metrics: time-to-first-detection, fraction of targets localised within tolerance, localisation error.
8. Run **Experiment E5**: does exploration strategy affect detection outcomes?
9. Update the visualisation to pin registered targets on the map.

**Acceptance:** a full run produces both a complete map and a correct, deduplicated target register.

**→ STOP. Report completion with detection metrics. Ask the user to review before proceeding.**

---

### PHASE 6 — ROS 2 / Gazebo port (optional upside)

**Goal:** a high-fidelity demo. **Do not start without the user's explicit go-ahead** — this phase has the highest setup-failure risk in the project.

1. **Ask the user** to confirm ROS 2 distro, OS, and that they have a working install. Guide them through setup if not; do not attempt to install system packages unprompted.
2. Get one drone model spawning and flying to a commanded waypoint. Nothing else until this works.
3. Wrap existing modules as ROS 2 nodes. **The algorithms do not change** — only the I/O boundary. Keep `src/` importable and unchanged; put ROS glue in a separate `ros2_ws/`.
4. Namespace two drones (`/drone1`, `/drone2`) with a single shared map node.
5. Record demo video. **Never live-demo Gazebo at a review.**

**Acceptance:** two drones exploring in Gazebo, recorded to video.

**→ STOP. Report completion and ask the user to review the recording.**

---

### PHASE 7 — Reproducibility and report artifacts

1. Freeze the code. Tag a release.
2. **Re-run every experiment from a clean clone** in a fresh venv. Confirm numbers match. Fix any non-determinism found.
3. Regenerate all final figures at publication resolution into `results/figures/`.
4. Write `docs/RESULTS.md`: every experiment, its setup, its numbers, and honest discussion including negative results and limitations.
5. Finalise `README.md`: exact reproduction instructions, one command per experiment.
6. Finalise `CITATIONS.md`: every repo, paper, dataset and pre-trained model used.
7. Generate a contribution summary from git history for the report's contribution matrix.

**Acceptance:** a stranger can clone the repo and reproduce every figure.

**→ STOP. Report completion.**

---

## 7. Standing Reminders

- **Honesty over impressiveness.** If a result is weak, say so. Fabricated or cherry-picked numbers are worse than a modest honest finding, and the user will be questioned on them in a viva.
- **Never tune on held-out seeds.** Use separate development seeds for tuning.
- **Explain, don't just implement.** When you write a non-obvious algorithm, add a docstring explaining *why* this approach, not just what it does. The user needs to defend every choice orally.
- **Prefer clarity over cleverness.** This code will be read by examiners.
- **Attribute everything** in `CITATIONS.md` as you go, not at the end.
- **When in doubt, ask the user.**
