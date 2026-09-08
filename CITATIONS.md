# Citations & Attributions

This is an academic project. **Every** third-party repository, paper, dataset,
or pre-trained model used must be recorded here as it is introduced — not at the
end. Unattributed reuse is a serious academic-integrity problem.

Each entry should record: what it is, where it came from (URL), its licence, how
it is used in this project, and the date it was introduced (YYYY-MM-DD).

---

## Algorithms & methods (textbook / foundational)

These are standard, hand-implemented from first principles (no code copied); we
still cite the sources our implementations follow.

- **A\*, BFS, DFS, Uniform-Cost Search** — Russell, S. & Norvig, P.,
  *Artificial Intelligence: A Modern Approach*. Course textbook (BCSE306L).
  Used for: path planning and search baselines (`src/planning/`). Implemented
  from scratch. _Introduced: 2026-08-04._
- **Frontier-based exploration** — Yamauchi, B. (1997), "A frontier-based
  approach for autonomous exploration," *IEEE CIRA*. Basis for the
  frontier-detection strategy (`src/frontier/`). _Introduced: Phase 2._
- **Hungarian algorithm (optimal assignment)** — Kuhn, H. W. (1955), "The
  Hungarian method for the assignment problem," *Naval Research Logistics
  Quarterly*. Used for optimal one-to-one agent→frontier allocation
  (`src/planning/allocation.py`), via `scipy.optimize.linear_sum_assignment`
  (we call the library solver; the cost/utility formulation is ours).
  _Introduced: Phase 4._
- **Multi-robot coordinated exploration** — Burgard, W., Moors, M., Stachniss,
  C. & Schneider, F. (2005), "Coordinated multi-robot exploration," *IEEE Trans.
  Robotics*. Conceptual basis for cost-vs-information-gain frontier assignment
  and the redundant-coverage argument (`src/planning/allocation.py`, Phase 4
  experiments). Implemented from the described method; no code copied.
  _Introduced: Phase 4._

## Software libraries

- **NumPy, SciPy, Matplotlib, pandas, OpenCV, PyYAML, tqdm** — standard
  open-source scientific-Python stack (BSD/Apache/MIT-family licences). Used as
  general-purpose tooling; see `requirements.txt`. _Introduced: 2026-08-04._

## Datasets

- _None yet._ (An aerial person-detection dataset will be added in Phase 5;
  its name, source, licence, and access date will be recorded here first.)

## Pre-trained models

- _None yet._ (A small pre-trained detector will be fine-tuned in Phase 5 and
  recorded here.)
