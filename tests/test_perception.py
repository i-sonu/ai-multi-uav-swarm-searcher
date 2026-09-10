"""Unit tests for Phase 5 half B: targets, simulated detector, registration,
and detection metrics. No trained model involved — this is the simulator side."""

import math

import numpy as np

from src.metrics.metrics import detection_metrics, match_registrations
from src.perception.detector import Detection, Detector, SimulatedDetector
from src.perception.registration import TargetRegister
from src.perception.targets import Target, place_targets


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #
def test_targets_are_reachable_distinct_and_deterministic():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True  # a 10x10 reachable block
    a = place_targets(6, seed=42, reachable_mask=mask, resolution=0.25, min_separation_m=0.5)
    b = place_targets(6, seed=42, reachable_mask=mask, resolution=0.25, min_separation_m=0.5)
    assert len(a) == 6
    assert [(t.r, t.c) for t in a] == [(t.r, t.c) for t in b]      # deterministic
    assert len({(t.r, t.c) for t in a}) == 6                        # distinct
    assert all(mask[t.r, t.c] for t in a)                           # reachable


def test_targets_respect_min_separation_when_space_allows():
    mask = np.ones((40, 40), dtype=bool)
    ts = place_targets(4, seed=1, reachable_mask=mask, resolution=0.25, min_separation_m=2.0)
    min_sep_cells = 2.0 / 0.25
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            d = math.hypot(ts[i].r - ts[j].r, ts[i].c - ts[j].c)
            assert d >= min_sep_cells - 1e-9


# --------------------------------------------------------------------------- #
# simulated detector geometry
# --------------------------------------------------------------------------- #
def _det(gt, targets, **kw):
    params = dict(seed=0, resolution=0.25, range_m=6.0, fov_deg=90.0,
                  recall=1.0, localisation_noise_m=0.0, false_positive_rate=0.0)
    params.update(kw)
    return SimulatedDetector(gt, targets, **params)


def test_simulated_detector_satisfies_protocol():
    assert isinstance(_det(np.zeros((5, 5), np.int8), []), Detector)


def test_target_out_of_range_not_seen():
    gt = np.zeros((60, 60), dtype=np.int8)
    t = Target(0, 0, 40)  # world x = 40*0.25 = 10 m away along +x
    d = _det(gt, [t])
    assert d.in_view((0.0, 0.0, 0.0), t) is False  # range is 6 m


def test_target_behind_heading_not_seen():
    gt = np.zeros((20, 20), dtype=np.int8)
    t = Target(0, 0, 8)  # 2 m along +x
    d = _det(gt, [t])
    assert d.in_view((0.0, 0.0, 0.0), t) is True          # facing +x -> seen
    assert d.in_view((0.0, 0.0, math.pi), t) is False     # facing -x -> outside cone


def test_wall_blocks_line_of_sight():
    gt = np.zeros((20, 20), dtype=np.int8)
    gt[:, 4] = 1  # vertical wall at column 4 (x = 1.0 m)
    t = Target(0, 0, 8)  # target at x = 2.0 m, behind the wall
    d = _det(gt, [t])
    assert d.in_view((0.0, 0.0, 0.0), t) is False


def test_recall_zero_returns_no_detections():
    gt = np.zeros((20, 20), dtype=np.int8)
    t = Target(0, 0, 8)
    d = _det(gt, [t], recall=0.0)
    assert d.detect((0.0, 0.0, 0.0), agent_id=0, step=0) == []


def test_detection_is_near_true_position_with_noise():
    gt = np.zeros((20, 20), dtype=np.int8)
    t = Target(0, 0, 8)  # true world (2.0, 0.0)
    d = _det(gt, [t], localisation_noise_m=0.1)
    dets = d.detect((0.0, 0.0, 0.0), 0, 5)
    assert len(dets) == 1
    assert math.hypot(dets[0].x - 2.0, dets[0].y - 0.0) < 1.0
    assert dets[0].step == 5 and dets[0].agent_id == 0


# --------------------------------------------------------------------------- #
# registration / dedup
# --------------------------------------------------------------------------- #
def test_repeat_sightings_fuse_into_one_entry():
    reg = TargetRegister(dedup_radius_m=1.0)
    for step in range(10):  # same target seen 10 frames in a row, jittered
        reg.add(Detection(2.0 + 0.05 * (step % 2), 0.0, "person", 0.8, agent_id=0, step=step))
    assert len(reg) == 1
    e = reg.entries()[0]
    assert e.n_obs == 10 and e.first_step == 0 and e.last_step == 9


def test_two_agents_same_target_fuse_and_record_both():
    reg = TargetRegister(dedup_radius_m=1.0)
    reg.add(Detection(2.0, 0.0, "person", 0.8, agent_id=0, step=1))
    reg.add(Detection(2.1, 0.1, "person", 0.9, agent_id=1, step=3))
    assert len(reg) == 1
    assert reg.entries()[0].agent_ids == {0, 1}
    assert reg.entries()[0].confidence == 0.9  # max confidence kept


def test_far_apart_detections_stay_separate():
    reg = TargetRegister(dedup_radius_m=1.0)
    reg.add(Detection(2.0, 0.0, "person", 0.8, 0, 0))
    reg.add(Detection(9.0, 9.0, "person", 0.8, 0, 0))
    assert len(reg) == 2


# --------------------------------------------------------------------------- #
# detection metrics
# --------------------------------------------------------------------------- #
def test_metrics_true_positive_and_localisation_error():
    targets = [Target(0, 0, 8), Target(1, 0, 40)]  # world (2,0) and (10,0)
    reg = TargetRegister(dedup_radius_m=1.0)
    reg.add(Detection(2.2, 0.0, "person", 0.9, agent_id=0, step=7))  # near target 0
    m = detection_metrics(reg, targets, resolution=0.25, tolerance_m=1.0)
    assert m["true_positives"] == 1 and m["false_negatives"] == 1 and m["false_positives"] == 0
    assert m["fraction_localised"] == 0.5
    assert m["time_to_first_detection"] == 7
    assert abs(m["mean_localisation_error_m"] - 0.2) < 1e-6


def test_metrics_false_positive_counts_and_censored_ttfd():
    targets = [Target(0, 0, 8)]
    reg = TargetRegister(dedup_radius_m=1.0)
    reg.add(Detection(9.0, 9.0, "person", 0.5, 0, 3))  # nowhere near the target
    m = detection_metrics(reg, targets, resolution=0.25, tolerance_m=1.0)
    assert m["true_positives"] == 0 and m["false_positives"] == 1
    assert m["time_to_first_detection"] is None  # never localised -> censored
    assert m["precision"] == 0.0 and m["recall"] == 0.0


def test_match_is_one_to_one():
    targets = [Target(0, 0, 8)]  # single true target
    reg = TargetRegister(dedup_radius_m=0.1)  # tiny radius -> two separate entries
    reg.add(Detection(2.1, 0.0, "person", 0.9, 0, 1))
    reg.add(Detection(1.9, 0.0, "person", 0.9, 0, 2))
    matches, unmatched_e, unmatched_t = match_registrations(reg, targets, 0.25, 1.0)
    assert len(matches) == 1 and len(unmatched_e) == 1 and len(unmatched_t) == 0


# --------------------------------------------------------------------------- #
# end-to-end: detection layer riding on a real exploration run
# --------------------------------------------------------------------------- #
def test_detection_layer_localises_targets_during_exploration():
    """A 2-agent coordinated run with the detector attached should localise most
    placed targets with small error — and, crucially, produce the *same* map
    coverage as a run without the detector (proving perception is decoupled)."""
    from src.agents.agent import Agent
    from src.constants import FREE
    from src.exploration.explorer import explore_team, reachable_free_mask, team_start_cells
    from src.mapping.occupancy_grid import OccupancyGrid
    from src.planning.astar import astar
    from src.world.map_generator import generate_map

    res, size, seed = 0.25, 60, 4000
    gt = generate_map("cluttered", size, seed)
    starts = team_start_cells(gt, 2)
    reach = np.zeros(gt.shape, dtype=bool)
    for s in starts:
        reach |= reachable_free_mask(gt, s)
    targets = place_targets(8, seed, reachable_mask=reach, resolution=res)

    def _run(with_detection):
        grid = OccupancyGrid(size, size, resolution=res)
        agents = [Agent(i, (s[1] * res, s[0] * res, 0.0), resolution=res) for i, s in enumerate(starts)]
        det = reg = None
        if with_detection:
            det = SimulatedDetector(gt, targets, seed=seed, resolution=res, recall=0.9, localisation_noise_m=0.3)
            reg = TargetRegister(dedup_radius_m=1.0)
        explore_team(gt, grid, agents, astar, method="hungarian",
                     n_beams=360, max_range=12.0, max_steps=9000, detector=det, register=reg)
        cov = int(((grid.grid == FREE) & reach).sum())
        return cov, reg

    cov_with, reg = _run(True)
    cov_without, _ = _run(False)

    # Decoupling: attaching the detector must not change exploration coverage.
    assert cov_with == cov_without

    m = detection_metrics(reg, targets, resolution=res, tolerance_m=1.0)
    assert m["fraction_localised"] >= 0.75            # most targets found
    assert m["mean_localisation_error_m"] < 1.0        # within tolerance
    assert m["time_to_first_detection"] is not None    # something was localised
