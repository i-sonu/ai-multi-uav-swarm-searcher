"""Target registration, sensor projection, and spatial deduplication (Task 5.2).

Simulates the onboard camera sensor (footprint-based detection) and a centralized
target registry with coordinate fusion/deduplication.
"""

from __future__ import annotations

import math
import numpy as np


class CameraSensor:
    def __init__(
        self,
        range_m: float = 6.0,
        recall_person: float = 0.26,
        recall_vehicle: float = 0.40,
        loc_noise_std_m: float = 0.3,
    ):
        """
        Args:
            range_m: side length of the square downward-facing camera footprint in meters.
            recall_person: probability of detecting a person (class 1) in FOV.
            recall_vehicle: probability of detecting a vehicle (class 4) in FOV.
            loc_noise_std_m: standard deviation of localization Gaussian noise in meters.
        """
        self.range_m = float(range_m)
        self.recalls = {1: float(recall_person), 4: float(recall_vehicle)}
        self.loc_noise_std_m = float(loc_noise_std_m)

    def sense_targets(
        self,
        agent_pose: tuple[float, float, float],
        agent_id: int,
        targets: list[object],
        seed: int | None = None,
    ) -> list[dict]:
        """Detect targets within the agent's camera footprint.

        Args:
            agent_pose: (x, y, theta) of the agent in meters.
            agent_id: ID of the sensing agent.
            targets: list of Target instances in the world.
            seed: optional random seed for determinism.

        Returns:
            list of detection dicts: [{"x": x, "y": y, "class_id": cid, "confidence": conf, "agent_id": aid}]
        """
        ax, ay, _ = agent_pose
        half_range = self.range_m / 2.0

        # Define bounding box for camera footprint
        min_x, max_x = ax - half_range, ax + half_range
        min_y, max_y = ay - half_range, ay + half_range

        detections = []
        # Use random state for reproducible probabilistic detections
        rng = np.random.RandomState(seed)

        for target in targets:
            # Check if target is inside the camera footprint
            if min_x <= target.x <= max_x and min_y <= target.y <= max_y:
                recall_prob = self.recalls.get(target.class_id, 0.30)
                
                # Check if detected (Recall probability)
                if rng.rand() < recall_prob:
                    # Add localization coordinate noise
                    dx = rng.normal(0, self.loc_noise_std_m)
                    dy = rng.normal(0, self.loc_noise_std_m)
                    
                    # Simulated confidence score
                    confidence = float(rng.uniform(0.50, 0.95))

                    detections.append({
                        "x": float(target.x + dx),
                        "y": float(target.y + dy),
                        "class_id": int(target.class_id),
                        "confidence": confidence,
                        "agent_id": int(agent_id),
                    })

        return detections


class TargetRegister:
    def __init__(self, dedup_threshold_m: float = 2.0):
        """
        Args:
            dedup_threshold_m: distance in meters below which repeat target sightings are merged.
        """
        self.dedup_threshold_m = float(dedup_threshold_m)
        self.records: list[dict] = []
        self.class_names = {1: "person", 4: "vehicle"}

    def register_detection(self, det: dict, step_idx: int) -> None:
        """Add or merge a target detection into the register.

        Uses confidence-weighted averaging to merge locations of repeat sightings.
        """
        cx = det["x"]
        cy = det["y"]
        cid = det["class_id"]
        c_conf = det["confidence"]
        aid = det["agent_id"]

        best_match = None
        min_dist = self.dedup_threshold_m

        # Search for closest registered target of the same category
        for record in self.records:
            if record["class_id"] != cid:
                continue
            dist = math.hypot(record["x"] - cx, record["y"] - cy)
            if dist < min_dist:
                min_dist = dist
                best_match = record

        if best_match is not None:
            # Merge coordinates using confidence-weighted average
            w_old = best_match["confidence"]
            w_new = c_conf
            best_match["x"] = float((best_match["x"] * w_old + cx * w_new) / (w_old + w_new))
            best_match["y"] = float((best_match["y"] * w_old + cy * w_new) / (w_old + w_new))
            # Take max confidence
            best_match["confidence"] = float(max(w_old, w_new))
            best_match["last_updated"] = int(step_idx)
            best_match["agents"].add(aid)
        else:
            # Register new target
            self.records.append({
                "id": len(self.records),
                "x": cx,
                "y": cy,
                "class_id": cid,
                "class_name": self.class_names.get(cid, "unknown"),
                "confidence": c_conf,
                "last_updated": step_idx,
                "agents": {aid},
            })

    def get_registered_targets(self) -> list[dict]:
        """Return all target records currently in the register."""
        return self.records
