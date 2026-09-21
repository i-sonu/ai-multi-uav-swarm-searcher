"""Unit tests for TargetRegister, CameraSensor, and spatial target deduplication."""

import numpy as np
from src.world.targets import Target
from src.perception.registration import CameraSensor, TargetRegister


def test_camera_sensor_visibility():
    # Camera at (10.0, 10.0) with range 6.0 (boundaries: [7.0, 13.0])
    sensor = CameraSensor(
        range_m=6.0,
        recall_person=1.0,
        recall_vehicle=1.0,
        loc_noise_std_m=0.0,
    )

    targets = [
        Target(id=0, x=9.0, y=9.0, class_id=1, class_name="person"),
        Target(id=1, x=6.5, y=10.0, class_id=4, class_name="vehicle"),  # Outside
    ]

    detections = sensor.sense_targets(
        agent_pose=(10.0, 10.0, 0.0),
        agent_id=0,
        targets=targets,
        seed=42,
    )
    assert len(detections) == 1
    assert detections[0]["class_id"] == 1
    assert detections[0]["x"] == 9.0
    assert detections[0]["y"] == 9.0


def test_target_register_dedup():
    register = TargetRegister(dedup_threshold_m=2.0)

    # First detection of person at (5.0, 5.0)
    det1 = {"x": 5.0, "y": 5.0, "class_id": 1, "confidence": 0.8, "agent_id": 0}
    register.register_detection(det1, step_idx=10)
    assert len(register.get_registered_targets()) == 1

    # Second detection of same person close by (5.2, 4.8) -> should merge
    det2 = {"x": 5.2, "y": 4.8, "class_id": 1, "confidence": 0.9, "agent_id": 1}
    register.register_detection(det2, step_idx=11)

    records = register.get_registered_targets()
    assert len(records) == 1
    # Check weighted coordinates:
    # x = (5.0 * 0.8 + 5.2 * 0.9) / 1.7 = 5.105882
    # y = (5.0 * 0.8 + 4.8 * 0.9) / 1.7 = 4.894117
    assert abs(records[0]["x"] - 5.105882) < 1e-4
    assert abs(records[0]["y"] - 4.894117) < 1e-4
    assert records[0]["confidence"] == 0.9
    assert records[0]["agents"] == {0, 1}

    # Third detection of vehicle close by -> should NOT merge (different class)
    det3 = {"x": 5.0, "y": 5.0, "class_id": 4, "confidence": 0.7, "agent_id": 0}
    register.register_detection(det3, step_idx=12)
    assert len(register.get_registered_targets()) == 2
