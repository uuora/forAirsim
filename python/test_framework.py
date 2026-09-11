import unittest
from unittest.mock import Mock

from coordinator import MissionCoordinator
from mission_config import MissionConfig
from target_provider import CameraTargetProvider, GroundTruthTargetProvider, TargetObservation
from mission_state import MissionState


class FrameworkTests(unittest.TestCase):
    def test_config_loads_and_separates_pads(self):
        cfg = MissionConfig.load("configs/mission.json")
        self.assertEqual(cfg.vehicles, ("DroneA", "DroneB"))
        self.assertGreaterEqual(abs(cfg.pads_world_ned_m["DroneA"][1] - cfg.pads_world_ned_m["DroneB"][1]), 2)
        self.assertEqual(cfg.perception_confirm_frames, 2)
        self.assertEqual(cfg.perception_lost_frames, 3)
        self.assertEqual(cfg.perception_alignment_deadband, 0.10)
        self.assertEqual(cfg.perception_stop_distance_m, 1.50)

    def test_ground_truth_provider_is_separate_from_collision_judge(self):
        client = Mock()
        client.simGetObjectPose.return_value.position.x_val = 0
        client.simGetObjectPose.return_value.position.y_val = 0
        client.simGetObjectPose.return_value.position.z_val = -3
        observation = GroundTruthTargetProvider(client, "BalloonTarget").observe("DroneA")
        self.assertEqual(observation, TargetObservation("DroneA", True, "ground_truth"))

    def test_camera_provider_handles_empty_frame(self):
        client = Mock()
        client.simGetImage.return_value = None
        observation, image, result = CameraTargetProvider(client).observe_with_frame("DroneA")
        self.assertFalse(observation.detected)
        self.assertEqual(observation.source, "camera")
        self.assertIsNone(image)
        self.assertIsNone(result)

    def test_camera_provider_estimates_median_depth_inside_detection_box(self):
        import numpy as np
        from unittest.mock import patch

        client = Mock()
        response = Mock(width=8, height=8, image_data_float=np.full(64, 4.25, dtype=np.float32))
        client.simGetImages.return_value = [response]
        selected = {"centre_px": [8, 8], "bbox_xywh": [0, 0, 16, 16], "shape_score": 0.9,
                    "normalised_error_xy": [0.0, 0.0], "truncated": False}
        detection = {"detected": True, "selected": selected, "candidate_count": 1}
        provider = CameraTargetProvider(client)
        with patch.object(provider, "observe_with_frame", return_value=(
                TargetObservation("DroneA", True, "camera", (8, 8), (0, 0, 16, 16), 1,
                                  0.9, (0.0, 0.0), False),
                np.zeros((16, 16, 3), dtype=np.uint8), detection)):
            observation, _, _ = provider.observe_with_depth("DroneA")
        self.assertAlmostEqual(observation.depth_m, 4.25)

    def test_camera_provider_rejects_invalid_depth_payload(self):
        import numpy as np
        from unittest.mock import patch

        client = Mock()
        client.simGetImages.return_value = [Mock(width=8, height=8, image_data_float=[1.0])]
        provider = CameraTargetProvider(client)
        base = TargetObservation("DroneA", True, "camera", (4, 4), (0, 0, 8, 8))
        with patch.object(provider, "observe_with_frame", return_value=(base, np.zeros((8, 8, 3)), {})):
            observation, _, _ = provider.observe_with_depth("DroneA")
        self.assertIsNone(observation.depth_m)

    def test_coordinator_requires_corridor_clear_before_fallback(self):
        coordinator = MissionCoordinator()
        self.assertEqual(coordinator.start().vehicle, "DroneA")
        coordinator.confirm_miss("DroneA")
        with self.assertRaises(ValueError):
            coordinator.confirm_hit("DroneB")
        dispatch = coordinator.mark_corridor_clear()
        self.assertEqual(dispatch.vehicle, "DroneB")
        self.assertEqual(coordinator.confirm_hit("DroneB"), "mission_complete")
        self.assertEqual(coordinator.state, MissionState.HIT_CONFIRMED)


class CoordinatorTerminationTests(unittest.TestCase):
    def test_hit_prevents_fallback(self):
        c = MissionCoordinator()
        c.start()
        c.confirm_hit("DroneA")
        with self.assertRaises(ValueError):
            c.mark_corridor_clear()
        self.assertEqual(c.state, MissionState.HIT_CONFIRMED)

    def test_fallback_timeout_terminates(self):
        c = MissionCoordinator()
        c.start()
        c.confirm_miss("DroneA")
        c.mark_corridor_clear()
        c.timeout()
        self.assertEqual(c.state, MissionState.FAILED)
        with self.assertRaises(ValueError):
            c.start()

    def test_runner_rejects_undispatched_contact(self):
        from run_balloon_mission import Experiment
        runner = Experiment.__new__(Experiment)
        runner.coordinator = MissionCoordinator()
        runner.coordinator.start()
        runner.client = Mock()
        with self.assertRaises(RuntimeError):
            runner.contact("DroneB")
        self.assertEqual(runner.client.mock_calls, [])

    def test_runner_routes_events_through_coordinator(self):
        from run_balloon_mission import Experiment
        runner = Experiment.__new__(Experiment)
        runner.coordinator = MissionCoordinator()
        runner.event = Mock()
        runner.change("start")
        self.assertEqual(runner.active, "DroneA")
        runner.change("miss")
        runner.change("fallback_ready")
        self.assertEqual(runner.active, "DroneB")
        runner.change("hit")
        self.assertEqual(runner.state, MissionState.HIT_CONFIRMED)

    def test_abort_after_hit_marks_execution_failure(self):
        c = MissionCoordinator()
        c.start()
        c.confirm_hit("DroneA")
        c.abort()
        self.assertEqual(c.state, MissionState.FAILED)
        self.assertIsNone(c.active_vehicle)


if __name__ == "__main__":
    unittest.main()
