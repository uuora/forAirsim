"""Deterministic failure-path checks for the first-version mission contract."""
import json
import tempfile
import unittest
from pathlib import Path

from coordinator import MissionCoordinator
from mission_config import MissionConfig


class FailurePathTests(unittest.TestCase):
    def test_fallback_timeout_is_terminal(self):
        c = MissionCoordinator()
        c.start()
        c.confirm_miss("DroneA")
        c.mark_corridor_clear()
        self.assertEqual(c.timeout(), "mission_failed")
        self.assertEqual(c.state.value, "FAILED")

    def test_fallback_cannot_dispatch_before_corridor_clear(self):
        c = MissionCoordinator()
        c.start()
        c.confirm_miss("DroneA")
        with self.assertRaises(ValueError):
            c.confirm_hit("DroneB")

    def test_invalid_safety_configuration_is_rejected(self):
        source = json.loads(Path("configs/mission.json").read_text(encoding="utf-8"))
        source["safety"]["minimum_drone_separation_m"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(ValueError):
                MissionConfig.load(path)

    def test_invalid_staging_geometry_is_rejected(self):
        source = json.loads(Path("configs/mission.json").read_text(encoding="utf-8"))
        source["staging_world_ned_m"]["DroneB"] = source["staging_world_ned_m"]["DroneA"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(ValueError):
                MissionConfig.load(path)

    def test_unsafe_deliberate_miss_offset_is_rejected(self):
        source = json.loads(Path("configs/mission.json").read_text(encoding="utf-8"))
        source["test_parameters"]["deliberate_miss_offset_y_m"] = 0.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(ValueError):
                MissionConfig.load(path)

    def test_invalid_perception_threshold_is_rejected(self):
        source = json.loads(Path("configs/mission.json").read_text(encoding="utf-8"))
        source["perception"]["lost_frames"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(ValueError):
                MissionConfig.load(path)

    def test_invalid_guidance_threshold_is_rejected(self):
        source = json.loads(Path("configs/mission.json").read_text(encoding="utf-8"))
        source["perception"]["alignment_deadband"] = 1.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(ValueError):
                MissionConfig.load(path)

    def test_runner_rejects_unsafe_runtime_miss_offset(self):
        from run_balloon_mission import Experiment
        with self.assertRaises(ValueError):
            Experiment("fallback", miss_offset_y=0.5)

    def test_runner_rejects_target_offset_outside_test_region(self):
        from run_balloon_mission import Experiment
        with self.assertRaises(ValueError):
            Experiment("fallback", target_offset=[2.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
