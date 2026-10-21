import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from run_single_batch import policy_order
from single_drone_experiment import load_config, summarize, terminal_success, limit_velocity, landing_ready


class SingleDroneExperimentTests(unittest.TestCase):
    def setUp(self):
        self.visual = load_config()["visual"]

    def row(self, **changes):
        value = {"tracking_state": "DETECTED", "detected": True,
                 "truncated": False, "frame_age_s": 0.1,
                 "depth_m": 2.4, "error_xy": (0.03, -0.02)}
        value.update(changes)
        return value

    def test_terminal_criterion_needs_fresh_stable_camera_evidence(self):
        good = [self.row() for _ in range(self.visual["stable_frames"])]
        self.assertTrue(terminal_success(good, self.visual))
        for bad in (self.row(depth_m=None), self.row(frame_age_s=0.8),
                    self.row(error_xy=(0.12, 0)), self.row(truncated=True),
                    self.row(tracking_state="TEMPORARILY_LOST")):
            self.assertFalse(terminal_success(good[:-1] + [bad], self.visual))

    def test_summary_does_not_turn_safe_landing_into_task_success(self):
        rows = [self.row(), self.row(depth_m=3.1), self.row()]
        summary = summarize(rows, 1.2, 12.0, self.visual)
        self.assertFalse(summary["task_success"])
        self.assertEqual(summary["observation_count"], 3)
        self.assertEqual(summary["approach_path_m"], 1.2)

    def test_rejects_unsafe_configuration_before_flight(self):
        source = load_config()
        source["visual"]["frame_max_age_s"] = 4.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsafe visual"):
                load_config(path)

    def test_rejects_endpoint_outside_workspace(self):
        source = load_config()
        source["coordinate_endpoint_world_ned_m"] = [1, 0, -3]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside workspace"):
                load_config(path)

    def test_counterbalanced_pair_order(self):
        self.assertEqual(policy_order(2),
                         ["coordinate", "visual", "visual", "coordinate"])
        with self.assertRaises(ValueError):
            policy_order(0)

    def test_common_speed_cap_limits_combined_three_axis_motion(self):
        applied = limit_velocity([0.15, 0.15, 0.15], 0.15)
        self.assertAlmostEqual(float(np.linalg.norm(applied)), 0.15)
        self.assertTrue(np.allclose(applied / applied[0], [1, 1, 1]))
        self.assertTrue(np.allclose(limit_velocity([0, 0, 0], 0.15), [0, 0, 0]))
        with self.assertRaises(ValueError):
            limit_velocity([float('nan'), 0, 0], 0.15)

    def test_landing_guard_needs_ground_actor_pad_height_and_low_speed(self):
        pad = [-5, -4, -3]
        self.assertTrue(landing_ready([-5, -4, 0.817], 0.0, 0.808, pad, "Ground_5"))
        for position, speed, actor in (
            ([-5, -4, -1], 0.0, "Ground_5"),
            ([-4, -4, 0.817], 0.0, "Ground_5"),
            ([-5, -4, 0.817], 0.1, "Ground_5"),
            ([-5, -4, 0.817], 0.0, "BalloonTarget"),
        ):
            self.assertFalse(landing_ready(position, speed, 0.808, pad, actor))

    def test_matched_speed_config_is_accepted_and_excess_speed_is_rejected(self):
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "single_drone_matched_speed.json")
        self.assertEqual(cfg["safety"]["approach_speed_cap_m_s"], 0.15)
        cfg["safety"]["coordinate_speed_m_s"] = 0.3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_cap.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "common approach speed cap"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
