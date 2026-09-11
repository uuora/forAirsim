import unittest

from analyze_perception import render_markdown, summarize


class PerceptionAnalysisTests(unittest.TestCase):
    def test_groups_frames_by_phase_and_vehicle(self):
        records = [
            {"phase": "APPROACH", "vehicles": {"DroneA": {"detected": True, "tracking_state": "LOST", "depth_m": 4.0, "guidance_advice": "HOLD"}}},
            {"phase": "APPROACH", "vehicles": {"DroneA": {"detected": True, "tracking_state": "DETECTED", "depth_m": 2.0, "guidance_advice": "ADVANCE"},
                                                  "DroneB": {"detected": False, "tracking_state": "LOST"}}},
        ]
        rows = {(row["phase"], row["vehicle"]): row for row in summarize(records)}
        self.assertEqual(rows[("APPROACH", "DroneA")]["raw_detections"], 2)
        self.assertEqual(rows[("APPROACH", "DroneA")]["tracking_states"]["DETECTED"], 1)
        self.assertEqual(rows[("APPROACH", "DroneA")]["depth_median_m"], 3.0)
        self.assertEqual(rows[("APPROACH", "DroneA")]["depth_delta_m"], -2.0)
        self.assertEqual(rows[("APPROACH", "DroneA")]["guidance_advice"]["ADVANCE"], 1)
        self.assertEqual(rows[("APPROACH", "DroneB")]["frames"], 1)

    def test_counts_empty_images_and_discloses_advisory_role(self):
        rows = summarize([{"phase": "TAKEOFF", "vehicles": {
            "DroneA": {"detected": False, "tracking_state": "LOST", "error": "empty_image"}}}])
        self.assertEqual(rows[0]["empty_images"], 1)
        text = render_markdown({"status": "PASS", "scenario": "a_hit"}, rows)
        self.assertIn("感知诊断", text)
        self.assertIn("TAKEOFF", text)


if __name__ == "__main__":
    unittest.main()
