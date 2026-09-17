import unittest

from pathlib import Path

from batch_validate import aggregate, event_delay_s, mission_row


class BatchAggregateTests(unittest.TestCase):
    def test_aggregate_counts_and_extrema(self):
        rows = [
            {"status": "PASS", "min_separation_m": 4.9, "max_waiting_drift_m": 0.05,
             "miss_to_dispatch_s": None},
            {"status": "FAIL", "min_separation_m": 4.7, "max_waiting_drift_m": 0.08,
             "miss_to_dispatch_s": 11.0},
        ]
        result = aggregate(rows)
        self.assertEqual(result["completed"], 2)
        self.assertEqual(result["passed"], 1)
        self.assertEqual(result["pass_rate"], 0.5)
        self.assertEqual(result["minimum_separation_m"], 4.7)
        self.assertEqual(result["maximum_waiting_drift_m"], 0.08)
        self.assertEqual(result["mean_miss_to_dispatch_s"], 11.0)

    def test_empty_aggregate(self):
        result = aggregate([])
        self.assertEqual(result["completed"], 0)
        self.assertIsNone(result["pass_rate"])

    def test_visual_mission_row_extracts_events_and_evidence(self):
        mission = {
            "status": "PASS",
            "evaluation": {"mode": "visual_fault_injection", "policy_id": "visual_a_first_fallback_v1"},
            "events": [
                {"event": "miss", "monotonic_s": 10.5},
                {"event": "fallback_ready", "monotonic_s": 13.0},
            ],
            "hit_evidence": {"vehicle": "DroneB"},
            "a_visual_records": [{}, {}],
            "b_visual_records": [{}],
            "b_visual_final_records": [{}, {}, {}],
            "b_visual_final_mode": "visual_gate_coordinate_contact",
            "visual_fallback_verified": True,
            "attempt_assessments": {"DroneA": {"state": "FAILED"}},
        }
        row = mission_row(mission, Path("logs/mission_visual/report.json"), 1,
                          "visual_fallback")
        self.assertEqual(event_delay_s(mission), 2.5)
        self.assertEqual(row["run"], "mission_visual")
        self.assertEqual(row["hit_vehicle"], "DroneB")
        self.assertEqual(row["b_visual_final_frames"], 3)
        self.assertEqual(row["evaluation_mode"], "visual_fault_injection")
        self.assertEqual(row["a_attempt_state"], "FAILED")
        result = aggregate([row])
        self.assertEqual(result["coordinate_contact_runs"], 1)
        self.assertEqual(result["visual_collision_runs"], 0)


if __name__ == "__main__":
    unittest.main()
