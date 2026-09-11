import unittest

from analyze_batches import render_markdown, summarize


class BatchAnalysisTests(unittest.TestCase):
    def test_groups_scenarios_and_computes_sample_statistics(self):
        rows = [
            {"scenario": "fallback", "status": "PASS", "min_separation_m": 4.8,
             "max_waiting_drift_m": 0.04, "miss_to_dispatch_s": 10.0},
            {"scenario": "fallback", "status": "FAIL", "min_separation_m": 4.6,
             "max_waiting_drift_m": 0.06, "miss_to_dispatch_s": 12.0},
            {"scenario": "a_hit", "status": "PASS", "min_separation_m": 4.9,
             "max_waiting_drift_m": 0.03, "miss_to_dispatch_s": None},
        ]
        result = {(row["scenario"], row["miss_offset_y_m"]): row for row in summarize(rows)}
        self.assertEqual(result[("fallback", None)]["pass_rate"], 0.5)
        self.assertEqual(result[("fallback", None)]["minimum_separation_m"], 4.6)
        self.assertEqual(result[("fallback", None)]["maximum_waiting_drift_m"], 0.06)
        self.assertEqual(result[("fallback", None)]["mean_miss_to_dispatch_s"], 11.0)
        self.assertIsNotNone(result[("fallback", None)]["stdev_miss_to_dispatch_s"])
        self.assertIsNone(result[("a_hit", None)]["mean_miss_to_dispatch_s"])

    def test_keeps_different_offsets_separate(self):
        rows = [
            {"scenario": "fallback", "miss_offset_y_m": -2.5, "status": "PASS",
             "min_separation_m": 4.9, "max_waiting_drift_m": 0.05, "miss_to_dispatch_s": 11.0},
            {"scenario": "fallback", "miss_offset_y_m": 2.0, "status": "PASS",
             "min_separation_m": 4.8, "max_waiting_drift_m": 0.05, "miss_to_dispatch_s": 14.0},
        ]
        result = summarize(rows)
        self.assertEqual(len(result), 2)
        self.assertEqual({row["miss_offset_y_m"] for row in result}, {-2.5, 2.0})

    def test_markdown_report_discloses_scope(self):
        result = {"unique_runs": 1, "by_scenario": [{
            "scenario": "fallback", "miss_offset_y_m": -2.5, "runs": 1,
            "target_offset_ned_m": [0.0, 0.0, 0.0],
            "passed": 1, "failed": 0, "pass_rate": 1.0,
            "minimum_separation_m": 4.9, "maximum_waiting_drift_m": 0.05,
            "mean_miss_to_dispatch_s": 11.0, "stdev_miss_to_dispatch_s": None,
        }]}
        text = render_markdown(result)
        self.assertIn("-2.5", text)
        self.assertIn("1/1", text)
        self.assertIn("固定 Blocks 场景", text)

    def test_keeps_target_positions_separate(self):
        base = {"scenario": "fallback", "miss_offset_y_m": -2.0, "status": "PASS",
                "min_separation_m": 4.9, "max_waiting_drift_m": 0.05,
                "miss_to_dispatch_s": 11.0}
        rows = [dict(base, target_offset_ned_m=[0.0, 0.0, 0.0]),
                dict(base, target_offset_ned_m=[0.5, 0.0, 0.0])]
        self.assertEqual(len(summarize(rows)), 2)


if __name__ == "__main__":
    unittest.main()
