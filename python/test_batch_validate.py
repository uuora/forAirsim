import unittest

from batch_validate import aggregate


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


if __name__ == "__main__":
    unittest.main()
