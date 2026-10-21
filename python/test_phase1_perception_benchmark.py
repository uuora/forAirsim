import unittest
from benchmark_phase1_perception import cloud_contract, checked_box, latency_summary


class PerceptionProbeTests(unittest.TestCase):
    def test_cloud_ambiguity_and_bad_geometry_are_not_accepted(self):
        box = {"xmin": 1, "ymin": 2, "xmax": 10, "ymax": 20}
        self.assertEqual(cloud_contract({"result_bbox_list": [box]}, 40, 40)["selection_status"], "SINGLE_BOX")
        self.assertEqual(cloud_contract({"result_bbox_list": [box, box]}, 40, 40)["selection_status"], "AMBIGUOUS")
        self.assertEqual(cloud_contract({}, 40, 40)["selection_status"], "NO_BOX")
        for values in ([1, 2, float('nan'), 20], [1, 2, 50, 20], [10, 2, 1, 20]):
            with self.assertRaises(ValueError):
                checked_box(values, 40, 40)

    def test_errors_remain_in_latency_budget_denominator(self):
        rows = [{"m": {"status": "RETURNED", "wall_seconds": .2, "processing_budget_met": True}},
                {"m": {"status": "ERROR", "wall_seconds": 30.0}},
                {"m": {"status": "RETURNED", "wall_seconds": 3.0, "processing_budget_met": False}}]
        summary = latency_summary(rows, "m")
        self.assertEqual(summary["attempted"], 3)
        self.assertEqual(summary["returned"], 2)
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["processing_budget_met"], 1)


if __name__ == "__main__":
    unittest.main()
