"""Population-mismatch regressions for the offline comparison evidence gate."""
from copy import deepcopy
from pathlib import Path
import unittest

from perception_trajectory_planner import plan_trajectory
from validate_trajectory_geometry import input_from_config, matched_mode_check


class MatchedModeEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        inputs = input_from_config(root / "configs/phase1_lateral_pilot/center.json", root / "configs/settings.json")
        cls.first = plan_trajectory(**inputs, mode="efficiency")
        cls.second = plan_trajectory(**inputs, mode="perception")
        cls.expected = len(inputs["candidate_endpoints"]) * len(inputs["params"].durations)

    def test_complete_same_input_pair_passes(self):
        self.assertTrue(matched_mode_check(self.first, self.second, self.expected)["passed"])

    def test_truncated_or_equally_incomplete_populations_fail(self):
        second = deepcopy(self.second)
        second["candidates"] = second["candidates"][:1]
        self.assertFalse(matched_mode_check(self.first, second, self.expected)["passed"])
        first = deepcopy(self.first)
        first["candidates"] = first["candidates"][:1]
        self.assertFalse(matched_mode_check(first, second, self.expected)["passed"])

    def test_reordered_candidate_identity_fails(self):
        second = deepcopy(self.second)
        second["candidates"][0], second["candidates"][1] = second["candidates"][1], second["candidates"][0]
        self.assertFalse(matched_mode_check(self.first, second, self.expected)["passed"])


if __name__ == "__main__":
    unittest.main()
