"""Regression tests for candidate evaluation, without AirSim or PyTorch."""
from copy import deepcopy
import json
import unittest

from dual_target_control.core import BezierParameters
from dual_target_control.trajectory_objective import (
    EvaluationCase, _score_rows, evaluate_parameters, make_training_cases,
    make_validation_cases,
)


class TrajectoryObjectiveTests(unittest.TestCase):
    def test_default_closed_loop_case_passes_independent_gates(self):
        case = make_training_cases()[0]
        result = evaluate_parameters(BezierParameters(), [case], dt=0.05)
        self.assertTrue(result["feasible"], result["summary"])
        self.assertLess(result["score"], 1.0)
        self.assertEqual(result["summary"]["first_entries_granted"], 2)
        self.assertIn("not AirSim", result["model"])
        row = result["cases"][0]
        self.assertEqual(row["initial_position_m"], [0.0, 0.0, 8.0])
        self.assertEqual(row["control_config"]["target_distance_min"], 6.0)
        self.assertEqual(row["arrival_radius_m"], 0.5)
        self.assertEqual(row["heading_tolerance_deg"], 20.0)
        json.dumps(result, allow_nan=False)

    def test_invalid_inputs_fail_before_rollout(self):
        for dt in (0, -0.01, float("nan"), True, "0.05"):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                evaluate_parameters(BezierParameters(), dt=dt)
        with self.assertRaises(ValueError):
            evaluate_parameters(BezierParameters(), [])
        with self.assertRaises(ValueError):
            evaluate_parameters([float("nan")] * 5)
        case = make_training_cases()[0]
        with self.assertRaises(ValueError):
            evaluate_parameters(BezierParameters(), [EvaluationCase(case.scenario, velocity_lag_s=0)])

    def test_training_and_validation_seed_split_and_explicit_intervals(self):
        train = make_training_cases()
        validation = make_validation_cases()
        self.assertEqual(len(train), 3)
        self.assertEqual(len(validation), 30)
        self.assertEqual(train[0].scenario.name, "close_range_seed_13")
        self.assertEqual({case.scenario.name for case in validation[-6:]},
                         {"close_range_seed_17", "close_range_seed_23", "close_range_seed_31"})
        self.assertTrue(all(case.intervals_s is not None for case in validation))
        self.assertEqual(train[0].scenario.targets, make_training_cases()[0].scenario.targets)

    def test_any_failed_gate_dominates_even_an_instantaneous_rollout(self):
        # Score arithmetic is tested separately from the real-controller case.
        row = dict(passed=True, elapsed_s=1e8,
                   maxima={"command_jerk_mps3": 1e6},
                   control_config={"max_acceleration": 4.0}, intervals_s=[0.05],
                   checks={"first_entries_granted": True, "command_limits": True})
        feasible_score, feasible, _ = _score_rows([row])
        failed = deepcopy(row)
        failed.update(passed=False, elapsed_s=0.0)
        failed["checks"]["first_entries_granted"] = False
        failed["maxima"]["command_jerk_mps3"] = 0.0
        failed_score, failed_feasible, _ = _score_rows([failed])
        self.assertTrue(feasible)
        self.assertFalse(failed_feasible)
        self.assertLess(feasible_score, 1.0)
        self.assertGreaterEqual(failed_score, 1.0)
        self.assertLess(feasible_score, failed_score)


if __name__ == "__main__":
    unittest.main()
