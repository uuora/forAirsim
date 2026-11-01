"""Envelope algebra and reproducibility checks, without optimizer outcomes."""
import unittest

import numpy as np

from trajectory_research.curve_task import evaluate_curve_parameters
from trajectory_research.uncertainty import (
    CENTRE_ERROR_RADIUS_M, CLEARANCE_M, EQUIVALENT_UNCERTAINTY_RADIUS_M,
    NOMINAL_CENTRE_M, NOMINAL_RADIUS_M, RADIUS_INCREMENT_M, STRESS_RADIUS_RANGE_M,
    evaluate_uncertainty_parameters, make_nominal_training_cases,
    make_uncertainty_stress_cases, make_uncertainty_training_cases,
)


class GeometricUncertaintyTests(unittest.TestCase):
    def test_equivalent_radius_has_one_clearance_and_equal_case_count(self):
        nominal = make_nominal_training_cases()
        margin = make_uncertainty_training_cases()
        self.assertEqual(len(nominal), 1)
        self.assertEqual(len(margin), 1)
        self.assertAlmostEqual(margin[0].radius_m, .75)
        self.assertAlmostEqual(margin[0].radius_m,
                               NOMINAL_RADIUS_M + CENTRE_ERROR_RADIUS_M + RADIUS_INCREMENT_M)
        self.assertEqual(nominal[0].clearance_m, CLEARANCE_M)
        self.assertEqual(margin[0].clearance_m, CLEARANCE_M)

    def test_seed_replay_unique_names_and_declared_error_envelope(self):
        first = make_uncertainty_stress_cases(2765)
        second = make_uncertainty_stress_cases(2765)
        self.assertEqual(first, second)
        self.assertNotEqual(first, make_uncertainty_stress_cases(2766))
        self.assertEqual(len({c.name for c in first}), 64)
        for case in first:
            case.validate()
            self.assertLessEqual(np.linalg.norm(np.array(case.centre_m) - NOMINAL_CENTRE_M),
                                 CENTRE_ERROR_RADIUS_M + 1e-14)
            self.assertGreaterEqual(case.radius_m, STRESS_RADIUS_RANGE_M[0])
            self.assertLessEqual(case.radius_m, STRESS_RADIUS_RANGE_M[1])

    def test_uniform_volume_uses_cubed_radius_not_uniform_radius(self):
        cases = make_uncertainty_stress_cases(9245, count=4096)
        radii = np.linalg.norm(np.array([c.centre_m for c in cases]) - NOMINAL_CENTRE_M, axis=1)
        # This checks the intended transform over a fixed deterministic sample,
        # not empirical calibration of a real-world uncertainty distribution.
        self.assertAlmostEqual(float(np.mean((radii / CENTRE_ERROR_RADIUS_M) ** 3)), .5, delta=.025)
        self.assertAlmostEqual(float(np.mean(radii / CENTRE_ERROR_RADIUS_M)), .75, delta=.025)

    def test_triangle_inequality_envelope_implies_each_sample_clearance(self):
        rng = np.random.default_rng(90210)
        directions = rng.normal(size=(32, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        nominal_points = np.array(NOMINAL_CENTRE_M) + directions * (
            EQUIVALENT_UNCERTAINTY_RADIUS_M + CLEARANCE_M + .025)
        cases = make_uncertainty_stress_cases(2854, count=128)
        for case in cases:
            distances = np.linalg.norm(nominal_points - case.centre_m, axis=1)
            nominal_distances = np.linalg.norm(nominal_points - NOMINAL_CENTRE_M, axis=1)
            lower = nominal_distances - CENTRE_ERROR_RADIUS_M
            self.assertTrue(np.all(distances >= lower - 1e-13))
            self.assertTrue(np.all(distances > case.radius_m + CLEARANCE_M))

    def test_margin_feasible_reference_passes_all_generated_stress_cases(self):
        parameters = [0., 1.1, 0., 5.]
        margin = evaluate_uncertainty_parameters(parameters, profile="margin")
        self.assertTrue(margin["feasible"], margin)
        stress = evaluate_curve_parameters(parameters, make_uncertainty_stress_cases(3571))
        self.assertTrue(stress["feasible"], stress)

    def test_profiles_change_feasibility_without_changing_quality(self):
        parameters = [0., .65, 0., 5.]
        nominal = evaluate_uncertainty_parameters(parameters, profile="nominal")
        margin = evaluate_uncertainty_parameters(parameters, profile="margin")
        self.assertTrue(nominal["feasible"], nominal)
        self.assertFalse(margin["feasible"])
        self.assertEqual(nominal["score_details"]["quality_term"], margin["score_details"]["quality_term"])
        self.assertEqual(nominal["score_details"]["total_checks"], margin["score_details"]["total_checks"])
        self.assertEqual(nominal["parameters"], margin["parameters"])

    def test_invalid_generator_arguments_and_profile_fail(self):
        for invalid in (-1, 1.5, True, np.nan, "2"):
            with self.subTest(seed=invalid), self.assertRaises(ValueError):
                make_uncertainty_stress_cases(invalid)
        for invalid in (0, -1, 1.5, True, np.nan, "2"):
            with self.subTest(count=invalid), self.assertRaises(ValueError):
                make_uncertainty_stress_cases(1, count=invalid)
        with self.assertRaises(ValueError):
            evaluate_uncertainty_parameters([0., .8, 0., 5.], profile="wind")


if __name__ == "__main__":
    unittest.main()
