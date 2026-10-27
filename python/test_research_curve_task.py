"""Mathematical and protocol checks for the reference-only curve search task."""
import json
import unittest

import numpy as np

from trajectory_research.constraints import derivative_norm_bound
from trajectory_research.curves import quintic_hermite
from trajectory_research.curve_task import (
    CurveCase, DEFAULT_PARAMETERS, END_POSITION, ENDPOINT_ACCELERATION,
    ENDPOINT_VELOCITY, START_POSITION, curve_from_parameters,
    evaluate_curve_parameters, jerk_energy, make_training_curve_cases,
    make_validation_curve_cases, parameter_bounds, subdivided_derivative_norm_bound,
)


class CurveTaskTests(unittest.TestCase):
    def test_anchor_passes_reference_certificates_and_has_finite_json(self):
        result = evaluate_curve_parameters(DEFAULT_PARAMETERS)
        self.assertTrue(result["feasible"], result)
        self.assertLess(result["score"], 1.)
        self.assertEqual(result["score_details"]["failed_cases"], 0)
        self.assertIn("no controller", result["scope"])
        json.dumps(result, allow_nan=False)

    def test_straight_reference_collides_and_cannot_outrank_feasible_anchor(self):
        straight = evaluate_curve_parameters([0, 0, 0, 3])
        anchor = evaluate_curve_parameters(DEFAULT_PARAMETERS)
        self.assertFalse(straight["feasible"])
        self.assertGreaterEqual(straight["score"], 1.)
        self.assertGreater(straight["score"], anchor["score"])
        self.assertTrue(any(row["sphere_certificate"]["status"] == "VIOLATED" for row in straight["cases"]))

    def test_endpoints_preserve_physical_position_velocity_acceleration(self):
        for parameters in ([.3, .8, -.2, 3.1], [-.7, -1.2, .5, 6.8]):
            curve = curve_from_parameters(parameters)
            for u, position in ((0., START_POSITION), (1., END_POSITION)):
                for order, expected in enumerate((position, ENDPOINT_VELOCITY, ENDPOINT_ACCELERATION)):
                    np.testing.assert_allclose(curve.evaluate(u, order, wrt="time"), expected, atol=2e-12)

    def test_energy_integrates_total_jerk_and_matches_independent_exact_quadrature(self):
        segment = curve_from_parameters([.2, .8, -.3, 4.2])
        nodes, weights = np.polynomial.legendre.leggauss(4)
        jerk = segment.evaluate((nodes + 1.) / 2., 3, wrt="time")
        expected = segment.duration_s * np.dot(weights, np.sum(jerk * jerk, axis=1)) / 2.
        self.assertAlmostEqual(jerk_energy(segment), expected, delta=1e-9)
        baseline = quintic_hermite(START_POSITION, ENDPOINT_VELOCITY, ENDPOINT_ACCELERATION,
                                   END_POSITION, ENDPOINT_VELOCITY, ENDPOINT_ACCELERATION, 4.2)
        self.assertGreater(jerk_energy(baseline), 0.)
        self.assertGreater(jerk_energy(segment), jerk_energy(baseline))

    def test_refined_derivative_bounds_enclose_dense_reference_values(self):
        curve = curve_from_parameters([.7, -.9, .6, 3.5])
        u = np.linspace(0., 1., 501)
        for order in (1, 2, 3):
            refined = subdivided_derivative_norm_bound(curve, order)
            self.assertLessEqual(refined, derivative_norm_bound(curve, order) + 1e-12)
            actual = np.max(np.linalg.norm(curve.evaluate(u, order, wrt="time"), axis=1))
            self.assertGreaterEqual(refined + 1e-11, actual)

    def test_case_split_is_explicit_and_bounds_are_defensive(self):
        train, validation = make_training_curve_cases(), make_validation_curve_cases()
        self.assertEqual(len(train), 3)
        self.assertGreaterEqual(len(validation), 8)
        self.assertFalse({c.name for c in train} & {c.name for c in validation})
        self.assertFalse({(c.centre_m, c.radius_m) for c in train} &
                         {(c.centre_m, c.radius_m) for c in validation})
        lower, upper = parameter_bounds()
        lower[:] = 100
        self.assertEqual(parameter_bounds()[0][0], -.8)
        self.assertEqual(upper[3], 7.)

    def test_invalid_inputs_fail_before_evaluation(self):
        for invalid in ([0, .8, 0], [0, .8, 0, 2.9], [0, 1.5, 0, 5],
                        [0, 0, 0, float("nan")], ["0", "0", "0", "5"],
                        [True, True, True, True]):
            with self.subTest(parameters=invalid), self.assertRaises(ValueError):
                evaluate_curve_parameters(invalid)
        for cases in ([], [object()], [CurveCase("bad", (0, 0, 0), -1)],
                      [make_training_curve_cases()[0]] * 2):
            with self.assertRaises(ValueError):
                evaluate_curve_parameters(DEFAULT_PARAMETERS, cases)
        with self.assertRaises(ValueError):
            subdivided_derivative_norm_bound(curve_from_parameters(DEFAULT_PARAMETERS), depth=True)


if __name__ == "__main__":
    unittest.main()
