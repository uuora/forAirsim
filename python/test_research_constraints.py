"""Numerical checks of conservative reference-curve certificates and budgets."""
import math
import unittest

import numpy as np

from trajectory_research.curves import PolynomialSegment, bezier_segment, c2_bump, quintic_hermite
from trajectory_research.constraints import (
    BUMP_DERIVATIVE_MAXIMA, c2_bump_amplitude_budget, convex_corridor_certificate,
    derivative_control_points, derivative_norm_bound, quintic_bump_jerk_energy_increment,
    sphere_clearance_certificate,
)


def energy(segment):
    # Four Gauss-Legendre nodes exactly integrate squared cubic jerk (degree 6).
    nodes, weights = np.polynomial.legendre.leggauss(4)
    jerk = segment.evaluate((nodes + 1) / 2, 3, wrt="time")
    return float(segment.duration_s * np.dot(weights, np.sum(jerk * jerk, axis=1)) / 2)


class ConstraintTests(unittest.TestCase):
    def setUp(self):
        self.segment = quintic_hermite([0, 0, 5], [2, 0, 0], [0, 0, 0],
                                      [8, 0, 5], [2, 0, 0], [0, 0, 0], 4)

    def test_derivative_controls_match_curve_and_enclose_norm(self):
        rng = np.random.default_rng(1701)
        u = np.linspace(0, 1, 201)
        for _ in range(8):
            segment = bezier_segment(rng.normal(size=(7, 3)), duration_s=2.3)
            for order in range(4):
                control = derivative_control_points(segment, order)
                actual = segment.evaluate(u, order, wrt="time")
                np.testing.assert_allclose(bezier_segment(control).evaluate(u), actual, atol=2e-10)
                self.assertLessEqual(float(np.max(np.linalg.norm(actual, axis=1))),
                                     derivative_norm_bound(segment, order) + 2e-10)

    def test_derivative_bounds_scale_with_physical_duration(self):
        slow = PolynomialSegment(self.segment.coefficients, 8)
        for order in range(4):
            np.testing.assert_allclose(derivative_control_points(slow, order),
                derivative_control_points(self.segment, order) / 2 ** order, atol=1e-13)
        zero = PolynomialSegment(np.array([[4., 2., 1.]]), 3)
        self.assertEqual(derivative_norm_bound(zero, 3), 0)

    def test_corridor_certifies_all_points_and_reports_real_violation(self):
        normals = np.vstack((np.eye(3), -np.eye(3)))
        offsets = np.array([9, 2, 6, 1, 2, -4])
        result = convex_corridor_certificate(self.segment, normals, offsets)
        self.assertEqual(result["status"], "CERTIFIED")
        offsets[0] = 7
        violation = convex_corridor_certificate(self.segment, normals, offsets, max_depth=3)
        self.assertEqual(violation["status"], "VIOLATED")
        self.assertIsNotNone(violation["violation_witness"])

    def test_subdivision_reduces_corridor_hull_conservatism(self):
        arch = bezier_segment([[0, 0], [1, 3], [2, 0]])
        # The actual arch peaks at y=1.5; its unrefined control hull reaches 3.
        coarse = convex_corridor_certificate(arch, [[0, 1]], [2], max_depth=0)
        refined = convex_corridor_certificate(arch, [[0, 1]], [2], max_depth=2)
        self.assertEqual(coarse["status"], "UNRESOLVED")
        self.assertEqual(refined["status"], "CERTIFIED")

    def test_sphere_subdivision_certifies_detour_and_catches_baseline_collision(self):
        baseline = sphere_clearance_certificate(self.segment, [4, 0, 5], .4)
        self.assertEqual(baseline["status"], "VIOLATED")
        detour = c2_bump(self.segment, [0, .8, 0])
        coarse = sphere_clearance_certificate(detour, [4, 0, 5], .4, max_depth=0)
        refined = sphere_clearance_certificate(detour, [4, 0, 5], .4, max_depth=12)
        self.assertEqual(coarse["status"], "UNRESOLVED")
        self.assertEqual(refined["status"], "CERTIFIED")
        self.assertEqual(refined["unresolved_intervals"], [])
        self.assertGreater(refined["minimum_certified_box_distance"], .4)

    def test_sphere_touch_is_unresolved_instead_of_certified(self):
        tangent = bezier_segment([[-1, 1], [1, 1]])
        result = sphere_clearance_certificate(tangent, [0, 0], 1, max_depth=5)
        self.assertEqual(result["status"], "UNRESOLVED")
        self.assertIsNone(result["violation_witness"])

    def test_amplitude_budget_is_sufficient_and_has_correct_scaling(self):
        margins = dict(speed_margin=.9, acceleration_margin=1.2, jerk_margin=8., clearance=2.)
        radius = c2_bump_amplitude_budget(2., **margins)["maximum_displacement_norm_m"]
        expected = min(2., .9 * 2 / BUMP_DERIVATIVE_MAXIMA[1], 1.2 * 4 / 24, 8 * 8 / 384)
        self.assertAlmostEqual(radius, expected)
        u = np.linspace(0, 1, 1001)
        baseline = quintic_hermite([0, 0, 0], [0, 0, 0], [0, 0, 0],
                                  [0, 0, 0], [0, 0, 0], [0, 0, 0], 2.)
        perturbed = c2_bump(baseline, [radius, 0, 0])
        for order, margin in enumerate((.9, 1.2, 8.), 1):
            self.assertLessEqual(float(np.max(np.linalg.norm(perturbed.evaluate(u, order, wrt="time"), axis=1))),
                                 margin + 1e-12)
        slow = c2_bump_amplitude_budget(4., **margins)["maximum_displacement_norm_m"]
        self.assertGreater(slow, radius)
        zero = c2_bump_amplitude_budget(2., **dict(margins, jerk_margin=0))
        self.assertEqual(zero["maximum_displacement_norm_m"], 0)

    def test_jerk_energy_increment_matches_exact_polynomial_quadrature(self):
        rng = np.random.default_rng(601)
        for _ in range(10):
            boundary = rng.normal(size=(6, 3))
            segment = quintic_hermite(*boundary, 1.7)
            displacement = rng.normal(size=3) * .3
            expected = quintic_bump_jerk_energy_increment(segment, displacement)
            measured = energy(c2_bump(segment, displacement)) - energy(segment)
            self.assertAlmostEqual(measured, expected, delta=max(1e-8, expected * 2e-11))
            slow = PolynomialSegment(segment.coefficients, 3.4)
            self.assertAlmostEqual(quintic_bump_jerk_energy_increment(slow, displacement), expected / 32)
        self.assertEqual(quintic_bump_jerk_energy_increment(self.segment, [0, 0, 0]), 0)
        with self.assertRaises(ValueError):
            quintic_bump_jerk_energy_increment(c2_bump(self.segment, [.1, 0, 0]), [1, 0, 0])

    def test_invalid_inputs_fail_before_certification(self):
        for invalid in (-1, math.nan, math.inf, True, "1", 2j):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                c2_bump_amplitude_budget(2, speed_margin=invalid, acceleration_margin=1,
                                        jerk_margin=1, clearance=1)
        for invalid in (0, -1, math.nan, math.inf, True):
            with self.assertRaises(ValueError):
                c2_bump_amplitude_budget(invalid, speed_margin=1, acceleration_margin=1,
                                        jerk_margin=1, clearance=1)
        for invalid in (-1, 4, True, 1.5):
            with self.assertRaises(ValueError):
                derivative_control_points(self.segment, invalid)
        with self.assertRaises(ValueError):
            sphere_clearance_certificate(self.segment, [0, 0], 1)
        with self.assertRaises(ValueError):
            convex_corridor_certificate(self.segment, [[0, 0, 0]], [1])
        with self.assertRaises(ValueError):
            sphere_clearance_certificate(self.segment, [0, 0, 0], 1, max_depth=21)


if __name__ == "__main__":
    unittest.main()
