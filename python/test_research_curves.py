"""Mathematical properties of the offline Hermite/Bézier research module."""
import unittest

import numpy as np

from trajectory_research.curves import (
    MAX_DEGREE, PolynomialSegment, bezier_segment, bump_weight, c2_bump,
    cubic_bezier_to_hermite, cubic_hermite, cubic_hermite_from_state,
    cubic_hermite_to_bezier, elevate_bezier, junction_residuals, quintic_hermite,
)


class ResearchCurveTests(unittest.TestCase):
    def setUp(self):
        self.p0 = np.array((0.2, -1.0, 4.0))
        self.p1 = np.array((7.0, 3.0, 6.0))
        self.v0 = np.array((0.7, -0.2, 0.1))
        self.v1 = np.array((0.4, 0.3, -0.1))
        self.a0 = np.array((0.1, -0.3, 0.2))
        self.a1 = np.array((-0.2, 0.1, 0.3))

    def quintic(self, duration=2.5):
        return quintic_hermite(self.p0, self.v0, self.a0,
                               self.p1, self.v1, self.a1, duration)

    def test_cubic_basis_equivalence_and_endpoint_u_tangents(self):
        rng = np.random.default_rng(916)
        u = np.linspace(0, 1, 31)
        for _ in range(20):
            p0, p1, m0, m1 = rng.normal(size=(4, 3))
            control = cubic_hermite_to_bezier(p0, p1, m0, m1)
            round_trip = cubic_bezier_to_hermite(control)
            for actual, expected in zip(round_trip, (p0, p1, m0, m1)):
                np.testing.assert_allclose(actual, expected, atol=2e-15)
            h, b = cubic_hermite(p0, p1, m0, m1), bezier_segment(control)
            for order in range(4):
                np.testing.assert_allclose(h.evaluate(u, order), b.evaluate(u, order), atol=2e-13)
            np.testing.assert_allclose(h.evaluate(0., 1), m0, atol=2e-14)
            np.testing.assert_allclose(h.evaluate(1., 1), m1, atol=2e-14)

    def test_cubic_physical_velocity_and_time_scaling(self):
        segment = cubic_hermite_from_state(self.p0, self.v0, self.p1, self.v1, 3.)
        np.testing.assert_allclose(segment.sample(0, 1), self.v0, atol=1e-13)
        np.testing.assert_allclose(segment.sample(3, 1), self.v1, atol=1e-13)
        same_shape = PolynomialSegment(segment.coefficients, 6.)
        u = np.linspace(0, 1, 21)
        for order in range(4):
            np.testing.assert_allclose(same_shape.evaluate(u, order, wrt="time"),
                segment.evaluate(u, order, wrt="time") / 2. ** order, atol=1e-13)
            np.testing.assert_allclose(segment.sample(3. * u, order),
                segment.evaluate(u, order, wrt="time"), atol=1e-13)

    def test_quintic_endpoint_position_velocity_acceleration(self):
        for duration in (.7, 2.5, 7.):
            segment = self.quintic(duration)
            self.assertEqual(segment.degree, 5)
            for u, expected in ((0, (self.p0, self.v0, self.a0)),
                                (1, (self.p1, self.v1, self.a1))):
                for order, value in enumerate(expected):
                    np.testing.assert_allclose(segment.evaluate(u, order, wrt="time"), value, atol=2e-12)

    def test_quintic_bezier_control_points_match_endpoint_formula(self):
        duration = 2.5
        control = self.quintic(duration).bezier_control_points()
        expected = np.array((self.p0,
            self.p0 + duration * self.v0 / 5.,
            self.p0 + 2. * duration * self.v0 / 5. + duration ** 2 * self.a0 / 20.,
            self.p1 - 2. * duration * self.v1 / 5. + duration ** 2 * self.a1 / 20.,
            self.p1 - duration * self.v1 / 5., self.p1))
        np.testing.assert_allclose(control, expected, atol=3e-14)
        equivalent = bezier_segment(control, duration_s=duration)
        for order in range(4):
            np.testing.assert_allclose(equivalent.evaluate(np.linspace(0, 1, 37), order),
                self.quintic(duration).evaluate(np.linspace(0, 1, 37), order), atol=1e-11)

    def test_nonzero_bump_preserves_c2_and_changes_jerk(self):
        segment = self.quintic()
        displacement = np.array((.7, -.3, .2))
        perturbed = c2_bump(segment, displacement)
        self.assertEqual(perturbed.degree, 6)
        for u in (0., 1.):
            for order in range(3):
                np.testing.assert_allclose(perturbed.evaluate(u, order, wrt="time"),
                    segment.evaluate(u, order, wrt="time"), atol=1e-12)
            sign = 1. if u == 0 else -1.
            np.testing.assert_allclose(perturbed.evaluate(u, 3, wrt="time")
                - segment.evaluate(u, 3, wrt="time"),
                sign * 384. * displacement / segment.duration_s ** 3, atol=1e-11)
        np.testing.assert_allclose(perturbed.evaluate(.5) - segment.evaluate(.5), displacement, atol=1e-13)
        u = np.linspace(0, 1, 101)
        differences = np.linalg.norm(perturbed.evaluate(u) - segment.evaluate(u), axis=1)
        self.assertLessEqual(float(differences.max()), np.linalg.norm(displacement) + 1e-12)
        self.assertGreater(float(differences.max()), .5)

    def test_bump_weight_and_analytic_derivatives(self):
        self.assertEqual(float(bump_weight(.5)), 1.)
        np.testing.assert_allclose(bump_weight([0, 1]), 0, atol=0)
        for order in (1, 2):
            np.testing.assert_allclose(bump_weight([0, 1], order), 0, atol=0)
        np.testing.assert_allclose(bump_weight([0, 1], 3), [384, -384], atol=0)
        u, h = .31, 1e-5
        for order in range(1, 4):
            finite_difference = (bump_weight(u + h, order - 1)
                                 - bump_weight(u - h, order - 1)) / (2 * h)
            np.testing.assert_allclose(bump_weight(u, order), finite_difference, rtol=2e-8, atol=2e-7)

    def test_degree_elevation_preserves_curve_and_hull_contains_samples(self):
        segment = c2_bump(self.quintic(), [.8, -.4, .3])
        control = segment.bezier_control_points()
        self.assertEqual(control.shape, (7, 3))
        elevated = elevate_bezier(control, 9)
        self.assertEqual(elevated.shape, (10, 3))
        sampled = segment.evaluate(np.linspace(0, 1, 301))
        lower, upper = segment.control_hull_bounds()
        self.assertTrue(np.all(sampled >= lower - 1e-11))
        self.assertTrue(np.all(sampled <= upper + 1e-11))
        np.testing.assert_allclose(bezier_segment(elevated).evaluate(np.linspace(0, 1, 31)),
            sampled[::10], atol=2e-10)
        # The deformation has a degree-six control displacement of 64/C(6,3)=3.2*d.
        base = self.quintic().bezier_control_points(target_degree=6)
        delta = control - base
        np.testing.assert_allclose(delta[[0, 1, 2, 4, 5, 6]], 0, atol=1e-13)
        np.testing.assert_allclose(delta[3], 3.2 * np.array([.8, -.4, .3]), atol=1e-13)

    def test_shared_node_uses_physical_derivatives_with_unequal_durations(self):
        left = self.quintic(2.)
        right = quintic_hermite(self.p1, self.v1, self.a1,
            [10, 5, 4], [0, 0, 0], [0, 0, 0], 3.7)
        result = junction_residuals(left, right)
        self.assertTrue(result["c2"], result)
        self.assertFalse(result["c3"], result)
        self.assertTrue(junction_residuals(c2_bump(left, [.1, .2, .3]), right)["c2"])
        # Equal du tangents at different durations are not equal physical velocities.
        wrong = PolynomialSegment(right.coefficients, 7.4)
        self.assertTrue(junction_residuals(left, wrong)["c0"])
        self.assertFalse(junction_residuals(left, wrong)["c1"])

    def test_analytic_derivatives_match_finite_differences(self):
        segment = c2_bump(self.quintic(), [.1, -.4, .2])
        u, h = np.array((.19, .43, .78)), 2e-6
        for order in range(1, 4):
            actual = segment.evaluate(u, order)
            numerical = (segment.evaluate(u + h, order - 1)
                         - segment.evaluate(u - h, order - 1)) / (2 * h)
            np.testing.assert_allclose(actual, numerical, rtol=2e-8, atol=2e-7)

    def test_invalid_inputs_and_extrapolation_are_rejected(self):
        for duration in (0, -1, True, np.bool_(False), "2", float("inf"), float("nan"), 2j):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                self.quintic(duration)
        for value in ([-1e-12], [1 + 1e-12], float("nan"), True, [False], [".2"], [[.2]]):
            with self.subTest(parameter=value), self.assertRaises(ValueError):
                self.quintic().evaluate(value)
        for order in (-1, 4, True, 1.5):
            with self.subTest(order=order), self.assertRaises(ValueError):
                self.quintic().evaluate(.5, order)
        with self.assertRaises(ValueError):
            self.quintic().sample(2.5 + 1e-12)
        with self.assertRaises(ValueError):
            self.quintic().evaluate(.5, wrt="position")
        for coefficients in ([], [[]], [[0, float("nan")]], [1, 2], [[True]], [["1"]]):
            with self.subTest(coefficients=coefficients), self.assertRaises(ValueError):
                PolynomialSegment(coefficients)
        with self.assertRaises(ValueError):
            cubic_hermite([0, 1], [2], [0, 1], [0, 1])
        with self.assertRaises(ValueError):
            cubic_bezier_to_hermite([[0, 1]] * 5)
        with self.assertRaises(ValueError):
            c2_bump(self.quintic(), [1, 2])
        with self.assertRaises(ValueError):
            elevate_bezier([[0], [1], [2]], 1)
        with self.assertRaises(ValueError):
            junction_residuals(self.quintic(), self.quintic(), atol=-1)

    def test_constructor_copies_input_and_degree_zero_derivative_is_zero(self):
        source = np.array([[1., 2., 3.]])
        segment = PolynomialSegment(source, 2.)
        source[0, 0] = 100.
        self.assertEqual(segment.evaluate(.5)[0], 1.)
        self.assertFalse(segment.coefficients.flags.writeable)
        for order in (1, 2, 3):
            np.testing.assert_array_equal(segment.sample([0., 1., 2.], order), np.zeros((3, 3)))

    def test_degree_scope_rejects_ill_conditioned_or_unbounded_conversions(self):
        self.assertEqual(MAX_DEGREE, 9)
        maximum = np.zeros((MAX_DEGREE + 1, 3))
        self.assertEqual(PolynomialSegment(maximum).degree, MAX_DEGREE)
        self.assertEqual(bezier_segment(maximum).degree, MAX_DEGREE)
        self.assertEqual(elevate_bezier([[0.], [1.]], MAX_DEGREE).shape,
                         (MAX_DEGREE + 1, 1))
        too_high = np.zeros((MAX_DEGREE + 2, 3))
        for constructor in (PolynomialSegment, bezier_segment):
            with self.subTest(constructor=constructor.__name__), self.assertRaises(ValueError):
                constructor(too_high)
        for source, target in ((maximum, MAX_DEGREE + 1),
                               (maximum, 1_000_000_000),
                               (too_high, MAX_DEGREE + 1)):
            with self.subTest(target=target), self.assertRaises(ValueError):
                elevate_bezier(source, target)
        with self.assertRaises(ValueError):
            self.quintic().bezier_control_points(target_degree=MAX_DEGREE + 1)


if __name__ == "__main__":
    unittest.main()
