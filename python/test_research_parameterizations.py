"""Independent mathematical checks for endpoint-preserving search coordinates."""
import math
import unittest

import numpy as np

from trajectory_research.parameterizations import (
    BoundaryStates, FAMILIES, REPRESENTATIONS, bezier_offsets_to_middle_state,
    curve_from_parameters, decode_normalized, default_parameters,
    middle_state_to_bezier_offsets, normalized_from_physical,
    parameter_bounds, parameter_names, physical_from_normalized,
)


class ParameterizationTests(unittest.TestCase):
    def setUp(self):
        self.boundary = BoundaryStates(
            p0=(.2, -.3, 4.1), v0=(.8, -.1, .2), a0=(.1, -.2, .05),
            p1=(7.5, 1.2, 5.4), v1=(.4, .2, -.1), a1=(-.1, .05, .1))

    def test_parameter_layout_bounds_and_common_initial_anchor(self):
        for family, size in (("single_bump", 4), ("paired_bump", 7)):
            with self.subTest(family=family):
                lower, upper = parameter_bounds(family)
                self.assertEqual(len(parameter_names(family)), size)
                self.assertEqual(lower.shape, (size,))
                self.assertTrue(np.all(lower < upper))
                anchor = default_parameters(family)
                self.assertTrue(np.all(anchor >= lower) and np.all(anchor <= upper))
                np.testing.assert_array_equal(anchor[:3], (0., 1.15, 0.))
                self.assertEqual(anchor[-1], 5.5)
                lower[0] = 123.
                self.assertEqual(parameter_bounds(family)[0][0], -.8)
                anchor[0] = 10.
                self.assertEqual(default_parameters(family)[0], 0.)
        self.assertIn("u_tangent", parameter_names()[3])

    def test_midpoint_mapping_is_inverse_and_matches_bernstein_values(self):
        rng = np.random.default_rng(236)
        # Values and derivatives of degree-seven Bernstein basis at u=.5,
        # evaluated from their binomial definitions, independently of decoder.
        basis = [math.comb(7, i) * .5 ** 7 for i in (3, 4)]
        derivative = [math.comb(7, i) * (i - (7-i)) * .5 ** 6 for i in (3, 4)]
        for _ in range(100):
            d, e = rng.normal(size=(2, 3))
            p3, p4 = middle_state_to_bezier_offsets(d, e)
            np.testing.assert_allclose(basis[0] * p3 + basis[1] * p4, d, atol=2e-15)
            np.testing.assert_allclose(derivative[0] * p3 + derivative[1] * p4, e, atol=2e-15)
            inverse = bezier_offsets_to_middle_state(p3, p4)
            np.testing.assert_allclose(inverse[0], d, atol=2e-15)
            np.testing.assert_allclose(inverse[1], e, atol=2e-15)

    def test_paired_family_matches_the_explicit_analytic_displacement(self):
        p = np.array((.3, -.7, .2, 1.2, -.8, .7, 4.3))
        u = np.linspace(0., 1., 101)
        baseline = self.boundary.baseline(p[-1])
        expected = baseline.evaluate(u) + (
            64. * u**3 * (1.-u)**3)[:, None] * (p[:3] + (u-.5)[:, None] * p[3:6])
        for representation in REPRESENTATIONS:
            curve = curve_from_parameters(p, boundary=self.boundary, representation=representation)
            np.testing.assert_allclose(curve.evaluate(u), expected, rtol=1e-11, atol=2e-11)
            np.testing.assert_allclose(curve.evaluate(.5)-baseline.evaluate(.5), p[:3], atol=2e-12)
            np.testing.assert_allclose(curve.evaluate(.5, 1)-baseline.evaluate(.5, 1), p[3:6], atol=2e-11)
            np.testing.assert_allclose(curve.evaluate(.5, 1, wrt="time")
                - baseline.evaluate(.5, 1, wrt="time"), p[3:6]/p[-1], atol=2e-11)

    def test_both_representations_preserve_physical_endpoint_states(self):
        rng = np.random.default_rng(429)
        for family in FAMILIES:
            lower, upper = parameter_bounds(family)
            for _ in range(12):
                parameters = rng.uniform(lower, upper)
                for representation in REPRESENTATIONS:
                    curve = curve_from_parameters(parameters, family=family,
                        representation=representation, boundary=self.boundary)
                    expected_sides = ((self.boundary.p0, self.boundary.v0, self.boundary.a0),
                                      (self.boundary.p1, self.boundary.v1, self.boundary.a1))
                    for u, expected in zip((0., 1.), expected_sides):
                        for order, value in enumerate(expected):
                            np.testing.assert_allclose(curve.evaluate(u, order, wrt="time"),
                                                       value, rtol=0, atol=2e-10)

    def test_coupled_decoders_have_the_same_position_and_derivatives(self):
        rng = np.random.default_rng(314)
        u = np.linspace(0, 1, 77)
        for family in FAMILIES:
            for _ in range(25):
                candidate = rng.uniform(-1, 1, len(parameter_names(family)))
                hermite = decode_normalized(candidate, family=family, boundary=self.boundary)
                bezier = decode_normalized(candidate, family=family, representation="bezier",
                                           boundary=self.boundary)
                self.assertEqual(hermite.degree, 6 if family == "single_bump" else 7)
                self.assertEqual(hermite.degree, bezier.degree)
                self.assertEqual(hermite.duration_s, bezier.duration_s)
                for order in range(4):
                    np.testing.assert_allclose(hermite.evaluate(u, order, wrt="time"),
                        bezier.evaluate(u, order, wrt="time"), rtol=1e-10, atol=4e-10)

    def test_bezier_perturbation_changes_only_the_free_controls(self):
        p = np.array((.2, .7, -.4, .6, -.3, 1.4, 4.7))
        baseline = self.boundary.baseline(p[-1]).bezier_control_points(target_degree=7)
        curve = curve_from_parameters(p, representation="bezier", boundary=self.boundary)
        delta = curve.bezier_control_points()-baseline
        np.testing.assert_allclose(delta[[0, 1, 2, 5, 6, 7]], 0, rtol=0, atol=2e-12)
        offsets = middle_state_to_bezier_offsets(p[:3], p[3:6])
        np.testing.assert_allclose(delta[3], offsets[0], atol=2e-12)
        np.testing.assert_allclose(delta[4], offsets[1], atol=2e-12)

    def test_single_bump_is_contained_in_paired_family_at_zero_tangent(self):
        u = np.linspace(0, 1, 43)
        for representation in REPRESENTATIONS:
            single = curve_from_parameters((.2, -.9, .3, 4.2), family="single_bump",
                                           representation=representation, boundary=self.boundary)
            paired = curve_from_parameters((.2, -.9, .3, 0., 0., 0., 4.2),
                                           representation=representation, boundary=self.boundary)
            for order in range(4):
                np.testing.assert_allclose(single.evaluate(u, order, wrt="time"),
                    paired.evaluate(u, order, wrt="time"), rtol=1e-10, atol=4e-10)

    def test_zero_shape_parameters_recover_the_baseline_at_every_duration(self):
        u = np.linspace(0, 1, 23)
        for family in FAMILIES:
            for duration in (3., 4.1, 7.):
                parameters = np.zeros(len(parameter_names(family)))
                parameters[-1] = duration
                for representation in REPRESENTATIONS:
                    decoded = curve_from_parameters(parameters, family=family,
                        representation=representation, boundary=self.boundary)
                    baseline = self.boundary.baseline(duration)
                    for order in range(4):
                        np.testing.assert_allclose(decoded.evaluate(u, order, wrt="time"),
                            baseline.evaluate(u, order, wrt="time"), rtol=1e-10, atol=4e-10)

    def test_changing_duration_rebuilds_baseline_instead_of_rescaling_endpoint_velocities(self):
        first = curve_from_parameters((0., .8, 0., 0., 0., 0., 3.))
        second = curve_from_parameters((0., .8, 0., 0., 0., 0., 7.))
        self.assertFalse(np.allclose(first.coefficients, second.coefficients))
        for curve in (first, second):
            np.testing.assert_allclose(curve.evaluate(0., 1, wrt="time"), (1., 0., 0.), atol=2e-12)
            np.testing.assert_allclose(curve.evaluate(1., 1, wrt="time"), (1., 0., 0.), atol=2e-12)

    def test_normalized_physical_round_trip_and_closed_box_endpoints(self):
        rng = np.random.default_rng(553)
        for family in FAMILIES:
            lower, upper = parameter_bounds(family)
            np.testing.assert_array_equal(physical_from_normalized(-np.ones(len(lower)), family=family), lower)
            np.testing.assert_array_equal(physical_from_normalized(np.ones(len(lower)), family=family), upper)
            for _ in range(50):
                point = rng.uniform(-1., 1., len(lower))
                physical = physical_from_normalized(point, family=family)
                np.testing.assert_allclose(normalized_from_physical(physical, family=family), point, atol=5e-16)

    def test_boundary_values_are_copied_and_validated(self):
        p0 = np.array((1., 2., 3.))
        boundary = BoundaryStates(p0=p0)
        p0[0] = 99
        self.assertEqual(boundary.p0, (1., 2., 3.))
        for invalid in ((1, 2), (True, True, False), (0, 1, np.inf), (0, 1, 2j), "abc"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                BoundaryStates(a1=invalid)
        for invalid_duration in (True, 0., -1., np.nan, np.inf, "5"):
            with self.subTest(T=invalid_duration), self.assertRaises(ValueError):
                boundary.baseline(invalid_duration)

    def test_invalid_family_representation_dimensions_and_out_of_bounds(self):
        for invalid in ("", "cubic", None, 1, [], ["paired_bump"]):
            with self.subTest(family=invalid), self.assertRaises(ValueError):
                parameter_bounds(invalid)
        for invalid in ("", "power", None, [], ["bezier"]):
            with self.subTest(representation=invalid), self.assertRaises(ValueError):
                curve_from_parameters(default_parameters(), representation=invalid)
        for invalid in ([0.]*6, [0.]*8, np.ones((1, 7)), [True]*7,
                        [0, 0, 0, 0, 0, 0, np.inf], [0, 0, 0, 0, 0, 0, 5+1j],
                        [.81, 0, 0, 0, 0, 0, 5], [0, 0, 0, 0, 0, 0, 7.01],
                        [0, 0, 0, 2.01, 0, 0, 5]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                curve_from_parameters(invalid)
        for invalid in ([0.]*6, [np.nan]*7, [1.000001]*7, [-1.000001]*7):
            with self.subTest(candidate=invalid), self.assertRaises(ValueError):
                decode_normalized(invalid)
        with self.assertRaises(ValueError):
            curve_from_parameters(default_parameters(), boundary={})
        with self.assertRaises(ValueError):
            middle_state_to_bezier_offsets([1, 2], [1, 2, 3])
        with self.assertRaises(ValueError):
            bezier_offsets_to_middle_state([1, 2, 3], [1, 2, np.inf])


if __name__ == "__main__":
    unittest.main()
