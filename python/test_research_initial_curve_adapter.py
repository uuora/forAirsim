"""Actual input-curve reconstruction and endpoint-state perturbation checks."""
import unittest

import numpy as np

from trajectory_research.curves import PolynomialSegment, bezier_segment, cubic_hermite_from_state
from trajectory_research.initial_curve_adapter import InitialCurveAdapter
from trajectory_research.parameterizations import BoundaryStates


class InitialCurveAdapterTests(unittest.TestCase):
    def test_arbitrary_bezier_zero_delta_reconstructs_degrees_three_five_seven(self):
        rng = np.random.default_rng(8827)
        u = np.linspace(0, 1, 53)
        for degree in (3, 5, 7):
            for _ in range(12):
                control = rng.uniform(-3., 3., (degree+1, 3))
                duration = rng.uniform(1., 7.)
                original = bezier_segment(control, duration_s=duration)
                reconstructed = InitialCurveAdapter.from_bezier(control, duration).decode_delta(np.zeros(7))
                for order in range(4):
                    np.testing.assert_allclose(reconstructed.evaluate(u, order, wrt="time"),
                        original.evaluate(u, order, wrt="time"), rtol=2e-10, atol=2e-9)

    def test_hermite_and_constant_input_zero_delta_reconstruct(self):
        state = BoundaryStates(p0=(.2, -.4, 4.), p1=(7., .6, 5.),
                               v0=(.3, .2, 0.), v1=(.7, -.1, .1))
        inputs = (state.baseline(4.2),
                  cubic_hermite_from_state(state.p0, state.v0, state.p1, state.v1, 3.2),
                  PolynomialSegment(np.array([[2., 3., 4.]]), 2.))
        for segment in inputs:
            adapter = InitialCurveAdapter.from_segment(segment)
            decoded = adapter.decode_delta([0.]*7)
            for order in range(4):
                np.testing.assert_allclose(decoded.evaluate(np.linspace(0, 1, 31), order, wrt="time"),
                    segment.evaluate(np.linspace(0, 1, 31), order, wrt="time"), atol=1e-10)

    def test_nonzero_delta_and_changed_duration_preserve_physical_endpoint_pva(self):
        rng = np.random.default_rng(337)
        initial = bezier_segment(rng.normal(size=(8, 3)), duration_s=3.4)
        adapter = InitialCurveAdapter.from_segment(initial)
        changed = adapter.decode_delta((.4, -.6, .8, 1.2, -.7, .3, 1.8))
        self.assertAlmostEqual(changed.duration_s, 5.2)
        for endpoint in (0., 1.):
            for order in range(3):
                np.testing.assert_allclose(changed.evaluate(endpoint, order, wrt="time"),
                    initial.evaluate(endpoint, order, wrt="time"), atol=2e-10)

    def test_same_duration_delta_has_declared_midpoint_position_and_u_tangent_increment(self):
        initial = bezier_segment(np.arange(24).reshape(8, 3)/3., duration_s=4.)
        adapter = InitialCurveAdapter.from_segment(initial)
        delta = np.array((3., -4., 5., -7., 6., 8., 0.))
        changed = adapter.decode_delta(delta)
        np.testing.assert_allclose(changed.evaluate(.5)-initial.evaluate(.5), delta[:3], atol=2e-11)
        np.testing.assert_allclose(changed.evaluate(.5, 1)-initial.evaluate(.5, 1), delta[3:6], atol=2e-10)
        np.testing.assert_allclose(changed.evaluate(.5, 1, wrt="time")-initial.evaluate(.5, 1, wrt="time"),
                                   delta[3:6]/4., atol=2e-10)

    def test_inputs_and_extracted_shape_are_independently_copied(self):
        control = np.arange(24, dtype=float).reshape(8, 3)
        adapter = InitialCurveAdapter.from_bezier(control, 3.)
        before = adapter.decode_delta([0.]*7).coefficients.copy()
        control[:] = -999.
        np.testing.assert_array_equal(adapter.decode_delta([0.]*7).coefficients, before)
        d0, e0 = np.ones(3)*20., np.ones(3)*-30.
        unbounded = InitialCurveAdapter(BoundaryStates(), d0, e0, 4.)
        d0[:] = 0.
        e0[:] = 0.
        self.assertEqual(unbounded.d0, (20., 20., 20.))
        self.assertEqual(unbounded.e0, (-30., -30., -30.))
        self.assertTrue(np.isfinite(unbounded.decode_delta([0.]*7).coefficients).all())

    def test_stored_degree_eight_non3d_and_invalid_inputs_are_rejected(self):
        for segment in (PolynomialSegment(np.zeros((9, 3)), 1.),
                        PolynomialSegment(np.zeros((4, 2)), 1.), None):
            with self.assertRaises(ValueError):
                InitialCurveAdapter.from_segment(segment)
        for control in (np.zeros((9, 3)), np.zeros((0, 3)), np.zeros((4, 2)),
                        np.full((4, 3), np.nan), np.zeros(3), [[True]*3]*4):
            with self.assertRaises(ValueError):
                InitialCurveAdapter.from_bezier(control, 3.)
        for duration in (0., -1., True, "3", np.inf, np.nan):
            with self.assertRaises(ValueError):
                InitialCurveAdapter.from_bezier(np.zeros((4, 3)), duration)
        for boundary, d0, e0 in (({}, [0.]*3, [0.]*3),
                                  (BoundaryStates(), [0.]*2, [0.]*3),
                                  (BoundaryStates(), [0.]*3, [np.inf]*3)):
            with self.assertRaises(ValueError):
                InitialCurveAdapter(boundary, d0, e0, 3.)

    def test_invalid_delta_and_nonpositive_result_duration_are_rejected(self):
        adapter = InitialCurveAdapter.from_bezier(np.zeros((4, 3)), 3.)
        for delta in ([0.]*6, [0.]*8, [np.nan]*7, [np.inf]*7, [True]*7,
                      [0., 0., 0., 0., 0., 0., -3.], [0., 0., 0., 0., 0., 0., -4.]):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                adapter.decode_delta(delta)


if __name__ == "__main__":
    unittest.main()
