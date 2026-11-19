"""Check the shape-preserving clock and between-sample certification."""
import unittest

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import BPoly

from .mapped_reference import bernstein, bound, compose, evaluate, original_mapping, optimize_mapping
from .planners import construct
from .reference import FrozenReference


class MappedReferenceTests(unittest.TestCase):
    def setUp(self):
        self.points = [[0, 0, -5], [3, 0, -5], [5, 2, -5], [7, 2, -5.2]]
        durations = [5., 4., 5.]
        self.source = dict(format='normalized_power_v1', method='hermite', points=self.points, yaw=0.,
                           coefficients=construct('hermite', self.points, durations).tolist(), segment_durations=durations)
        self.policy = dict(max_speed_mps=1.6, max_acceleration_mps2=1.5, max_jerk_mps3=3., min_knot_speed_mps=.65)
        self.limits = dict(bounds_ned_min=[-20, -20, -15], bounds_ned_max=[20, 20, -1])

    def test_global_clock_and_chain_derivatives_match_direct_evaluation(self):
        warp = Polynomial([0., .7, .2, .1])
        mapped = original_mapping(self.source, warp.coef, 17.)
        base = FrozenReference(self.source)
        times = np.linspace(0., mapped.duration, 101)
        expected = [base.sample(float(warp(t / mapped.duration) * base.duration)) for t in times]
        samples = mapped.sample_batch(times)
        np.testing.assert_allclose(samples[:, 0], [row['p'] for row in expected], atol=1.e-10)
        rates = [warp.deriv(k)(times / 17.)[:, None] * base.duration / 17.**k for k in (1, 2, 3)]
        speed, accel, jerk = [np.array([row[key] for row in expected]) for key in ('v', 'a', 'jerk')]
        q, r, s = rates
        np.testing.assert_allclose(samples[:, 1], speed*q, atol=1.e-10)
        np.testing.assert_allclose(samples[:, 2], accel*q*q+speed*r, atol=1.e-10)
        np.testing.assert_allclose(samples[:, 3], jerk*q**3+3*accel*q*r+speed*s, atol=1.e-10)
        self.assertLess(max(mapped.certify(self.policy, self.limits)['max_abs_knot_jumps']), 1.e-9)

    def test_bernstein_composition_and_off_grid_peak(self):
        path = np.array([[0., 0., 0.], [2., 0., 0.], [1., 1., 0.]])
        clock = bernstein([0., .7, .3])
        ts = np.linspace(0., 1., 51)
        np.testing.assert_allclose(evaluate(compose(path, clock), ts), evaluate(path, evaluate(clock, ts)), atol=1.e-13)
        # The maximum is interior and the two endpoints alone satisfy the cap.
        interior_peak = np.array([0., 2.2, 0.])
        self.assertEqual(bound(interior_peak, lambda b: np.all(b <= 1.)), 'violation')
        self.assertEqual(bound(interior_peak, lambda b: np.all(b <= 1.2)), 'pass')

    def test_segment_solution_preserves_geometry_and_knot_speed(self):
        source = original_mapping(self.source, [0., 1.], 19.)
        final, certificate, receipt = optimize_mapping(self.source, source, self.policy, self.limits)
        self.assertEqual(receipt['selected'], 'optimized')
        self.assertLess(final.duration, 14.)
        self.assertEqual(certificate['issues'], [])
        self.assertGreaterEqual(min(certificate['waypoint_speeds_mps']), .65)
        expected = np.asarray(self.points)
        np.testing.assert_allclose(final.sample_batch(final.times)[:, 0], expected, atol=1.e-10)
        for index, clock in enumerate(final.clocks):
            spline = BPoly(clock[:, None], final.times[index:index+2])
            np.testing.assert_allclose(spline(final.times[index:index+2], nu=2), 0., atol=1.e-10)
            np.testing.assert_allclose(spline(final.times[index:index+2], nu=3), 0., atol=1.e-10)
