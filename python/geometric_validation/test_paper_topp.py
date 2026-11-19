import unittest

import numpy as np
from scipy.integrate import quad

from .heading_toppquad import RouteHeading, identity_reference, sample_heading
from .paper_topp import SpeedReference, flat_thrust
from .planners import construct


class PaperToppTests(unittest.TestCase):
    def setUp(self):
        points = [[0., 0., -6.], [3., 1., -7.], [5., 4., -7.]]
        durations = [1.5, 2.]
        self.base = dict(format='normalized_power_v1', method='bezier', points=points, yaw=0.,
                         segment_durations=durations, coefficients=construct('bezier', points, durations).tolist())
        self.s = np.linspace(0., sum(durations), 21)

    def test_timing_inversion_and_alpha_derivatives(self):
        h = .5 + .15*self.s
        original = SpeedReference(self.base, self.s, h)
        alpha = 1.7
        slower = SpeedReference(self.base, self.s, h, alpha)
        times = np.sort(np.r_[np.linspace(0., original.duration, 19), original.times])
        expected_duration = quad(lambda s: 1/np.sqrt(.5 + .15*s), 0., 3.5)[0]
        self.assertAlmostEqual(original.duration, expected_duration)
        np.testing.assert_allclose(slower.sample_batch(times*alpha),
                                   original.sample_batch(times)/alpha**np.arange(5)[None,:,None], atol=1.e-8)
        np.testing.assert_allclose(slower.sample_batch(slower.times)[:, 0], self.base['points'], atol=1.e-9)

    def test_hover_thrust_uses_actual_actuator_order(self):
        class Controller:
            mass, g = .03, 9.81
            inertia = np.eye(3)
            TM_to_f = np.array([[.25, 1, 0, 1], [.25, 0, 1, 0], [.25, -1, 0, -1], [.25, 0, -1, 0]])

        np.testing.assert_allclose(flat_thrust(np.zeros((2,5,3)), Controller(), 1.57), .03*9.81/4)
        forces = flat_thrust(np.zeros((2,5,3)), Controller(), [0., 1.], [1., 1.], [.2, .3])
        np.testing.assert_allclose(forces[:, 0] - .03*9.81/4, [.2, .3])
        np.testing.assert_allclose(forces[:, 2] - .03*9.81/4, [-.2, -.3])
        np.testing.assert_allclose(forces.sum(axis=1), .03*9.81)

    def test_heading_changes_with_path_and_holds_beyond_end(self):
        original = identity_reference(self.base, [1.5, 2.])
        slow = identity_reference(self.base, [3., 4.])
        heading = RouteHeading(original)
        timeline = np.linspace(0., original.duration + 1., 130)
        yaw, rate = sample_heading(original, heading, timeline)
        slow_yaw, slow_rate = sample_heading(slow, heading, timeline * 2.)
        self.assertGreater(np.ptp(yaw), .2)
        np.testing.assert_allclose(yaw, slow_yaw, atol=1.e-10)
        np.testing.assert_allclose(rate / 2., slow_rate, atol=1.e-10)
        self.assertEqual(rate[-1], 0.)

    def test_positive_interpolation_when_cubic_overshoots(self):
        h = np.ones_like(self.s)
        h[9:11] = 1.e-7
        reference = SpeedReference(self.base, self.s, h)
        self.assertEqual(reference.interpolation, 'log_cubic_floor_1e-6')
        self.assertTrue(np.all(reference.profile(np.linspace(0., 3.5, 200)) > 0))
