import unittest

import numpy as np

from fast_reference import decode, ordinary_plan, retimed
from prepared_route_planner import prepared_spec
from segment_retiming import SegmentTimeMap, sampled_motor_forces, search_profiles, tracking_reserves


class SegmentRetimingTests(unittest.TestCase):
    def setUp(self):
        spec = prepared_spec([[8., 0., 5.], [11., 5., 6.], [6., 11., 6.]], policy="pass")
        self.base = ordinary_plan(spec, "hermite")
        self.segments = decode(self.base).segments
        self.knots = np.r_[0., np.cumsum([s.duration_s for s in self.segments])]
        anchors = np.linspace(0., self.knots[-1], 15)
        self.solution = {"anchors": anchors.tolist(), "h": (1.1 + .7*np.sin(anchors)).tolist()}
        self.blocks = [(self.segments, self.solution)]

    def test_positive_c2_blend_and_physical_derivative(self):
        weights = [0., 1., .125, .85]
        timing = SegmentTimeMap(self.solution, self.knots, weights)
        np.testing.assert_allclose(timing.blend(self.knots), weights)
        for order in (1, 2):
            np.testing.assert_allclose(timing.blend(self.knots, order), 0., atol=1.e-12)
        s = np.linspace(.01, self.knots[-1]-.01, 150)
        self.assertTrue(np.all(timing(s) > 0.))
        self.assertTrue(np.all((timing.blend(s) >= 0.) & (timing.blend(s) <= 1.)))
        np.testing.assert_allclose(timing(s, 1), (timing(s+1.e-5)-timing(s-1.e-5))/2.e-5, atol=1.e-8)

    def test_uniform_recovers_fixed_timing_and_legacy_decode(self):
        old = retimed(self.base, self.blocks, .25)
        new = retimed(self.base, self.blocks, profile=SegmentTimeMap(self.solution, self.knots, [.25]*4))
        a, b = decode(old), decode(new)
        t = np.linspace(0., min(a.duration_s, b.duration_s), 99)
        np.testing.assert_allclose(a.ends, b.ends, atol=1.e-12)
        for derivative in (0, 1, 2):
            np.testing.assert_allclose(a.sample(t, derivative), b.sample(t, derivative), atol=1.e-10)
        self.assertNotIn("retiming_profile", old)
        self.assertIsNone(new["retiming_fraction"])

    def test_via_velocity_is_free_and_terminal_is_at_rest(self):
        weights = [.5, 1., 0., .2]
        timing = SegmentTimeMap(self.solution, self.knots, weights)
        plan = retimed(self.base, self.blocks, profile=timing)
        curve = decode(plan)
        for i, event in enumerate(curve.waypoint_times):
            phase = self.knots[i+1]
            end_speed = np.linalg.norm(self.segments[i].sample(self.segments[i].duration_s, 1))
            self.assertAlmostEqual(event["reference_speed_mps"], end_speed*np.sqrt(timing(phase)), places=9)
            np.testing.assert_allclose(curve.sample(event["arrival_time_s"]), event["point"], atol=1.e-10)
        for a, b in zip(curve.segments, curve.segments[1:]):
            for derivative in (0, 1, 2):
                np.testing.assert_allclose(a.sample(a.duration_s, derivative), b.sample(0., derivative), atol=1.e-8)
        self.assertLess(curve.waypoint_times[-1]["reference_speed_mps"], 1.e-10)

    def test_search_returns_fastest_checked_feasible_profile(self):
        def evaluate(w):
            return dict(model_feasible=w[1] <= .25, duration_s=10. - sum(w), fractions=w)
        best, trials = search_profiles(3, [.25, .125], evaluate)
        self.assertEqual(best, (1., .25, 1.))
        self.assertEqual(evaluate(best)["duration_s"], min(x["duration_s"] for x in trials if x["model_feasible"]))
        self.assertEqual(len(trials), len({x["fractions"] for x in trials}))
        best, _ = search_profiles(2, [.25], lambda w: dict(model_feasible=False, duration_s=1.))
        self.assertIsNone(best)

    def test_fixed_stop_keeps_dwell_and_rest_conditions(self):
        spec = prepared_spec([[7., 1., 5.], [11., 6., 6.]], policy="stop", dwell_s=1.)
        base = ordinary_plan(spec, "bezier")
        motion = decode(base).segments[::2]
        blocks = [([piece], {"anchors": [0., piece.duration_s], "h": [1., 1.3]}) for piece in motion]
        curve = decode(retimed(base, blocks, .125))
        first, last = curve.waypoint_times
        self.assertAlmostEqual(first["evaluation_time_s"] - first["arrival_time_s"], 1.)
        np.testing.assert_allclose(curve.sample(first["arrival_time_s"]+.5, 1), 0., atol=1.e-10)
        np.testing.assert_allclose(curve.sample(last["evaluation_time_s"], 1), 0., atol=1.e-10)

    def test_single_leg_supports_two_independent_coefficients(self):
        end = self.segments[0].duration_s
        solution = {"anchors": [0., end/2., end], "h": [1.4, 1.2, .8]}
        timing = SegmentTimeMap(solution, [0., end], [.1, .9])
        np.testing.assert_allclose(timing([0., end]), [1.04, .82])
        with self.assertRaises(ValueError):
            SegmentTimeMap(solution, [0., end], [.1, 1.01])

    def test_tracking_constraints_and_sampled_thrust(self):
        t = np.linspace(0., 1., 51)
        zero = np.zeros((len(t), 3))
        model = dict(nominal_gravity_mps2=9.8, mass_kg=1., inertia_diag_kgm2=[.01]*3,
                     thrust_allocation_flu=[[1,1,1,1],[-1,1,1,-1],[-1,1,-1,1],[-1,-1,1,1]],
                     max_thrust_per_rotor_n=6.)
        motors = sampled_motor_forces(t, zero, model)
        self.assertTrue(motors["passed"])
        self.assertAlmostEqual(motors["min_n"], 9.8/4)
        self.assertFalse(sampled_motor_forces(t, zero, dict(model, max_thrust_per_rotor_n=2.))["passed"])
        v = np.ones_like(zero)
        spec = dict(command_limits={"max_slew_mps2": 6.}, peak_position_limit_m=.3)
        checks = tracking_reserves(t, zero, zero, zero, v, spec, .18, .13)
        self.assertFalse(checks["peak_margin_passed"])
        self.assertFalse(checks["braking_passed"])


if __name__ == "__main__":
    unittest.main()
