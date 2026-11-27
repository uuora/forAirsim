import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from dual_target_control.small_field import (inside_field, ordered_waypoints, reference_checks,
                                            target_distance, terminal_hold, total_tilt_deg)
from dual_target_control.native_scene import NativeScene
from fast_panel import configuration
from fast_reference import decode, ordinary_plan


class SmallFieldChecks(unittest.TestCase):
    def setUp(self):
        config = configuration("pass", "hermite", "ordinary", small_field=True)
        self.spec = config["curve_experiment"]["plan"]["route"]
        self.rules = self.spec["small_field"]
        self.point_rules = {key: value for key, value in self.rules.items() if key != "vehicle_bounds_frd_m"}

    def test_order_and_speed_must_hold_on_same_sample(self):
        w1, w2, _ = self.spec["waypoints"]
        samples = np.array([w2, w1, w1, w2])
        times = np.arange(4)
        speeds = np.array([.6, .49, .5, .6])
        attitudes = np.tile([0., 0., 0., 1.], (len(times), 1))
        results = ordered_waypoints(times, samples, speeds, self.spec, attitudes)
        self.assertEqual([row["hit_time_s"] for row in results], [2., 3.])
        self.assertEqual(results[0]["hit_speed_mps"], .5)
        out_of_order = ordered_waypoints(times[:3], samples[:3], speeds[:3], self.spec, attitudes[:3])
        self.assertFalse(out_of_order[1]["passed"])
        self.assertEqual(out_of_order[1]["minimum_distance_m"], 0.)

    def test_terminal_hold_starts_at_reference_end_and_cannot_restart(self):
        times = np.linspace(0., 1.5, 31)
        p = np.zeros((len(times), 3))
        goal = np.zeros(3)
        reference_end = .21
        v = np.zeros_like(p)
        good = terminal_hold(times, p, v, goal, reference_end, self.point_rules)
        self.assertTrue(good["passed"])
        p[4, 0] = .5
        self.assertFalse(terminal_hold(times, p, v, goal, reference_end, self.point_rules)["reference_end_passed"])
        p[4, 0] = 0.
        v[8, 2] = .51
        broken = terminal_hold(times, p, v, goal, reference_end, self.point_rules)
        self.assertTrue(broken["reference_end_passed"])
        self.assertTrue(broken["end_passed"])
        self.assertFalse(broken["passed"])
        self.assertAlmostEqual(broken["continuous_s"], .14)

    def test_terminal_gap_breaks_the_hold(self):
        times = np.r_[np.arange(0., .41, .05), np.arange(.55, 1.51, .05)]
        p = np.zeros((len(times), 3))
        attitudes = np.tile([0., 0., 0., 1.], (len(times), 1))
        result = terminal_hold(times, p, p, np.zeros(3), .21, self.rules, attitudes)
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["first_failed_time_s"], .55)

    def test_vehicle_box_tracks_display_scale_and_attitude(self):
        bounds = np.asarray(self.rules["vehicle_bounds_frd_m"])
        np.testing.assert_allclose(bounds[1] - bounds[0], [.549749, .549749, .164075], atol=1e-6)
        position = np.zeros(3)
        identity = [0., 0., 0., 1.]
        diagonal = Rotation.from_euler("z", 45, degrees=True).as_quat()
        self.assertGreater(target_distance(position, [.60, 0., 0.], self.rules, identity), .30)
        self.assertLess(target_distance(position, [.60, 0., 0.], self.rules, diagonal), .30)
        self.assertEqual(target_distance(position, [.60, 0., 0.], self.point_rules), .60)
        self.assertAlmostEqual(target_distance(position, [0., 0., .20], self.rules, identity), .20 + bounds[0, 2])
        waypoint = np.array(self.spec["waypoints"][0])
        samples = np.array([waypoint - [.50, 0., 0.]])
        result = ordered_waypoints([0.], samples, np.array([.50]), self.spec, [identity])
        self.assertTrue(result[0]["passed"])

    def test_vehicle_terminal_uses_interpolated_attitude_and_hold(self):
        times = np.linspace(0., 1.5, 31)
        goal = np.zeros(3)
        positions = np.tile([.45, 0., 0.], (len(times), 1))
        velocities = np.zeros_like(positions)
        attitudes = Rotation.from_euler("z", np.zeros(len(times))).as_quat()
        result = terminal_hold(times, positions, velocities, goal, .225, self.rules, attitudes)
        self.assertTrue(result["passed"])
        self.assertFalse(terminal_hold(times, positions, velocities, goal, .225, self.point_rules)["passed"])
        positions[4:6, 0] = .55
        attitudes[5] = Rotation.from_euler("z", 90, degrees=True).as_quat()
        result = terminal_hold(times, positions, velocities, goal, .225, self.rules, attitudes)
        self.assertTrue(result["reference_end_passed"])
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["first_failed_time_s"], .25)

    def test_body_down_tilt_combines_roll_and_pitch_and_ignores_yaw(self):
        half = np.deg2rad(8)
        s, c = np.sin(half), np.cos(half)
        quaternion = [s*c, s*c, -s*s, c*c]
        self.assertGreater(total_tilt_deg(quaternion), 20.)
        self.assertAlmostEqual(total_tilt_deg([0., 0., s, c]), 0.)
        self.assertAlmostEqual(total_tilt_deg([s, 0., 0., c]), 16.)

    def test_ordinary_reference_covers_all_continuous_polynomial_segments(self):
        config = configuration("pass", "bezier", "ordinary", small_field=True)
        spec = config["curve_experiment"]["plan"]["route"]
        curve = decode(ordinary_plan(spec, "bezier"))
        self.assertTrue(reference_checks(curve)["passed"])
        np.testing.assert_allclose(curve.sample(0.), [0., 0., .5])
        np.testing.assert_allclose(self.rules['launch'], [0., 0., 0.])
        for event in curve.waypoint_times[:-1]:
            self.assertEqual(event["evaluation_time_s"], event["arrival_time_s"])

    def test_small_field_requires_continuous_pass(self):
        with self.assertRaisesRegex(ValueError, "仅支持连续过点"):
            configuration("stop", "hermite", "ordinary", small_field=True)

    def test_origin_roundoff_is_distinct_from_out_of_bounds(self):
        self.assertTrue(inside_field([-4.7e-10, -6.3e-14, .5], self.rules))
        self.assertFalse(inside_field([-2e-9, 0., .5], self.rules))

    def test_custom_small_field_bspline_keeps_route_and_pass_rules(self):
        points = [[1.8, 2., .8], [3.3, 3.4, .8], [4.55, 1.8, .8], [5.55, 4.65, .8], [6.45, 3.8, .8]]
        config = configuration("pass", "bspline", "ordinary", points=points, small_field=True)
        spec = config["curve_experiment"]["plan"]["route"]
        curve = decode(ordinary_plan(spec, "bspline"))
        np.testing.assert_allclose(curve.sample([event["arrival_time_s"] for event in curve.waypoint_times]), points)
        self.assertEqual(spec["small_field"]["minimum_pass_speed_mps"], .5)
        self.assertTrue(reference_checks(curve)["passed"])

    def test_positive_field_matches_native_observer(self):
        config = configuration("pass", "hermite", "ordinary", small_field=True)
        self.assertEqual(config["small_field"]["bounds_zup_m"], [[0., 0., 0.], [8., 8., 2.5]])
        command = NativeScene(config, None)._field_bounds_command()
        self.assertTrue(command.endswith("SetFieldBounds 0.0 8.0 0.0 8.0 0.0 2.5"))


if __name__ == "__main__":
    unittest.main()
