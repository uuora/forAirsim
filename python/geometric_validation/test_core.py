"""No-AirSim mathematical and contract tests for the isolated validation code."""
import math
import unittest

import numpy as np

from geometric_validation.core import (
    command, desired_rotation, euler_to_matrix, matrix_to_euler,
    quaternion_to_matrix, so3_angle,
)
from geometric_validation.reference import Reference, build_reference


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"kp": [2, 2, 3], "kv": [2.2, 2.2, 2.5],
                    "gravity": 9.80665, "hover_throttle": .4,
                    "max_tilt_rad": math.radians(25), "min_throttle": .08,
                    "max_throttle": .8, "max_acceleration": 5}
        self.ref = {"p": [0, 0, -5], "v": [0, 0, 0], "a": [0, 0, 0], "yaw": 0.}

    def cmd(self, **changes):
        return command(self.ref["p"], self.ref["v"], [1, 0, 0, 0],
                       {**self.ref, **changes}, self.cfg)

    def test_hover_balance(self):
        result = self.cmd()
        self.assertAlmostEqual(result["throttle"], .4)
        self.assertAlmostEqual(result["attitude_error_rad"], 0)
        np.testing.assert_allclose(result["R_des"], np.eye(3))
        self.assertFalse(any(result["saturation"].values()))

    def test_ned_direction_signs_and_vertical_balance(self):
        north = self.cmd(a=[1, 0, 0])
        east = self.cmd(a=[0, 1, 0])
        self.assertLess(north["pitch"], 0)
        self.assertGreater(east["roll"], 0)
        self.assertLess(self.cmd(a=[0, 0, 1])["throttle"], .4)
        self.assertGreater(self.cmd(a=[0, 0, -1])["throttle"], .4)
        # The desired thrust vector really accelerates in the requested world direction.
        for ref_a, result in (([1, 0, 0], north), ([0, 1, 0], east)):
            rd = np.asarray(result["R_des"])
            recovered = np.array([0., 0., self.cfg["gravity"]]) - np.linalg.norm(result["force_specific"]) * rd[:, 2]
            np.testing.assert_allclose(recovered, ref_a, atol=1e-12)

    def test_nonzero_yaw_uses_world_force_and_euler_reconstructs_attitude(self):
        result = self.cmd(a=[1, 0, 0], yaw=math.pi/2)
        self.assertLess(result["roll"], 0)
        self.assertAlmostEqual(result["pitch"], 0, places=12)
        np.testing.assert_allclose(euler_to_matrix(result["roll"], result["pitch"], result["yaw"]), result["R_des"], atol=1e-12)

    def test_quaternion_sign_and_scale_do_not_change_attitude(self):
        q = np.array([math.cos(.3), 0, 0, math.sin(.3)])
        r = quaternion_to_matrix(q)
        np.testing.assert_allclose(quaternion_to_matrix(-q), r)
        np.testing.assert_allclose(quaternion_to_matrix(q * 1e300), r)
        np.testing.assert_allclose(r, euler_to_matrix(0, 0, .6), atol=1e-12)

    def test_so3_geodesic_at_pi_and_very_small_angle(self):
        self.assertAlmostEqual(so3_angle(np.eye(3), euler_to_matrix(math.pi, 0, 0)), math.pi)
        self.assertAlmostEqual(so3_angle(np.eye(3), euler_to_matrix(0, 1e-10, 0)), 1e-10, places=18)
        # Opposite Euler representatives describe the same physical orientation.
        self.assertLess(so3_angle(euler_to_matrix(0, 0, math.pi), euler_to_matrix(0, 0, -math.pi)), 1e-12)

    def test_feedback_signs_restore_position_and_velocity(self):
        result = command([1, -1, -4], [1, -1, 1], [1, 0, 0, 0], self.ref, self.cfg)
        requested = np.asarray(result["a_requested"])
        self.assertLess(requested[0], 0)
        self.assertGreater(requested[1], 0)
        self.assertLess(requested[2], 0)

    def test_acceleration_tilt_and_collective_bounds(self):
        result = self.cmd(a=[100, 0, 0])
        self.assertTrue(result["saturation"]["acceleration"])
        self.assertTrue(result["saturation"]["tilt"])
        self.assertLessEqual(result["desired_tilt_rad"], self.cfg["max_tilt_rad"] + 1e-12)
        for acceleration in ([0, 0, 1e6], [1e6, -1e6, -1e6]):
            result = self.cmd(a=acceleration)
            self.assertGreaterEqual(result["throttle"], .08)
            self.assertLessEqual(result["throttle"], .8)
            self.assertTrue(np.all(np.isfinite(result["R_des"])))

    def test_thrust_cap_vertical_floor_and_negative_projection_are_visible(self):
        cfg = {**self.cfg, "max_acceleration": 100}
        high = command(self.ref["p"], self.ref["v"], [1, 0, 0, 0], {**self.ref, "a": [0, 0, -100]}, cfg)
        self.assertTrue(high["saturation"]["force_magnitude"])
        self.assertAlmostEqual(high["throttle"], .8)
        down = command(self.ref["p"], self.ref["v"], [1, 0, 0, 0], {**self.ref, "a": [0, 0, 100]}, cfg)
        self.assertTrue(down["saturation"]["vertical_force"])
        inverted = command(self.ref["p"], self.ref["v"], [0, 1, 0, 0], self.ref, self.cfg)
        self.assertTrue(inverted["saturation"]["nonpositive_projection"])
        self.assertTrue(inverted["saturation"]["collective"])
        self.assertAlmostEqual(inverted["throttle"], .08)

    def test_singular_heading_fallback_is_orthonormal(self):
        r, singular = desired_rotation([1, 0, 0], 0)
        self.assertTrue(singular)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r), 1)

    def test_euler_roundtrip_over_upright_domain(self):
        for roll in (-.4, 0, .4):
            for pitch in (-.4, 0, .4):
                for yaw in (-math.pi, -.8, 0, 1.5, math.pi):
                    r = euler_to_matrix(roll, pitch, yaw)
                    np.testing.assert_allclose(euler_to_matrix(*matrix_to_euler(r)), r, atol=1e-12)

    def test_invalid_data_and_bad_configuration_rejected(self):
        for q in ([0, 0, 0, 0], [math.nan, 0, 0, 1]):
            with self.assertRaises(ValueError):
                quaternion_to_matrix(q)
        for key, value in (("kp", [-1, 1, 1]), ("max_tilt_rad", math.pi/2),
                           ("max_throttle", .2), ("gravity", math.nan)):
            with self.assertRaises(ValueError):
                command(self.ref["p"], self.ref["v"], [1, 0, 0, 0], self.ref, {**self.cfg, key: value})
        with self.assertRaises(ValueError):
            self.cmd(a=[math.inf, 0, 0])


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.reference = Reference({"start": [0, 0, -5], "offsets": [[3, 0, -.5], [3, 3, -1]],
                                    "segment_durations": [5, 5],
                                    "velocities": [[0, 0, 0], [.4, .4, -.1], [0, 0, 0]], "yaw": .4})

    def test_endpoints_and_c3_continuity(self):
        ref = self.reference
        np.testing.assert_allclose(ref.sample(0)["p"], [0, 0, -5])
        np.testing.assert_allclose(ref.sample(10)["p"], [3, 3, -6], atol=1e-12)
        for t in (0, 10):
            for key in ("v", "a", "jerk"):
                np.testing.assert_allclose(ref.sample(t)[key], [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(ref.sample(5)["v"], [.4, .4, -.1], atol=1e-12)
        for key in ("p", "v", "a", "jerk"):
            np.testing.assert_allclose(ref.sample(5-1e-8)[key], ref.sample(5+1e-8)[key], atol=1e-7)

    def test_analytic_derivatives_against_central_differences(self):
        ref = self.reference
        for t in (1.3, 3.2, 6.8, 8.7):
            for key, derivative in (("p", "v"), ("v", "a"), ("a", "jerk"), ("jerk", "snap")):
                h = 1e-4
                numerical = (np.asarray(ref.sample(t+h)[key]) - np.asarray(ref.sample(t-h)[key])) / (2*h)
                np.testing.assert_allclose(numerical, ref.sample(t)[derivative], rtol=2e-6, atol=1e-7)

    def test_outside_reference_holds_rest_and_hover_has_zero_derivatives(self):
        for t, point in ((-1, self.reference.start), (11, self.reference.end)):
            result = self.reference.sample(t)
            np.testing.assert_allclose(result["p"], point)
            for key in ("v", "a", "jerk", "snap"):
                np.testing.assert_allclose(result[key], [0, 0, 0])
        hover = Reference({"offsets": [[0, 0, 0]], "segment_durations": [3]})
        for t in np.linspace(0, 3, 50):
            for key in ("v", "a", "jerk", "snap"):
                np.testing.assert_allclose(hover.sample(t)[key], [0, 0, 0], atol=1e-10)

    def test_build_reference_and_path(self):
        ref = build_reference({"start": [1, 2, -5], "yaw": 1., "cases": {"line": {"offsets": [[2, 0, 0]], "segment_durations": [5]}}}, "line")
        self.assertEqual(len(ref.sample_path(15)), 15)
        np.testing.assert_allclose(ref.end, [3, 2, -5])
        self.assertEqual(ref.yaw, 1.)

    def test_invalid_reference_rejected(self):
        for cfg in ({"segment_durations": [0]}, {"yaw": math.nan},
                    {"velocities": [[1, 0, 0], [0, 0, 0]]},
                    {"offsets": [[1, 2, math.inf]]}):
            with self.assertRaises(ValueError):
                Reference(cfg)


if __name__ == "__main__":
    unittest.main()
