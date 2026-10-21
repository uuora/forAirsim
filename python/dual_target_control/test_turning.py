"""Measured-yaw and first-pass regressions using a lagged point-mass plant."""
import math
import unittest

import numpy as np

from dual_target_control import core
from validate_dual_target_turning import (
    INTERVALS, SCENARIOS, YAW_LAGS, run_scenario, sphere_entry,
)


class TurningTests(unittest.TestCase):
    def test_independent_sphere_detector_finds_swept_entry(self):
        self.assertAlmostEqual(sphere_entry(np.array([-2., 0, 5]), np.array([2., 0, 5]),
                                            np.array([0., 0, 5]), 0.5), 0.375)
        self.assertIsNone(sphere_entry(np.array([-2., 0.6, 5]), np.array([2., 0.6, 5]),
                                       np.array([0., 0, 5]), 0.5))

    def test_nonradial_entry_is_on_sphere_and_uses_entry_measured_yaw(self):
        # A 3-4-5 triangle fixes the entry at (-.4, .3), exactly .5m from
        # the target. The segment ends inside the sphere at (-.2, .3), so
        # its nearest point is an endpoint rather than a perpendicular foot.
        # This geometry catches a sphere-entry approximation using that foot.
        target = np.array([0., 0., 5.])
        entry = np.array([-0.4, 0.3, 5.])
        previous = np.array([-1., 0.3, 5.])
        position = np.array([-0.2, 0.3, 5.])
        self.assertAlmostEqual(sphere_entry(previous, position, target, 0.5), 0.75)
        bearing = math.atan2(-0.3, 0.4)
        for initial_yaw_deg, final_yaw_deg, expected_hit in ((-10., -20., True), (-8., -18., False)):
            with self.subTest(initial_yaw=initial_yaw_deg, final_yaw=final_yaw_deg):
                controller = core.RollingController(core.ControlConfig())
                controller.reset(previous, [4., 0., 0.], math.radians(initial_yaw_deg))
                controller.set_targets([target, [8., 0., 5.]])
                result = controller.update(position, [4., 0., 0.], math.radians(final_yaw_deg), 0.2)
                # At the known entry fraction the yaw is -17.5 or -15.5
                # degrees: errors 19.37 and 21.37 straddle the 20-degree gate.
                measured = math.radians(initial_yaw_deg + 0.75 * (final_yaw_deg - initial_yaw_deg))
                expected_error = abs(math.degrees(bearing - measured))
                self.assertEqual(result["hits"], int(expected_hit))
                if expected_hit:
                    hit = result["hit_event"]
                    self.assertIsNotNone(hit)
                    np.testing.assert_allclose(hit["position"], entry, atol=1e-12, rtol=0)
                    self.assertAlmostEqual(np.linalg.norm(hit["position"] - target), 0.5, places=12)
                    self.assertAlmostEqual(hit["yaw_error_deg"], expected_error, places=10)
                    self.assertLess(expected_error, 20.0)
                else:
                    self.assertIsNone(result["hit_event"])
                    np.testing.assert_array_equal(result["targets"][0], target)
                    self.assertGreater(expected_error, 20.0)

    def test_large_cross_track_correction_preserves_approach_and_measured_yaw_slowdown(self):
        position = np.array([9.4, 2., 5.])
        target = np.array([10., 0., 5.])
        facing_target = math.atan2(float(target[1] - position[1]), float(target[0] - position[0]))

        def stalled_measurement(yaw):
            controller = core.RollingController(core.ControlConfig())
            controller.reset([0., 0., 5.])
            controller.set_targets([target, [15., 0., 5.]])
            # A disturbance has moved the aircraft two metres off a straight
            # approach. Hold the measured state to expose the requested steady
            # velocity, independently of a plant's ability to follow it.
            for _ in range(100):
                state = controller.update(position, [0., 0., 0.], yaw, 0.02)
                self.assertEqual(state["hits"], 0)
            return state

        aligned = stalled_measurement(facing_target)
        away = stalled_measurement(math.pi / 2)
        aligned_speed = float(np.linalg.norm(aligned["velocity_command"]))
        away_speed = float(np.linalg.norm(away["velocity_command"]))
        # Correcting the position still calls for motion, but it must not
        # restore full cruise speed next to the goal or defeat yaw slowdown.
        self.assertGreater(aligned_speed, 0.1)
        self.assertLess(aligned_speed, 1.0)
        self.assertGreater(away_speed, 0.0)
        self.assertLess(away_speed, 0.25 * aligned_speed)
        # Commanded yaw has had two seconds to settle; measured yaw remains
        # facing away. This prevents mistaking command alignment for body yaw.
        self.assertGreater(away["yaw_error_deg"], 140.0)

    def test_unqualified_passed_target_recovers_by_measured_yaw_inside_sphere(self):
        controller = core.RollingController(core.ControlConfig())
        target = np.array([0., 0., 5.])
        previous = np.array([-1., 0., 5.])
        position = np.array([0.2, 0., 5.])
        yaw = math.pi / 2
        controller.reset(previous, [0., 0., 0.], yaw)
        controller.set_targets([target, [8., 0., 5.]])
        self.assertAlmostEqual(sphere_entry(previous, position, target, 0.5), 5 / 12)
        state = controller.update(position, [0., 0., 0.], yaw, 0.02)
        self.assertEqual(state["hits"], 0)
        self.assertIsNone(state["hit_event"])
        np.testing.assert_array_equal(state["targets"][0], target)
        # The missed aircraft has stopped inside the sphere on the far side.
        # Its body must turn towards the still-pending target at 180 degrees.
        # Command yaw can lead measured yaw; it cannot qualify the arrival.
        for _ in range(300):
            previous_yaw = yaw
            commanded = math.radians(state["yaw_command_deg"])
            yaw += ((commanded - yaw + math.pi) % (2 * math.pi) - math.pi) * (1 - math.exp(-0.02 / 0.25))
            entry_error = abs(math.degrees((math.pi - previous_yaw + math.pi) % (2 * math.pi) - math.pi))
            state = controller.update(position, [0., 0., 0.], yaw, 0.02)
            if entry_error > 20.0:
                self.assertEqual(state["hits"], 0)
                self.assertIsNone(state["hit_event"])
                np.testing.assert_array_equal(state["targets"][0], target)
            if state["hits"]:
                self.assertLessEqual(entry_error, 20.0)
                self.assertAlmostEqual(state["hit_event"]["yaw_error_deg"], entry_error, places=10)
                self.assertLessEqual(np.linalg.norm(state["hit_event"]["position"] - target), 0.5)
                np.testing.assert_array_equal(state["targets"][0], [8., 0., 5.])
                break
        else:
            self.fail("Passed target stalled inside the sphere despite measured-yaw convergence")

    def test_small_position_error_with_lateral_inertia_slows_before_target(self):
        # The aircraft is already almost on the terminal centreline, three
        # metres before the target. Its lateral velocity, rather than the tiny
        # current offset, predicts an impending miss and fast bearing change.
        position = np.array([7., 0.045, 5.])
        target = np.array([10., 0., 5.])
        yaw = math.atan2(-0.045, 3.0)
        lateral_speed = 1.045
        total_speed = math.hypot(3.5, lateral_speed)

        def requested_command(measured_velocity):
            controller = core.RollingController(core.ControlConfig())
            controller.reset([0., 0., 5.])
            controller.set_targets([target, [15., 0., 5.]])
            # Hold the observation to inspect the commanded envelope after
            # its acceleration-limited transient. Both observations have the
            # same position, bearing and total measured speed.
            for _ in range(100):
                state = controller.update(position, measured_velocity, yaw, 0.02)
                self.assertEqual(state["hits"], 0)
            return state["velocity_command"]

        settled = requested_command([total_speed, 0., 0.])
        drifting = requested_command([3.5, lateral_speed, 0.])
        self.assertAlmostEqual(np.linalg.norm([3.5, lateral_speed, 0.]), total_speed)
        settled_speed = float(np.linalg.norm(settled))
        drifting_speed = float(np.linalg.norm(drifting))
        self.assertGreater(settled_speed, 3.0)
        self.assertLess(drifting_speed, settled_speed - 0.25)
        self.assertLess(drifting[1], 0.0)
        self.assertGreater(drifting_speed, 0.1)

    def test_two_targets_first_pass_with_lagged_velocity_and_measured_yaw(self):
        for scenario in SCENARIOS:
            for yaw_lag in YAW_LAGS:
                for intervals in INTERVALS:
                    with self.subTest(scenario=scenario.name, yaw_lag=yaw_lag, intervals=intervals):
                        result = run_scenario(core, scenario, yaw_lag=yaw_lag, intervals=intervals)
                        failures = [key for key, passed in result["checks"].items() if not passed]
                        self.assertTrue(result["passed"],
                                        f"Failed {failures}; legs={result['legs']}; violations={result['violations']}")


if __name__ == "__main__":
    unittest.main()
