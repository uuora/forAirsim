"""Meaningful geometric/control regressions, without an AirSim dependency."""

import math
import unittest

import numpy as np

from dual_target_control.core import ControlConfig, RollingController


class CoreTests(unittest.TestCase):
    def make_controller(self, **settings):
        controller = RollingController(ControlConfig(**settings))
        controller.reset([0, 0, 5])
        return controller

    def test_invalid_config_and_state_are_rejected(self):
        for options in ({"max_speed": 0}, {"margin": 12}, {"seed": 1.3},
                        {"max_acceleration": float("nan")}, {"heading_tolerance_deg": 181},
                        {"height_max": 3}, {"control_hz": "fast"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ControlConfig(**options).validate()
        controller = self.make_controller()
        with self.assertRaises(ValueError):
            controller.update([0, float("nan"), 5], [0, 0, 0], 0, 0.02)
        with self.assertRaises(ValueError):
            controller.update([0, 0, 5], [0, 0, 0], 0, 0)
        self.assertEqual(ControlConfig.from_dict({"seed": 4, "vehicle": "Drone"}).seed, 4)

    def test_entire_two_leg_curve_and_targets_stay_in_buffered_box(self):
        for seed in range(75):
            controller = self.make_controller(seed=seed)
            controller.reset([18, -17, 8.9], [-4, 3, 0.1], yaw=0)
            path = controller.planned_path
            self.assertTrue(np.all(path >= controller.config.inner_lower - 1e-10))
            self.assertTrue(np.all(path <= controller.config.inner_upper + 1e-10))
            self.assertTrue(np.allclose(path[80], controller.targets[0]))
            self.assertTrue(np.allclose(path[-1], controller.targets[1]))
            self.assertGreater(np.linalg.norm(controller.targets[0] - controller.targets[1]), 1.0)

    def test_second_target_changes_path_before_first_hit(self):
        controller = self.make_controller()
        controller.set_targets([[10, 0, 5], [10, 10, 6]])
        left_path = controller.planned_path[:80]
        controller.set_targets([[10, 0, 5], [10, -10, 6]])
        right_path = controller.planned_path[:80]
        self.assertGreater(np.max(np.linalg.norm(left_path - right_path, axis=1)), 0.5)
        self.assertEqual(controller.snapshot()["hits"], 0)

    def test_commands_obey_speed_acceleration_and_yaw_step_limits(self):
        controller = self.make_controller()
        position, velocity, yaw = np.array([0., 0., 5.]), np.zeros(3), math.pi
        controller.reset(position, velocity, yaw)
        last_command = np.zeros(3)
        last_yaw_command = yaw
        dt = 0.02
        for _ in range(600):
            state = controller.update(position, velocity, yaw, dt)
            command = state["velocity_command"]
            self.assertLessEqual(np.linalg.norm(command), controller.config.max_speed + 1e-9)
            self.assertLessEqual(np.linalg.norm(command - last_command), controller.config.max_acceleration * dt + 1e-9)
            commanded_yaw = math.radians(state["yaw_command_deg"])
            error = (commanded_yaw - last_yaw_command + math.pi) % (2 * math.pi) - math.pi
            self.assertLessEqual(abs(error), math.radians(controller.config.max_yaw_rate_deg) * dt + 1e-9)
            velocity += (command - velocity) * (1 - math.exp(-dt / 0.18))
            position += velocity * dt
            yaw = commanded_yaw
            last_command = command
            last_yaw_command = commanded_yaw

    def test_pause_replan_preserves_queue_hits_and_discards_stale_sweep(self):
        controller = self.make_controller()
        controller.reset([-1, 0, 5], [2, 0, 0], 0)
        controller.set_targets([[0, 0, 5], [8, 0, 5]])
        self.assertEqual(controller.update([-0.3, 0, 5], [2, 0, 0], 0, 0.02)["hits"], 1)
        queued = controller.targets
        resumed = controller.replan([10, 0, 5], [0, 0, 0], math.pi)
        self.assertEqual(resumed["hits"], 1)
        self.assertTrue(np.allclose(resumed["targets"], queued))
        self.assertTrue(np.allclose(resumed["velocity_command"], np.zeros(3)))
        self.assertTrue(np.allclose(resumed["planned_path"][0], [10, 0, 5]))
        result = controller.update([10, 0, 5], [0, 0, 0], math.pi, 0.02)
        self.assertEqual(result["hits"], 1)

    def test_qualified_hit_rolls_queue_and_retains_motion(self):
        controller = self.make_controller()
        controller.reset([-1, 0, 5], [2, 0, 0], yaw=0)
        controller.set_targets([[0, 0, 5], [8, 2, 6]])
        previous_second = controller.targets[1]
        state = controller.update([-0.3, 0, 5], [2, 0, 0], 0, 0.02)
        self.assertEqual(state["hits"], 1)
        self.assertIsNotNone(state["hit_event"])
        self.assertTrue(np.allclose(state["targets"][0], previous_second))
        self.assertGreater(np.linalg.norm(state["targets"][0] - state["targets"][1]), 1)
        self.assertGreater(np.linalg.norm(state["velocity_command"]), 1)
        self.assertTrue(np.allclose(state["planned_path"][80], previous_second))

    def test_wrong_measured_yaw_cannot_grant_hit(self):
        controller = self.make_controller()
        controller.reset([-1, 0, 5], yaw=math.pi)
        controller.set_targets([[0, 0, 5], [8, 0, 5]])
        state = controller.update([-0.2, 0, 5], [2, 0, 0], math.pi, 0.02)
        self.assertEqual(state["hits"], 0)
        self.assertIsNone(state["hit_event"])

    def test_fast_swept_crossing_does_not_skip_target(self):
        controller = self.make_controller()
        controller.reset([-2, 0, 5], [4, 0, 0], yaw=0)
        controller.set_targets([[0, 0, 5], [8, 0, 5]])
        state = controller.update([2, 0, 5], [4, 0, 0], 0, 0.02)
        self.assertEqual(state["hits"], 1)
        self.assertTrue(state["hit_event"]["swept"])
        self.assertAlmostEqual(state["hit_event"]["yaw_error_deg"], 0)

    def test_boundary_immediately_requests_bounded_braking(self):
        controller = self.make_controller()
        controller.reset([18.5, 0, 5], [4, 0, 0], yaw=0)
        controller.set_targets([[18.9, 1, 5], [-10, 0, 5]])
        state = controller.update([18.55, 0, 5], [4, 0, 0], 0, 0.02)
        self.assertTrue(state["boundary_limited"])
        self.assertLess(state["velocity_command"][0], 4)
        self.assertLessEqual(np.linalg.norm(state["velocity_command"] - np.array([4, 0, 0])), 0.080000001)

    def test_planning_timing_persists_without_replanning_and_copies_are_isolated(self):
        controller = self.make_controller()
        before = controller.snapshot()
        state = controller.update([0, 0, 5], [0, 0, 0], 0, 0.02)
        self.assertEqual(before["planning_ms"], state["planning_ms"])
        state["targets"][0][:] = 1000
        state["planned_path"][:] = 1000
        state["velocity_command"][:] = 1000
        fresh = controller.snapshot()
        self.assertLess(np.max(fresh["targets"][0]), 100)
        self.assertLess(np.max(fresh["planned_path"]), 100)
        self.assertLess(np.linalg.norm(fresh["velocity_command"]), 5)

    def test_default_lagged_plant_reaches_multiple_goals_in_sixty_seconds(self):
        controller = self.make_controller()
        position, velocity, yaw = np.array([0., 0., 5.]), np.zeros(3), 0.0
        dt = 0.02
        for _ in range(int(60 / dt)):
            state = controller.update(position, velocity, yaw, dt)
            command = state["velocity_command"]
            velocity += (command - velocity) * (1 - math.exp(-dt / 0.18))
            position += velocity * dt
            desired_yaw = math.radians(state["yaw_command_deg"])
            yaw_delta = (desired_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
            yaw += yaw_delta * (1 - math.exp(-dt / 0.10))
            self.assertTrue(np.all(position >= controller.config.lower - 1e-6))
            self.assertTrue(np.all(position <= controller.config.upper + 1e-6))
        self.assertGreaterEqual(state["hits"], 3)

    def test_distant_sharp_turn_does_not_force_slow_whole_initial_leg(self):
        controller = self.make_controller(seed=4, max_speed=12)
        initial = controller.snapshot()
        position, velocity, yaw = np.array([0., 0., 5.]), np.zeros(3), 0.0
        dt = 0.02
        peak_speed = 0.0
        for _ in range(int(5 / dt)):
            state = controller.update(position, velocity, yaw, dt)
            velocity += (state["velocity_command"] - velocity) * (1 - math.exp(-dt / 0.18))
            position += velocity * dt
            desired_yaw = math.radians(state["yaw_command_deg"])
            yaw_delta = (desired_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
            yaw += yaw_delta * (1 - math.exp(-dt / 0.10))
            peak_speed = max(peak_speed, float(np.linalg.norm(velocity)))
        self.assertGreater(peak_speed, 4)
        self.assertGreater(np.linalg.norm(position - np.array([0, 0, 5])), 8)
        self.assertTrue(state["hits"] >= 1 or state["distance"] < initial["distance"])


if __name__ == "__main__":
    unittest.main()
