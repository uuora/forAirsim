"""Parameter search must preserve the controller's existing arrival contract."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from itertools import product
import math
import unittest

import numpy as np

from dual_target_control.core import BezierParameters, ControlConfig, RollingController


class BezierParameterTests(unittest.TestCase):
    def test_serialization_validation_and_immutable_bounds(self):
        defaults = BezierParameters()
        np.testing.assert_array_equal(defaults.to_vector(), [.4, .3, .2, .45, 1.])
        self.assertEqual(BezierParameters.from_vector(defaults.to_vector()), defaults)
        self.assertEqual(BezierParameters.from_dict(defaults.to_dict()), defaults)
        self.assertEqual(BezierParameters.from_dict({"outgoing_weight": .7}).outgoing_weight, .7)
        self.assertTrue(all(type(value) is float for value in defaults.to_dict().values()))
        with self.assertRaises(FrozenInstanceError):
            defaults.outgoing_weight = .5
        bounds = defaults.lower_bounds()
        bounds[:] = -99
        self.assertEqual(defaults.lower_bounds()[0], 0)
        for value in (True, np.bool_(False), "0.4", float("nan"), float("inf"), 1j, -.1, .9):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BezierParameters(outgoing_weight=value)
        for value in ([.4] * 4, [[.4] * 5], [.4, .3, .2, .45, "1.0"],
                      [.4, .3, .2, .45, True], [.4, .3, .2, .45, 1.31]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BezierParameters.from_vector(value)
        for value in ({"typo": .3}, None, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BezierParameters.from_dict(value)

    def test_default_geometry_matches_pre_parameterization_fixture(self):
        # Fixture from the original literal coefficient formulas, including the
        # curved approach, straight terminal corridor, and second cubic segment.
        controller = RollingController(ControlConfig())
        controller.reset([1, -2, 5], [1, 2, .2], .35)
        controller.set_targets([[11, 3, 6], [-4, 12, 8]])
        indices = [0, 10, 25, 40, 60, 70, 80, 100, 120, 140, 160]
        expected = [
            [1., -2., 5.],
            [1.923312294821146, -.9875983172889417, 5.095519676364036],
            [3.7518123430319283, -.7899244010092428, 5.100800506797845],
            [5.7162824791851445, -1.2028398565442604, 5.067783772980398],
            [7.7765642212368, -.8082898471063831, 5.211716915939635],
            [9.3882821106184, 1.0958550764468085, 5.605858457969818],
            [11., 3., 6.],
            [10.161863559834602, 5.1179181712642, 6.454334984450984],
            [5.90325160351067, 7.493636866484675, 6.984085083172442],
            [.4430138454314032, 9.872537128462811, 7.5217926403076785],
            [-4., 12., 8.],
        ]
        np.testing.assert_allclose(controller.planned_path[indices], expected, rtol=0, atol=1e-13)
        np.testing.assert_allclose(controller._speed_profile[indices],
            [3.084066098289715, 2.2940714281460695, 4., 3.466847760441224,
             4., 4., 1.8931854477336212, 4., 4., 4., 4.], rtol=0, atol=1e-11)

    def test_parameter_extremes_keep_endpoints_and_curve_inside_buffer(self):
        for choices in product((0, 1), repeat=5):
            vector = np.where(choices, BezierParameters.upper_bounds(), BezierParameters.lower_bounds())
            parameters = BezierParameters.from_vector(vector)
            for seed in range(8):
                with self.subTest(choices=choices, seed=seed):
                    controller = RollingController(ControlConfig(seed=seed), parameters)
                    controller.reset([18.9, -18.8, 8.9], [-3, 1, -.5])
                    path = controller.planned_path
                    np.testing.assert_array_equal(path[0], [18.9, -18.8, 8.9])
                    np.testing.assert_array_equal(path[80], controller.targets[0])
                    np.testing.assert_array_equal(path[-1], controller.targets[1])
                    self.assertTrue(np.all(path >= controller.config.inner_lower - 1e-10))
                    self.assertTrue(np.all(path <= controller.config.inner_upper + 1e-10))
                    self.assertLessEqual(controller._target_speed, controller.config.max_speed)

    def test_parameter_change_keeps_sweep_commands_rng_and_queue(self):
        controller = RollingController(ControlConfig())
        controller.reset([-2, 0, 5], [2, 0, 0], .1)
        controller.set_targets([[0, 0, 5], [8, 2, 6]])
        before = controller.update([-1, 0, 5], [2, .1, 0], .1, .02)
        untouched = deepcopy(controller)
        revision = controller.planning_revision
        result = controller.set_parameters(BezierParameters(outgoing_weight=.65, pass_speed_scale=1.2))
        self.assertEqual(controller.planning_revision, revision + 1)
        self.assertEqual(result["planning_revision"], controller.planning_revision)
        self.assertIsNone(result["hit_event"])
        self.assertEqual(result["hits"], before["hits"])
        np.testing.assert_array_equal(result["targets"], before["targets"])
        np.testing.assert_array_equal(result["velocity_command"], before["velocity_command"])
        self.assertEqual(result["yaw_command_deg"], before["yaw_command_deg"])
        np.testing.assert_array_equal(controller.current_position, [-1, 0, 5])
        np.testing.assert_array_equal(controller.current_velocity, [2, .1, 0])
        self.assertAlmostEqual(controller.current_yaw, .1)
        # A complete sphere crossing on the very next sample must still count.
        after = controller.update([1, 0, 5], [2, 0, 0], .1, .02)
        reference = untouched.update([1, 0, 5], [2, 0, 0], .1, .02)
        self.assertEqual(after["hits"], 1)
        self.assertTrue(after["hit_event"]["swept"])
        np.testing.assert_array_equal(after["targets"], reference["targets"])
        self.assertLessEqual(np.linalg.norm(after["velocity_command"] - before["velocity_command"]), .08 + 1e-9)
        yaw_change = math.radians(after["yaw_command_deg"] - before["yaw_command_deg"])
        yaw_change = (yaw_change + math.pi) % (2 * math.pi) - math.pi
        self.assertLessEqual(abs(yaw_change), math.radians(120) * .02 + 1e-9)
        # Reading the state/parameter diagnostics must not expose mutable core state.
        controller.current_position[:] = 500
        controller.current_velocity[:] = 500
        after["bezier_parameters"]["outgoing_weight"] = 500
        self.assertLess(np.max(controller.current_position), 10)
        self.assertLess(np.max(controller.current_velocity), 10)
        self.assertEqual(controller.snapshot()["bezier_parameters"]["outgoing_weight"], .65)
        cleared = controller.set_parameters(BezierParameters())
        self.assertEqual(cleared["hits"], 1)
        self.assertIsNone(cleared["hit_event"])

    def test_selection_before_reset_and_invalid_selection_are_atomic(self):
        controller = RollingController(ControlConfig())
        parameters = BezierParameters(outgoing_weight=.1)
        self.assertIsNone(controller.set_parameters(parameters))
        self.assertEqual(controller.parameters, parameters)
        self.assertEqual(controller.planning_revision, 0)
        controller.reset([0, 0, 5])
        revision, path = controller.planning_revision, controller.planned_path
        with self.assertRaises(ValueError):
            controller.set_parameters({"outgoing_weight": .4})
        self.assertEqual(controller.planning_revision, revision)
        np.testing.assert_array_equal(controller.planned_path, path)
        with self.assertRaises(AttributeError):
            controller.parameters = BezierParameters()


if __name__ == "__main__":
    unittest.main()
