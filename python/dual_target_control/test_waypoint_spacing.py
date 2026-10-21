"""Three-dimensional local waypoint generation, independent of a simulator."""

import unittest
from unittest.mock import Mock

import numpy as np

from dual_target_control.core import ControlConfig, RollingController


class WaypointSpacingTests(unittest.TestCase):
    def local_config(self, **options):
        return ControlConfig(target_distance_min=6.0, target_distance_max=12.0, **options)

    def assert_local_target(self, config, anchor, target):
        distance = float(np.linalg.norm(target - anchor))
        self.assertGreaterEqual(distance, config.target_distance_min - 1e-9)
        self.assertLessEqual(distance, config.target_distance_max + 1e-9)
        self.assertGreater(distance, 2 * config.arrival_radius)
        self.assertTrue(np.all(target >= config.inner_lower - 1e-10))
        self.assertTrue(np.all(target <= config.inner_upper + 1e-10))

    def test_hundreds_of_seeds_keep_initial_and_rolling_legs_local(self):
        starts = ([0., 0., 5.], [19., 19., 9.], [-19., -19., 3.],
                  [18.99, -18.99, 3.01], [0., 19., 6.], [21., 0., 5.])
        blue_can_be_farther_than_max_from_plant = False
        for seed in range(240):
            with self.subTest(seed=seed):
                config = self.local_config(seed=seed)
                controller = RollingController(config)
                start = np.array(starts[seed % len(starts)])
                initial = controller.reset(start)
                red, blue = initial["targets"]
                self.assert_local_target(config, start, red)
                self.assert_local_target(config, red, blue)
                blue_can_be_farther_than_max_from_plant |= np.linalg.norm(blue - start) > 12
                self.assertTrue(np.all(initial["planned_path"] >= config.inner_lower - 1e-10))
                self.assertTrue(np.all(initial["planned_path"] <= config.inner_upper + 1e-10))
                for index in range(12):
                    previous_blue = controller.targets[1]
                    # Observe an already completed arrival to isolate generation
                    # from plant dynamics, which have their own turning tests.
                    controller.replan(controller.targets[0], [0., 0., 0.], 0.)
                    state = controller.update(controller.targets[0], [0., 0., 0.], 0., .02)
                    self.assertEqual(state["hits"], index + 1)
                    np.testing.assert_array_equal(state["targets"][0], previous_blue)
                    self.assert_local_target(config, previous_blue, state["targets"][1])
                    np.testing.assert_allclose(state["planned_path"][80], previous_blue)
                    np.testing.assert_allclose(state["planned_path"][-1], state["targets"][1])
        # The shell follows the queue: the blue point need not stay inside a
        # sphere centred on the current measured aircraft position.
        self.assertTrue(blue_can_be_farther_than_max_from_plant)

    def test_distance_is_three_dimensional_in_a_tall_narrow_field(self):
        config = self.local_config(x_min=-1.1, x_max=1.1, y_min=-1.1, y_max=1.1,
                                   height_min=0., height_max=20., margin=.05)
        for seed in range(50):
            config.seed = seed
            controller = RollingController(config)
            origin = np.array([0., 0., 10.])
            red, blue = controller.reset(origin)["targets"]
            self.assert_local_target(config, origin, red)
            self.assert_local_target(config, red, blue)
            self.assertLess(np.linalg.norm((red - origin)[:2]), 2.)
            self.assertGreater(abs(red[2] - origin[2]), 5.)

    def test_external_long_queue_rolls_from_blue_even_after_replan(self):
        controllers = [RollingController(self.local_config(seed=17)) for _ in range(2)]
        supplied = np.array([[-15., 0., 5.], [15., 0., 5.]])
        for controller in controllers:
            controller.reset([-16., 0., 5.])
            controller.set_targets(supplied)
        resumed = controllers[1].replan([0., 18., 8.], [0., 0., 0.], 1.)
        np.testing.assert_array_equal(resumed["targets"], supplied)
        self.assertEqual(resumed["hits"], 0)
        controllers[1].replan([-16., 0., 5.], [0., 0., 0.], 0.)
        rolled = [controller.update([-15.3, 0., 5.], [0., 0., 0.], 0., .02)
                  for controller in controllers]
        for state in rolled:
            self.assertEqual(state["hits"], 1)
            np.testing.assert_array_equal(state["targets"][0], supplied[1])
            self.assert_local_target(controllers[0].config, supplied[1], state["targets"][1])
            self.assertGreater(np.linalg.norm(state["targets"][1] - [-15.3, 0., 5.]), 12.)
        # Replanning never consumes RNG state or regenerates the queue.
        np.testing.assert_array_equal(rolled[0]["targets"], rolled[1]["targets"])

    def test_seed_and_reset_are_reproducible(self):
        for seed in range(30):
            controller = RollingController(self.local_config(seed=seed))
            first = controller.reset([18.99, -18.99, 3.01])["targets"]
            controller.replan([0., 0., 5.])
            repeated = controller.reset([18.99, -18.99, 3.01])["targets"]
            twin = RollingController(self.local_config(seed=seed)).reset(
                [18.99, -18.99, 3.01])["targets"]
            np.testing.assert_array_equal(first, repeated)
            np.testing.assert_array_equal(first, twin)

    def test_reset_uses_measured_anchor_even_outside_the_buffered_box(self):
        config = ControlConfig(target_distance_min=6., target_distance_max=6.)
        controller = RollingController(config)
        measured = np.array([22., 0., 5.])
        red, blue = controller.reset(measured)["targets"]
        self.assert_local_target(config, measured, red)
        self.assertAlmostEqual(np.linalg.norm(red - measured), 6., places=10)
        self.assertAlmostEqual(np.linalg.norm(blue - red), 6., places=10)
        self.assertGreater(abs(np.linalg.norm(red - np.clip(
            measured, config.inner_lower, config.inner_upper)) - 6.), .01)

    def test_deterministic_fallback_cannot_create_a_long_leg(self):
        config = self.local_config()
        controller = RollingController(config)
        # Force every proposed direction away from the box. A whole-field
        # corner fallback would exceed 12m; the shell intersection must not.
        impossible = Mock()
        impossible.normal.return_value = np.ones(3)
        impossible.uniform.side_effect = AssertionError("outward rays have no admissible radius")
        controller._rng = impossible
        for anchor in (config.inner_upper, config.inner_upper + [3., 3., 3.],
                       config.inner_upper + [12., 0., 0.]):
            with self.subTest(anchor=anchor):
                first = controller._sample_target(anchor)
                second = controller._sample_target(anchor)
                self.assert_local_target(config, anchor, first)
                np.testing.assert_array_equal(first, second)
        self.assertEqual(impossible.normal.call_count, 6 * 128)

    def test_small_feasible_field_and_fixed_distance_interval(self):
        for seed in range(100):
            config = ControlConfig(x_min=-1., x_max=1., y_min=-1., y_max=1.,
                                   height_min=3., height_max=5., margin=.2, seed=seed,
                                   target_distance_min=1.1, target_distance_max=1.1)
            controller = RollingController(config)
            anchor = np.array([0., 0., 4.])
            targets = controller.reset(anchor)["targets"]
            for target in targets:
                self.assert_local_target(config, anchor, target)
                self.assertAlmostEqual(np.linalg.norm(target - anchor), 1.1, places=10)
                anchor = target
            for _ in range(8):
                controller.replan(controller.targets[0], [0., 0., 0.], 0.)
                state = controller.update(controller.targets[0], [0., 0., 0.], 0., .02)
                self.assertIsNotNone(state["hit_event"])
                self.assert_local_target(config, state["targets"][0], state["targets"][1])

    def test_invalid_or_globally_unreachable_distance_settings_are_rejected(self):
        for minimum, maximum in ((-1., 12.), (6., -1.), (0., 12.), (6., 0.),
                                 (12., 6.), (1., 12.), (30., 40.),
                                 (float("nan"), 12.), (6., float("inf"))):
            with self.subTest(minimum=minimum, maximum=maximum), self.assertRaises(ValueError):
                ControlConfig(target_distance_min=minimum, target_distance_max=maximum).validate()
        with self.assertRaisesRegex(ValueError, "unreachable"):
            ControlConfig(x_min=-1., x_max=1., y_min=-1., y_max=1.,
                          height_min=3., height_max=5., margin=.2,
                          target_distance_min=2., target_distance_max=3.).validate()
        config = ControlConfig.from_dict({"target_distance_min": 6., "target_distance_max": 12.})
        self.assertEqual((config.target_distance_min, config.target_distance_max), (6., 12.))
        with self.assertRaisesRegex(ValueError, "unreachable.*measured"):
            RollingController(config).reset([40., 0., 5.])
        # The legacy opt-out remains available for existing small-field users.
        ControlConfig(x_min=-1., x_max=1., y_min=-1., y_max=1.,
                      height_min=3., height_max=5., margin=.2).validate()


if __name__ == "__main__":
    unittest.main()
