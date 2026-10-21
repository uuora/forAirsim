"""Terminal-path regressions using recorded geometry and a lagged point mass.

The G19 fixture is the state immediately after event 18 in
logs/dual_target_airsim_20261015_195157_981361/telemetry.jsonl. The first target
has only 2.4 mm of +y clearance, while its inbound side has room to settle.
These checks do not claim to simulate AirSim thrust, attitude or rotor physics.
"""

import math
import unittest

import numpy as np

from dual_target_control import core
from validate_dual_target_turning import TurningScenario, run_scenario, wrap


FIELD = dict(x_min=-50.0, x_max=50.0, y_min=-50.0, y_max=50.0,
             height_min=2.0, height_max=25.0)
G19_POSITION = np.array([-25.185871124267578, 46.93100357055664, 12.353456497192383])
G19_VELOCITY = np.array([-0.46439051628112793, 2.064114809036255, 0.6681298017501831])
G19_YAW = 1.9326490181624882
G19_TARGETS = np.array([
    [-26.425193156993743, 48.99762374466296, 18.640622101518055],
    [-32.38706916466101, 46.30562764579211, 23.47873229484489],
])


def first_leg(controller):
    """Locate the target in the public path without assuming a sample count."""
    path = controller.planned_path
    distances = np.linalg.norm(path - controller.targets[0], axis=1)
    index = int(np.argmin(distances))
    if distances[index] > 1e-9:
        raise AssertionError("the planned path must pass through its first target")
    return path[:index + 1]


def straight_terminal_length(points):
    """Measure the contiguous collinear suffix of the sampled public path."""
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    nonzero = np.flatnonzero(lengths > 1e-12)
    if not len(nonzero):
        return 0.0
    direction = segments[nonzero[-1]] / lengths[nonzero[-1]]
    result = 0.0
    for segment, length in zip(segments[::-1], lengths[::-1]):
        if length <= 1e-12:
            continue
        if np.dot(segment, direction) <= 0 or np.linalg.norm(np.cross(segment, direction)) > 1e-8 * length:
            break
        result += float(length)
    return result


class TerminalCorridorTests(unittest.TestCase):
    def controller(self, position, targets, velocity=None, yaw=0.0):
        config = core.ControlConfig(**FIELD)
        controller = core.RollingController(config)
        controller.reset(position, velocity, yaw)
        controller.set_targets(targets)
        return controller

    def assert_path_inside(self, controller):
        path = controller.planned_path
        self.assertTrue(np.isfinite(path).all())
        self.assertTrue(np.all(path >= controller.config.inner_lower - 1e-9))
        self.assertTrue(np.all(path <= controller.config.inner_upper + 1e-9))
        np.testing.assert_allclose(first_leg(controller)[-1], controller.targets[0], atol=1e-10)
        np.testing.assert_allclose(path[-1], controller.targets[1], atol=1e-10)

    def test_wall_clearance_does_not_remove_inbound_final_metre(self):
        # Rotate the measured case onto every horizontal wall. A blocked
        # outgoing continuation must not remove feasible inbound straightness.
        for turns in range(4):
            angle = turns * math.pi / 2
            rotation = np.array([[math.cos(angle), -math.sin(angle), 0.0],
                                 [math.sin(angle), math.cos(angle), 0.0],
                                 [0.0, 0.0, 1.0]])
            with self.subTest(wall=turns):
                controller = self.controller(rotation @ G19_POSITION,
                    G19_TARGETS @ rotation.T, rotation @ G19_VELOCITY, wrap(G19_YAW + angle))
                self.assert_path_inside(controller)
                self.assertGreaterEqual(straight_terminal_length(first_leg(controller)), 1.0)

    def test_short_successor_distance_does_not_shorten_inbound_corridor(self):
        # Both successors request the same 90-degree turn. Their distances
        # differ, but neither changes the available space before the red target.
        legs = []
        for successor in ([8.0, 1.2, 8.0], [8.0, 6.0, 8.0]):
            with self.subTest(successor=successor):
                controller = self.controller([0.0, 0.0, 8.0], [[8.0, 0.0, 8.0], successor])
                self.assert_path_inside(controller)
                leg = first_leg(controller)
                legs.append(leg)
                self.assertGreaterEqual(straight_terminal_length(leg), 1.0)
        np.testing.assert_allclose(legs[0], legs[1], atol=1e-10)

    def test_successor_direction_still_shapes_inbound_path(self):
        left = self.controller([0.0, 0.0, 8.0], [[8.0, 0.0, 8.0], [8.0, 6.0, 8.0]])
        right = self.controller([0.0, 0.0, 8.0], [[8.0, 0.0, 8.0], [8.0, -6.0, 8.0]])
        for controller in (left, right):
            self.assert_path_inside(controller)
            self.assertGreaterEqual(straight_terminal_length(first_leg(controller)), 1.0)
        difference = np.linalg.norm(first_leg(left) - first_leg(right), axis=1)
        self.assertGreater(float(np.max(difference)), 0.5)
        left_entry = first_leg(left)[-1] - first_leg(left)[-2]
        right_entry = first_leg(right)[-1] - first_leg(right)[-2]
        self.assertGreater(left_entry[1], 0.0)
        self.assertLess(right_entry[1], 0.0)

    def test_recorded_g19_first_pass_with_velocity_and_measured_yaw_lag(self):
        scenario = TurningScenario("recorded_g19_near_wall", tuple(G19_POSITION),
            tuple(G19_VELOCITY), tuple(map(tuple, G19_TARGETS)), FIELD,
            yaw=G19_YAW, deadline=25.0)
        for velocity_lag in (0.18, 0.30, 0.35):
            for intervals in ((0.02,), (0.02, 0.02, 0.04, 0.06)):
                with self.subTest(velocity_lag=velocity_lag, intervals=intervals):
                    result = run_scenario(core, scenario, velocity_lag=velocity_lag,
                        yaw_lag=0.35, intervals=intervals)
                    # The independent detector examines measured yaw at actual
                    # sphere entry, not whether the command points at the goal.
                    self.assertEqual(result["arrival_radius_m"], 0.5)
                    self.assertEqual(result["heading_tolerance_deg"], 20.0)
                    self.assertTrue(result["passed"], result)
                    self.assertTrue(result["checks"]["queue_roll_preserves_blue"])
                    for leg in result["legs"]:
                        self.assertEqual(leg["geometric_entry_count"], 1)
                        self.assertEqual(leg["near_reversal_count"], 0)
                        self.assertEqual(leg["plane_crossings_before_hit"], 0)
                        self.assertLessEqual(leg["first_entry_yaw_error_deg"], 20.0)

    def test_aligned_command_cannot_replace_wrong_measured_yaw(self):
        controller = self.controller([6.8, 0.0, 8.0], [[8.0, 0.0, 8.0], [14.0, 0.0, 8.0]],
                                     yaw=math.pi / 2)
        # Stay inside the sphere while the yaw command converges, deliberately
        # withholding body rotation. A relaxed yaw gate would count this hit.
        for _ in range(100):
            state = controller.update([7.8, 0.0, 8.0], [0.0, 0.0, 0.0], math.pi / 2, 0.02)
            self.assertEqual(state["hits"], 0)
            self.assertIsNone(state["hit_event"])
        self.assertLess(abs(state["yaw_command_deg"]), 1e-8)
        self.assertGreater(state["yaw_error_deg"], controller.config.heading_tolerance_deg)
        # Only a subsequent actual body-yaw measurement can qualify arrival.
        controller.update([7.8, 0.0, 8.0], [0.0, 0.0, 0.0], 0.0, 0.02)
        accepted = controller.update([7.8, 0.0, 8.0], [0.0, 0.0, 0.0], 0.0, 0.02)
        self.assertEqual(accepted["hits"], 1)


if __name__ == "__main__":
    unittest.main()
