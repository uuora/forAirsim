"""Acceptance and command boundaries for pre-approach state matching."""
import json
import math
import unittest
from dataclasses import replace

import numpy as np

from matched_trajectory_start import (
    MatchedStartGate, MatchedStartParams, initial_state_check,
    reposition_xy_command, validate_matched_start_config,
)


class MatchedTrajectoryStartTests(unittest.TestCase):
    def setUp(self):
        self.params = MatchedStartParams()
        self.desired = [-5., -.5, -3.]

    def check(self, position=None, velocity=None, acceleration=None, quaternion=None):
        return initial_state_check(self.desired if position is None else position,
            [0, 0, 0] if velocity is None else velocity,
            [0, 0, 0] if acceleration is None else acceleration,
            [0, 0, 0, 1] if quaternion is None else quaternion,
            self.desired, self.params)

    def test_inside_position_tolerance_but_moving_cannot_enter_approach(self):
        result = self.check(position=[-4.99, -.5, -3], velocity=[.02, 0, 0])
        self.assertTrue(result["valid"])
        self.assertFalse(result["accepted"])
        self.assertFalse(result["protection_required"])
        self.assertIn("SPEED_TOLERANCE_EXCEEDED", result["reasons"])

    def test_position_speed_and_acceleration_acceptance_use_vector_norms(self):
        for kwargs, reason in (
            ({"position": [-4.99, -.49, -2.99]}, "POSITION_TOLERANCE_EXCEEDED"),
            ({"velocity": [.008, .008, 0]}, "SPEED_TOLERANCE_EXCEEDED"),
            ({"acceleration": [.04, .04, 0]}, "ACCELERATION_TOLERANCE_EXCEEDED"),
        ):
            with self.subTest(reason=reason):
                result = self.check(**kwargs)
                self.assertFalse(result["accepted"])
                self.assertIn(reason, result["reasons"])
                self.assertFalse(result["protection_required"])

    def test_physical_budget_violation_protects_without_clipping_actual_vectors(self):
        result = self.check(velocity=[.11, .11, 0], acceleration=[.22, .22, 0])
        self.assertTrue(result["protection_required"])
        self.assertEqual(result["velocity_ned_m_s"], [.11, .11, 0])
        self.assertEqual(result["acceleration_ned_m_s2"], [.22, .22, 0])
        self.assertIn("ACTUAL_SPEED_BUDGET_EXCEEDED", result["reasons"])
        self.assertIn("ACTUAL_ACCELERATION_BUDGET_EXCEEDED", result["reasons"])

    def test_nan_and_malformed_state_are_explicit_and_json_serializable(self):
        for field, value in (("position", [float("nan"), 0, 0]),
                             ("velocity", [0, float("inf"), 0]),
                             ("acceleration", [0, 0]), ("quaternion", [0, 0, 0, 0])):
            with self.subTest(field=field):
                result = self.check(**{field: value})
                self.assertFalse(result["valid"])
                self.assertFalse(result["accepted"])
                self.assertTrue(result["protection_required"])
                self.assertTrue(result["reasons"])
                json.dumps(result, allow_nan=False)

    def test_attitude_tolerance_waits_but_model_range_violation_protects(self):
        def yaw(degrees):
            return [0, 0, math.sin(math.radians(degrees)/2), math.cos(math.radians(degrees)/2)]
        near = self.check(quaternion=yaw(1.1))
        self.assertFalse(near["accepted"])
        self.assertFalse(near["protection_required"])
        self.assertAlmostEqual(near["body_rpy_deg"][2], 1.1)
        self.assertTrue(self.check(quaternion=yaw(8))["protection_required"])

    def test_dispatched_xy_command_uses_norm_cap_and_acceleration_ramp(self):
        previous = np.array([.025, 0.])
        result = reposition_xy_command([-9, -9, -3], self.desired, previous, .1, self.params)
        applied = np.asarray(result["command_velocity_xy_m_s"])
        self.assertLessEqual(np.linalg.norm(applied), .05 + 1e-12)
        self.assertLessEqual(np.linalg.norm(applied - previous), .005 + 1e-12)
        self.assertTrue(result["speed_norm_limited"])
        self.assertTrue(result["command_delta_limited"])

    def test_ramp_uses_actual_small_elapsed_and_caps_long_elapsed(self):
        short = reposition_xy_command([-9, -9, -3], self.desired, [0, 0], .02, self.params)
        long = reposition_xy_command([-9, -9, -3], self.desired, [0, 0], 20, self.params)
        self.assertAlmostEqual(short["command_delta_norm_m_s"], .001)
        self.assertAlmostEqual(long["command_delta_norm_m_s"], .005)
        self.assertEqual(long["command_ramp_effective_dt_s"], .1)
        with self.assertRaises(ValueError):
            reposition_xy_command(self.desired, self.desired, [.051, 0], .1, self.params)

    def test_reposition_never_commands_vertical_error(self):
        result = reposition_xy_command([-5, -.5, -2.9], self.desired, [0, 0], .1, self.params)
        self.assertEqual(result["command_velocity_xy_m_s"], [0., 0.])
        self.assertFalse(self.check(position=[-5, -.5, -2.9])["accepted"])

    def test_position_deadband_ramps_dispatched_velocity_to_zero(self):
        first = reposition_xy_command([-4.99, -.5, -3], self.desired, [.01, 0], .1, self.params)
        self.assertTrue(first["position_deadband_active"])
        np.testing.assert_allclose(first["command_velocity_xy_m_s"], [.005, 0])
        second = reposition_xy_command([-4.99, -.5, -3], self.desired,
            first["command_velocity_xy_m_s"], .1, self.params)
        np.testing.assert_allclose(second["command_velocity_xy_m_s"], [0, 0])
        self.assertLessEqual(second["command_delta_norm_m_s"], .005 + 1e-12)

    def test_deadband_matches_full_position_acceptance_with_fixed_height_residual(self):
        outside = [-5 + .01495, -.5, -3 + .002]
        inside = [-5 + .0145, -.5, -3 + .002]
        first = reposition_xy_command(outside, self.desired, [0, 0], .1, self.params)
        second = reposition_xy_command(inside, self.desired, [0, 0], .1, self.params)
        self.assertFalse(first["position_deadband_active"])
        self.assertGreater(first["position_error_norm_m"], .015)
        self.assertLess(first["command_velocity_xy_m_s"][0], 0)
        self.assertFalse(self.check(position=outside)["accepted"])
        self.assertTrue(second["position_deadband_active"])
        self.assertEqual(second["command_velocity_xy_m_s"], [0., 0.])
        self.assertTrue(self.check(position=inside)["accepted"])

    def test_stability_is_continuous_and_resets_on_failed_measurement(self):
        gate = MatchedStartGate(self.params, start_time_s=0)
        for now in (0, .1, .2):
            self.assertFalse(gate.update(self.check(), now)["gate_accepted"])
        failed = gate.update(self.check(velocity=[.02, 0, 0]), .3)
        self.assertEqual(failed["continuous_stable_s"], 0)
        for now in (.4, .5, .6, .7, .8):
            self.assertFalse(gate.update(self.check(), now)["gate_accepted"])
        self.assertTrue(gate.update(self.check(), .9)["gate_accepted"])

    def test_long_sample_gap_cannot_mint_unobserved_stability(self):
        gate = MatchedStartGate(self.params, start_time_s=0)
        gate.update(self.check(), 0)
        result = gate.update(self.check(), .6)
        self.assertFalse(result["gate_accepted"])
        self.assertEqual(result["stable_reset_reason"], "SAMPLE_GAP_EXCEEDED")
        self.assertEqual(result["continuous_stable_s"], 0)

    def test_timeout_and_nonmonotonic_time_never_accept(self):
        gate = MatchedStartGate(self.params, start_time_s=0)
        gate.update(self.check(), 29.6)
        gate.update(self.check(), 29.8)
        result = gate.update(self.check(), 30.)
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["gate_status"], "TIMEOUT")
        self.assertFalse(result["gate_accepted"])
        other = MatchedStartGate(self.params)
        other.update(self.check(), 10)
        backwards = other.update(self.check(), 9)
        self.assertTrue(backwards["protection_required"])
        self.assertEqual(backwards["stable_reset_reason"], "NONMONOTONIC_SAMPLE_TIME")

    def test_validator_preserves_inherited_limits_and_rejects_numeric_text(self):
        cfg = {"initial_state_match": {"enabled": True},
               "planning": {"execution_speed_cap_m_s": .15, "max_acceleration_m_s2": .3}}
        self.assertEqual(validate_matched_start_config(cfg), self.params)
        for invalid in (replace(self.params, physical_speed_cap_m_s=.151),
                        replace(self.params, physical_acceleration_cap_m_s2=.31),
                        replace(self.params, reposition_speed_cap_m_s=.06),
                        replace(self.params, reposition_acceleration_cap_m_s2=.06),
                        replace(self.params, command_dt_s=.11),
                        replace(self.params, position_tolerance_m=".015"),
                        replace(self.params, proportional_gain=float("nan"))):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_matched_start_config(invalid)


if __name__ == "__main__":
    unittest.main()
