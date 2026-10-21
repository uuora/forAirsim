"""Developmental command-policy tests with explicit fake clock/state receipts."""
import copy
from dataclasses import replace
import hashlib
import json
import math
import unittest

from t1_identity_motion_policy import IdentityMotionPolicy, MotionPolicyConfig
from t1_semantic_perception import Candidate, SemanticFrame


DESCRIPTION = "带白色横条的红色气球"
DESCRIPTION_SHA = hashlib.sha256(DESCRIPTION.encode("utf-8")).hexdigest()


def bound_fixture(*, frame_id="frame-1", sequence=1, receipt=1.0, ready=1.1):
    candidate = Candidate("candidate-1", (10.0, 10.0, 30.0, 30.0), 0.9, "balloon")
    frame = SemanticFrame("session", 1, frame_id, sequence, "a"*64, 64, 64,
                          receipt, ready, (candidate,))
    binding = {"session_id": "session", "target_generation": 1,
        "description_sha256": DESCRIPTION_SHA, "frame_id": frame_id,
        "frame_sequence": sequence, "image_sha256": "a"*64,
        "readback_receipt_monotonic_s": receipt, "candidates_ready_monotonic_s": ready}
    selection = {"status": "TEMPORAL_LOCAL_ASSOCIATION_VALID", "state": "TRACKING",
        "session_id": "session", "target_generation": 1, "description_sha256": DESCRIPTION_SHA,
        "current_frame_id": frame_id, "current_frame_sequence": sequence,
        "current_image_sha256": "a"*64, "current_binding": binding,
        "selected_candidate_id": candidate.candidate_id, "selected_candidate": candidate.to_dict(),
        "track_id": "track-1", "semantic_anchor": "request-1",
        "semantic_review_deadline_monotonic_s": 8.0,
        "semantic_binding_provenance": "CLIENT_SIDE_BINDING; NOT SERVER_ECHO",
        "identity_probability_calibrated": False, "instance_identity_guaranteed": False,
        "appearance_continuity_verified": False, "flight_authorized": False}
    surface = {"status": "SURFACE_GEOMETRY_ACCEPTED", "session_id": "session",
        "target_generation": 1, "frame_id": frame_id, "frame_sequence": sequence,
        "image_sha256": "a"*64, "selected_candidate_id": candidate.candidate_id,
        "selected_bbox_xyxy_pixels": list(candidate.bbox_xyxy_pixels),
        "selection_status": "TEMPORAL_LOCAL_ASSOCIATION_VALID", "freshness_checked": True,
        "identity_probability_calibrated": False, "instance_identity_guaranteed": False,
        "flight_authorized": False, "execution_guard_checked": False,
        "depth_kind": "DepthPerspective", "ray_range_m": 3.0,
        "surface_position_world_ned_m": [3.0, 0.0, -3.0],
        "surface_position_local_ned_m": [3.0, 0.0, -3.0],
        "rgb_reference_world_ned_m": [3.0, 0.0, -3.0],
        "temporal_track_id": "track-1", "semantic_anchor": "request-1",
        "description_sha256": DESCRIPTION_SHA, "current_binding": copy.deepcopy(binding)}
    rebind_surface(selection, surface)
    return frame, selection, surface


def rebind_surface(selection, surface):
    surface["selection_certificate_sha256"] = hashlib.sha256(json.dumps(selection,
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


class MotionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = IdentityMotionPolicy("session", 1, DESCRIPTION, clock=lambda: 1.12)
        self.frame, self.selection, self.surface = bound_fixture()
        self.args = dict(frame=self.frame, selection=self.selection, surface=self.surface,
            desired_xy_m_s=[0.03, 0.0], actual_velocity_xy_m_s=[0.0, 0.0],
            self_position_ned_m=[0.0, 0.0, -3.0], previous_command_xy_m_s=[0.0, 0.0],
            previous_command_monotonic_s=1.02, self_state_receipt_monotonic_s=1.12, now=1.12)

    def decide(self, **changes):
        return self.policy.decide(**(self.args | changes))

    def test_accepts_geometry_false_flight_flag_without_promoting_identity(self):
        result = self.decide()
        self.assertEqual(result["status"], "ALLOW_BOUNDED_COMMAND")
        self.assertTrue(result["target_command_authorized"])
        self.assertFalse(result["flight_authorized"])
        self.assertFalse(result["instance_identity_guaranteed"])
        self.assertAlmostEqual(result["command_velocity_xy_m_s"][0], 0.015)
        self.assertLessEqual(result["command_duration_s"], 0.12)
        self.assertGreaterEqual(result["latest_dispatch_monotonic_s"], result["decision_monotonic_s"])

    def test_injected_clock_is_used_when_now_omitted(self):
        arguments = dict(self.args)
        arguments.pop("now")
        self.assertEqual(self.policy.decide(**arguments)["decision_monotonic_s"], 1.12)

    def test_actual_elapsed_drives_ramp_not_nominal_period(self):
        result = self.decide(previous_command_monotonic_s=1.10)
        self.assertAlmostEqual(result["command_velocity_xy_m_s"][0], 0.003)
        self.assertAlmostEqual(result["elapsed_previous_command_s"], 0.02)

    def test_initial_zero_elapsed_cannot_accelerate(self):
        result = self.decide(previous_command_monotonic_s=1.12)
        self.assertEqual(result["status"], "WAIT_NO_TARGET")
        self.assertEqual(result["reason"], "INITIAL_ZERO_ELAPSED_PREVENTS_POSITIVE_CHANGE")
        self.assertEqual(result["command_velocity_xy_m_s"], [0.0, 0.0])

    def test_diagonal_norm_and_ramp_limits(self):
        result = self.decide(desired_xy_m_s=[3.0, 4.0], previous_command_monotonic_s=0.92)
        norm = math.hypot(*result["command_velocity_xy_m_s"])
        self.assertLessEqual(norm, 0.03+1e-12)
        self.assertAlmostEqual(norm, 0.03)

    def test_near_expiry_shortens_lease(self):
        result = self.decide(now=1.45, self_state_receipt_monotonic_s=1.45,
                             previous_command_monotonic_s=1.35)
        self.assertEqual(result["status"], "ALLOW_BOUNDED_COMMAND")
        self.assertAlmostEqual(result["command_duration_s"], 0.03)
        self.assertAlmostEqual(result["command_end_deadline_monotonic_s"], 1.48)

    def test_insufficient_lease_waits_then_requires_brake_if_moving(self):
        result = self.decide(now=1.49, self_state_receipt_monotonic_s=1.49,
                            previous_command_monotonic_s=1.39)
        self.assertEqual(result["status"], "WAIT_NO_TARGET")
        result = self.decide(now=1.491, self_state_receipt_monotonic_s=1.491,
            previous_command_monotonic_s=1.391, previous_command_xy_m_s=[0.02, 0.0])
        self.assertEqual(result["status"], "REQUIRE_SELF_STATE_BRAKE")
        self.assertFalse(result["target_command_authorized"])

    def test_missing_target_brakes_previous_nonzero_command(self):
        result = self.decide(selection={"status": "NO_TARGET_OUTPUT"}, surface=None,
                             previous_command_xy_m_s=[0.02, 0.0])
        self.assertEqual(result["status"], "REQUIRE_SELF_STATE_BRAKE")
        self.assertAlmostEqual(result["command_velocity_xy_m_s"][0], 0.005)
        self.assertTrue(result["self_state_brake_authorized"])
        self.assertFalse(result["target_observation_used_for_brake"])

    def test_missing_target_with_residual_actual_velocity_still_brakes(self):
        result = self.decide(selection=None, actual_velocity_xy_m_s=[0.01, 0.0])
        self.assertEqual(result["status"], "REQUIRE_SELF_STATE_BRAKE")
        self.assertEqual(result["command_velocity_xy_m_s"], [0.0, 0.0])

    def test_lost_target_when_stationary_waits(self):
        result = self.decide(frame=None, selection=None, surface=None)
        self.assertEqual(result["status"], "WAIT_NO_TARGET")
        self.assertEqual(result["command_duration_s"], 0.0)

    def test_gpu_delayed_result_cannot_authorize_motion(self):
        frame, selection, surface = bound_fixture(ready=1.6)
        result = self.decide(frame=frame, selection=selection, surface=surface, now=1.61,
            self_state_receipt_monotonic_s=1.61, previous_command_monotonic_s=1.51,
            previous_command_xy_m_s=[0.02, 0.0])
        self.assertEqual(result["status"], "REQUIRE_SELF_STATE_BRAKE")
        self.assertEqual(result["reason"], "CURRENT_FRAME_EXPIRED")

    def test_future_result_rejected(self):
        frame, selection, surface = bound_fixture(ready=1.2)
        result = self.decide(frame=frame, selection=selection, surface=surface)
        self.assertEqual(result["status"], "REJECT")
        self.assertEqual(result["reason"], "FRAME_READY_IN_FUTURE")

    def test_same_frame_cannot_renew_target_command(self):
        self.decide()
        result = self.decide(now=1.13, self_state_receipt_monotonic_s=1.13)
        self.assertEqual(result["status"], "REJECT")
        self.assertEqual(result["reason"], "TARGET_FRAME_ALREADY_USED_FOR_COMMAND")

    def test_next_sequence_requires_actually_new_readback(self):
        self.decide()
        frame, selection, surface = bound_fixture(frame_id="frame-2", sequence=2, receipt=1.0)
        result = self.decide(frame=frame, selection=selection, surface=surface,
                            now=1.13, self_state_receipt_monotonic_s=1.13)
        self.assertEqual(result["reason"], "FRAME_RECEIPT_NOT_ACTUALLY_NEWER")

    def test_frame_id_changed_content_is_rejected_even_before_command(self):
        self.decide(selection=None)
        other = replace(self.frame, candidates=(Candidate("candidate-2", (11.0, 10.0, 30.0, 30.0), 0.9),))
        result = self.decide(frame=other)
        self.assertEqual(result["reason"], "FRAME_ID_REUSED_WITH_DIFFERENT_BINDING")

    def test_sequence_regression_rejected(self):
        self.decide()
        frame, selection, surface = bound_fixture(frame_id="old", sequence=0)
        result = self.decide(frame=frame, selection=selection, surface=surface)
        self.assertEqual(result["reason"], "FRAME_SEQUENCE_REGRESSION")

    def test_context_and_description_mismatches_rejected(self):
        for field, value in (("session_id", "other"), ("target_generation", 2),
                             ("description_sha256", "b"*64), ("current_image_sha256", "b"*64)):
            with self.subTest(field=field):
                policy = IdentityMotionPolicy("session", 1, DESCRIPTION)
                selection = dict(self.selection, **{field: value})
                result = policy.decide(**(self.args | {"selection": selection}))
                self.assertEqual(result["status"], "REJECT")
                self.assertEqual(result["reason"], "SELECTION_CURRENT_CONTEXT_MISMATCH")

    def test_bad_receipt_binding_and_bool_integer_rejected(self):
        for field, value in (("readback_receipt_monotonic_s", 1.01), ("target_generation", True)):
            with self.subTest(field=field):
                selection = copy.deepcopy(self.selection)
                selection["current_binding"][field] = value
                policy = IdentityMotionPolicy("session", 1, DESCRIPTION)
                result = policy.decide(**(self.args | {"selection": selection}))
                self.assertEqual(result["reason"], "SELECTION_RECEIPT_BINDING_MISMATCH")

    def test_surface_temporal_provenance_and_digest_bound(self):
        for field, value in (("frame_id", "other"), ("selected_candidate_id", "other"),
                             ("temporal_track_id", "other"), ("semantic_anchor", "other"),
                             ("selection_certificate_sha256", "b"*64)):
            with self.subTest(field=field):
                policy = IdentityMotionPolicy("session", 1, DESCRIPTION)
                result = policy.decide(**(self.args | {"surface": dict(self.surface, **{field: value})}))
                self.assertEqual(result["status"], "REJECT")

    def test_identity_claims_cannot_be_upgraded(self):
        for field in ("identity_probability_calibrated", "instance_identity_guaranteed",
                      "appearance_continuity_verified"):
            with self.subTest(field=field):
                policy = IdentityMotionPolicy("session", 1, DESCRIPTION)
                result = policy.decide(**(self.args | {"selection": dict(self.selection, **{field: True})}))
                self.assertEqual(result["reason"], "SELECTION_IDENTITY_CLAIM_EXCEEDS_EVIDENCE")

    def test_semantic_review_deadline_limits_lease(self):
        self.selection["semantic_review_deadline_monotonic_s"] = 1.20
        rebind_surface(self.selection, self.surface)
        result = self.decide()
        self.assertEqual(result["status"], "ALLOW_BOUNDED_COMMAND")
        self.assertAlmostEqual(result["command_duration_s"], 0.06)

    def test_review_deadline_expiry_requires_reanchor(self):
        self.selection["semantic_review_deadline_monotonic_s"] = 1.11
        result = self.decide()
        self.assertEqual(result["status"], "WAIT_NO_TARGET")
        self.assertEqual(result["reason"], "SEMANTIC_REVIEW_DEADLINE_EXPIRED")

    def test_expired_own_state_does_not_generate_braking_command(self):
        result = self.decide(now=1.3, self_state_receipt_monotonic_s=1.12,
            previous_command_xy_m_s=[0.02, 0.0], selection=None)
        self.assertEqual(result["status"], "REJECT")
        self.assertEqual(result["reason"], "SELF_STATE_EXPIRED_REMEASURE_BEFORE_BRAKE")
        self.assertIsNone(result["command_velocity_xy_m_s"])

    def test_previous_dispatch_future_or_self_state_before_command_rejected(self):
        for previous, receipt in ((1.13, 1.12), (1.1, 1.09)):
            with self.subTest(previous=previous):
                policy = IdentityMotionPolicy("session", 1, DESCRIPTION)
                result = policy.decide(**(self.args | {"previous_command_monotonic_s": previous,
                                      "self_state_receipt_monotonic_s": receipt}))
                self.assertEqual(result["reason"], "SELF_OR_COMMAND_CLOCK_ORDER_INVALID")

    def test_long_interval_never_uses_nominal_ramp_for_target(self):
        result = self.decide(previous_command_monotonic_s=0.7)
        self.assertEqual(result["status"], "WAIT_NO_TARGET")
        self.assertEqual(result["reason"], "PREVIOUS_COMMAND_INTERVAL_EXCEEDED")
        policy = IdentityMotionPolicy("session", 1, DESCRIPTION)
        result = policy.decide(**(self.args | {"previous_command_monotonic_s": 0.7,
            "previous_command_xy_m_s": [0.02, 0.0], "actual_velocity_xy_m_s": [0.01, 0.0]}))
        self.assertEqual(result["status"], "REQUIRE_SELF_STATE_BRAKE")
        self.assertTrue(result["brake_interval_capped_for_long_gap"])
        self.assertEqual(result["command_velocity_xy_m_s"], [0.0, 0.0])

    def test_clock_regression_nonfinite_and_bool_rejected(self):
        self.decide(selection=None)
        result = self.decide(now=1.11, self_state_receipt_monotonic_s=1.11)
        self.assertEqual(result["reason"], "DECISION_CLOCK_REGRESSION")
        for value in (math.nan, math.inf, -1, True):
            result = self.decide(now=value)
            self.assertEqual(result["status"], "REJECT")
            self.assertEqual(result["reason"], "INVALID_DECISION_TIME")

    def test_invalid_vectors_and_model_speed_ceiling_rejected(self):
        changes = (("desired_xy_m_s", [math.nan, 0.0]), ("actual_velocity_xy_m_s", [0.061, 0.0]),
            ("previous_command_xy_m_s", [0.031, 0.0]), ("self_position_ned_m", [2.0, 0.0, -3.0]))
        for key, value in changes:
            with self.subTest(key=key):
                result = IdentityMotionPolicy("session", 1, DESCRIPTION).decide(**(self.args | {key: value}))
                self.assertEqual(result["status"], "REJECT")

    def test_position_envelope_reduces_near_boundary_command(self):
        result = self.decide(self_position_ned_m=[0.97, 0.0, -3.0])
        self.assertEqual(result["status"], "ALLOW_BOUNDED_COMMAND")
        self.assertLess(result["command_velocity_xy_m_s"][0], 0.01)
        self.assertTrue(result["boundary_envelope"]["boundary_limited"])

    def test_no_feasible_boundary_brake_declines_command(self):
        result = self.decide(self_position_ned_m=[0.999, 0.0, -3.0],
            previous_command_xy_m_s=[0.02, 0.0], actual_velocity_xy_m_s=[0.02, 0.0],
            previous_command_monotonic_s=1.11, selection=None)
        self.assertEqual(result["status"], "REJECT")
        self.assertEqual(result["reason"], "NO_FEASIBLE_SELF_STATE_BRAKE")
        self.assertIsNone(result["command_velocity_xy_m_s"])

    def test_closed_policy_cannot_command(self):
        self.policy.close()
        self.assertEqual(self.decide()["reason"], "MOTION_POLICY_CLOSED")

    def test_nonfinite_or_unserializable_certificate_rejected_without_exception(self):
        for value in (math.nan, {1, 2}):
            with self.subTest(value=value):
                selection = dict(self.selection, annotation=value)
                result = IdentityMotionPolicy("session", 1, DESCRIPTION).decide(**(self.args | {"selection": selection}))
                self.assertEqual(result["reason"], "SELECTION_NOT_FINITE_JSON_CERTIFICATE")
                self.assertEqual(result["status"], "REJECT")

    def test_nonjson_receipt_binding_rejected_without_exception(self):
        selection = copy.deepcopy(self.selection)
        selection["current_binding"]["readback_receipt_monotonic_s"] = {1, 2}
        result = self.decide(selection=selection)
        self.assertEqual(result["reason"], "SELECTION_RECEIPT_BINDING_MISMATCH")

    def test_nonfinite_frame_report_rejected_without_exception(self):
        frame = replace(self.frame, detector_report_json='{"latency":NaN}')
        result = self.decide(frame=frame)
        self.assertEqual(result["reason"], "INVALID_FRAME_JSON_BINDING")

    def test_unreadable_surface_or_boolean_box_rejected_without_exception(self):
        for change in (dict(surface_position_world_ned_m=[math.nan, 0.0, -3.0]),
                       dict(ray_range_m=0.0), dict(selected_bbox_xyxy_pixels=[True, 10.0, 30.0, 30.0])):
            with self.subTest(change=change):
                result = IdentityMotionPolicy("session", 1, DESCRIPTION).decide(**(self.args |
                    {"surface": dict(self.surface, **change)}))
                self.assertEqual(result["status"], "REJECT")

    def test_config_cannot_widen_observed_xy_envelope(self):
        changes = (dict(current_frame_max_age_s=0.51), dict(command_lease_s=0.13),
            dict(command_acceleration_cap_m_s2=0.16), dict(max_previous_command_interval_s=0.41),
            dict(speed_cap_m_s=0.061), dict(freshness_reserve_s=0.01),
            dict(boundary_horizon_s=0.5), dict(braking_acceleration_model_m_s2=0.2),
            dict(self_state_max_age_s=0.2), dict(position_reserve_m=0.01))
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                MotionPolicyConfig(**change)


if __name__ == "__main__":
    unittest.main()
