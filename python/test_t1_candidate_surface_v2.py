"""Contract failures and independent geometry checks for the temporal adapter."""
import hashlib
import json
import unittest
from copy import deepcopy
from dataclasses import replace

import numpy as np

from t1_candidate_surface import SurfaceConfig, SurfaceRejection
from t1_candidate_surface_v2 import temporal_candidate_surface_estimate
from t1_pose_geometry import CameraPose, SceneCameraModel
from t1_semantic_perception import Candidate, SemanticFrame
from replay_t1_temporal_surfaces import geometry_inputs, restore_frame


class TemporalSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.description = "designated stripe balloon"
        self.frame = SemanticFrame("session", 2, "current", 4,
            hashlib.sha256(b"image").hexdigest(), 100, 80, 10., 10.1,
            (Candidate("chosen", (40., 30., 60., 50.), .7, "balloon"),
             Candidate("higher_score_rival", (70., 30., 90., 50.), .95, "balloon")), b"image")
        self.camera = SceneCameraModel(100, 80, 50., 50., 50., 40.)
        self.pose = CameraPose([1, 2, -3], [0, 0, 0, 1], [10, 20, 30])
        self.depth = np.full((80, 100), 5.)
        self.pairing = {"frame_id": "current", "image_sha256": self.frame.image_sha256,
            "depth_kind": "DepthPerspective", "rgb_readback_timestamp_ns": 1_000_000_000,
            "depth_readback_timestamp_ns": 1_005_000_000,
            "intrinsics_pairing_scope": "caller_declared_equal_RGB_depth_Scene_intrinsics"}
        sha = hashlib.sha256(self.description.encode()).hexdigest()
        self.cert = {"status": "TEMPORAL_LOCAL_ASSOCIATION_VALID", "state": "TRACKING",
            "session_id": "session", "target_generation": 2, "description_sha256": sha,
            "current_frame_id": "current", "current_frame_sequence": 4,
            "current_image_sha256": self.frame.image_sha256,
            "selected_candidate_id": "chosen", "selected_candidate": self.frame.candidates[0].to_dict(),
            "current_binding": {"session_id": "session", "target_generation": 2,
                "description_sha256": sha, "frame_id": "current", "frame_sequence": 4,
                "image_sha256": self.frame.image_sha256, "readback_receipt_monotonic_s": 10.,
                "candidates_ready_monotonic_s": 10.1},
            "track_id": "track-1", "semantic_anchor": "request-1",
            "semantic_expires_monotonic_s": 10., "semantic_review_deadline_monotonic_s": 16.,
            "semantic_review_remaining_s": 5.8, "confirmation_count": 2, "confirmation_required": 2,
            "configured_timeouts": {"current_frame_max_age_s": .5,
                "semantic_source_max_age_s": 2., "max_link_gap_s": .5, "max_unreviewed_track_s": 8.},
            "identity_probability_calibrated": False, "instance_identity_guaranteed": False,
            "appearance_continuity_verified": False, "flight_authorized": False,
            "semantic_binding_provenance": "CLIENT_SIDE_BINDING; NOT SERVER_ECHO"}

    def estimate(self, frame=None, **overrides):
        args = dict(description=self.description, now=10.2, depth=self.depth,
            camera=self.camera, rgb_pose=self.pose, depth_pose=self.pose,
            selection_certificate=self.cert, pairing_contract=self.pairing)
        args.update(overrides)
        return temporal_candidate_surface_estimate(frame or self.frame, **args)

    def test_temporal_lower_score_selection_preserved_and_no_motion_permission(self):
        result = self.estimate()
        self.assertEqual(result["selected_candidate_id"], "chosen")
        self.assertEqual(result["selection_status"], "TEMPORAL_LOCAL_ASSOCIATION_VALID")
        self.assertFalse(result["semantic_local_association_certificate_accepted"])
        self.assertTrue(result["temporal_local_association_certificate_accepted"])
        self.assertTrue(result["freshness_checked"])
        for key in ("flight_authorized", "online_control_enabled", "execution_guard_checked",
                    "identity_probability_calibrated", "instance_identity_guaranteed"):
            self.assertFalse(result[key])
        np.testing.assert_allclose(result["surface_position_world_ned_m"], [16, 22, 27])
        self.assertEqual(result["current_binding"], self.cert["current_binding"])
        expected = hashlib.sha256(json.dumps(self.cert, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.assertEqual(result["selection_certificate_sha256"], expected)

    def test_temporal_track_outlives_semantic_reply_but_not_review_window(self):
        # A track is not mislabeled as a fresh semantic reply: source expiry=10,
        # local track now=10.2 remains inside the separately checked review window.
        self.assertGreater(10.2, self.cert["semantic_expires_monotonic_s"])
        self.assertEqual(self.estimate()["status"], "SURFACE_GEOMETRY_ACCEPTED")

    def test_non_temporal_or_not_tracking_certificate_rejected(self):
        for status in ("NO_TARGET_OUTPUT", "SEMANTIC_LOCAL_ASSOCIATION_VALID", "DETECTOR_ONLY_UNCALIBRATED"):
            with self.subTest(status=status), self.assertRaisesRegex(SurfaceRejection, "NOT_ACCEPTED"):
                self.estimate(selection_certificate=dict(self.cert, status=status))
        for state in ("LOST", "AMBIGUOUS", "RECOVERY_PENDING", "UNINITIALIZED"):
            with self.subTest(state=state), self.assertRaisesRegex(SurfaceRejection, "NOT_TRACKING"):
                self.estimate(selection_certificate=dict(self.cert, state=state))

    def test_outer_context_and_description_are_bound(self):
        for key, value in (("session_id", "other"), ("target_generation", 3),
            ("current_frame_id", "old"), ("current_frame_sequence", 3),
            ("current_image_sha256", "0"*64), ("description_sha256", "0"*64)):
            with self.subTest(key=key), self.assertRaisesRegex(SurfaceRejection, "BINDING_MISMATCH"):
                self.estimate(selection_certificate=dict(self.cert, **{key: value}))
        with self.assertRaisesRegex(SurfaceRejection, "BINDING_MISMATCH"):
            self.estimate(description="different target")

    def test_nested_binding_requires_all_timestamps_and_exact_fields(self):
        for key in self.cert["current_binding"]:
            cert = deepcopy(self.cert)
            cert["current_binding"].pop(key)
            with self.subTest(key=key), self.assertRaisesRegex(SurfaceRejection, "INCOMPLETE"):
                self.estimate(selection_certificate=cert)
        for key in ("readback_receipt_monotonic_s", "candidates_ready_monotonic_s"):
            cert = deepcopy(self.cert)
            cert["current_binding"][key] += .01
            with self.assertRaisesRegex(SurfaceRejection, "BINDING_MISMATCH"):
                self.estimate(selection_certificate=cert)
        cert = deepcopy(self.cert)
        cert["current_binding"]["truth"] = "injected"
        with self.assertRaisesRegex(SurfaceRejection, "INCOMPLETE"):
            self.estimate(selection_certificate=cert)

    def test_bool_integer_bindings_do_not_equal_one(self):
        for nested, key in ((False, "target_generation"), (False, "current_frame_sequence"),
            (True, "target_generation"), (True, "frame_sequence")):
            cert = deepcopy(self.cert)
            (cert["current_binding"] if nested else cert)[key] = True
            with self.assertRaisesRegex(SurfaceRejection, "INTEGER"):
                self.estimate(selection_certificate=cert)

    def test_current_age_and_future_ready_are_checked_at_use_time(self):
        self.assertEqual(self.estimate(now=10.5)["status"], "SURFACE_GEOMETRY_ACCEPTED")
        with self.assertRaisesRegex(SurfaceRejection, "CURRENT_FRAME_EXPIRED"):
            self.estimate(now=10.500001)
        with self.assertRaisesRegex(SurfaceRejection, "READY_IN_FUTURE"):
            self.estimate(now=10.099999)

    def test_deadline_inflation_and_untrusted_timeout_override_rejected(self):
        with self.assertRaisesRegex(SurfaceRejection, "DEADLINE_INCONSISTENT"):
            self.estimate(selection_certificate=dict(self.cert, semantic_review_deadline_monotonic_s=1000.))
        cert = deepcopy(self.cert)
        cert["configured_timeouts"]["max_unreviewed_track_s"] = 999.
        with self.assertRaisesRegex(SurfaceRejection, "TIMEOUT_CONFIG_MISMATCH"):
            self.estimate(selection_certificate=cert)
        cert = deepcopy(self.cert)
        cert["semantic_expires_monotonic_s"] = 12.
        cert["semantic_review_deadline_monotonic_s"] = 18.
        with self.assertRaisesRegex(SurfaceRejection, "DEADLINE_INCONSISTENT"):
            self.estimate(selection_certificate=cert)

    def test_fresh_current_frame_cannot_revive_expired_review(self):
        frame = replace(self.frame, readback_receipt_monotonic_s=16.1, candidates_ready_monotonic_s=16.2)
        cert = deepcopy(self.cert)
        cert["current_binding"]["readback_receipt_monotonic_s"] = 16.1
        cert["current_binding"]["candidates_ready_monotonic_s"] = 16.2
        with self.assertRaisesRegex(SurfaceRejection, "REVIEW_DEADLINE_EXPIRED"):
            self.estimate(frame, now=16.2, selection_certificate=cert)

    def test_certificate_flags_and_provenance_fail_closed(self):
        for key in ("identity_probability_calibrated", "instance_identity_guaranteed",
                    "appearance_continuity_verified", "flight_authorized"):
            for value in (True, None, 0):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(SurfaceRejection, "EVIDENCE_SCOPE"):
                    self.estimate(selection_certificate=dict(self.cert, **{key: value}))
        with self.assertRaisesRegex(SurfaceRejection, "PROVENANCE"):
            self.estimate(selection_certificate=dict(self.cert, semantic_binding_provenance="SERVER_VERIFIED"))

    def test_confirmation_anchor_and_track_are_required(self):
        for key in ("semantic_anchor", "track_id"):
            with self.assertRaisesRegex(SurfaceRejection, "MISSING_TEMPORAL"):
                self.estimate(selection_certificate=dict(self.cert, **{key: ""}))
        for updates in ({"confirmation_count": 1}, {"confirmation_required": 1}, {"confirmation_count": True}):
            with self.assertRaisesRegex(SurfaceRejection, "CONFIRMATION"):
                self.estimate(selection_certificate=dict(self.cert, **updates))

    def test_candidate_full_content_and_identity_match_required(self):
        for key, value in (("detector_score", .71), ("source_class_name", "other"),
                           ("class_name", "other"), ("score_interpretation", "identity probability")):
            cert = deepcopy(self.cert)
            cert["selected_candidate"][key] = value
            with self.assertRaisesRegex(SurfaceRejection, "CONTENT_MISMATCH"):
                self.estimate(selection_certificate=cert)
        with self.assertRaisesRegex(SurfaceRejection, "NO_UNIQUE_SELECTED"):
            self.estimate(selection_certificate=dict(self.cert, selected_candidate_id="absent"))

    def test_nonfinite_timing_and_payload_rejected(self):
        for key in ("semantic_expires_monotonic_s", "semantic_review_deadline_monotonic_s",
                    "semantic_review_remaining_s"):
            for value in (float("nan"), float("inf"), True):
                with self.subTest(key=key, value=value), self.assertRaises(SurfaceRejection):
                    self.estimate(selection_certificate=dict(self.cert, **{key: value}))
        with self.assertRaisesRegex(SurfaceRejection, "NONFINITE_OR_NONJSON"):
            self.estimate(selection_certificate=dict(self.cert, gap_age_s=float("nan")))
        with self.assertRaises(SurfaceRejection):
            self.estimate(now=float("nan"))

    def test_truncated_or_edge_touching_candidate_rejected(self):
        for chosen in (replace(self.frame.candidates[0], truncated=True),
                       Candidate("chosen", (0., 30., 60., 50.), .7, "balloon")):
            frame = replace(self.frame, candidates=(chosen, self.frame.candidates[1]))
            cert = deepcopy(self.cert)
            cert["selected_candidate"] = chosen.to_dict()
            with self.assertRaises(SurfaceRejection):
                self.estimate(frame, selection_certificate=cert)

    def test_depth_and_pairing_reject_misaligned_inputs(self):
        for updates in ({"frame_id": "other"}, {"image_sha256": "0"*64},
            {"depth_kind": "DepthPlanar"}, {"intrinsics_pairing_scope": None},
            {"depth_readback_timestamp_ns": 1_030_000_000}, {"rgb_readback_timestamp_ns": True}):
            with self.subTest(updates=updates), self.assertRaises(SurfaceRejection):
                self.estimate(pairing_contract=dict(self.pairing, **updates))
        with self.assertRaisesRegex(SurfaceRejection, "POSE_SKEW"):
            self.estimate(depth_pose=CameraPose([1.03, 2, -3], [0, 0, 0, 1], [10, 20, 30]))
        with self.assertRaisesRegex(SurfaceRejection, "TRANSLATION_MISMATCH"):
            self.estimate(depth_pose=CameraPose([1, 2, -3], [0, 0, 0, 1], [10, 20, 31]))

    def test_invalid_depth_fraction_and_dimensions_retained(self):
        for invalid in (None, np.ones((80, 99)), np.full((80, 100), np.nan),
                        np.zeros((80, 100)), np.full((80, 100), 1000.)):
            with self.assertRaises(SurfaceRejection):
                self.estimate(depth=invalid)

    def test_actual_depth_pixel_is_not_fabricated_median(self):
        depth = self.depth.copy()
        depth[35:45, 45:55] = np.where(np.indices((10, 10))[1] < 5, 4., 6.)
        result = self.estimate(depth=depth)
        x, y = result["depth_statistics"]["representative_pixel_xy"]
        self.assertEqual(result["depth_statistics"]["median_ray_range_m"], 5.)
        self.assertEqual(result["ray_range_m"], depth[y, x])
        self.assertIn(result["ray_range_m"], (4., 6.))
        self.assertLess(result["rgb_reference_reprojection_max_axis_error_px"], 1e-10)

    def test_replay_projection_excludes_phase_actor_and_evaluation_inputs(self):
        raw = {key: None for key in ("session_id", "target_generation", "frame_id", "frame_sequence",
            "description", "image_size_wh", "media", "scene_intrinsics", "world_from_local_translation_ned_m",
            "response_camera_poses", "capture_received_host_s", "observer_record")}
        raw.update(evaluation_only={"target_world_ned_m": [999, 999, 999]},
            phase_id="would_leak_recovery_answer", scene_acceptance={"status": "PASS"},
            camera_world_ned_m=[999, 999, 999])
        allowed = geometry_inputs(raw)
        for key in ("evaluation_only", "phase_id", "scene_acceptance", "camera_world_ned_m"):
            self.assertNotIn(key, allowed)

    def test_saved_frame_restoration_keeps_candidate_and_byte_binding(self):
        restored = restore_frame(self.frame.to_dict(), b"image")
        self.assertEqual(restored, self.frame)
        with self.assertRaisesRegex(ValueError, "bound digest"):
            restore_frame(self.frame.to_dict(), b"changed PNG")


if __name__ == "__main__":
    unittest.main()
