import hashlib
import math
import unittest
from copy import deepcopy
from dataclasses import replace

import numpy as np

from t1_candidate_surface import (SurfaceConfig, SurfaceRejection, candidate_surface_estimate,
                                 detector_only_selection_certificate)
from t1_pose_geometry import CameraPose, SceneCameraModel, project_world
from t1_semantic_perception import Candidate, SemanticFrame


class CandidateSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.frame = SemanticFrame("session", 2, "current-1", 4, hashlib.sha256(b"image").hexdigest(),
            100, 80, 10., 10.1, (Candidate("a", (40., 30., 60., 50.), .7),))
        self.camera = SceneCameraModel(100, 80, 50., 50., 50., 40.)
        self.pose = CameraPose([1, 2, -3], [0, 0, 0, 1], [10, 20, 30])
        self.depth = np.full((80, 100), 5.)
        self.pairing = {"frame_id": self.frame.frame_id, "image_sha256": self.frame.image_sha256,
            "depth_kind": "DepthPerspective", "rgb_readback_timestamp_ns": 1_000_000_000,
            "depth_readback_timestamp_ns": 1_005_000_000,
            "intrinsics_pairing_scope": "caller_declared_equal_RGB_depth_Scene_intrinsics"}
        self.certificate = detector_only_selection_certificate(self.frame)

    def estimate(self, **kwargs):
        arguments = dict(depth=self.depth, camera=self.camera, rgb_pose=self.pose, depth_pose=self.pose,
            selection_certificate=self.certificate, pairing_contract=self.pairing)
        arguments.update(kwargs)
        return candidate_surface_estimate(self.frame, **arguments)

    def test_detector_only_is_bound_current_surface_and_not_control_evidence(self):
        result = self.estimate()
        np.testing.assert_allclose(result["surface_position_world_ned_m"], [16, 22, 27])
        self.assertEqual(result["depth_statistics"]["representative_pixel_xy"], [50, 40])
        self.assertEqual(result["selection_status"], "DETECTOR_ONLY_UNCALIBRATED")
        for key in ("flight_authorized", "online_control_enabled", "freshness_checked", "execution_guard_checked",
                    "identity_probability_calibrated", "instance_identity_guaranteed"):
            self.assertFalse(result[key])
        self.assertFalse(result["pairing_metadata_checks"]["synchronization_verified"])

    def test_semantic_certificate_requires_same_current_frame_and_box(self):
        cert = dict(self.certificate, status="SEMANTIC_LOCAL_ASSOCIATION_VALID",
                    request_mode="image_text", proposals_transmitted=False, server_proposal_ranking=False)
        self.assertTrue(self.estimate(selection_certificate=cert)["semantic_local_association_certificate_accepted"])
        for key, value in (("current_frame_id", "old"), ("current_image_sha256", "0"*64),
                           ("target_generation", 1), ("current_frame_sequence", 3), ("session_id", "other")):
            with self.assertRaisesRegex(SurfaceRejection, "BINDING_MISMATCH"):
                self.estimate(selection_certificate=dict(cert, **{key: value}))
        bad = deepcopy(cert)
        bad["selected_candidate"]["bbox_xyxy_pixels"][0] += 1
        with self.assertRaisesRegex(SurfaceRejection, "CONTENT_MISMATCH"):
            self.estimate(selection_certificate=bad)

    def test_uncertain_no_selection_or_excess_identity_claim_rejected(self):
        for updates in ({"status": "UNCERTAIN"}, {"selected_candidate_id": "missing"},
                        {"identity_probability_calibrated": True}, {"online_control_enabled": True}):
            with self.assertRaises(SurfaceRejection):
                self.estimate(selection_certificate=dict(self.certificate, **updates))

    def test_no_candidate_and_tied_highest_score_are_not_unique(self):
        with self.assertRaisesRegex(SurfaceRejection, "NO_YOLO_CANDIDATE"):
            detector_only_selection_certificate(replace(self.frame, candidates=()))
        with self.assertRaisesRegex(SurfaceRejection, "AMBIGUOUS"):
            detector_only_selection_certificate(replace(self.frame, candidates=(self.frame.candidates[0],
                Candidate("b", (70., 30., 90., 50.), .7))))

    def test_truncation_flag_and_unmarked_edge_touch_are_rejected(self):
        for candidate in (replace(self.frame.candidates[0], truncated=True),
                          Candidate("a", (0., 30., 60., 50.), .7)):
            current = replace(self.frame, candidates=(candidate,))
            with self.assertRaises(SurfaceRejection):
                candidate_surface_estimate(current, depth=self.depth, camera=self.camera,
                    rgb_pose=self.pose, depth_pose=self.pose,
                    selection_certificate=detector_only_selection_certificate(current), pairing_contract=self.pairing)

    def test_geometry_hash_and_frame_are_checked_independently(self):
        for updates in ({"frame_id": "other"}, {"image_sha256": "0"*64}):
            with self.assertRaisesRegex(SurfaceRejection, "GEOMETRY_CURRENT_FRAME_BINDING_MISMATCH"):
                self.estimate(pairing_contract=dict(self.pairing, **updates))

    def test_missing_depth_and_wrong_resolution_are_rejected(self):
        for depth in (None, np.ones((80, 99)), np.ones((80, 100, 1))):
            with self.assertRaisesRegex(SurfaceRejection, "DEPTH_DIMENSIONS"):
                self.estimate(depth=depth)

    def test_invalid_depth_fraction_nan_inf_zero_and_range_limits(self):
        depth = self.depth.copy()
        depth[35:45, 45:55] = np.nan
        depth[35:37, 45:55] = 5.
        with self.assertRaisesRegex(SurfaceRejection, "INSUFFICIENT"):
            self.estimate(depth=depth)
        for invalid in (np.inf, 0, -.5, .05, 1000.):
            depth[35:45, 45:55] = invalid
            with self.assertRaisesRegex(SurfaceRejection, "INSUFFICIENT"):
                self.estimate(depth=depth)

    def test_median_does_not_fabricate_depth_at_another_pixel(self):
        depth = self.depth.copy()
        depth[35:45, 45:55] = np.where(np.indices((10, 10))[1] < 5, 4., 6.)
        result = self.estimate(depth=depth)
        x, y = result["depth_statistics"]["representative_pixel_xy"]
        self.assertEqual(result["depth_statistics"]["median_ray_range_m"], 5.)
        self.assertEqual(result["ray_range_m"], depth[y, x])
        self.assertIn(result["ray_range_m"], (4., 6.))

    def test_range_is_not_axial_depth_for_off_axis_candidate(self):
        current = replace(self.frame, candidates=(Candidate("a", (70., 30., 90., 50.), .7),))
        result = candidate_surface_estimate(current, depth=self.depth, camera=self.camera,
            rgb_pose=self.pose, depth_pose=self.pose,
            selection_certificate=detector_only_selection_certificate(current), pairing_contract=self.pairing)
        self.assertLess(result["rgb_reference_axial_depth_m"], result["ray_range_m"])
        projected = project_world(result["surface_position_world_ned_m"], self.pose, self.camera)
        self.assertAlmostEqual(projected["ray_range_m"], 5.)
        np.testing.assert_allclose(projected["pixel_xy"], [80., 40.])

    def test_nontrivial_camera_quaternion_and_translation_roundtrip(self):
        # Mixed rotation quaternion derived independently from Euler(.2,-.3,.5).
        r, p, y = .2, -.3, .5
        cr, cp, cy = (math.cos(a/2) for a in (r, p, y))
        sr, sp, sy = (math.sin(a/2) for a in (r, p, y))
        q = [sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
             cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy]
        pose = CameraPose([1, 2, -3], q, [10, 20, 30])
        result = self.estimate(rgb_pose=pose, depth_pose=pose)
        np.testing.assert_allclose(pose.world_to_camera(result["surface_position_world_ned_m"]), [5, 0, 0], atol=1e-12)
        self.assertLess(result["depth_pixel_reprojection_max_axis_error_px"], 1e-10)
        self.assertLess(result["rgb_reference_reprojection_max_axis_error_px"], 1e-10)

    def test_pose_and_readback_skew_gates_do_not_claim_exact_synchronization(self):
        with self.assertRaisesRegex(SurfaceRejection, "READBACK_SKEW"):
            self.estimate(pairing_contract=dict(self.pairing, depth_readback_timestamp_ns=1_030_000_000))
        with self.assertRaisesRegex(SurfaceRejection, "POSE_SKEW"):
            self.estimate(depth_pose=CameraPose([1.03, 2, -3], [0, 0, 0, 1], [10, 20, 30]))
        yaw = math.radians(2)
        with self.assertRaisesRegex(SurfaceRejection, "POSE_SKEW"):
            self.estimate(depth_pose=CameraPose([1, 2, -3], [0, 0, math.sin(yaw/2), math.cos(yaw/2)], [10, 20, 30]))
        with self.assertRaisesRegex(SurfaceRejection, "TRANSLATION_MISMATCH"):
            self.estimate(depth_pose=CameraPose([1, 2, -3], [0, 0, 0, 1], [10, 20, 31]))

    def test_missing_pairing_and_axial_type_must_not_silently_pass(self):
        for updates in ({"depth_kind": "DepthPlanar"}, {"intrinsics_pairing_scope": None},
                        {"rgb_readback_timestamp_ns": 0}, {"depth_readback_timestamp_ns": True}):
            with self.assertRaises(SurfaceRejection):
                self.estimate(pairing_contract=dict(self.pairing, **updates))
        with self.assertRaises(SurfaceRejection):
            self.estimate(pairing_contract=None)

    def test_invalid_config_and_scene_size(self):
        for args in ({"min_valid_depth_fraction": 1.1}, {"min_valid_depth_fraction": 0},
                     {"min_ray_range_m": 1000}, {"max_camera_pose_rotation_deg": float("nan")}):
            with self.assertRaises(SurfaceRejection):
                SurfaceConfig(**args)
        with self.assertRaisesRegex(SurfaceRejection, "INTRINSICS_SIZE"):
            self.estimate(camera=SceneCameraModel(200, 80, 50, 50, 100, 40))


if __name__ == "__main__":
    unittest.main()
