"""Independent delayed-proof gate checks; no API, simulator or truth inputs."""
from copy import deepcopy
from concurrent.futures import Future
import hashlib
import inspect
import json
import unittest
from unittest.mock import patch

import numpy as np

from t1_candidate_surface import SurfaceRejection
from t1_candidate_surface_v2 import temporal_candidate_surface_estimate as estimate_v2
from t1_candidate_surface_v3 import temporal_candidate_surface_estimate
from t1_delayed_semantic_bridge import (DelayedSemanticPolicy,
    DelayedChainIdentityTracker, chain_proof_sha256)
from t1_pose_geometry import CameraPose, SceneCameraModel
from t1_semantic_perception import Candidate, SemanticFrame, SemanticReply


DESCRIPTION = "designated striped balloon"


def make_frame(sequence, receipt):
    encoded = f"RGB-{sequence}".encode()
    return SemanticFrame("surface-session", 2, f"frame-{sequence}", sequence,
        hashlib.sha256(encoded).hexdigest(), 100, 80, receipt, receipt + .01,
        (Candidate(f"chosen-{sequence}", (40., 30., 60., 50.), .7, "balloon"),
         Candidate(f"higher-score-{sequence}", (70., 30., 90., 50.), .95, "balloon")),
        encoded)


class ManualExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, operation, request):
        future = Future()
        future.set_running_or_notify_cancel()
        self.calls.append((future, request))
        return future


def confirmed_selection(post_adoption_frames=2):
    """Use production policy and tracker, a controlled future, and actual frames."""
    executor = ManualExecutor()
    policy = DelayedSemanticPolicy(lambda request: None, session_id="surface-session",
        target_generation=2, description=DESCRIPTION, executor=executor, clock=lambda: 13.)
    tracker = DelayedChainIdentityTracker("surface-session", 2, DESCRIPTION)
    source = make_frame(0, 10.)
    try:
        submitted = policy.request(source, now=10.02)
        if submitted["status"] != "SUBMITTED":
            raise AssertionError(submitted)
        for sequence in range(1, 11):
            current = make_frame(sequence, 10. + .2 * sequence)
            policy.poll(current, now=current.readback_receipt_monotonic_s + .02)
            tracker.update(current, now=current.readback_receipt_monotonic_s + .03)
        future, request = executor.calls[0]
        future.set_result(SemanticReply(request.binding(), 12.18,
            {"answer": "synthetic accepted source box", "result_bbox_list": [
                dict(zip(("xmin", "ymin", "xmax", "ymax"), (40., 30., 60., 50.)))]}))
        current = make_frame(11, 12.2)
        accepted = policy.poll(current, now=12.22)
        if accepted["status"] != "SEMANTIC_LOCAL_ASSOCIATION_VALID":
            raise AssertionError(accepted)
        certificate = tracker.update(current, semantic_decision=accepted, now=12.23)
        for offset in range(1, post_adoption_frames + 1):
            current = make_frame(11 + offset, 12.2 + .2 * offset)
            certificate = tracker.update(current, now=current.readback_receipt_monotonic_s + .03)
        return current, certificate, current.readback_receipt_monotonic_s + .03
    finally:
        policy.close(wait=False)
        tracker.close()


class DelayedSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.cert, self.now = confirmed_selection()
        self.camera = SceneCameraModel(100, 80, 50., 50., 50., 40.)
        self.pose = CameraPose([1, 2, -3], [0, 0, 0, 1], [10, 20, 30])
        self.depth = np.full((80, 100), 5.)
        self.pairing = {"frame_id": self.frame.frame_id,
            "image_sha256": self.frame.image_sha256, "depth_kind": "DepthPerspective",
            "rgb_readback_timestamp_ns": 1_000_000_000,
            "depth_readback_timestamp_ns": 1_005_000_000,
            "intrinsics_pairing_scope": "caller_declared_equal_RGB_depth_Scene_intrinsics"}

    def estimate(self, **overrides):
        args = dict(description=DESCRIPTION, now=self.now, depth=self.depth,
            camera=self.camera, rgb_pose=self.pose, depth_pose=self.pose,
            selection_certificate=self.cert, pairing_contract=self.pairing)
        args.update(overrides)
        return temporal_candidate_surface_estimate(self.frame, **args)

    def test_missing_delayed_proof_fails_closed(self):
        certificate = deepcopy(self.cert)
        certificate.pop("delayed_chain_proof")
        with self.assertRaises(SurfaceRejection) as caught:
            self.estimate(selection_certificate=certificate)
        self.assertEqual(caught.exception.reason, "DELAYED_CHAIN_SELECTION_PROOF_REJECTED")

    def test_configuration_cannot_relax_current_or_original_deadlines(self):
        for key, value in (("current_frame_max_age_s", .6),
                           ("semantic_source_max_age_s", 2.),
                           ("semantic_source_max_age_s", 9.),
                           ("max_unreviewed_track_s", 9.)):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(
                    SurfaceRejection, "DELAYED_CHAIN_FRESHNESS_CONFIG_MISMATCH"):
                self.estimate(**{key: value})

    def test_nonfinite_or_bool_configuration_rejected(self):
        for key in ("now", "current_frame_max_age_s", "semantic_source_max_age_s",
                    "max_unreviewed_track_s"):
            for value in (True, float("nan"), float("inf"), -1.):
                with self.subTest(key=key, value=value), self.assertRaises(SurfaceRejection):
                    self.estimate(**{key: value})

    def test_verifier_error_is_a_closed_surface_reason(self):
        with patch("t1_candidate_surface_v3.verify_delayed_selection",
                   side_effect=ValueError("raw response text must not escape")):
            with self.assertRaises(SurfaceRejection) as caught:
                self.estimate()
        self.assertEqual(str(caught.exception), "DELAYED_CHAIN_SELECTION_PROOF_REJECTED")
        self.assertIsNone(caught.exception.__cause__)

    def test_surface_interface_accepts_no_target_truth_or_segmentation(self):
        fields = inspect.signature(temporal_candidate_surface_estimate).parameters
        self.assertEqual(set(fields), {"frame", "description", "now", "depth", "camera",
            "rgb_pose", "depth_pose", "selection_certificate", "pairing_contract",
            "config", "current_frame_max_age_s", "semantic_source_max_age_s",
            "max_unreviewed_track_s"})
        self.assertEqual(inspect.signature(estimate_v2).parameters[
            "semantic_source_max_age_s"].default, 2.)

    def test_real_delayed_chain_geometry_preserves_lower_score_and_no_motion_permission(self):
        self.assertEqual(self.cert["status"], "TEMPORAL_LOCAL_ASSOCIATION_VALID")
        result = self.estimate()
        self.assertEqual(result["selected_candidate_id"], self.frame.candidates[0].candidate_id)
        self.assertGreater(self.frame.candidates[1].detector_score,
            self.frame.candidates[0].detector_score)
        self.assertTrue(result["delayed_chain_proof_independently_checked"])
        self.assertEqual(result["delayed_chain_verification"]["post_adoption_confirmation_count"], 2)
        for key in ("flight_authorized", "online_control_enabled", "execution_guard_checked",
                    "identity_probability_calibrated", "instance_identity_guaranteed"):
            self.assertFalse(result[key])
        np.testing.assert_allclose(result["surface_position_world_ned_m"], [16., 22., 27.])

    def test_full_certificate_hash_includes_delayed_proof(self):
        before = deepcopy(self.cert)
        result = self.estimate()
        expected = hashlib.sha256(json.dumps(self.cert, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        stripped = deepcopy(self.cert)
        stripped.pop("delayed_chain_proof")
        stripped_hash = hashlib.sha256(json.dumps(stripped, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.assertEqual(result["selection_certificate_sha256"], expected)
        self.assertNotEqual(expected, stripped_hash)
        self.assertEqual(self.cert, before)

    def test_valid_geometry_equals_v2_explicit_eight_second_contract(self):
        result = self.estimate()
        previous = estimate_v2(self.frame, description=DESCRIPTION, now=self.now,
            depth=self.depth, camera=self.camera, rgb_pose=self.pose, depth_pose=self.pose,
            selection_certificate=self.cert, pairing_contract=self.pairing,
            semantic_source_max_age_s=8., max_unreviewed_track_s=8.)
        for key in ("surface_position_world_ned_m", "rgb_reference_world_ned_m",
                    "bbox_plane_corners_world_ned_m", "depth_statistics",
                    "selection_certificate_sha256", "current_binding"):
            self.assertEqual(result[key], previous[key])

    def test_original_source_and_reply_hashes_are_checked(self):
        for field in ("request_sha256", "semantic_reply_sha256"):
            certificate = deepcopy(self.cert)
            certificate["delayed_chain_proof"][field] = "0" * 64
            certificate["delayed_chain_proof_sha256"] = chain_proof_sha256(
                certificate["delayed_chain_proof"])
            with self.subTest(field=field), self.assertRaises(SurfaceRejection):
                self.estimate(selection_certificate=certificate)

    def test_missing_and_changed_proof_hash_fail_closed(self):
        for field in ("delayed_chain_proof", "delayed_chain_proof_sha256"):
            certificate = deepcopy(self.cert)
            certificate.pop(field)
            with self.subTest(field=field), self.assertRaises(SurfaceRejection):
                self.estimate(selection_certificate=certificate)
        certificate = deepcopy(self.cert)
        certificate["delayed_chain_proof"]["reply_completed_monotonic_s"] += .001
        with self.assertRaises(SurfaceRejection):
            self.estimate(selection_certificate=certificate)

    def test_zero_or_one_post_adoption_frame_cannot_authorize_geometry(self):
        for count in (0, 1):
            frame, certificate, now = confirmed_selection(count)
            self.assertEqual(certificate["state"], "RECOVERY_PENDING")
            self.assertEqual(len(certificate["delayed_chain_proof"]["post_adoption_frame_ids"]), count)
            pairing = dict(self.pairing, frame_id=frame.frame_id, image_sha256=frame.image_sha256)
            with self.subTest(count=count), self.assertRaises(SurfaceRejection):
                temporal_candidate_surface_estimate(frame, description=DESCRIPTION, now=now,
                    depth=self.depth, camera=self.camera, rgb_pose=self.pose, depth_pose=self.pose,
                    selection_certificate=certificate, pairing_contract=pairing)

    def test_forged_tracking_count_does_not_replace_newer_frames(self):
        frame, certificate, now = confirmed_selection(1)
        certificate.update(status="TEMPORAL_LOCAL_ASSOCIATION_VALID", state="TRACKING",
            confirmation_count=3, confirmation_required=3)
        pairing = dict(self.pairing, frame_id=frame.frame_id, image_sha256=frame.image_sha256)
        with self.assertRaises(SurfaceRejection):
            temporal_candidate_surface_estimate(frame, description=DESCRIPTION, now=now,
                depth=self.depth, camera=self.camera, rgb_pose=self.pose, depth_pose=self.pose,
                selection_certificate=certificate, pairing_contract=pairing)

    def test_adoption_frame_is_not_a_post_adoption_confirmation(self):
        certificate = deepcopy(self.cert)
        proof = certificate["delayed_chain_proof"]
        proof["post_adoption_frame_ids"].insert(0, proof["adoption"]["accepted_frame_id"])
        certificate["delayed_chain_proof_sha256"] = chain_proof_sha256(proof)
        with self.assertRaises(SurfaceRejection):
            self.estimate(selection_certificate=certificate)

    def test_deadlines_cannot_be_rebased_to_reply_or_adoption(self):
        source_receipt = self.cert["delayed_chain_proof"]["request_binding"][
            "readback_receipt_monotonic_s"]
        self.assertEqual(self.cert["semantic_review_deadline_monotonic_s"], source_receipt + 8.)
        self.assertEqual(self.cert["semantic_expires_monotonic_s"], source_receipt + 8.)
        self.assertGreater(self.now - source_receipt, 2.)
        for origin in (self.cert["delayed_chain_proof"]["reply_completed_monotonic_s"],
                       self.cert["delayed_chain_proof"]["adoption"]["accepted_monotonic_s"]):
            certificate = dict(self.cert, semantic_expires_monotonic_s=origin + 8.,
                semantic_review_deadline_monotonic_s=origin + 8.)
            with self.subTest(origin=origin), self.assertRaises(SurfaceRejection):
                self.estimate(selection_certificate=certificate)

    def test_expired_source_and_stale_current_frame_are_rejected(self):
        for now in (self.frame.readback_receipt_monotonic_s + .500001, 18.000001):
            with self.subTest(now=now), self.assertRaises(SurfaceRejection):
                self.estimate(now=now)

    def test_current_candidate_content_and_rgb_hash_are_exactly_bound(self):
        for key, value in (("current_image_sha256", "0" * 64),
                           ("selected_candidate_id", self.frame.candidates[1].candidate_id)):
            with self.subTest(key=key), self.assertRaises(SurfaceRejection):
                self.estimate(selection_certificate=dict(self.cert, **{key: value}))
        certificate = deepcopy(self.cert)
        certificate["selected_candidate"]["detector_score"] += .01
        with self.assertRaises(SurfaceRejection):
            self.estimate(selection_certificate=certificate)

    def test_node_metadata_and_iou_audits_recomputed_after_new_proof_hash(self):
        changes = ("frame_sha256", "selected_candidate_sha256", "association_audit")
        for field in changes:
            certificate = deepcopy(self.cert)
            node = certificate["delayed_chain_proof"]["nodes"][3]
            if field == "association_audit":
                node[field][0]["iou"] = .999
            else:
                node[field] = "0" * 64
            certificate["delayed_chain_proof_sha256"] = chain_proof_sha256(
                certificate["delayed_chain_proof"])
            with self.subTest(field=field), self.assertRaises(SurfaceRejection):
                self.estimate(selection_certificate=certificate)

    def test_missing_chain_node_and_ambiguous_competitor_fail_closed(self):
        certificate = deepcopy(self.cert)
        del certificate["delayed_chain_proof"]["nodes"][4]
        certificate["delayed_chain_proof_sha256"] = chain_proof_sha256(
            certificate["delayed_chain_proof"])
        with self.assertRaises(SurfaceRejection):
            self.estimate(selection_certificate=certificate)
        certificate = deepcopy(self.cert)
        node = certificate["delayed_chain_proof"]["nodes"][4]
        node["frame"]["candidates"][1]["bbox_xyxy_pixels"] = [41., 30., 61., 50.]
        node["frame_sha256"] = chain_proof_sha256(node["frame"])
        certificate["delayed_chain_proof_sha256"] = chain_proof_sha256(
            certificate["delayed_chain_proof"])
        with self.assertRaises(SurfaceRejection):
            self.estimate(selection_certificate=certificate)

    def test_unknown_truth_fields_in_proof_are_rejected(self):
        certificate = deepcopy(self.cert)
        certificate["delayed_chain_proof"]["target_actor_truth"] = [999., 999., 999.]
        certificate["delayed_chain_proof_sha256"] = chain_proof_sha256(
            certificate["delayed_chain_proof"])
        with self.assertRaises(SurfaceRejection):
            self.estimate(selection_certificate=certificate)

    def test_confirmation_counter_cannot_be_bool_or_inflated(self):
        for field in ("confirmation_count", "confirmation_required"):
            for value in (True, 2, 100):
                with self.subTest(field=field, value=value), self.assertRaises(SurfaceRejection):
                    self.estimate(selection_certificate=dict(self.cert, **{field: value}))


if __name__ == "__main__":
    unittest.main()
