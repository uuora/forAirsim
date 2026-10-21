"""Controlled future/tracker tests for background semantic confirmation."""
from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np
import test_t1_candidate_surface_v3 as fixtures
from t1_candidate_surface_v3 import temporal_candidate_surface_estimate
from t1_delayed_semantic_bridge import (DelayedSemanticPolicy, DelayedBridgeConfig,
    chain_proof_sha256, verify_delayed_selection)
from t1_pose_geometry import CameraPose, SceneCameraModel
from t1_review_handoff import ReviewHandoffIdentityTracker, HANDOFF_EVENTS, HANDOFF_STATUSES
from t1_semantic_perception import SemanticReply, Candidate


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.executor = fixtures.ManualExecutor()
        self.policy = DelayedSemanticPolicy(lambda request: None,
            session_id="surface-session", target_generation=2,
            description=fixtures.DESCRIPTION, executor=self.executor,
            clock=lambda: 100., config=DelayedBridgeConfig(min_request_interval_s=.01))
        self.tracker = ReviewHandoffIdentityTracker("surface-session", 2, fixtures.DESCRIPTION)
        self.results = []

    def tearDown(self):
        self.policy.close()
        self.tracker.close()

    def frame(self, sequence):
        return fixtures.make_frame(sequence, 10. + .2 * sequence)

    def submit(self, frame):
        self.assertEqual(self.policy.request(frame,
            now=frame.readback_receipt_monotonic_s + .04)["status"], "SUBMITTED")

    def complete(self, *, box=(40., 30., 60., 50.)):
        future, request = self.executor.calls[-1]
        future.set_result(SemanticReply(request.binding(), request.submitted_monotonic_s + .10,
            {"result_bbox_list": [dict(zip(("xmin", "ymin", "xmax", "ymax"), box))]}))

    def step(self, sequence, *, frame=None, override="POLICY"):
        frame = frame or self.frame(sequence)
        semantic = self.policy.poll(frame, now=frame.readback_receipt_monotonic_s + .02)
        if override != "POLICY":
            semantic = override
        allowed = semantic if isinstance(semantic, dict) and semantic.get(
            "status") == "SEMANTIC_LOCAL_ASSOCIATION_VALID" else None
        result = self.tracker.update(frame, allowed, now=frame.readback_receipt_monotonic_s + .03)
        self.results.append(result)
        meta = result["review_handoff"]
        self.assertEqual(set(meta), {"event", "status", "active_request_id",
            "pending_request_id", "confirmation_count", "reason"})
        self.assertIn(meta["event"], HANDOFF_EVENTS)
        self.assertIn(meta["status"], HANDOFF_STATUSES)
        if result["status"] == "TEMPORAL_LOCAL_ASSOCIATION_VALID":
            verify_delayed_selection(frame, result, fixtures.DESCRIPTION,
                frame.readback_receipt_monotonic_s + .03)
        return result

    def initial(self):
        self.step(0)
        self.submit(self.frame(0)); self.complete()
        self.assertEqual(self.step(1)["review_handoff"]["event"], "INITIAL_ADOPTION")
        self.assertEqual(self.step(2)["status"], "NO_TARGET_OUTPUT")
        result = self.step(3)
        self.assertEqual(result["review_handoff"]["event"], "INITIAL_CONFIRMED")
        return result

    def advance(self, first, last):
        return [self.step(sequence) for sequence in range(first, last + 1)]

    def review(self, source_sequence, *, box=(40., 30., 60., 50.)):
        self.submit(self.frame(source_sequence)); self.complete(box=box)
        return self.step(source_sequence + 1)

    def test_initial_confirmation_requires_three_distinct_actual_frames(self):
        result = self.initial()
        self.assertEqual(result["confirmation_count"], 3)
        self.assertEqual(result["delayed_chain_proof"]["post_adoption_frame_ids"],
            ["frame-2", "frame-3"])

    def test_same_target_background_review_preserves_output_then_atomic_swap(self):
        first = self.initial()
        self.advance(4, 15)
        pending = self.review(15)
        middle = self.step(17)
        committed = self.step(18)
        for result in (pending, middle, committed):
            self.assertEqual(result["status"], "TEMPORAL_LOCAL_ASSOCIATION_VALID")
        self.assertEqual(pending["semantic_anchor"], first["semantic_anchor"])
        self.assertEqual(middle["semantic_anchor"], first["semantic_anchor"])
        self.assertNotEqual(committed["semantic_anchor"], first["semantic_anchor"])
        self.assertNotEqual(committed["track_id"], first["track_id"])
        self.assertEqual(pending["review_handoff"]["confirmation_count"], 1)
        self.assertEqual(middle["review_handoff"]["confirmation_count"], 2)
        self.assertEqual(committed["review_handoff"]["event"], "REVIEW_COMMITTED")
        self.assertEqual(committed["delayed_chain_proof"]["post_adoption_frame_ids"],
            ["frame-17", "frame-18"])
        self.assertEqual(pending["semantic_review_deadline_monotonic_s"], 18.)
        self.assertEqual(committed["semantic_review_deadline_monotonic_s"], 21.)
        self.assertEqual(committed["delayed_chain_proof"]["adoption"][
            "accepted_frame_id"], "frame-16")

    def test_same_target_chain_proof_is_accepted_by_existing_surface_v3(self):
        self.initial(); self.advance(4, 15); self.review(15); self.step(17)
        certificate = self.step(18); frame = self.frame(18)
        surface = temporal_candidate_surface_estimate(frame, description=fixtures.DESCRIPTION,
            now=frame.readback_receipt_monotonic_s + .03, depth=np.full((80, 100), 5.),
            camera=SceneCameraModel(100, 80, 50., 50., 50., 40.),
            rgb_pose=CameraPose([1, 2, -3], [0, 0, 0, 1], [10, 20, 30]),
            depth_pose=CameraPose([1, 2, -3], [0, 0, 0, 1], [10, 20, 30]),
            selection_certificate=certificate, pairing_contract={"frame_id": frame.frame_id,
                "image_sha256": frame.image_sha256, "depth_kind": "DepthPerspective",
                "rgb_readback_timestamp_ns": 1_000_000_000,
                "depth_readback_timestamp_ns": 1_005_000_000,
                "intrinsics_pairing_scope": "caller_declared_equal_RGB_depth_Scene_intrinsics"})
        self.assertEqual(surface["selection_certificate_sha256"], chain_proof_sha256(certificate))
        self.assertTrue(surface["delayed_chain_proof_independently_checked"])
        self.assertFalse(surface["flight_authorized"])

    def test_old_expiry_during_pending_confirmation_clears_both_chains(self):
        self.initial(); self.advance(4, 38)
        pending = self.review(38)
        self.assertEqual(pending["status"], "TEMPORAL_LOCAL_ASSOCIATION_VALID")
        expired = self.step(40)
        self.assertEqual(expired["review_handoff"]["event"], "CHAIN_LOST")
        self.assertEqual(expired["status"], "NO_TARGET_OUTPUT")
        self.assertIsNone(expired["review_handoff"]["pending_request_id"])
        self.assertEqual(self.step(41)["status"], "NO_TARGET_OUTPUT")

    def test_expiry_cannot_be_refreshed_by_a_review_accepted_on_expired_frame(self):
        self.initial(); self.advance(4, 39)
        self.submit(self.frame(39)); self.complete()
        result = self.step(40)
        self.assertEqual(result["status"], "NO_TARGET_OUTPUT")
        self.assertEqual(result["review_handoff"]["event"], "CHAIN_LOST")

    def test_conflicting_semantic_review_clears_active_output_immediately(self):
        self.initial(); self.advance(4, 15)
        result = self.review(15, box=(70., 30., 90., 50.))
        self.assertEqual(result["state"], "AMBIGUOUS")
        self.assertEqual(result["review_handoff"]["event"], "REVIEW_CONFLICT")
        self.assertIsNone(result["selected_candidate_id"])
        self.assertEqual(self.step(17)["status"], "NO_TARGET_OUTPUT")

    def test_sequence_gap_drops_pending_and_never_reuses_it(self):
        self.initial(); self.advance(4, 15); self.review(15)
        result = self.step(18)
        self.assertEqual(result["review_handoff"]["event"], "CHAIN_LOST")
        self.assertEqual(result["reason"], "HANDOFF_FRAME_CHAIN_GAP")
        self.assertEqual(self.step(19)["status"], "NO_TARGET_OUTPUT")

    def test_missing_candidate_drops_both_and_cannot_resume_old_evidence(self):
        self.initial(); self.advance(4, 15); self.review(15)
        result = self.step(17, frame=replace(self.frame(17), candidates=()))
        self.assertEqual(result["status"], "NO_TARGET_OUTPUT")
        self.assertEqual(self.step(18)["status"], "NO_TARGET_OUTPUT")

    def test_context_change_drops_both_without_cross_context_output(self):
        self.initial(); self.advance(4, 15); self.review(15)
        result = self.step(17, frame=replace(self.frame(17), target_generation=3))
        self.assertEqual(result["reason"], "FRAME_CONTEXT_MISMATCH")
        self.assertIsNone(result["review_handoff"]["active_request_id"])

    def test_duplicate_frame_does_not_count_as_confirmation(self):
        self.initial(); self.advance(4, 15); pending = self.review(15)
        duplicate = self.tracker.update(self.frame(16), now=13.24)
        self.assertEqual(duplicate["review_handoff"]["event"], "DUPLICATE_FRAME")
        self.assertEqual(duplicate["review_handoff"]["confirmation_count"], 1)
        self.assertEqual(duplicate["status"], "NO_TARGET_OUTPUT")
        self.assertEqual(self.step(17)["review_handoff"]["confirmation_count"], 2)
        self.assertEqual(self.step(18)["review_handoff"]["event"], "REVIEW_COMMITTED")
        self.assertEqual(pending["semantic_anchor"], "chain-semantic-000001")

    def test_pending_replaced_by_fresh_review_restarts_its_own_confirmations(self):
        self.initial(); self.advance(4, 15); first = self.review(15)
        self.submit(self.frame(16)); self.complete()
        replacement = self.step(17)
        self.assertEqual(replacement["review_handoff"]["event"], "REVIEW_REPLACED")
        self.assertNotEqual(first["review_handoff"]["pending_request_id"],
            replacement["review_handoff"]["pending_request_id"])
        self.assertEqual(replacement["review_handoff"]["confirmation_count"], 1)
        self.assertEqual(self.step(18)["semantic_anchor"], "chain-semantic-000001")
        self.assertEqual(self.step(19)["semantic_anchor"], "chain-semantic-000003")

    def test_explicit_backend_rejection_cancels_pending_but_keeps_old_deadline(self):
        self.initial(); self.advance(4, 15); self.review(15)
        frame = self.frame(17)
        result = self.tracker.update(frame, {"status": "REJECTED", "reason": "SOURCE_ATTRIBUTE_REJECTED"},
            now=frame.readback_receipt_monotonic_s + .03)
        self.assertEqual(result["status"], "TEMPORAL_LOCAL_ASSOCIATION_VALID")
        self.assertEqual(result["semantic_anchor"], "chain-semantic-000001")
        self.assertEqual(result["semantic_review_deadline_monotonic_s"], 18.)
        self.assertEqual(result["review_handoff"]["event"], "REVIEW_REJECTED")
        self.assertIsNone(result["review_handoff"]["pending_request_id"])

    def test_no_valid_result_does_not_extend_original_deadline(self):
        self.initial(); self.advance(4, 39)
        result = self.step(40, override=None)
        self.assertEqual(result["status"], "NO_TARGET_OUTPUT")
        self.assertIsNone(result["semantic_anchor"])

    def test_genuine_fresh_request_after_loss_requires_new_confirmations(self):
        self.initial(); self.advance(4, 15)
        self.step(16, frame=replace(self.frame(16), candidates=()))
        self.step(17); self.submit(self.frame(17)); self.complete()
        adopted = self.step(18)
        self.assertEqual(adopted["state"], "RECOVERY_PENDING")
        self.assertEqual(self.step(19)["status"], "NO_TARGET_OUTPUT")
        recovered = self.step(20)
        self.assertEqual(recovered["status"], "TEMPORAL_LOCAL_ASSOCIATION_VALID")
        self.assertEqual(recovered["semantic_anchor"], "chain-semantic-000002")

    def test_inflight_request_from_before_loss_cannot_reanchor_after_loss(self):
        self.initial(); self.advance(4, 15); self.submit(self.frame(15))
        # The controlled accepted proof itself is geometrically intact, but
        # the tracker lost output due to an explicit clock error in between.
        self.tracker.update(self.frame(15), now=float("nan"))
        self.complete(); result = self.step(16)
        self.assertEqual(result["status"], "NO_TARGET_OUTPUT")
        self.assertEqual(result["reason"], "HANDOFF_FRESH_SOURCE_AFTER_LOSS_REQUIRED")

    def test_invalid_time_loss_allows_genuinely_new_source_recovery(self):
        self.initial(); self.advance(4, 15)
        self.tracker.update(self.frame(15), now=float("nan"))
        self.step(16, override=None); self.submit(self.frame(16)); self.complete()
        self.assertEqual(self.step(17)["state"], "RECOVERY_PENDING")
        self.step(18)
        self.assertEqual(self.step(19)["status"], "TEMPORAL_LOCAL_ASSOCIATION_VALID")

    def test_adoption_or_source_replay_tamper_is_rejected_before_output(self):
        self.initial(); self.advance(4, 15); self.review(15)
        frame = self.frame(17)
        accepted = self.policy.poll(frame, now=13.42)
        proof = accepted["delayed_chain_proof"]
        proof["adoption"]["accepted_monotonic_s"] += .001
        accepted["delayed_chain_proof_sha256"] = chain_proof_sha256(proof)
        result = self.tracker.update(frame, accepted, now=13.43)
        self.assertEqual(result["status"], "NO_TARGET_OUTPUT")
        self.assertEqual(result["reason"], "HANDOFF_EXISTING_REQUEST_PROOF_REBOUND")

    def test_selection_proof_tamper_fails_independent_verifier(self):
        self.initial(); self.advance(4, 15); self.review(15); self.step(17)
        cert = self.step(18); forged = deepcopy(cert)
        forged["delayed_chain_proof"]["post_adoption_frame_ids"].pop()
        forged["delayed_chain_proof_sha256"] = chain_proof_sha256(forged["delayed_chain_proof"])
        with self.assertRaises(ValueError):
            verify_delayed_selection(self.frame(18), forged, fixtures.DESCRIPTION, 13.63)

    def test_reset_clears_pending_and_rejects_old_context(self):
        self.initial(); self.advance(4, 15); self.review(15)
        result = self.tracker.reset(session_id="surface-session", target_generation=3,
            description=fixtures.DESCRIPTION, now=13.3)
        self.assertEqual(result["review_handoff"]["event"], "CONTEXT_RESET")
        self.assertEqual(self.tracker.state, "UNINITIALIZED")
        self.assertEqual(self.tracker.update(self.frame(17), now=13.43)["reason"], "FRAME_CONTEXT_MISMATCH")

    def test_closed_tracker_has_no_output_or_pending_resurrection(self):
        self.initial(); self.tracker.close()
        result = self.tracker.update(self.frame(4), now=10.83)
        self.assertEqual(result["status"], "NO_TARGET_OUTPUT")
        self.assertEqual(result["review_handoff"]["event"], "TRACKER_CLOSED")


if __name__ == "__main__":
    unittest.main()
