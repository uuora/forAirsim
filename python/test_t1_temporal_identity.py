"""Contracts for conservative temporal association; no simulator/model imports."""
import copy
from concurrent.futures import Future
from dataclasses import replace
import hashlib
import unittest

from t1_semantic_perception import (BoundedSemanticPolicy, Candidate, SemanticFrame,
                                    SemanticReply)
from t1_temporal_identity import TemporalIdentityTracker, TrackerConfig


DESCRIPTION = "the red and white striped balloon"


def frame(sequence, receipt, boxes=((100, 100, 200, 200),), *, session="episode", generation=0):
    image = f"image-{sequence}".encode()
    return SemanticFrame(session, generation, f"f-{sequence}", sequence, hashlib.sha256(image).hexdigest(),
                         640, 480, receipt, receipt + .001,
                         tuple(Candidate(f"f-{sequence}-c-{index}", tuple(box), .99 - index * .1,
                                         source_class_name="balloon") for index, box in enumerate(boxes)),
                         image)


def semantic(current, request="review-1", selected_index=0, *, source_receipt=None, source_sequence=None):
    selected = current.candidates[selected_index]
    source_receipt = current.readback_receipt_monotonic_s - .1 if source_receipt is None else source_receipt
    source_sequence = current.frame_sequence - 1 if source_sequence is None else source_sequence
    return {"status": "SEMANTIC_LOCAL_ASSOCIATION_VALID", "session_id": current.session_id,
            "target_generation": current.target_generation,
            "description_sha256": hashlib.sha256(DESCRIPTION.encode()).hexdigest(),
            "current_frame_id": current.frame_id, "current_frame_sequence": current.frame_sequence,
            "current_image_sha256": current.image_sha256,
            "selected_candidate_id": selected.candidate_id, "selected_candidate": selected.to_dict(),
            "request_binding": {"request_id": request, "session_id": current.session_id,
                "target_generation": current.target_generation,
                "description_sha256": hashlib.sha256(DESCRIPTION.encode()).hexdigest(),
                "frame_id": f"source-{source_sequence}", "frame_sequence": source_sequence,
                "image_sha256": "a" * 64, "readback_receipt_monotonic_s": source_receipt,
                "submitted_monotonic_s": source_receipt + .01},
            "reply_completed_monotonic_s": source_receipt + .05,
            "reply_binding_provenance": "CLIENT_SIDE_BINDING; NOT SERVER_ECHO",
            "request_mode": "image_text", "proposals_transmitted": False, "server_proposal_ranking": False,
            "semantic_box_xyxy_pixels": list(selected.bbox_xyxy_pixels), "source_candidate_id": "source-candidate",
            "source_association": [{"candidate_id": "source-candidate", "iou": 1.0}],
            "current_association": [{"candidate_id": candidate.candidate_id,
                                      "iou": 1.0 if candidate == selected else 0.0}
                                     for candidate in current.candidates],
            "identity_probability_calibrated": False, "instance_identity_guaranteed": False}


def rebind(decision, current, selected_index=0):
    output = copy.deepcopy(decision)
    selected = current.candidates[selected_index]
    output.update(current_frame_id=current.frame_id, current_frame_sequence=current.frame_sequence,
                  current_image_sha256=current.image_sha256, selected_candidate_id=selected.candidate_id,
                  selected_candidate=selected.to_dict(), current_association=[
                      {"candidate_id": candidate.candidate_id, "iou": 1.0 if candidate == selected else 0.0}
                      for candidate in current.candidates])
    return output


class TemporalIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tracker = TemporalIdentityTracker("episode", 0, DESCRIPTION)

    def start(self):
        first = frame(1, 10.0)
        decision = semantic(first)
        self.assertEqual(self.tracker.update(first, decision, 10.01)["state"], "RECOVERY_PENDING")
        second = frame(2, 10.1, ((102, 100, 202, 200),))
        result = self.tracker.update(second, now=10.11)
        self.assertEqual(result["state"], "TRACKING")
        return result, decision

    def test_no_initial_highest_score_designation(self):
        result = self.tracker.update(frame(1, 10), now=10.01)
        self.assertEqual(result["state"], "UNINITIALIZED")
        self.assertIsNone(result["selected_candidate"])
        self.assertTrue(result["needs_semantic_review"])

    def test_initial_confirmation_requires_two_new_frames(self):
        first = frame(1, 10)
        result = self.tracker.update(first, semantic(first), 10.01)
        self.assertIsNone(result["selected_candidate"])
        result = self.tracker.update(frame(2, 10.1), now=10.11)
        self.assertEqual(result["confirmation_count"], 2)
        self.assertIsNotNone(result["selected_candidate"])

    def test_same_request_does_not_restart_confirmation(self):
        first = frame(1, 10)
        decision = semantic(first)
        self.tracker.update(first, decision, 10.01)
        second = frame(2, 10.1)
        result = self.tracker.update(second, rebind(decision, second), 10.11)
        self.assertEqual(result["state"], "TRACKING")
        self.assertEqual(result["semantic_expires_monotonic_s"], 11.9)

    def test_duplicate_does_not_confirm_or_output(self):
        first = frame(1, 10)
        self.tracker.update(first, semantic(first), 10.01)
        result = self.tracker.update(first, semantic(first), 10.02)
        self.assertEqual(result["reason"], "DUPLICATE_FRAME")
        self.assertEqual(result["confirmation_count"], 1)
        self.assertIsNone(result["selected_candidate"])

    def test_duplicate_in_tracking_never_reissues_output(self):
        self.start()
        result = self.tracker.update(frame(2, 10.1, ((102, 100, 202, 200),)), now=10.12)
        self.assertEqual(result["state"], "TRACKING")
        self.assertIsNone(result["selected_candidate"])

    def test_same_sequence_modified_box_invalidates_chain(self):
        self.start()
        result = self.tracker.update(frame(2, 10.1, ((104, 100, 204, 200),)), now=10.12)
        self.assertEqual(result["state"], "LOST")
        self.assertEqual(result["reason"], "FRAME_SEQUENCE_REUSED_WITH_DIFFERENT_BINDING")

    def test_slow_unique_motion_does_not_use_detector_score(self):
        previous, _ = self.start()
        current = frame(3, 10.2, ((105, 100, 205, 200), (350, 100, 450, 200)))
        current = replace(current, candidates=(replace(current.candidates[0], detector_score=.2),
                                               replace(current.candidates[1], detector_score=.99)))
        result = self.tracker.update(current, now=10.21)
        self.assertEqual(result["selected_candidate_id"], current.candidates[0].candidate_id)
        self.assertEqual(result["track_id"], previous["track_id"])

    def test_overlapping_distractor_makes_output_ambiguous(self):
        self.start()
        result = self.tracker.update(frame(3, 10.2, ((104, 100, 204, 200), (108, 100, 208, 200))), now=10.21)
        self.assertEqual(result["state"], "AMBIGUOUS")
        self.assertIsNone(result["selected_candidate"])
        self.assertTrue(result["needs_semantic_review"])

    def test_unique_above_threshold_still_needs_margin(self):
        self.start()
        result = self.tracker.update(frame(3, 10.2, ((150, 100, 250, 200), (158, 100, 258, 200))), now=10.21)
        self.assertEqual(result["state"], "AMBIGUOUS")
        self.assertIsNone(result["selected_candidate"])

    def test_no_box_immediately_loses_and_single_return_cannot_recover(self):
        self.start()
        result = self.tracker.update(frame(3, 10.2, ()), now=10.21)
        self.assertEqual(result["state"], "LOST")
        result = self.tracker.update(frame(4, 10.3), now=10.31)
        self.assertEqual(result["reason"], "FRESH_SEMANTIC_ANCHOR_REQUIRED")
        self.assertIsNone(result["selected_candidate"])

    def test_old_anchor_cannot_bridge_occlusion(self):
        _, decision = self.start()
        self.tracker.update(frame(3, 10.2, ()), now=10.21)
        returned = frame(4, 10.3)
        result = self.tracker.update(returned, rebind(decision, returned), 10.31)
        self.assertEqual(result["reason"], "SEMANTIC_ANCHOR_REUSED_AFTER_LOSS")
        self.assertIsNone(result["selected_candidate"])

    def test_recovery_needs_new_anchor_and_two_frames(self):
        initial, _ = self.start()
        self.tracker.update(frame(3, 10.2, ()), now=10.21)
        fourth = frame(4, 10.3)
        result = self.tracker.update(fourth, semantic(fourth, "review-2"), 10.31)
        self.assertEqual(result["state"], "RECOVERY_PENDING")
        self.assertIsNone(result["selected_candidate"])
        result = self.tracker.update(frame(5, 10.4), now=10.41)
        self.assertEqual(result["state"], "TRACKING")
        self.assertNotEqual(result["track_id"], initial["track_id"])

    def test_pending_chain_itself_cannot_bridge_loss(self):
        first = frame(1, 10)
        decision = semantic(first)
        self.tracker.update(first, decision, 10.01)
        self.tracker.update(frame(2, 10.1, ()), now=10.11)
        third = frame(3, 10.2)
        result = self.tracker.update(third, rebind(decision, third), 10.21)
        self.assertEqual(result["reason"], "SEMANTIC_ANCHOR_REUSED_AFTER_LOSS")

    def test_new_review_of_same_track_preserves_output_and_id(self):
        initial, _ = self.start()
        third = frame(3, 10.2)
        result = self.tracker.update(third, semantic(third, "review-2"), 10.21)
        self.assertEqual(result["reason"], "SEMANTIC_REVIEW_REFRESHED_CONTINUOUS_TRACK")
        self.assertEqual(result["track_id"], initial["track_id"])
        self.assertEqual(result["semantic_anchor"], "review-2")
        self.assertIsNotNone(result["selected_candidate"])

    def test_new_review_selecting_other_box_is_conflict(self):
        self.start()
        third = frame(3, 10.2, ((104, 100, 204, 200), (350, 100, 450, 200)))
        decision = semantic(third, "review-2", 1)
        result = self.tracker.update(third, decision, 10.21)
        self.assertEqual(result["state"], "AMBIGUOUS")
        self.assertIsNone(result["selected_candidate"])

    def test_semantic_request_id_cannot_change_source_binding(self):
        self.start()
        third = frame(3, 10.2)
        result = self.tracker.update(third, semantic(third, "review-1"), 10.21)
        self.assertEqual(result["reason"], "SEMANTIC_REQUEST_ID_REBOUND")

    def test_expired_semantic_certificate_does_not_initially_anchor(self):
        current = frame(10, 10)
        result = self.tracker.update(current, semantic(current, source_receipt=7.9), 10.01)
        self.assertEqual(result["reason"], "SEMANTIC_CERTIFICATE_INVALID_OR_EXPIRED")
        self.assertIsNone(result["selected_candidate"])

    def test_long_link_gap_requires_semantic_recovery(self):
        self.start()
        result = self.tracker.update(frame(3, 10.7), now=10.71)
        self.assertEqual(result["reason"], "TEMPORAL_LINK_GAP_EXCEEDED")
        self.assertIsNone(result["selected_candidate"])

    def test_fresh_semantic_after_gap_seeds_confirmation(self):
        self.start()
        current = frame(3, 10.7)
        result = self.tracker.update(current, semantic(current, "review-2"), 10.71)
        self.assertEqual(result["state"], "RECOVERY_PENDING")
        self.assertIsNone(result["selected_candidate"])

    def test_old_review_window_never_extended_by_local_frames(self):
        self.start()
        result = None
        for index in range(3, 84):
            receipt = 10.1 + (index - 2) * .1
            result = self.tracker.update(frame(index, receipt), now=receipt + .01)
            if result["state"] == "LOST":
                break
        self.assertEqual(result["reason"], "SEMANTIC_REVIEW_INTERVAL_EXCEEDED")
        self.assertIsNone(result["selected_candidate"])

    def test_proactive_review_flag_before_deadline(self):
        self.start()
        result = None
        for index in range(3, 64):
            receipt = 10.1 + (index - 2) * .1
            result = self.tracker.update(frame(index, receipt), now=receipt + .01)
        self.assertEqual(result["state"], "TRACKING")
        self.assertTrue(result["needs_semantic_review"])

    def test_uncertain_policy_does_not_masquerade_as_new_anchor(self):
        initial, _ = self.start()
        result = self.tracker.update(frame(3, 10.2), {"status": "UNCERTAIN", "reason": "SEMANTIC_SOURCE_EXPIRED"}, 10.21)
        self.assertEqual(result["semantic_anchor"], "review-1")
        self.assertEqual(result["track_id"], initial["track_id"])

    def test_stale_current_frame_immediately_clears_track(self):
        self.start()
        result = self.tracker.update(frame(3, 10.2), now=10.8)
        self.assertEqual(result["reason"], "CURRENT_FRAME_EXPIRED")
        self.assertIsNone(result["semantic_anchor"])

    def test_frame_or_clock_regression_never_outputs(self):
        self.start()
        result = self.tracker.update(frame(1, 10.3), now=10.31)
        self.assertEqual(result["reason"], "FRAME_SEQUENCE_OR_RECEIPT_REGRESSION")
        self.assertIsNone(result["selected_candidate"])
        result = self.tracker.update(frame(3, 10.4), now=10.0)
        self.assertEqual(result["reason"], "TRACKER_CLOCK_REGRESSION")

    def test_new_sequence_requires_strictly_new_receipt(self):
        self.start()
        result = self.tracker.update(frame(3, 10.1), now=10.12)
        self.assertEqual(result["reason"], "FRAME_SEQUENCE_OR_RECEIPT_REGRESSION")

    def test_reused_older_frame_id_is_rejected(self):
        self.start()
        current = replace(frame(3, 10.2), frame_id="f-1")
        result = self.tracker.update(current, now=10.21)
        self.assertEqual(result["reason"], "FRAME_ID_REUSED")

    def test_reset_cancels_anchor_and_old_generation(self):
        self.start()
        result = self.tracker.reset(session_id="episode", target_generation=1, description=DESCRIPTION, now=10.2)
        self.assertEqual(result["state"], "UNINITIALIZED")
        result = self.tracker.update(frame(3, 10.3), now=10.31)
        self.assertEqual(result["reason"], "FRAME_CONTEXT_MISMATCH")
        result = self.tracker.update(frame(1, 10.4, generation=1), now=10.41)
        self.assertIsNone(result["selected_candidate"])

    def test_reset_requires_advanced_generation_and_monotonic_time(self):
        self.start()
        with self.assertRaises(ValueError):
            self.tracker.reset(session_id="episode", target_generation=0, description=DESCRIPTION, now=10.2)
        with self.assertRaises(ValueError):
            self.tracker.reset(session_id="episode", target_generation=1, description=DESCRIPTION, now=9)

    def test_closed_tracker_cannot_output_or_reset(self):
        self.start()
        self.tracker.close()
        result = self.tracker.update(frame(3, 10.2), now=10.21)
        self.assertEqual(result["reason"], "TRACKER_CLOSED")
        self.assertIsNone(result["semantic_anchor"])
        with self.assertRaises(ValueError):
            self.tracker.reset(session_id="new", target_generation=0, description=DESCRIPTION, now=10.3)

    def test_invalid_binding_fields_fail_closed(self):
        cases = [
            ("session_id", "other"), ("target_generation", True), ("description_sha256", "b" * 64),
            ("current_frame_id", "other"), ("current_frame_sequence", True), ("current_image_sha256", "b" * 64),
            ("reply_binding_provenance", "SERVER_ECHO"), ("request_mode", "text_only"),
            ("proposals_transmitted", True), ("identity_probability_calibrated", True),
            ("instance_identity_guaranteed", True), ("reply_completed_monotonic_s", 100),
            ("selected_candidate_id", "absent"), ("source_association", []), ("current_association", []),
        ]
        for key, value in cases:
            with self.subTest(key=key):
                tracker = TemporalIdentityTracker("episode", 0, DESCRIPTION)
                current = frame(1, 10)
                decision = semantic(current)
                decision[key] = value
                result = tracker.update(current, decision, 10.01)
                self.assertEqual(result["reason"], "SEMANTIC_CERTIFICATE_INVALID_OR_EXPIRED")
                self.assertIsNone(result["selected_candidate"])

    def test_changed_current_selected_box_is_rejected(self):
        current = frame(1, 10)
        decision = semantic(current)
        decision["selected_candidate"]["bbox_xyxy_pixels"][0] = 101
        result = self.tracker.update(current, decision, 10.01)
        self.assertIsNone(result["selected_candidate"])

    def test_false_unique_semantic_audit_is_rejected(self):
        current = frame(1, 10, ((100, 100, 200, 200), (110, 100, 210, 200)))
        decision = semantic(current)
        decision["current_association"][1]["iou"] = .5
        result = self.tracker.update(current, decision, 10.01)
        self.assertIsNone(result["selected_candidate"])

    def test_no_identity_probability_or_motion_claim(self):
        result, _ = self.start()
        self.assertFalse(result["identity_probability_calibrated"])
        self.assertFalse(result["instance_identity_guaranteed"])
        self.assertFalse(result["appearance_continuity_verified"])
        self.assertFalse(result["flight_authorized"])
        self.assertNotIn("identity_probability", result)
        self.assertEqual(result["status"], "TEMPORAL_LOCAL_ASSOCIATION_VALID")

    def test_more_than_two_confirmations_is_supported(self):
        tracker = TemporalIdentityTracker("episode", 0, DESCRIPTION,
                                         TrackerConfig(initialization_confirm_frames=3))
        first = frame(1, 10)
        tracker.update(first, semantic(first), 10.01)
        self.assertIsNone(tracker.update(frame(2, 10.1), now=10.11)["selected_candidate"])
        self.assertIsNotNone(tracker.update(frame(3, 10.2), now=10.21)["selected_candidate"])

    def test_config_invalid_values(self):
        for kwargs in ({"initialization_confirm_frames": 1}, {"recovery_confirm_frames": True},
                       {"max_link_gap_s": 0}, {"association_min_iou": 1.1},
                       {"current_frame_max_age_s": float("nan")},
                       {"semantic_source_max_age_s": .1}, {"max_unreviewed_track_s": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrackerConfig(**kwargs)

    def test_actual_bounded_policy_certificate_is_accepted(self):
        class Executor:
            def submit(self, fn, request):
                future = Future()
                future.set_result(fn(request))
                return future
        def backend(request):
            return SemanticReply(request.binding(), 10.04, {"result_bbox_list": [
                {"xmin": 100, "ymin": 100, "xmax": 200, "ymax": 200}]})
        policy = BoundedSemanticPolicy(backend, session_id="episode", target_generation=0,
                                      description=DESCRIPTION, executor=Executor())
        source = frame(0, 10)
        policy.request(source, now=10.01)
        current = frame(1, 10.05)
        first = policy.poll(current, now=10.06)
        result = self.tracker.update(current, first, 10.06)
        self.assertEqual(result["state"], "RECOVERY_PENDING")
        current = frame(2, 10.1)
        second = policy.poll(current, now=10.11)
        result = self.tracker.update(current, second, 10.11)
        self.assertEqual(result["state"], "TRACKING")
        self.assertEqual(result["semantic_anchor"], first["request_binding"]["request_id"])
        policy.close()


if __name__ == "__main__":
    unittest.main()

