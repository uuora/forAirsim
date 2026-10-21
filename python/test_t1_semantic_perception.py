"""Fail-closed tests for finite T1 semantic observation, without API or flight."""
import hashlib
import tempfile
import unittest
from dataclasses import replace
from concurrent.futures import Future
from pathlib import Path

from t1_semantic_perception import (BoundedSemanticPolicy, Candidate, PolicyConfig,
                                    SemanticFrame, SemanticReply, YoloVlxBackend)


class ManualExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, operation, request):
        future = Future()
        future.set_running_or_notify_cancel()
        self.calls.append((future, request))
        return future


def frame(sequence=1, receipt=10.0, *, boxes=((10, 10, 30, 30),), session="session-a",
          generation=0, frame_id=None, ready=None, digest=None, size=(100, 80)):
    candidates = tuple(Candidate(f"f{sequence}-yolo-{index}", tuple(float(x) for x in box), 0.7)
                       for index, box in enumerate(boxes))
    return SemanticFrame(session, generation, frame_id or f"frame-{sequence}", sequence,
                         digest or hashlib.sha256(f"frame-{sequence}".encode()).hexdigest(),
                         *size, receipt, ready if ready is not None else receipt + 0.05, candidates)


def result(box=(10, 10, 30, 30)):
    return {"answer": "object", "result_bbox_list": [dict(zip(
        ("xmin", "ymin", "xmax", "ymax"), box))]}


class BoundedSemanticTests(unittest.TestCase):
    def setUp(self):
        self.executor = ManualExecutor()
        self.policy = BoundedSemanticPolicy(lambda request: None, session_id="session-a",
            target_generation=0, description="指定红色气球", executor=self.executor)

    def tearDown(self):
        self.policy.close()

    def submit(self, source=None):
        return self.policy.request(source or frame(), now=10.1)

    def complete(self, *, payload=None, completed=10.8, binding=None, mode="image_text", proposals=False):
        future, request = self.executor.calls[-1]
        future.set_result(SemanticReply(binding or request.binding(), completed,
                                       payload if payload is not None else result(), mode, proposals))

    def test_current_frame_required_then_fresh_association(self):
        self.assertEqual(self.submit()["status"], "SUBMITTED")
        self.complete(completed=10.15)
        old = self.policy.poll(frame(), now=10.2)
        self.assertEqual(old["reason"], "NEWER_CURRENT_FRAME_REQUIRED")
        accepted = self.policy.poll(frame(2, 10.8, boxes=((11, 10, 31, 30),)), now=10.9)
        self.assertEqual(accepted["status"], "SEMANTIC_LOCAL_ASSOCIATION_VALID")
        self.assertEqual(accepted["selected_candidate_id"], "f2-yolo-0")
        self.assertFalse(accepted["online_control_enabled"])
        self.assertFalse(accepted["server_proposal_ranking"])
        self.assertFalse(accepted["identity_probability_calibrated"])

    def test_request_bound_to_frame_description_and_digest(self):
        submitted = self.submit()
        binding = submitted["request_binding"]
        self.assertEqual(binding["image_sha256"], frame().image_sha256)
        self.assertEqual(binding["description_sha256"], hashlib.sha256("指定红色气球".encode()).hexdigest())
        self.assertEqual(binding["frame_sequence"], 1)
        self.assertFalse(submitted["proposals_transmitted"])

    def test_one_pending_no_new_submission(self):
        self.submit()
        refused = self.policy.request(frame(2, 10.2), now=10.3)
        self.assertEqual(refused["reason"], "TRANSPORT_SLOT_OCCUPIED")
        self.assertEqual(len(self.executor.calls), 1)

    def test_timeout_does_not_free_running_transport(self):
        self.submit()
        timeout = self.policy.poll(frame(2, 12.2), now=12.3)
        self.assertEqual(timeout["reason"], "RESPONSE_TIMEOUT")
        self.assertTrue(timeout["transport_slot_occupied"])
        refused = self.policy.request(frame(3, 12.3), now=12.4)
        self.assertEqual(refused["reason"], "TRANSPORT_SLOT_OCCUPIED")
        self.complete(completed=12.5)
        discarded = self.policy.poll(frame(4, 12.5), now=12.6)
        self.assertEqual(discarded["reason"], "RESPONSE_TIMEOUT")
        self.assertIsNone(discarded["selected_candidate_id"])
        self.assertFalse(discarded["transport_slot_occupied"])
        self.assertEqual(self.policy.request(frame(5, 12.6), now=12.7)["status"], "SUBMITTED")

    def test_completed_late_result_still_times_out(self):
        self.submit()
        self.complete(completed=12.2)
        decision = self.policy.poll(frame(2, 12.2), now=12.3)
        self.assertEqual(decision["reason"], "RESPONSE_TIMEOUT")
        self.assertIsNone(decision["selected_candidate_id"])

    def test_reset_invalidates_inflight_and_keeps_slot(self):
        self.submit()
        decision = self.policy.reset(session_id="session-a", target_generation=1,
                                     description="另一个指定气球", now=10.2)
        self.assertTrue(decision["transport_slot_occupied"])
        self.complete()
        discarded = self.policy.poll(frame(2, 10.8, generation=1), now=10.9)
        self.assertEqual(discarded["reason"], "CONTEXT_RESET")
        self.assertIsNone(discarded["selected_candidate_id"])

    def test_same_context_reset_rejected(self):
        with self.assertRaises(ValueError):
            self.policy.reset(session_id="session-a", target_generation=0,
                              description="new text", now=10.2)

    def test_mismatched_reply_frame_digest_rejected(self):
        self.submit()
        binding = self.executor.calls[-1][1].binding()
        binding["image_sha256"] = "0" * 64
        self.complete(binding=binding)
        self.assertEqual(self.policy.poll(frame(2, 10.8), now=10.9)["reason"], "RESPONSE_BINDING_MISMATCH")

    def test_mismatched_reply_generation_rejected(self):
        self.submit()
        binding = self.executor.calls[-1][1].binding()
        binding["target_generation"] = 2
        self.complete(binding=binding)
        self.assertEqual(self.policy.poll(frame(2, 10.8), now=10.9)["reason"], "RESPONSE_BINDING_MISMATCH")

    def test_unexpected_server_proposal_contract_rejected(self):
        self.submit()
        self.complete(proposals=True)
        self.assertEqual(self.policy.poll(frame(2, 10.8), now=10.9)["reason"], "UNSUPPORTED_TRANSPORT_CONTRACT")

    def test_backend_exception_sanitized(self):
        self.submit()
        self.executor.calls[-1][0].set_exception(RuntimeError("credential-or-private-URL"))
        decision = self.policy.poll(frame(2, 10.8), now=10.9)
        self.assertEqual(decision["reason"], "SEMANTIC_BACKEND_ERROR")
        self.assertEqual(decision["error_type"], "RuntimeError")
        self.assertNotIn("credential", str(decision))

    def test_cloud_none_or_multiple_boxes_uncertain(self):
        for boxes in ([], [result()["result_bbox_list"][0]] * 2):
            with self.subTest(boxes=len(boxes)):
                executor = ManualExecutor()
                policy = BoundedSemanticPolicy(lambda request: None, session_id="session-a",
                    target_generation=0, description="red", executor=executor)
                policy.request(frame(), now=10.1)
                future, request = executor.calls[0]
                future.set_result(SemanticReply(request.binding(), 10.8, {"result_bbox_list": boxes}))
                self.assertEqual(policy.poll(frame(2, 10.8), now=10.9)["reason"], "NO_SINGLE_SEMANTIC_BOX")
                policy.close()

    def test_cloud_invalid_box_rejected(self):
        self.submit()
        self.complete(payload=result((10, 10, float("nan"), 30)))
        self.assertEqual(self.policy.poll(frame(2, 10.8), now=10.9)["reason"], "INVALID_SEMANTIC_RESULT")

    def test_semantic_box_unmatched_no_yolo_fallback(self):
        self.submit()
        self.complete(payload=result((50, 10, 70, 30)))
        self.assertEqual(self.policy.poll(frame(2, 10.8), now=10.9)["reason"], "SOURCE_BOX_ASSOCIATION_UNRESOLVED")

    def test_ambiguous_source_duplicate_boxes(self):
        self.submit(frame(boxes=((10, 10, 30, 30), (10, 10, 30, 30))))
        self.complete()
        self.assertEqual(self.policy.poll(frame(2, 10.8), now=10.9)["reason"], "SOURCE_BOX_ASSOCIATION_UNRESOLVED")

    def test_second_above_threshold_is_ambiguous_even_with_margin(self):
        self.submit(frame(boxes=((10, 10, 30, 30), (14, 10, 34, 30))))
        self.complete()
        self.assertEqual(self.policy.poll(frame(2, 10.8), now=10.9)["reason"],
                         "SOURCE_BOX_ASSOCIATION_UNRESOLVED")

    def test_ambiguous_current_or_missing_detection_clears_evidence(self):
        self.submit()
        self.complete()
        decision = self.policy.poll(frame(2, 10.8, boxes=((10, 10, 30, 30), (10, 10, 30, 30))), now=10.9)
        self.assertEqual(decision["reason"], "CURRENT_BOX_ASSOCIATION_UNRESOLVED")
        recovered = self.policy.poll(frame(3, 11.0), now=11.1)
        self.assertEqual(recovered["reason"], "NO_VALID_SEMANTIC_EVIDENCE")

    def test_no_current_yolo_candidate_is_uncertain(self):
        self.submit()
        self.complete()
        self.assertEqual(self.policy.poll(frame(2, 10.8, boxes=()), now=10.9)["reason"], "CURRENT_BOX_ASSOCIATION_UNRESOLVED")

    def test_expired_current_frame_rejected_before_selection(self):
        self.submit()
        self.complete()
        self.assertEqual(self.policy.poll(frame(2, 10.3), now=10.9)["reason"], "CURRENT_FRAME_EXPIRED")

    def test_semantic_age_includes_source_acquisition_not_reply_only(self):
        self.submit()
        self.complete()
        self.assertEqual(self.policy.poll(frame(2, 10.8), now=10.9)["status"], "SEMANTIC_LOCAL_ASSOCIATION_VALID")
        decision = self.policy.poll(frame(3, 12.0), now=12.1)
        self.assertEqual(decision["reason"], "SEMANTIC_SOURCE_EXPIRED")
        self.assertIsNone(decision["selected_candidate_id"])

    def test_low_frequency_gate_and_no_candidate_request(self):
        self.submit()
        self.complete()
        self.policy.poll(frame(2, 10.8), now=10.9)
        self.assertEqual(self.policy.request(frame(3, 11.0), now=11.1)["reason"], "LOW_FREQUENCY_INTERVAL")
        self.assertEqual(self.policy.request(frame(4, 12.2, boxes=()), now=12.3)["reason"], "NO_YOLO_BALLOON_CANDIDATE")

    def test_frame_context_time_and_sequence_rejections(self):
        self.assertEqual(self.policy.request(frame(session="another"), now=10.1)["reason"], "FRAME_CONTEXT_MISMATCH")
        self.assertEqual(self.policy.request(frame(ready=10.3), now=10.2)["reason"], "FRAME_TIME_IN_FUTURE")
        self.assertEqual(self.policy.request(frame(), now=10.3)["status"], "SUBMITTED")
        self.assertEqual(self.policy.poll(frame(frame_id="rebound"), now=10.4)["reason"],
                         "FRAME_SEQUENCE_REUSED_WITH_DIFFERENT_BINDING")

    def test_monotonic_clock_regression_rejected(self):
        self.submit()
        with self.assertRaises(ValueError):
            self.policy.poll(frame(), now=10.0)

    def test_response_completion_outside_interval_rejected(self):
        self.submit()
        self.complete(completed=10.0)
        self.assertEqual(self.policy.poll(frame(2, 10.8), now=10.9)["reason"], "INVALID_SEMANTIC_RESULT")

    def test_frame_contract_and_config_reject_invalid_values(self):
        for operation in (lambda: frame(receipt=float("nan")), lambda: frame(size=(0, 80)),
                          lambda: frame(boxes=((0, 0, 110, 20),)), lambda: frame(digest="invalid"),
                          lambda: frame(receipt="10.0", ready=10.05), lambda: frame(receipt=True),
                          lambda: PolicyConfig(response_timeout_s=0), lambda: PolicyConfig(association_margin=2)):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                operation()

    def test_close_prevents_new_work(self):
        self.policy.close()
        self.assertEqual(self.policy.request(frame(), now=10.1)["reason"], "POLICY_CLOSED")
        self.assertEqual(len(self.executor.calls), 0)


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return {"candidates": [{"class_name": "balloon", "bbox_xyxy_pixels": [10, 10, 30, 30],
                                 "confidence": 0.7, "source_class_name": "sports ball"},
                                {"class_name": "vessel", "bbox_xyxy_pixels": [40, 10, 60, 30],
                                 "confidence": 0.9, "source_class_name": "boat"}]}


class FakeCloud:
    request_mode = "image_text"
    proposals_transmitted = False

    def __init__(self):
        self.calls = []

    def ground(self, image, boxes, description, **options):
        self.calls.append((image.size, boxes, description, options))
        return result()


class BackendContractTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "frame.png"
        Image.new("RGB", (100, 80), "white").save(self.path)
        self.detector, self.cloud = FakeDetector(), FakeCloud()
        self.backend = YoloVlxBackend(detector=self.detector, vlx_client=self.cloud, clock=lambda: 10.1)

    def tearDown(self):
        self.temp.cleanup()

    def detect(self, expected=None):
        return self.backend.detect_frame(self.path, session_id="session-a", target_generation=0,
            frame_id="frame-1", frame_sequence=1, readback_receipt_monotonic_s=10.0,
            expected_image_sha256=expected)

    def test_explicit_warmup_before_timed_frame(self):
        with self.assertRaises(ValueError):
            self.detect()
        warmup = self.backend.warmup(self.path)
        self.assertFalse(warmup["semantic_api_called"])
        self.assertEqual(self.cloud.calls, [])
        current = self.detect()
        self.assertEqual(len(current.candidates), 1)
        self.assertEqual(current.candidates[0].source_class_name, "sports ball")
        self.assertEqual(self.detector.calls, 2)

    def test_real_image_hash_binding_and_empty_server_proposals(self):
        self.backend.warmup(self.path)
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        current = self.detect(digest)
        executor = ManualExecutor()
        policy = BoundedSemanticPolicy(self.backend, session_id="session-a", target_generation=0,
                                       description="红色气球", executor=executor)
        policy.request(current, now=10.1)
        request = executor.calls[0][1]
        reply = self.backend(request)
        self.assertEqual(reply.binding, request.binding())
        self.assertEqual(self.cloud.calls[0][1], [])
        self.assertEqual(self.cloud.calls[0][2], "红色气球")
        self.assertFalse(reply.proposals_transmitted)
        self.assertNotIn("encoded_image", current.to_dict())
        policy.close()

    def test_changed_saved_png_rejected(self):
        self.backend.warmup(self.path)
        with self.assertRaises(ValueError):
            self.detect("0" * 64)

    def test_backend_rejects_modified_description_binding(self):
        self.backend.warmup(self.path)
        executor = ManualExecutor()
        policy = BoundedSemanticPolicy(self.backend, session_id="session-a", target_generation=0,
                                       description="red", executor=executor)
        policy.request(self.detect(), now=10.1)
        request = executor.calls[0][1]
        with self.assertRaises(ValueError):
            self.backend(replace(request, description="changed"))
        self.assertEqual(self.cloud.calls, [])
        policy.close()


if __name__ == "__main__":
    unittest.main()
