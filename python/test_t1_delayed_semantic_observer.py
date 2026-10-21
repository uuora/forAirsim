"""Real observer composition and redacted HTTP telemetry, with no network."""
from concurrent.futures import Future
from types import SimpleNamespace
import unittest
import requests

import test_t1_continuous_semantic_observer as observer_fixture
from t1_delayed_semantic_observer import (
    DelayedSemanticObserver, TracedGroundingBackend, transport_error_category,
)
from t1_delayed_semantic_bridge import DelayedSemanticPolicy
from t1_semantic_perception import SemanticReply


class ManualExecutor:
    def __init__(self):
        self.calls = []
    def submit(self, function, request):
        future = Future()
        future.set_running_or_notify_cancel()
        self.calls.append((future, request))
        return future
    def shutdown(self, **kwargs):
        pass
    def complete(self, clock):
        future, request = self.calls[-1]
        future.set_result(SemanticReply(request.binding(), clock(), {
            "result_bbox_list": [dict(xmin=10, ymin=10, xmax=30, ymax=30)]}))


class DelayedObserverTests(unittest.TestCase):
    observation = observer_fixture.ObserverTests.observation
    def setUp(self):
        self.clock, self.executor = observer_fixture.Clock(), ManualExecutor()
        self.backend = observer_fixture.Backend(self.clock)
        def factory(backend, **kw):
            return DelayedSemanticPolicy(backend, executor=self.executor, **kw)
        self.observer = DelayedSemanticObserver(self.backend, clock=self.clock, policy_factory=factory)
        self.observer.begin("session", "constant description", "warmup", recovery_observability="VISUAL_ATTRIBUTE")
    def tearDown(self):
        for future, _ in self.executor.calls:
            if not future.done():
                future.set_exception(ValueError("synthetic"))
        self.observer.close()
    def step(self, seq):
        self.clock.now += .2
        return self.observer.process(self.observation(seq))
    def test_initialization_needs_two_frames_after_reply_adoption(self):
        first = self.step(1)
        self.assertEqual(first["semantic_request"]["status"], "SUBMITTED")
        self.executor.complete(self.clock)
        second = self.step(2)
        third = self.step(3)
        fourth = self.step(4)
        self.assertEqual(second["temporal_identity"]["state"], "RECOVERY_PENDING")
        self.assertEqual(third["temporal_identity"]["state"], "RECOVERY_PENDING")
        self.assertEqual(fourth["temporal_identity"]["state"], "TRACKING")
        self.observer.close()
        self.assertEqual(self.observer.close_receipt["policy_config"]["max_source_age_s"], 8.)
        self.assertEqual(self.observer.close_receipt["tracker_config"]["semantic_source_max_age_s"], 8.)
    def test_truth_is_rejected_before_detector(self):
        row = self.observation(1)
        row["target_pose"] = [0,0,0]
        with self.assertRaises(ValueError):
            self.observer.process(row)
        self.assertEqual(self.backend.detect_calls, 0)


class FakeResponse:
    status_code = 200
    def raise_for_status(self):
        pass
    def json(self):
        return {"choices": [{}], "secret_response_text": "never serialize"}


class FakeSession:
    def post(self, url, **kwargs):
        return FakeResponse()


class TraceTests(unittest.TestCase):
    def test_clocks_without_payload_or_endpoint(self):
        clock = observer_fixture.Clock()
        class Grounder:
            vlx_client = SimpleNamespace(session=FakeSession())
            def __call__(self, request):
                response = self.vlx_client.session.post("private_endpoint", json={"private": "key"})
                response.raise_for_status()
                response.json()
                return SemanticReply(request.binding(), clock(), {"result_bbox_list": []})
        request = SimpleNamespace(binding=lambda: {"request_id": "r1"})
        backend = TracedGroundingBackend(Grounder(), clock=clock)
        backend(request)
        trace = backend.snapshot()[0]
        self.assertEqual(trace["http_status_class"], "HTTP_2XX")
        self.assertEqual(trace["json_parse_status"], "PARSED")
        self.assertEqual(trace["grounding_conversion_status"], "CONVERTED")
        self.assertEqual(trace["service_payload_status"], "CHOICES_PRESENT_CONVERSION_PENDING")
        self.assertNotIn("private", str(trace))
        self.assertNotIn("secret_response_text", str(trace))
    def test_json_exception_category_precedes_request_exception(self):
        error = requests.exceptions.JSONDecodeError("private body", "private", 0)
        self.assertEqual(transport_error_category(error), "JSON_DECODE_FAILURE")
        self.assertEqual(transport_error_category(requests.exceptions.ReadTimeout("private")), "READ_TIMEOUT")


if __name__ == "__main__":
    unittest.main()
