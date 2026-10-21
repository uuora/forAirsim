"""Observer boundary tests with the real policy/tracker, no cloud or simulator."""
from concurrent.futures import Future
import hashlib
import unittest

from t1_continuous_semantic_observer import ContinuousSemanticObserver
from t1_semantic_perception import BoundedSemanticPolicy, Candidate, SemanticFrame, SemanticReply


class Clock:
    now = 10.
    def __call__(self):
        return self.now


class ManualExecutor:
    def __init__(self):
        self.calls = []
    def submit(self, backend, request):
        future = Future()
        future.set_running_or_notify_cancel()
        self.calls.append((future, request))
        return future
    def complete(self, clock, box=(10, 10, 30, 30)):
        future, request = self.calls[-1]
        future.set_result(SemanticReply(request.binding(), clock(), {
            "answer": "target", "result_bbox_list": [dict(zip(("xmin","ymin","xmax","ymax"),box))]}))


class Backend:
    def __init__(self, clock):
        self.clock = clock
        self.boxes = ((10, 10, 30, 30),)
        self.detect_calls = 0
    def warmup(self, path):
        return {"status":"WARMED", "semantic_api_called":False}
    def detect_frame(self, path, **kw):
        self.detect_calls += 1
        return SemanticFrame(kw["session_id"], kw["target_generation"], kw["frame_id"],
            kw["frame_sequence"], kw["expected_image_sha256"], 100, 80,
            kw["readback_receipt_monotonic_s"], self.clock(), tuple(
                Candidate(f"f{kw['frame_sequence']}-c{i}", tuple(map(float, box)), .8)
                for i,box in enumerate(self.boxes)))
    def __call__(self, request):
        raise AssertionError("No real cloud in tests")


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.clock, self.executor = Clock(), ManualExecutor()
        self.backend = Backend(self.clock)
        self.observers = []
    def tearDown(self):
        for observer in self.observers:
            for future,request in self.executor.calls:
                if not future.done():
                    future.set_exception(RuntimeError("synthetic final drain"))
            observer.close()
    def observer(self, mode="VISUAL_ATTRIBUTE"):
        def factory(backend, **kw):
            return BoundedSemanticPolicy(backend, executor=self.executor, **kw)
        observer = ContinuousSemanticObserver(self.backend, clock=self.clock, policy_factory=factory)
        observer.begin("session", "constant description", "warmup", recovery_observability=mode)
        self.observers.append(observer)
        return observer
    def observation(self, sequence):
        return {"session_id":"session", "target_generation":1, "description":"constant description",
            "frame_id":f"f{sequence}", "frame_sequence":sequence, "rgb_path":"image.png",
            "rgb_sha256":hashlib.sha256(f"{sequence}".encode()).hexdigest(), "image_size_wh":[100,80],
            "readback_receipt_monotonic_s":self.clock(), "capture_requested_monotonic_s":self.clock(),
            "capture_completed_monotonic_s":self.clock(),
            "camera":{"scene_intrinsics":{}, "pose_local_ned_m":[0,0,0],
                "orientation_xyzw":[0,0,0,1], "world_from_local_translation_ned_m":[0,0,0]},
            "capture_metadata":{"rgb_render_readback_timestamp_ns":100, "timestamp_scope":"readback"}}
    def step(self, observer, sequence, advance=.2):
        self.clock.now += advance
        return observer.process(self.observation(sequence))
    def initialized(self, mode="VISUAL_ATTRIBUTE"):
        observer = self.observer(mode)
        first = self.step(observer,1)
        self.assertEqual(first["semantic_request"]["status"],"SUBMITTED")
        self.executor.complete(self.clock)
        second = self.step(observer,2)
        self.assertEqual(second["temporal_identity"]["state"],"RECOVERY_PENDING")
        third = self.step(observer,3)
        self.assertEqual(third["temporal_identity"]["state"],"TRACKING")
        return observer
    def test_no_automatic_high_confidence_designation(self):
        result=self.step(self.observer(),1)
        self.assertEqual(result["temporal_identity"]["status"],"NO_TARGET_OUTPUT")
        self.assertFalse(result["flight_authorized"])
    def test_real_policy_initialization_and_frame_binding(self):
        observer=self.initialized()
        result=self.step(observer,4)
        self.assertEqual(result["temporal_identity"]["current_binding"]["frame_sequence"],4)
        self.assertEqual(observer._submitted_count,1)
    def test_forbidden_truth_rejected_before_detection(self):
        observer=self.observer()
        for key in ("phase_id","evaluation_only","target_world_ned_m","depth"):
            row=self.observation(1)
            row[key]={}
            with self.assertRaises(ValueError):
                observer.process(row)
        self.assertEqual(self.backend.detect_calls,0)
    def test_nested_camera_truth_rejected(self):
        observer=self.observer()
        row=self.observation(1)
        row["camera"]["target_pose"]=[1,2,3]
        with self.assertRaises(ValueError):
            observer.process(row)
    def test_constant_description_and_generation(self):
        observer=self.observer()
        for key,value in (("description","later right"),("target_generation",2)):
            row=self.observation(1)
            row[key]=value
            with self.assertRaises(ValueError):
                observer.process(row)
    def test_identical_loss_blocks_new_recovery_requests(self):
        observer=self.initialized("UNRESOLVABLE_IDENTICAL")
        self.backend.boxes=((50,10,70,30),)
        result=self.step(observer,4)
        self.assertTrue(result["unobservable_recovery_blocked"])
        self.assertEqual(result["temporal_identity"]["status"],"NO_TARGET_OUTPUT")
        self.backend.boxes=((10,10,30,30),)
        result=self.step(observer,5,advance=3.)
        self.assertEqual(result["semantic_request"]["reason"],"IDENTITY_UNOBSERVABLE_AFTER_LOSS")
        self.assertEqual(result["temporal_identity"]["status"],"NO_TARGET_OUTPUT")
        self.assertEqual(len(self.executor.calls),1)
    def test_gap_with_new_semantic_reply_cannot_reanchor_identical(self):
        observer=self.initialized("UNRESOLVABLE_IDENTICAL")
        # Keep an intact track up to a periodic review at three seconds.
        for sequence in range(4,18):
            result=self.step(observer,sequence)
            if result["semantic_request"]["status"] == "SUBMITTED":
                break
        self.assertEqual(result["semantic_request"]["status"],"SUBMITTED")
        self.executor.complete(self.clock)
        result=self.step(observer,sequence+1,advance=.6)
        self.assertTrue(result["unobservable_recovery_blocked"])
        self.assertEqual(result["temporal_identity"]["status"],"NO_TARGET_OUTPUT")
    def test_cue_loss_requests_fresh_semantic_evidence(self):
        observer=self.initialized()
        self.backend.boxes=((50,10,70,30),)
        lost=self.step(observer,4)
        self.assertEqual(lost["temporal_identity"]["status"],"NO_TARGET_OUTPUT")
        requested=self.step(observer,5,advance=2.6)
        self.assertEqual(requested["semantic_request"]["status"],"SUBMITTED")
        self.executor.complete(self.clock,box=(50,10,70,30))
        pending=self.step(observer,6)
        recovered=self.step(observer,7)
        self.assertEqual(pending["temporal_identity"]["state"],"RECOVERY_PENDING")
        self.assertEqual(recovered["temporal_identity"]["state"],"TRACKING")
        self.assertFalse(recovered["unobservable_recovery_blocked"])
    def test_identical_pending_loss_cannot_restart_positional_anchor(self):
        observer=self.observer("UNRESOLVABLE_IDENTICAL")
        self.step(observer,1)
        self.executor.complete(self.clock)
        pending=self.step(observer,2)
        self.assertEqual(pending["temporal_identity"]["state"],"RECOVERY_PENDING")
        self.backend.boxes=((50,10,70,30),)
        result=self.step(observer,3)
        self.assertTrue(result["unobservable_recovery_blocked"])
        result=self.step(observer,4,advance=3.)
        self.assertEqual(result["semantic_request"]["reason"],"IDENTITY_UNOBSERVABLE_AFTER_LOSS")
        self.assertEqual(len(self.executor.calls),1)
    def test_initialization_never_shortens_three_second_review_period(self):
        observer=self.observer()
        self.step(observer,1)
        self.executor.complete(self.clock,box=(50,10,70,30))
        self.step(observer,2)
        result=self.step(observer,3,advance=2.)
        self.assertEqual(result["semantic_request"]["status"],"NOT_REQUESTED")
        self.assertEqual(len(self.executor.calls),1)
    def test_close_retains_counters_without_cloud_credentials(self):
        observer=self.initialized()
        receipt=observer.close()
        self.assertEqual(receipt["request_submissions"],1)
        self.assertFalse(receipt["flight_authorized"])
        self.assertIs(observer.close(),receipt)
        with self.assertRaises(ValueError):
            self.step(observer,4)


if __name__ == "__main__":
    unittest.main()
