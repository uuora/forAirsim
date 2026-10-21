"""Independent scoring must retain missing/late/ambiguous observations."""
import copy
import unittest

from analyze_t1_identity_sequence import _decision_label, analyze_report, expected_slots, rate_bounds
from prepare_t1_identity_sequence import build_protocol


class EvaluationTests(unittest.TestCase):
    def test_missing_session_retains_sixty_unknown_slots(self):
        config=build_protocol("missing",6100410,"VISUAL_ATTRIBUTE")
        result=analyze_report(None,config,episode_id="missing")
        self.assertEqual(result["planned_slots"],60)
        self.assertEqual(result["captured_slots"],0)
        self.assertEqual(result["scheduled_identity"]["unknown_slots"],60)
        self.assertIsNone(result["scheduled_identity"]["fraction"])
        self.assertEqual(result["scheduled_identity"]["upper_bound"],1)
    def test_bounds_include_missing_and_unresolved(self):
        result=rate_bounds([{"x":label} for label in
            ("CORRECT","WRONG","NO_OUTPUT","PENDING","UNKNOWN_TRUTH","UNRESOLVED")],"x")
        self.assertEqual(result["planned_slot_denominator"],6)
        self.assertEqual(result["lower_bound"],1/6)
        self.assertEqual(result["upper_bound"],3/6)
        self.assertIsNone(result["fraction"])
    def test_empty_grid_not_zero_success(self):
        self.assertIsNone(rate_bounds([],"x")["fraction"])
    def test_duplicate_grid_rejected(self):
        config=build_protocol("duplicate",6100410,"VISUAL_ATTRIBUTE")
        slot=expected_slots(config)[0]
        with self.assertRaises(ValueError):
            analyze_report({"scheduled_slots":[slot,slot]},config,episode_id="x")
    def fixture(self):
        bindings={"session_id":"s", "target_generation":1, "frame_id":"f", "frame_sequence":1,
                  "image_sha256":"a"*64}
        candidate={"candidate_id":"c", "bbox_xyxy_pixels":[10.,10.,30.,30.],"detector_score":.9}
        semantic={**bindings,"readback_receipt_monotonic_s":10.,"candidates":[candidate]}
        tracker={"status":"TEMPORAL_LOCAL_ASSOCIATION_VALID", "state":"TRACKING",
            "selected_candidate":candidate,"current_binding":bindings,"flight_authorized":False,
            "instance_identity_guaranteed":False}
        frame={"frame_id":"f", "frame_sequence":1,"capture_received_host_s":10.,
            "media":{"rgb":{"sha256":"a"*64}},"observer_record":{"status":"RETURNED",
            "decision":{"semantic_frame":semantic,"temporal_identity":tracker}}}
        truth={"target":{"bbox_xyxy_pixels":[10,10,30,30]},
               "distractor":{"bbox_xyxy_pixels":[50,10,70,30]}}
        return frame,{"session_id":"s"},truth
    def test_unique_target_and_distractor_scores(self):
        frame,report,truth=self.fixture()
        self.assertEqual(_decision_label(frame,report,truth)[0],"CORRECT")
        truth["target"],truth["distractor"]=truth["distractor"],truth["target"]
        self.assertEqual(_decision_label(frame,report,truth)[0],"WRONG")
    def test_ambiguous_truth_never_assumed_correct(self):
        frame,report,truth=self.fixture()
        truth["distractor"]=copy.deepcopy(truth["target"])
        self.assertEqual(_decision_label(frame,report,truth)[0],"UNRESOLVED")
    def test_no_visible_match_unresolved(self):
        frame,report,truth=self.fixture()
        truth["target"]["bbox_xyxy_pixels"]=None
        self.assertEqual(_decision_label(frame,report,truth)[0],"UNRESOLVED")
    def test_changed_frame_candidate_rejected(self):
        frame,report,truth=self.fixture()
        frame["observer_record"]["decision"]["semantic_frame"]["image_sha256"]="b"*64
        self.assertEqual(_decision_label(frame,report,truth)[0],"UNRESOLVED")


if __name__=="__main__":
    unittest.main()
