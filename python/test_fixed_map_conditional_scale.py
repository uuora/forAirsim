import unittest
from unittest.mock import MagicMock,patch
from types import SimpleNamespace
import numpy as np
from fixed_map_conditional_scale import ConditionalScaleBalloonDetector


def candidate(box,score=.8):return dict(bbox_xyxy_pixels=box,confidence=score,truncated=False,class_name='balloon')
def report(candidates):return dict(candidates=candidates,candidate_count=len(candidates),designated_instance_identity_established=False)


class ConditionalScaleTests(unittest.TestCase):
    def test_yellow_profile_starts_high_resolution_and_preserves_other_candidates(self):
        seen=[]
        yellow=candidate([0,0,40,40]);red=candidate([50,0,90,40])
        detector=SimpleNamespace(imgsz=640)
        def detect(image):
            seen.append(detector.imgsz)
            return report([yellow,red])
        detector.detect=detect
        with patch('fixed_map_target_profiles.yellow_evidence',side_effect=lambda image,box,truncated:dict(matched=box[0]==0)):
            result=ConditionalScaleBalloonDetector(detector,description='黄色气球').detect(np.zeros((100,100,3),np.uint8))
        self.assertEqual(seen,[1280])
        self.assertEqual(len(result['candidates']),2)
        self.assertFalse(result['designated_instance_identity_established'])
    def test_existing_matching_candidates_are_preserved_including_ambiguity(self):
        original=report([candidate([0,0,30,30]),candidate([40,0,70,30])]);detector=SimpleNamespace(imgsz=640,detect=MagicMock(return_value=original))
        with patch('fixed_map_conditional_scale.band_evidence',return_value=dict(status='ACCEPTED')):
            result=ConditionalScaleBalloonDetector(detector).detect(np.zeros((90,90,3),np.uint8))
        self.assertEqual(result['candidates'],original['candidates']);self.assertEqual(result['candidate_count'],2);detector.detect.assert_called_once();self.assertFalse(result['designated_instance_identity_established'])
    def test_fallback_failure_restores_primary_scale(self):
        detector=SimpleNamespace(imgsz=640,detect=MagicMock(side_effect=[report([candidate([0,0,40,40])]),RuntimeError('predictor failed')]))
        with patch('fixed_map_conditional_scale.band_evidence',return_value=dict(status='REJECTED')):
            with self.assertRaisesRegex(RuntimeError,'predictor failed'):ConditionalScaleBalloonDetector(detector).detect(np.zeros((90,90,3),np.uint8))
        self.assertEqual(detector.imgsz,640)
    def test_empty_640_recovers_current_target_at_960(self):
        found=candidate([0,0,40,40])
        detector=SimpleNamespace(imgsz=640,detect=MagicMock(side_effect=[report([]),report([found])]))
        with patch('fixed_map_conditional_scale.band_evidence',return_value=dict(status='ACCEPTED')):
            result=ConditionalScaleBalloonDetector(detector).detect(np.zeros((90,90,3),np.uint8))
        self.assertEqual(detector.detect.call_count,2)
        self.assertEqual(result['candidates'],[found])
        self.assertEqual(result['conditional_scale_receipt']['next_primary_imgsz'],960)
        self.assertFalse(result['designated_instance_identity_established'])
    def test_duplicate_geometry_prefers_observed_band_without_creating_two_tracks(self):
        a=candidate([0,0,40,40],.9);b=candidate([1,1,41,41],.8)
        detector=SimpleNamespace(imgsz=640,detect=MagicMock(side_effect=[report([a]),report([b])]))
        def band(image,box):return dict(status='ACCEPTED' if box[0]==1 else 'REJECTED')
        with patch('fixed_map_conditional_scale.band_evidence',side_effect=band):result=ConditionalScaleBalloonDetector(detector).detect(np.zeros((90,90,3),np.uint8))
        self.assertEqual(result['candidate_count'],1);self.assertEqual(result['candidates'][0],b);self.assertEqual(a['bbox_xyxy_pixels'],[0,0,40,40]);self.assertFalse(result['designated_instance_identity_established'])
    def test_other_primary_scale_is_refused(self):
        with self.assertRaises(ValueError):ConditionalScaleBalloonDetector(SimpleNamespace(imgsz=960))
    def test_resolution_memory_does_not_reuse_a_past_candidate(self):
        a=candidate([0,0,40,40]);b=candidate([1,1,41,41])
        detector=SimpleNamespace(imgsz=640,detect=MagicMock(side_effect=[report([a]),report([b]),report([])]))
        def band(image,box):return dict(status='ACCEPTED' if box[0]==1 else 'REJECTED')
        wrapper=ConditionalScaleBalloonDetector(detector)
        with patch('fixed_map_conditional_scale.band_evidence',side_effect=band):
            first=wrapper.detect(np.zeros((90,90,3),np.uint8));second=wrapper.detect(np.ones((90,90,3),np.uint8))
        self.assertEqual(first['conditional_scale_receipt']['next_primary_imgsz'],960)
        self.assertEqual(second['conditional_scale_receipt']['primary_imgsz'],960);self.assertEqual(second['candidate_count'],0);self.assertEqual(detector.imgsz,640);self.assertEqual(detector.detect.call_count,3)


if __name__=='__main__':unittest.main()
