import unittest
import json
import numpy as np
from unittest.mock import patch
from types import SimpleNamespace
from fixed_map_target_profiles import instruction_target,yellow_evidence

class TargetProfileTests(unittest.TestCase):
    def test_language_cannot_change_yellow_to_red_or_preview_to_flight(self):
        from fixed_map_sensor_language import interpret
        cfg=dict(perception_backend='yolo_only_v8')
        safety=dict(height_world_m=3.1,max_command_m_s=.65)
        for text,description in [('接近黄色气球','红色气球'),('预览黄色气球的接近路径','黄色气球')]:
            proposal=dict(disposition='ACCEPT',action='start',vehicle='DroneA',target_description=description,
                          requested_height_m=None,requested_speed_limit_m_s=None,reason='mock')
            response=dict(message=dict(content=json.dumps(proposal)),done=True,done_reason='stop')
            import io
            with patch('fixed_map_sensor_language.memory_headroom',return_value=dict(physical_available_bytes=4*2**30,commit_available_bytes=4*2**30)), patch('urllib.request.urlopen',return_value=io.StringIO(json.dumps(response))):
                record=interpret(text,safety,cfg)
            self.assertFalse(record['validated'])
    def test_colors_are_preserved(self):
        self.assertEqual(instruction_target('无人机A接近黄色气球'),'黄色气球')
        self.assertEqual(instruction_target('预览普通红色气球'),'普通红色气球')
        self.assertEqual(instruction_target('接近带白色横条的红色气球'),'带白色横条的红色气球')
    def test_conflicting_and_unsupported_colors_are_refused(self):
        for text in ['接近红色和黄色气球','接近蓝色气球']:
            with self.assertRaises(ValueError):instruction_target(text)
    def test_yellow_profile_refuses_red_and_white(self):
        for bgr in [(0,0,255),(255,255,255),(255,0,0)]:
            image=np.full((60,60,3),bgr,np.uint8)
            self.assertFalse(yellow_evidence(image,[0,0,60,60])['matched'])
        image=np.full((60,60,3),(0,255,255),np.uint8)
        self.assertTrue(yellow_evidence(image,[0,0,60,60])['matched'])
    def test_small_and_truncated_boxes_refused(self):
        image=np.full((60,60,3),(0,255,255),np.uint8)
        self.assertFalse(yellow_evidence(image,[0,0,15,15])['matched'])
        self.assertFalse(yellow_evidence(image,[0,0,60,60],True)['matched'])
    def test_actual_attribute_selector_binds_only_yellow(self):
        from fixed_map_perception_execution import current_attribute_audit
        image=np.zeros((80,160,3),np.uint8)
        image[:,:80]=(0,255,255);image[:,80:]=(0,0,255)
        candidates=[SimpleNamespace(candidate_id='yellow',bbox_xyxy_pixels=(0,0,80,80),truncated=False),
                    SimpleNamespace(candidate_id='red',bbox_xyxy_pixels=(80,0,160,80),truncated=False)]
        frame=SimpleNamespace(frame_id='f1',image_sha256='a'*64,encoded_image=b'',height=80,width=160,candidates=candidates)
        with patch('fixed_map_frame_image.decode_bound_rgb',return_value=image):
            audit=current_attribute_audit(frame,'黄色气球','yellow')
        self.assertEqual(audit['status'],'ACCEPTED')
        self.assertEqual(audit['matching_candidate_ids'],['yellow'])

if __name__=='__main__':unittest.main()
