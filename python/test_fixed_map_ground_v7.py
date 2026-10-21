"""Ground/replay isolation, bounded multiscale NMS, and engine-lifetime guards."""
import copy,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from fixed_map_multiscale_detector import MultiScaleBalloonDetector
from fixed_map_scene_binding import bind_scene_config,engine_identity_alive
from fixed_map_perception_geometry import make_replay_preview
from fixed_map_perception_task import live_estimate
from fixed_map_perception_execution import PerceptionExecutor
from test_fixed_map_perception_execution import row,geometry,CFG,SAFETY
class GroundTests(unittest.TestCase):
    def test_multiscale_deduplicates_and_restores_original_size(self):
        class Detector:
            imgsz=1280;nms_iou=.5
            def detect(self,image):return dict(candidates=[dict(confidence=.9 if self.imgsz==640 else .8,bbox_xyxy_pixels=[10.,10.,30.,30.])],candidate_count=1,source='fake')
        d=Detector();r=MultiScaleBalloonDetector(d).detect(np.zeros((80,100,3),np.uint8));self.assertEqual(r['candidate_count'],1);self.assertEqual(r['candidates'][0]['inference_imgsz'],640);self.assertEqual(d.imgsz,1280)
    def test_dual_scale_inference_failure_is_not_partial_success(self):
        class Detector:
            imgsz=1280;nms_iou=.5
            def detect(self,image):
                if self.imgsz==640:raise ValueError('synthetic failure')
                return dict(candidates=[])
        d=Detector()
        with self.assertRaises(ValueError):MultiScaleBalloonDetector(d).detect(np.zeros((80,100,3),np.uint8))
        self.assertEqual(d.imgsz,1280)
    def test_engine_restart_rejects_old_public_receipt(self):
        with patch('fixed_map_scene_binding.current_engine_identity',return_value=dict(pid=2,created=20)):
            with self.assertRaisesRegex(ValueError,'LIFETIME_CHANGED'):bind_scene_config({},dict(status='PASS_RUNTIME_RANDOM_SCENE',scene_version='old',engine_identity=dict(pid=1,created=10)))
    def test_failed_scene_receipt_cannot_bind(self):
        with self.assertRaisesRegex(ValueError,'NOT_VALID'):bind_scene_config({},dict(status='INVALID_ENGINE_RESTART_PENDING_REBIND'))
    def test_pid_reuse_not_treated_as_same_engine(self):
        with patch('fixed_map_scene_binding.psutil.Process',return_value=SimpleNamespace(create_time=lambda:20)):
            self.assertFalse(engine_identity_alive(dict(pid=1,created=10)))
    def test_replay_preview_preserves_mode_and_live_interface_rejects(self):
        r=row();r['mode']=r['position_estimate']['mode']='replay'
        result=make_replay_preview(r['position_estimate'],[0.,0.,3.1],geometry(),SAFETY,CFG,10.1)
        self.assertEqual(result['status'],'REPLAY_STATIC_PREVIEW_VALID');self.assertFalse(result['execution_allowed']);self.assertEqual(r['position_estimate']['mode'],'replay')
        with self.assertRaisesRegex(ValueError,'ONLINE'):live_estimate(r,CFG,10.1)
    def test_ground_only_blocks_execution_before_planning_or_motion(self):
        e=PerceptionExecutor.__new__(PerceptionExecutor);e.service=SimpleNamespace(cfg=dict(ground_development_only=True))
        with self.assertRaisesRegex(ValueError,'GROUND_DEVELOPMENT_EXECUTION_DISABLED'):e.request_plan(True,'synthetic')
    def test_live_estimate_rejects_engine_death_even_matching_scene(self):
        r=row();cfg=dict(CFG,scene_epoch_required=True,runtime_engine_identity=dict(pid=1,created=10));r['capture']['runtime_engine_identity']=cfg['runtime_engine_identity']
        with patch('fixed_map_scene_binding.engine_identity_alive',return_value=False):
            with self.assertRaisesRegex(ValueError,'LIFETIME_CHANGED'):live_estimate(r,cfg,10.1)
if __name__=='__main__':unittest.main()
