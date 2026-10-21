"""Interface guards; synthetic checks are not live mission evidence."""
import copy,time,unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from dataclasses import replace
from fixed_map_yolo_identity import YoloAssociation,AssociationConfig,YoloImageBackend
from t1_semantic_perception import SemanticFrame,Candidate
from fixed_map_sensor_language import validate_proposal,validate_record,context
from fixed_map_sensor_mission import SensorMission
from fixed_map_perception_geometry import digest_json
from fixed_map_perception_task import live_estimate,build_preview,execution_evidence
from test_fixed_map_perception_execution import row,geometry,SAFETY,CFG
from fixed_map_perception import save

def frame(seq=1,candidates=1,t=None):
    return SemanticFrame('unit-session',3,f'f-{seq}',seq,'a'*64,640,480,seq*.2 if t is None else t,seq*.2+.01,tuple(Candidate(f'c-{seq}-{i}',(40+i*100,40,100+i*100,120),.9,'balloon',False) for i in range(candidates)),b'', '{}')
def proposal(action='start'):
    return dict(disposition='ACCEPT',action=action,vehicle='DroneA',target_description='气球' if action in ['start','preview'] else 'unspecified',requested_height_m=None,requested_speed_limit_m_s=None,reason='test')
def cfg():return dict(CFG,perception_backend='yolo_only_v8',association_confirmation_frames=3,ground_development_only=True,flight_authorized=False)
def yrow():
    r=row();e=r['position_estimate'];e.update(perception_backend='yolo_only_v8',track_id='local-1');r['temporal_identity']['track_id']='local-1'
    e['semantic_anchor']=dict(kind='yolo_current_rgb_attributes_v8',**{k:e[k] for k in ['session_id','target_generation','description_sha256','frame_id','image_sha256']},selected_candidate_id='own-candidate-1',track_id='local-1',consecutive_frames=3)
    return r
def record(config):
    r=dict(validated=True,context=context(SAFETY,config),proposal=proposal(),created_monotonic_s=time.monotonic());r['record_sha256']=digest_json(r);return r

class CpuThreadBudgetTests(unittest.TestCase):
    def test_predictor_reset_is_restored_after_warmup(self):
        from fixed_map_perception import restore_inference_threads
        from unittest.mock import MagicMock
        torch=MagicMock();torch.get_num_threads.side_effect=[8,2]
        receipt=restore_inference_threads({'torch_threads':2},torch)
        torch.set_num_threads.assert_called_once_with(2);self.assertEqual(receipt['actual_threads'],2);self.assertEqual(receipt['before_restoration_threads'],8)
    def test_invalid_or_ineffective_budget_is_rejected(self):
        from fixed_map_perception import restore_inference_threads
        from unittest.mock import MagicMock
        for value in [True,0,17,2.5]:
            torch=MagicMock()
            with self.assertRaisesRegex(ValueError,'INVALID'):restore_inference_threads({'torch_threads':value},torch)
            torch.set_num_threads.assert_not_called()
        torch=MagicMock();torch.get_num_threads.side_effect=[8,8]
        with self.assertRaisesRegex(ValueError,'NOT_APPLIED'):restore_inference_threads({'torch_threads':2},torch)

class YoloTests(unittest.TestCase):
    def setUp(self):self.p=patch('fixed_map_yolo_identity.current_attribute_audit',return_value=dict(candidates=[],status='REJECTED',reason='unused'));self.p.start();self.addCleanup(self.p.stop)
    def test_three_current_frames_required(self):
        t=YoloAssociation('unit-session',3,'气球');self.assertEqual([t.update(frame(i),now=i*.2+.01)['state'] for i in [1,2,3]],['ACQUIRING','ACQUIRING','TRACKING'])
    def test_ambiguous_never_selected(self):
        t=YoloAssociation('unit-session',3,'气球');r=t.update(frame(1,2),now=.21);self.assertEqual(r['state'],'LOST');self.assertIsNone(r['selected_candidate_id'])
    def test_absent_restarts_confirmation(self):
        t=YoloAssociation('unit-session',3,'气球')
        for i in [1,2,3]:t.update(frame(i),now=i*.2+.01)
        self.assertEqual(t.update(frame(4,0),now=.81)['state'],'LOST');self.assertEqual(t.update(frame(5),now=1.01)['consecutive_frames'],1)
    def test_stale_and_other_generation_rejected(self):
        for f,n in [(frame(),1.),(replace(frame(),target_generation=4),.21)]:self.assertEqual(YoloAssociation('unit-session',3,'气球').update(f,now=n)['state'],'LOST')
    def test_large_box_jump_restarts(self):
        t=YoloAssociation('unit-session',3,'气球');t.update(frame(),now=.21);f=frame(2);f=replace(f,candidates=(replace(f.candidates[0],bbox_xyxy_pixels=(300,200,360,280)),));self.assertEqual(t.update(f,now=.41)['consecutive_frames'],1)
    def test_duplicate_frame_does_not_confirm(self):
        t=YoloAssociation('unit-session',3,'气球');self.assertEqual([t.update(frame(),now=.21)['consecutive_frames'] for _ in range(3)],[1,1,1])
    def test_backend_never_constructs_semantic_client(self):
        with patch('vlx_api_client.VLXSeekAPIClient',side_effect=AssertionError('network client')):
            b=YoloImageBackend(Mock());self.assertFalse(hasattr(b,'vlx_client'))
            with self.assertRaises(RuntimeError):b(None)
    def test_anchor_digest_and_current_frame_required(self):
        c=cfg();live_estimate(yrow(),c,10.1)
        for key,v in [('frame_id','old'),('consecutive_frames',2),('track_id','other')]:
            r=yrow();r['position_estimate']['semantic_anchor'][key]=v
            with self.assertRaises(ValueError):live_estimate(r,c,10.1)
    def test_yolo_task_source_and_execution_binding(self):
        r=yrow();p=build_preview(r,[0,0,3.1],geometry(),SAFETY,cfg(),10.1);self.assertEqual(p['task']['source_kind'],'online_onboard_rgb_depth_yolo_attributes_v8');binding=execution_evidence(dict(estimate=r['position_estimate'],path=p['path']),r,cfg(),10.1);self.assertIn('YOLO',binding['source_kind'])
    def test_replay_not_executable(self):
        r=yrow();r['mode']='replay'
        with self.assertRaises(ValueError):live_estimate(r,cfg(),10.1)

class LanguageTests(unittest.TestCase):
    def test_allowed_commands(self):
        for a in ['start','preview','stop','return']:validate_proposal(proposal(a),SAFETY)
    def test_no_extra_code_or_coordinates(self):
        for k in ['python','goal_xyz_m']:
            p=proposal();p[k]='arbitrary'
            with self.assertRaises(ValueError):validate_proposal(p,SAFETY)
    def test_vehicle_and_parameter_rejected(self):
        for k,v in [('vehicle','DroneB'),('requested_height_m',5),('requested_speed_limit_m_s',2),('requested_height_m',True)]:
            p=proposal();p[k]=v
            with self.assertRaises(ValueError):validate_proposal(p,SAFETY)
    def test_sealed_record_stale_changed_config(self):
        c=cfg();r=record(c);validate_record(r,SAFETY,c)
        with self.assertRaises(ValueError):validate_record(r,SAFETY,c,r['created_monotonic_s']+121)
        with self.assertRaises(ValueError):validate_record(r,SAFETY,dict(c,target_generation=9))
        r['proposal']['vehicle']='DroneB'
        with self.assertRaises(ValueError):validate_record(r,SAFETY,c)
    def test_ground_mission_cannot_arm(self):
        c=cfg();e=Mock(stage='READY',safety=SAFETY,service=SimpleNamespace(cfg=dict(c,target_description='气球')))
        with self.assertRaisesRegex(ValueError,'FLIGHT_DISABLED'):SensorMission(e).submit(record(c),c)
        e.prepare_observation.assert_not_called()
    def test_preview_uses_sensor_interface(self):
        c=cfg();e=Mock(stage='READY',safety=SAFETY,service=SimpleNamespace(cfg=dict(c,target_description='气球')));r=record(c);r['proposal']=proposal('preview');r.pop('record_sha256');r['record_sha256']=digest_json(r)
        SensorMission(e).submit(r,c);e.plan_perception.assert_called_once_with(False,'language_sensor_preview_v8');e.prepare_observation.assert_not_called()
    def test_stop_cancels_mission(self):
        c=cfg();e=Mock(stage='APPROACH',safety=SAFETY);r=record(c);r['proposal']=proposal('stop');r.pop('record_sha256');r['record_sha256']=digest_json(r);m=SensorMission(e);m.active=True;m.submit(r,c);self.assertFalse(m.active);e.command.assert_called_once_with('stop','language_sensor_v8')

class RecorderTests(unittest.TestCase):
    def test_transient_windows_read_lock_is_bounded(self):
        original=Path.replace;calls=[]
        def replace(path,dest):
            calls.append(1)
            if len(calls)<=2:raise PermissionError('synthetic file share lock')
            return original(path,dest)
        with tempfile.TemporaryDirectory() as folder,patch.object(Path,'replace',replace),patch('fixed_map_perception.time.sleep') as sleep:
            p=Path(folder)/'publication.json';save(p,dict(valid=False,reason='test'));self.assertTrue(p.exists());self.assertEqual(sleep.call_count,2)
    def test_persistent_lock_is_failure_not_fake_success(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(Path,'replace',side_effect=PermissionError('synthetic persistent lock')),patch('fixed_map_perception.time.sleep') as sleep:
            p=Path(folder)/'publication.json'
            with self.assertRaises(PermissionError):save(p,dict(valid=True))
            self.assertFalse(p.exists());self.assertEqual(sleep.call_count,20)

class BoundImageTests(unittest.TestCase):
    def test_capture_and_inference_interleaving_reuses_exact_immutable_image(self):
        import cv2,hashlib,numpy as np
        from fixed_map_frame_image import decode_bound_rgb,_decode
        _decode.cache_clear();encoded=[cv2.imencode('.png',np.full((8,8,3),n,np.uint8))[1].tobytes() for n in [10,20,30]]
        source=decode_bound_rgb(encoded[0],hashlib.sha256(encoded[0]).hexdigest())
        for b in encoded[1:]:decode_bound_rgb(b,hashlib.sha256(b).hexdigest())
        self.assertIs(source,decode_bound_rgb(encoded[0],hashlib.sha256(encoded[0]).hexdigest()));self.assertFalse(source.flags.writeable);self.assertLessEqual(_decode.cache_info().currsize,3)
    def test_cache_keys_exact_bytes_and_is_readonly(self):
        import cv2,hashlib,numpy as np
        from fixed_map_frame_image import decode_bound_rgb
        encoded=cv2.imencode('.png',np.zeros((40,60,3),np.uint8))[1].tobytes();digest=hashlib.sha256(encoded).hexdigest()
        a=decode_bound_rgb(encoded,digest);self.assertIs(a,decode_bound_rgb(encoded,digest));self.assertFalse(a.flags.writeable)
        other=cv2.imencode('.png',np.full((40,60,3),255,np.uint8))[1].tobytes()
        self.assertIsNot(a,decode_bound_rgb(other,hashlib.sha256(other).hexdigest()))
        with self.assertRaises(ValueError):decode_bound_rgb(other,digest)
    def test_unique_rgb_audit_is_bound_once_without_changing_selection(self):
        from fixed_map_yolo_identity import select
        f=frame();audit=dict(status='REJECTED',reason='CURRENT_ATTRIBUTE_VISIBLE_BUT_INSTANCE_NOT_CONFIRMED',frame_id=f.frame_id,image_sha256=f.image_sha256,matching_candidate_ids=[f.candidates[0].candidate_id],candidates=[dict(candidate_id=f.candidates[0].candidate_id,matches_description=True)])
        with patch('fixed_map_yolo_identity.current_attribute_audit',return_value=audit) as check:
            selected,accepted=select(f,'带白色横条的红色气球');check.assert_called_once_with(f,'带白色横条的红色气球',None)
            self.assertEqual(selected,f.candidates[0].candidate_id);self.assertEqual(accepted['status'],'ACCEPTED');self.assertEqual(accepted['image_sha256'],f.image_sha256)

if __name__=='__main__':unittest.main()
