"""Synthetic interface checks; never counted as live perception evidence."""
import copy, json, queue, unittest
from pathlib import Path
import numpy as np
from fixed_map_perception_geometry import surface_from_box,validate_pair,sanitize_obstacles,make_proposal,preview_task,perception_safety_parameters
from fixed_map_mission import Geometry,Executor
ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/fixed_map_perception_v1.json').read_text(encoding='utf-8'))
TASK=json.loads((ROOT/'reports/fixed_map_flight_20261009/fixed_version/task.json').read_text(encoding='utf-8'))
def meta():
    pose=dict(position_local_ned_m=[0,0,0],orientation_xyzw=[0,0,0,1],world_from_local_translation_ned_m=[0,0,-3.1])
    return dict(rgb_timestamp_ns=1000000000,depth_timestamp_ns=1001000000,received_monotonic_s=10.,timestamps_advanced=True,capture_errors=[],rgb_pose=pose,depth_pose=copy.deepcopy(pose),rgb_K=[[800,0,800],[0,800,450],[0,0,1]],depth_K=[[320,0,320],[0,320,240],[0,0,1]],rgb_size_wh=[1600,900],depth_size_wh=[640,480])
def estimate():return dict(mode='online',session_id='test',target_generation=1,description_sha256='a'*64,frame_id='test-frame',image_sha256='b'*64,received_monotonic_s=10.,valid=True,target_state='TRACKING',semantic_review_deadline_monotonic_s=17.,surface_world_xyz_m=[10.,0.,3.1])
class PerceptionInterfaceTests(unittest.TestCase):
    def test_central_robust_pixel_is_actual_depth_and_not_center_estimate(self):
        cfg=dict(CFG,surface_point_policy='central_robust_actual_pixel_v11',surface_depth_band_floor_m=.04,surface_depth_band_mad_factor=3.)
        p=surface_from_box([700,350,900,550],np.full((480,640),10.),meta(),cfg,10.1)
        self.assertEqual(p['representative_depth_pixel_xy'],[320,240]);self.assertFalse(p['center_estimated']);self.assertEqual(p['selection']['policy'],cfg['surface_point_policy'])
    def test_central_depth_outlier_is_rejected_by_robust_band(self):
        cfg=dict(CFG,surface_point_policy='central_robust_actual_pixel_v11',surface_depth_band_floor_m=.04,surface_depth_band_mad_factor=3.)
        depth=np.full((480,640),10.);depth[240,320]=20.
        p=surface_from_box([700,350,900,550],depth,meta(),cfg,10.1);self.assertNotEqual(p['representative_depth_pixel_xy'],[320,240]);self.assertEqual(p['ray_range_m'],10.)
    def test_surface_policy_does_not_relax_expiry_or_accept_unknown_model(self):
        cfg=dict(CFG,surface_point_policy='unknown')
        with self.assertRaisesRegex(ValueError,'POLICY'):surface_from_box([700,350,900,550],np.full((480,640),10.),meta(),cfg,10.1)
        cfg.update(surface_point_policy='central_robust_actual_pixel_v11',surface_depth_band_floor_m=.04,surface_depth_band_mad_factor=3.)
        with self.assertRaisesRegex(ValueError,'EXPIRED'):surface_from_box([700,350,900,550],np.full((480,640),10.),meta(),cfg,10.6)
    def test_different_dimensions_use_actual_depth_sample_and_axes(self):
        p=surface_from_box([700,350,900,550],np.full((480,640),10.),meta(),CFG,10.1)
        self.assertTrue(p['valid']);self.assertFalse(p['center_estimated']);self.assertAlmostEqual(p['surface_world_xyz_m'][2],3.1,places=5)
        self.assertAlmostEqual(p['ray_range_m'],10);self.assertIn('NOT_balloon_center',p['meaning'])
    def test_off_axis_and_rotated_pose_world_conversion(self):
        m=meta();q=[0,0,np.sin(np.pi/4),np.cos(np.pi/4)];m['rgb_pose']['orientation_xyzw']=q;m['depth_pose']['orientation_xyzw']=q
        p=surface_from_box([1000,350,1200,550],np.full((480,640),10.),m,CFG,10.1)
        self.assertGreater(p['surface_world_xyz_m'][1],8);self.assertLess(p['surface_world_xyz_m'][0],0)
    def test_invalid_depth_and_missing_depth_coverage_rejected(self):
        for d in [np.full((480,640),np.nan),np.zeros((480,640)),np.ones((10,10))]:
            with self.assertRaises(ValueError):surface_from_box([700,350,900,550],d,meta(),CFG,10.1)
    def test_skew_stale_and_nonadvancing_marks_reject(self):
        m=meta();m['depth_timestamp_ns']+=30000000;self.assertIn('RGB_DEPTH_READBACK_SKEW_EXCEEDED',validate_pair(m,CFG,10.1))
        self.assertIn('CURRENT_FRAME_EXPIRED',validate_pair(meta(),CFG,10.6))
        m=meta();m['timestamps_advanced']=False;self.assertIn('NONADVANCING_RENDER_TIMESTAMPS',validate_pair(m,CFG,10.1))
        m=meta();m['render_age_sim_s']=.6;self.assertIn('RENDER_TIMESTAMP_STALE_OR_FUTURE',validate_pair(m,CFG,10.1))
    def test_target_geometry_changes_cannot_change_static_planner_input(self):
        safety=perception_safety_parameters(TASK)
        self.assertFalse(any(k in safety for k in ['goal_xyz_m','target_xyz_m','target_actor','target_source']))
        static=dict(id='wall',display_label='wall',shape='sphere',center_xyz_m=[3,8,3],radius_m=1,group='obstacles')
        target=dict(id='EL1_Balloon_Target',display_label='Target',shape='sphere',center_xyz_m=[48,-2,3.1],radius_m=1,group='balloons')
        a,_=sanitize_obstacles([static,target]);target['center_xyz_m']=[1,2,3];b,_=sanitize_obstacles([static,target]);self.assertEqual(a,b)
        self.assertEqual(a[0]['id'],'static-000')
    def test_preview_source_expiry_workspace_and_replay_boundaries(self):
        e=estimate();start=[0,0,.8];g=Geometry([dict(id='static-000',shape='sphere',center_xyz_m=[100,0,3.1],radius_m=1,group='obstacles')])
        proposal=make_proposal(e,start,TASK,CFG,10.1)
        result=preview_task(proposal,e,start,g,TASK,CFG,10.2);self.assertFalse(result['flight_authorized'])
        for key,value in [('image_sha256','c'*64),('target_generation',2),('goal_xyz_m',[52,0,3.1])]:
            bad=copy.deepcopy(proposal);bad[key]=value
            with self.assertRaises(ValueError):preview_task(bad,e,start,g,TASK,CFG,10.2)
        with self.assertRaisesRegex(ValueError,'EXPIRED'):preview_task(proposal,e,start,g,TASK,CFG,10.6)
        replay=copy.deepcopy(e);replay['mode']='replay'
        with self.assertRaisesRegex(ValueError,'REPLAY'):preview_task(proposal,replay,start,g,TASK,CFG,10.2)
    def test_blocked_goal_rejected_at_original_clearance(self):
        e=estimate();start=[0,0,.8];p=make_proposal(e,start,TASK,CFG,10.1)
        g=Geometry([dict(id='static-000',shape='sphere',center_xyz_m=p['goal_xyz_m'],radius_m=1,group='obstacles')])
        with self.assertRaises(ValueError):preview_task(p,e,start,g,TASK,CFG,10.2)
    def test_fixed_baseline_validation_is_preserved_and_preview_does_not_queue(self):
        ex=Executor.__new__(Executor);ex.task=TASK;ex.commands=queue.Queue()
        bad=copy.deepcopy(TASK);bad['goal_xyz_m']=[9,0,3.1]
        with self.assertRaisesRegex(ValueError,'PROPOSED_TASK'):ex.submit_task(bad)
        self.assertTrue(ex.commands.empty())
        e=estimate();start=[0,0,.8];p=make_proposal(e,start,TASK,CFG,10.1);g=Geometry([dict(id='static-000',shape='sphere',center_xyz_m=[100,0,3.1],radius_m=1,group='obstacles')])
        ex.submit_task(p,'perception_preview',dict(estimate=e,start=start,geometry=g,cfg=CFG,now=10.2));self.assertTrue(ex.commands.empty());self.assertEqual(ex.task,TASK)
if __name__=='__main__':unittest.main()
