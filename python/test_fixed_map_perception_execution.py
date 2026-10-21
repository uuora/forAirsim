"""Bounded synthetic task/stop checks; NOT live flight evidence."""
import copy,hashlib,unittest
from types import SimpleNamespace
from unittest.mock import MagicMock,patch
import numpy as np
from fixed_map_perception_task import live_estimate,build_preview,revalidate_preview,execution_evidence
from fixed_map_perception_execution import PerceptionExecutor,static_geometry_v2
from fixed_map_mission import Geometry
from fixed_map_perception import ROOT,read
from test_fixed_map_perception import meta

SAFETY=read(ROOT/'reports/fixed_map_flight_20261009/fixed_version/task.json')
CFG=dict(read(ROOT/'configs/fixed_map_perception_v1.json'),target_generation=2,target_description='带白色横条的红色气球',surface_stability_max_m=.25,sensor_surface_exclusion_m=3.25)

def row(receipt=10.,frame='source-1',surface=None):
    surface=surface or [10.,0.,3.1];m=meta();m.update(mode='online',session_id='own-online',target_generation=2,description_sha256=hashlib.sha256(CFG['target_description'].encode()).hexdigest(),frame_id=frame,image_sha256='b'*64,received_monotonic_s=receipt)
    p=dict(valid=True,surface_world_xyz_m=surface,center_estimated=False)
    e={k:m[k] for k in ['mode','session_id','target_generation','description_sha256','frame_id','image_sha256','received_monotonic_s']}
    e.update(valid=True,target_state='TRACKING',semantic_review_deadline_monotonic_s=17.,surface_world_xyz_m=surface,meaning='visible_surface_point_not_center',surface=p,semantic_anchor='cloud-request-1')
    return dict(mode='online',valid=True,capture=m,position_estimate=e,semantic_frame=dict(frame_id=frame,image_sha256='b'*64),temporal_identity=dict(selected_candidate_id='own-candidate-1'),current_attribute_audit=dict(status='ACCEPTED',frame_id=frame,image_sha256='b'*64,selected_candidate_id='own-candidate-1'),candidate_positions=[dict(candidate_id='own-candidate-1',position=p)])

def geometry():return Geometry([dict(id='static-000',shape='sphere',center_xyz_m=[100,0,3.1],radius_m=1,group='obstacles')])

class ExecutionTests(unittest.TestCase):
    def test_recovery_height_handoff_waits_without_relaxing_plane_guard(self):
        import tempfile
        from pathlib import Path
        from concurrent.futures import Future
        r=row();preview=build_preview(r,[0,0,3.1],geometry(),SAFETY,CFG,10.1);future=Future();future.set_result(preview)
        e=self.recovery_owner();e.service=SimpleNamespace(latest=r,cfg=dict(CFG,plan_fresh_handoff_timeout_s=2.,recovery_height_handoff_wait_v17=True));e.static_geometry=geometry();e.perception_mode=True;e.pending_plan=dict(future=future,execute=True,actor='fresh_reconfirmation_replan',submitted_monotonic_s=10.,validation_finished_monotonic_s=10.);e.plan_version=0;e.run_dir=None;e.raw_state=SimpleNamespace(timestamp=1000);e.reason=''
        with patch('fixed_map_perception_execution.time.monotonic',return_value=10.1):e.poll_plan()
        self.assertEqual(e.preview_result['status'],'WAITING_FRESH_REVALIDATION');self.assertEqual(e.preview_result['reason'],'SELF_NOT_ON_FIXED_HEIGHT_PLANE');self.assertEqual(e.stage,'ABORTED')
        e.p=np.array([0,0,3.095]);e.service.latest=row(10.15,'new-fresh')
        with tempfile.TemporaryDirectory() as temp,patch('fixed_map_perception_execution.mission.OUT',Path(temp)),patch('fixed_map_perception_execution.time.monotonic',return_value=10.2):e.poll_plan()
        self.assertEqual(e.stage,'APPROACH');self.assertEqual(e.active_plan['estimate']['frame_id'],'new-fresh');self.assertIsNone(e.pending_plan)
    def test_recovery_height_handoff_cannot_extend_existing_deadline(self):
        from concurrent.futures import Future
        r=row();future=Future();future.set_result(build_preview(r,[0,0,3.1],geometry(),SAFETY,CFG,10.1))
        e=self.recovery_owner();e.service=SimpleNamespace(latest=row(12.1,'fresh-but-wrong-height'),cfg=dict(CFG,plan_fresh_handoff_timeout_s=2.,recovery_height_handoff_wait_v17=True));e.static_geometry=geometry();e.perception_mode=True;e.reason='';e.pending_plan=dict(future=future,execute=True,actor='recovery',submitted_monotonic_s=10.,validation_finished_monotonic_s=10.,finished_monotonic_s=10.1,handoff_deadline_monotonic_s=12.1)
        with patch('fixed_map_perception_execution.time.monotonic',return_value=12.2):e.poll_plan()
        self.assertEqual(e.preview_result['status'],'REJECTED');self.assertEqual(e.preview_result['reason'],'SELF_NOT_ON_FIXED_HEIGHT_PLANE');self.assertIsNone(e.pending_plan);self.assertEqual(e.stage,'ABORTED')
    def initial_wait_owner(self):
        e=self.recovery_owner();e.stage='ARRIVED';e.perception_mode=False;e.service=SimpleNamespace(cfg=dict(initial_plan_fresh_wait_v14=True,plan_fresh_handoff_timeout_s=2.));e.pending_plan=None;e.reason=''
        return e
    def test_initial_expiry_waits_for_new_evidence_without_motion(self):
        e=self.initial_wait_owner();e.request_plan=MagicMock(side_effect=ValueError('CURRENT_FRAME_EXPIRED'))
        with patch('fixed_map_perception_execution.time.monotonic',return_value=10.):e.process_request(dict(action='execute_perception',actor='language'))
        self.assertEqual(e.preview_result['status'],'WAITING_FRESH_INITIAL_PLAN');self.assertEqual(e.initial_plan_request['deadline'],12.)
        e.request_plan=MagicMock()
        with patch('fixed_map_perception_execution.time.monotonic',return_value=10.1):e.poll_initial_plan_request()
        e.request_plan.assert_called_once_with(True,'language');self.assertIsNone(e.initial_plan_request)
    def test_initial_wait_is_bounded_and_geometry_refusal_is_terminal(self):
        e=self.initial_wait_owner();e.request_plan=MagicMock(side_effect=ValueError('CURRENT_FRAME_EXPIRED'))
        with patch('fixed_map_perception_execution.time.monotonic',return_value=10.):e.process_request(dict(action='execute_perception',actor='language'))
        with patch('fixed_map_perception_execution.time.monotonic',return_value=12.):e.poll_initial_plan_request()
        self.assertEqual(e.preview_result['reason'],'INITIAL_FRESH_PLAN_HANDOFF_DEADLINE');self.assertIsNone(e.initial_plan_request)
        e.request_plan=MagicMock(side_effect=ValueError('REBOUND_PATH_CLEARANCE_FAILED'));e.process_request(dict(action='execute_perception',actor='language'))
        self.assertEqual(e.preview_result['status'],'REJECTED');self.assertIsNone(e.initial_plan_request)
    def test_v13_initial_expiry_still_rejects_without_opt_in(self):
        e=self.initial_wait_owner();e.service.cfg.pop('initial_plan_fresh_wait_v14');e.request_plan=MagicMock(side_effect=ValueError('CURRENT_FRAME_EXPIRED'))
        e.process_request(dict(action='execute_perception',actor='language'));self.assertEqual(e.preview_result['status'],'REJECTED');self.assertIsNone(getattr(e,'initial_plan_request',None))
    def stationary_owner(self):
        e=self.recovery_owner();e.p=np.array([0.,0.,3.092]);e.v=np.array([0.,0.,-.006]);e.vcmd=np.zeros(3)
        e.perception_mode=True;e.service=SimpleNamespace(cfg=dict(recovery_policy_version='fixed_height_stationary_hold_v12'));e.g=geometry();e.c=MagicMock();e.api=MagicMock();e.ground_b=[0,2,0]
        e.state.update(landed=False,state_timestamp_ns=1000,DroneB_position_xyz_m=[0,2,0])
        return e
    def test_stationary_recovery_holds_actual_xy_and_original_height(self):
        e=self.stationary_owner();e.active_plan=dict(task=dict(goal_xyz_m=[38,5,3.1]))
        with patch('fixed_map_perception_execution.time.monotonic',return_value=10.):e.stationary_recovery_hold()
        command=e.c.moveByVelocityAsync.call_args
        self.assertEqual(command.args[:2],(0.,0.));self.assertLess(command.args[2],0.)
        self.assertEqual(e.stationary_hold_origin.tolist(),[0,0,3.1]);self.assertLessEqual(np.linalg.norm(e.vcmd)/.1,.4000001)
    def test_stationary_recovery_requires_fresh_state_before_any_new_command(self):
        e=self.stationary_owner()
        with patch('fixed_map_perception_execution.time.monotonic',return_value=10.):e.stationary_recovery_hold()
        e.c.reset_mock()
        with patch('fixed_map_perception_execution.time.monotonic',return_value=10.1):
            with self.assertRaisesRegex(ValueError,'STALE_STATE'):e.stationary_recovery_hold()
        e.c.moveByVelocityAsync.assert_not_called()
    def test_stationary_recovery_rejects_blocked_column_and_state_gap(self):
        e=self.stationary_owner();e.g.segment=MagicMock(return_value=(False,{}))
        with self.assertRaisesRegex(ValueError,'COLUMN_BLOCKED'):e.stationary_recovery_hold()
        e.c.moveByVelocityAsync.assert_not_called();e.g=geometry();e.stationary_hold_previous=1.
        with patch('fixed_map_perception_execution.time.monotonic',return_value=2.):
            with self.assertRaisesRegex(ValueError,'STATE_GAP'):e.stationary_recovery_hold()
        e.c.moveByVelocityAsync.assert_not_called()
    def test_stationary_recovery_does_not_run_for_v11_or_after_return(self):
        e=self.stationary_owner();e.service.cfg['recovery_policy_version']='fixed_height_new_fresh_plan_v11';e.stationary_recovery_hold()
        e.service.cfg['recovery_policy_version']='fixed_height_stationary_hold_v12';e.stage='RETURN';e.stationary_recovery_hold()
        e.c.moveByVelocityAsync.assert_not_called();self.assertIsNone(e.stationary_hold_previous)
    def test_stationary_recovery_rejects_horizontal_drift_and_drone_b_motion(self):
        e=self.stationary_owner();e.stationary_hold_origin=np.array([.3,0,3.1])
        with self.assertRaisesRegex(ValueError,'XY_DRIFT'):e.stationary_recovery_hold()
        e.stationary_hold_origin=None;e.state['DroneB_position_xyz_m']=[0,2.2,0]
        with self.assertRaisesRegex(ValueError,'DRONE_B_GROUND'):e.stationary_recovery_hold()
        e.c.moveByVelocityAsync.assert_not_called()
    def recovery_owner(self):
        e=PerceptionExecutor.__new__(PerceptionExecutor);e.stage='ABORTED';e.p=np.array([0.,0.,3.05]);e.safety=SAFETY;e.state=dict(speed_m_s=0.,DroneB_landed=True,DroneB_api_control=False);e.event=MagicMock();e.sample=MagicMock()
        return e
    def test_recovery_reuses_height_controller_before_restoring_stop_state(self):
        e=self.recovery_owner();e.align_return_height=MagicMock(side_effect=lambda:setattr(e,'p',np.array([0.,0.,3.095])))
        e.restore_recovery_plane();e.align_return_height.assert_called_once();self.assertEqual(e.stage,'ABORTED');self.assertIsNone(e.prev)
    def test_recovery_rejects_uncleared_column_without_planning(self):
        e=self.recovery_owner();e.g=MagicMock();e.g.segment.return_value=(False,{});e.c=MagicMock()
        with self.assertRaisesRegex(ValueError,'COLUMN_BLOCKED'):e.restore_recovery_plane()
        e.c.moveByVelocityAsync.assert_not_called();self.assertEqual(e.stage,'ABORTED')
    def test_recovery_does_not_relax_height_tolerance(self):
        e=self.recovery_owner();e.align_return_height=MagicMock()
        with self.assertRaisesRegex(ValueError,'STABILITY'):e.restore_recovery_plane()
    def test_evidence_expiring_during_alignment_cannot_start_plan(self):
        e=self.recovery_owner();e.service=SimpleNamespace(latest=row(),cfg=CFG);e.perception_mode=True;e.active_plan=dict(estimate=row()['position_estimate']);e.pending_plan=None;e.baseline_mode=False;e.planner=MagicMock()
        e.restore_recovery_plane=MagicMock(side_effect=lambda:e.service.latest.update(valid=False))
        with patch('fixed_map_perception_execution.time.monotonic',return_value=10.1):
            with self.assertRaisesRegex(ValueError,'NO_VALID'):e.request_plan(True,'fresh_reconfirmation_replan')
        e.planner.submit.assert_not_called()
    def test_recovery_rebuilds_from_current_surface_not_cached_path(self):
        import fixed_map_perception_execution as adapter
        from fixed_map_perception_task import build_preview
        e=self.recovery_owner();e.p=np.array([0.,0.,3.1]);e.service=SimpleNamespace(latest=row(surface=[11,0,3.1]),cfg=CFG);e.perception_mode=True;e.active_plan=dict(estimate=row()['position_estimate']);e.pending_plan=None;e.baseline_mode=False;e.planner=MagicMock();e.source_epoch=adapter.mission.sha(adapter.__file__);e.static_geometry=geometry();e.preview_result=dict(status='FRESH_REVALIDATED_PERCEPTION_PLAN',task={},path=[[0,0,3.1],[6.5,0,3.1]],estimate=row()['position_estimate'])
        with patch('fixed_map_perception_execution.time.monotonic',return_value=10.1):e.request_plan(True,'fresh_reconfirmation_replan')
        self.assertIs(e.planner.submit.call_args.args[0],build_preview);self.assertEqual(e.planner.submit.call_args.args[1]['position_estimate']['surface_world_xyz_m'],[11,0,3.1])
    def test_live_expired_replay_lost_and_unconfirmed_reject(self):
        for changes in [dict(mode='replay'),dict(valid=False)]:
            r=row();r.update(changes)
            with self.assertRaises(ValueError):live_estimate(r,CFG,10.1)
        with self.assertRaisesRegex(ValueError,'EXPIRED'):live_estimate(row(),CFG,10.6)
        for state in ['LOST','UNCONFIRMED','AMBIGUOUS']:
            r=row();r['position_estimate']['target_state']=state
            with self.assertRaises(ValueError):live_estimate(r,CFG,10.1)
    def test_version_hash_and_depth_bindings_fail_closed(self):
        for key,value in [('target_generation',1),('description_sha256','a'*64),('image_sha256','c'*64)]:
            r=row();r['position_estimate'][key]=value
            with self.assertRaises(ValueError):live_estimate(r,CFG,10.1)
        r=row();r['candidate_positions'][0]['position']=dict(r['candidate_positions'][0]['position'],surface_world_xyz_m=[11,0,3.1])
        with self.assertRaisesRegex(ValueError,'DEPTH_SOURCE'):live_estimate(r,CFG,10.1)
    def test_fresh_current_attribute_required_after_delayed_reply(self):
        for changes in [dict(status='REJECTED'),dict(frame_id='old-frame'),dict(selected_candidate_id='wrong-local-candidate')]:
            r=row();r['current_attribute_audit'].update(changes)
            with self.assertRaises(ValueError):live_estimate(r,CFG,10.1)
    def test_old_semantic_source_expiry_rejected(self):
        r=row();r['position_estimate']['semantic_review_deadline_monotonic_s']=10.05
        with self.assertRaisesRegex(ValueError,'SEMANTIC_EVIDENCE_EXPIRED'):live_estimate(r,CFG,10.1)
    def test_long_planning_rebinds_new_frame_without_rebasing_old_source(self):
        old=row();p=build_preview(old,[0,0,3.1],geometry(),SAFETY,CFG,10.1)
        with self.assertRaisesRegex(ValueError,'EXPIRED'):revalidate_preview(p,old,[0,0,3.1],geometry(),SAFETY,CFG,10.8)
        result=revalidate_preview(p,row(10.7,'source-2',[10.1,0,3.1]),[0,0,3.1],geometry(),SAFETY,CFG,10.8)
        self.assertEqual(result['estimate']['frame_id'],'source-2');self.assertEqual(p['planned_from_estimate']['received_monotonic_s'],10.)
        self.assertAlmostEqual(result['task']['goal_xyz_m'][0],6.6)
    def test_changed_goal_and_context_require_new_planning(self):
        p=build_preview(row(),[0,0,3.1],geometry(),SAFETY,CFG,10.1)
        with self.assertRaisesRegex(ValueError,'SURFACE_CHANGED'):revalidate_preview(p,row(10.2,'source-2',[11,0,3.1]),[0,0,3.1],geometry(),SAFETY,CFG,10.3)
        r=row(10.2,'source-2');r['position_estimate']['session_id']=r['capture']['session_id']='other-session'
        with self.assertRaisesRegex(ValueError,'CONTEXT'):revalidate_preview(p,r,[0,0,3.1],geometry(),SAFETY,CFG,10.3)
    def test_unknown_candidate_surface_rejects_path(self):
        r=row();r['candidate_positions'].append(dict(candidate_id='distractor',position=dict(valid=True,surface_world_xyz_m=[5,0,3.1])))
        with self.assertRaisesRegex(ValueError,'SURFACE_EXCLUSION'):build_preview(r,[0,0,3.1],geometry(),SAFETY,CFG,10.1)
    def test_geometry_retains_physical_target_platform_without_semantic_label(self):
        parts=read(ROOT/'reports/fixed_map_flight_20261009/fixed_version/editor_audit/planning_geometry.json')['primitives']
        p=next(p for p in parts if 'targetplatform' in p['id'].lower());kept,removed=static_geometry_v2(parts)
        found=[k for k in kept if k['center_xyz_m']==p['center_xyz_m'] and k.get('local_size_xyz_m')==p.get('local_size_xyz_m')]
        self.assertTrue(found);self.assertNotIn(p['id'],removed)
        self.assertTrue(all('target' not in k['id'].lower() and k['group']!='balloons' for k in kept))
    def test_invalid_evidence_cancels_old_motion_and_hovers(self):
        e=PerceptionExecutor.__new__(PerceptionExecutor);e.pending_plan=None;e.perception_mode=True;e.baseline_mode=False;e.active_plan=dict(plan_version=1);e.stage='APPROACH';e.service=SimpleNamespace(latest=dict(valid=False),cfg=CFG)
        e.events=[];e.event=lambda kind,**kw:e.events.append((kind,kw));e.c=MagicMock();e.p=np.array([0,0,3.1]);e.vcmd=np.ones(3);e.reason='';e.hold_since=None;e.prev=None;e.safety=e.task=SAFETY
        e.ground_b=[0,0,0];e.state=dict(DroneB_api_control=False,DroneB_landed=True,DroneB_position_xyz_m=[0,0,0],speed_m_s=0.,state_timestamp_ns=1000);e.raw_state=SimpleNamespace(timestamp=1000)
        e.control();self.assertEqual(e.stage,'STOP_HOVER');e.c.cancelLastTask.assert_called_once_with(vehicle_name='DroneA');e.c.hoverAsync.assert_called_once_with(vehicle_name='DroneA');self.assertTrue(e.resume_required)
    def test_synthetic_changed_estimate_changes_task_goal_not_fixed_baseline(self):
        baseline=copy.deepcopy(SAFETY)
        a=build_preview(row(),[0,0,3.1],geometry(),SAFETY,CFG,10.1)
        b=build_preview(row(surface=[12,2,3.1]),[0,0,3.1],geometry(),SAFETY,CFG,10.1)
        self.assertNotEqual(a['task']['goal_xyz_m'],b['task']['goal_xyz_m']);self.assertEqual(SAFETY,baseline)

if __name__=='__main__':unittest.main()
