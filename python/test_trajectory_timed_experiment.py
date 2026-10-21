import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import cv2
import numpy as np
from perception_trajectory_planner import PlanningParams,plan_trajectory
from trajectory_single_experiment import ProtectionRequired,TrajectorySingleExperiment
from trajectory_timed_experiment import TimedTrajectoryExperiment


class Clock:
    def __init__(self):self.now=100.
    def __call__(self):return self.now
    def advance(self,value):self.now+=value


class TimedExperimentTests(unittest.TestCase):
    def test_capture_crossing_inner_boundary_is_saved_then_protected_before_planning(self):
        from concurrent.futures import ThreadPoolExecutor
        with TemporaryDirectory() as folder,ThreadPoolExecutor(max_workers=1) as writer:
            clock=Clock();run,state,_,_=self.run_fixture(folder,clock)
            run._maintain_fixed_goal=Mock();run.media_futures=[];run.media_writer=writer
            run.report['observations']=[]
            run.target_source='sensor_surface_estimate'
            run.tracker=Mock();run.tracker.update.return_value=SimpleNamespace(state=SimpleNamespace(value='DETECTED'))
            positions=iter(([-5,.42,-3],[-5,.46,-3]))
            run.state.side_effect=lambda **kw:(np.array(next(positions)),0.,state)
            observation=SimpleNamespace(valid=True,detected=True,truncated=False,candidate_count=1,confidence=.7,
                normalised_error_xy=[0,0],depth_m=3.,image=np.zeros((2,2,3),dtype=np.uint8),
                depth=np.ones((2,2),dtype=np.float32),state_sample=None,age_at=lambda:.1,metadata=lambda:{})
            run.timestamped_camera=Mock();run.timestamped_camera.capture.return_value=observation
            with patch('trajectory_timed_experiment.time.monotonic',clock),self.assertRaisesRegex(ProtectionRequired,'ENTERED_PROTECTION_MARGIN'):
                run._sample_timestamped()
            for future in run.media_futures:future.result()
            self.assertEqual(len(run.report['observations']),1)
            self.assertEqual(run.report['observations'][0]['media_persistence']['status'],'SAVED')
            self.assertIn('ACTUAL_POSITION_ENTERED_PROTECTION_MARGIN',run.report['observations'][0]['planning_input_body_state']['protection_reasons'])
            run.client.moveByVelocityZAsync.assert_not_called()

    def run_fixture(self,folder,clock):
        run=object.__new__(TimedTrajectoryExperiment)
        run.cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/trajectory_guarded_pilot_v5p1/target_y_plus_0p4.json').read_text(encoding='utf-8'))
        run.planning=run.cfg['planning'];run.guard_cfg=run.planning['execution_envelope']
        run.schedule=dict(next_cycle_budget_s=.17,rpc_overhead_budget_s=.035,state_query_budget_s=.015)
        run.ramp_time_basis='fixed_sample_quantum';run.last_approach_request_host_s=None
        run.context=None;run.pending_reobservation=None;run.active_observation=None
        run.active_xy_command=np.zeros(2);run.freshness_replans=0;run.phase='APPROACH'
        run.fixed_height_execution=True;run.origin_translation=np.array([0,-3,-.8535])
        run.folder=Path(folder);run.vehicle='DroneA';run._record_path=Mock()
        run.report=dict(timed_command_records=[],timing_events=[],execution_guard_samples=[],goal_keepalive_records=[],
            freshness_replan_events=[],settling_records=[])
        run.client=Mock()
        def vec(v):return SimpleNamespace(**dict(zip(('x_val','y_val','z_val'),v)))
        state=SimpleNamespace(timestamp=100,kinematics_estimated=SimpleNamespace(
            position=vec([-5,-.5,-3]),linear_velocity=vec([0,0,0]),linear_acceleration=vec([0,0,0]),
            orientation=SimpleNamespace(x_val=0,y_val=0,z_val=0,w_val=1)))
        def query(**kw):
            clock.advance(.004)
            state.timestamp+=1
            return np.array([-5.,-.5,-3.]),0.,state
        run.state=Mock(side_effect=query)
        run.client.moveByVelocityZAsync.return_value.join.side_effect=lambda:clock.advance(.08)
        plan=plan_trajectory(position=[-5,-.5,-3],velocity=[0,0,0],acceleration=[0,0,0],
            target_position=[0,0,-3],target_velocity=[0,0,0],candidate_endpoints=[[-4.8,-.38,-3]],
            params=PlanningParams(durations=(3,),max_speed_mps=.15,max_acceleration_mps2=.3))
        obs=Mock();obs.age_at.side_effect=lambda host_now=None:.14+(clock.now if host_now is None else host_now)-100
        row=dict(frame_id='frame:A',planning_input_body_state={'host_monotonic_s':99.92})
        run.latest_input=(obs,row)
        record=dict(plan=plan,plan_id='plan:A',frame_id='frame:A')
        return run,state,obs,record

    def test_actual_request_interval_resolves_recorded_boundary_transition(self):
        with TemporaryDirectory() as folder:
            clock=Clock();run,_,_,_=self.run_fixture(folder,clock)
            run.ramp_time_basis='elapsed_since_previous_request'
            run.schedule['command_ramp_interval_cap_s']=.2
            run.last_approach_request_host_s=99.86
            position=[-4.395049571990967,.4187329113483429,-2.9979729652404785]
            previous=[.045248544113207625,.028470689341194552]
            desired=[.04482652618567173,.03750139461323364]
            with patch('trajectory_timed_experiment.time.monotonic',clock):
                result=run._boundary_command(position,desired,previous,.05)
            self.assertTrue(result['accepted'])
            self.assertLessEqual(result['command_velocity_xy_m_s'][1],result['upper_velocity_xy_m_s'][1])
            self.assertAlmostEqual(result['command_ramp_timing']['effective_interval_s'],.14)
            self.assertLessEqual(result['command_change_norm_m_s'],.15*.14+1e-12)

    def test_zero_request_interval_does_not_borrow_future_command_duration(self):
        with TemporaryDirectory() as folder:
            clock=Clock();run,_,_,_=self.run_fixture(folder,clock)
            run.ramp_time_basis='elapsed_since_previous_request'
            run.schedule['command_ramp_interval_cap_s']=.2
            run.last_approach_request_host_s=100.
            with patch('trajectory_timed_experiment.time.monotonic',clock):
                result=run._boundary_command([-5,-.5,-3],[.1,.1],[0,0],.05)
            self.assertEqual(result['command_velocity_xy_m_s'],[0.,0.])
            self.assertEqual(result['command_ramp_timing']['effective_interval_s'],0.)

    def test_one_segment_then_yield_preserves_bound_plan_frame_and_does_not_brake(self):
        with TemporaryDirectory() as folder:
            clock=Clock();run,state,obs,record=self.run_fixture(folder,clock)
            run._freshness_brake_and_reobserve=Mock()
            with patch('trajectory_timed_experiment.time.monotonic',clock):
                run._execute_fixed_height({},record,state,obs)
            self.assertEqual(record['execution_status'],'YIELDED_FOR_NEW_OBSERVATION')
            self.assertEqual(len(record['execution_samples']),1)
            run._freshness_brake_and_reobserve.assert_not_called()
            ledger=run.report['timed_command_records']
            self.assertEqual(len(ledger),2)
            self.assertTrue(all(c['plan_id']=='plan:A' and c['frame_id']=='frame:A' for c in ledger))
            self.assertLess(ledger[0]['reference']['reference_time_s'],ledger[1]['reference']['reference_time_s'])
            self.assertTrue(all(c['observation_age_at_dispatch_s']+c['requested_duration_s']<=.44+1e-9 for c in ledger))

    def test_rpc_delay_spike_is_saved_and_no_more_target_commands_sent(self):
        with TemporaryDirectory() as folder:
            clock=Clock();run,state,obs,record=self.run_fixture(folder,clock)
            run.client.moveByVelocityZAsync.return_value.join.side_effect=lambda:clock.advance(.6)
            run._freshness_brake_and_reobserve=Mock(return_value=(np.array([-5.,-.5,-3]),state))
            with patch('trajectory_timed_experiment.time.monotonic',clock):
                run._execute_fixed_height({},record,state,obs)
            self.assertEqual(run.client.moveByVelocityZAsync.call_count,1)
            self.assertGreater(record['execution_samples'][0]['observation_age_after_rpc_and_state_s'],.5)
            self.assertEqual(record['execution_status'],'SELF_STATE_BRAKE_AND_REOBSERVE')
            run._freshness_brake_and_reobserve.assert_called_once()

    def test_expired_frame_or_wrong_frame_never_dispatches_target_motion(self):
        for wrong in (False,True):
            with self.subTest(wrong=wrong),TemporaryDirectory() as folder:
                clock=Clock();run,state,obs,record=self.run_fixture(folder,clock)
                if wrong:
                    record['frame_id']='other'
                    with self.assertRaises(ProtectionRequired):run._execute_fixed_height({},record,state,obs)
                else:
                    obs.age_at.side_effect=None;obs.age_at.return_value=.49
                    run._freshness_brake_and_reobserve=Mock(return_value=(np.array([-5.,-.5,-3]),state))
                    with patch('trajectory_timed_experiment.time.monotonic',clock):run._execute_fixed_height({},record,state,obs)
                run.client.moveByVelocityZAsync.assert_not_called()

    def test_new_observation_does_not_refresh_old_plan_context(self):
        with TemporaryDirectory() as folder:
            clock=Clock();run,state,old,record=self.run_fixture(folder,clock)
            run.context={'observation':old}
            new=Mock()
            def parent_input(*args):
                run.active_observation=new
                return {'inputs':'new'}
            with patch.object(TrajectorySingleExperiment,'_planning_inputs',side_effect=parent_input):
                run._planning_inputs(new,{'frame_id':'new'},np.zeros(3),state)
            self.assertIs(run.active_observation,old)
            self.assertIs(run.latest_input[0],new)

    def test_file_worker_preserves_rgb_depth_bytes_and_hashes(self):
        with TemporaryDirectory() as folder:
            clock=Clock();run,_,_,_=self.run_fixture(folder,clock)
            image=np.arange(36,dtype=np.uint8).reshape(3,4,3)
            depth=np.arange(12,dtype=np.float32).reshape(3,4)
            row={}
            run._save_media(SimpleNamespace(image=image,depth=depth),row,1)
            np.testing.assert_array_equal(cv2.imread(row['raw_image']),image)
            np.testing.assert_array_equal(np.load(row['raw_depth']),depth)
            self.assertEqual(hashlib.sha256(Path(row['raw_image']).read_bytes()).hexdigest(),row['raw_image_sha256'])
            self.assertEqual(row['media_persistence']['status'],'SAVED')

    def test_parent_approach_contract_runs_three_plans_with_timed_dispatch(self):
        with TemporaryDirectory() as folder:
            clock=Clock();run,state,_,first=self.run_fixture(folder,clock)
            run.report.update(observations=[],planning_records=[],control_iterations=[],protective_return_reason=None,episode_id='test')
            run.planning['max_interface_planning_steps']=3
            run.target_source='sensor_surface_estimate';run.planning_mode='perception'
            run.media_futures=[];run.media_writer=Mock()
            run._prepare_matched_start=Mock();run._settle=Mock()
            run.last_position=None;run.path_m=0.
            def capture():
                clock.advance(.08)
                number=len(run.report['observations'])+1
                anchor=clock.now
                obs=Mock(valid=True)
                obs.age_at.side_effect=lambda host_now=None:.1+(clock.now if host_now is None else host_now)-anchor
                row=dict(frame_id=f'f{number}',detected=True,truncated=False,tracking_state='DETECTED',
                    frame_age_s=.1,error_xy=[.2,0],depth_m=4.,world_ned_m=[-5,-.5,-3],
                    planning_input_body_state={'host_monotonic_s':anchor})
                run.report['observations'].append(row);run.latest_input=(obs,row)
                return obs,row,np.array([-5.,-.5,-3]),state
            run._sample_timestamped=Mock(side_effect=capture)
            run._planning_inputs=Mock(return_value={})
            with patch('trajectory_timed_experiment.time.monotonic',clock),patch('trajectory_single_experiment.plan_trajectory',return_value=first['plan']):
                run.approach()
            self.assertIsNone(run.report['protective_return_reason'])
            self.assertEqual(len(run.report['planning_records']),3)
            self.assertEqual(run.report['approach_stop_reason'],'configured_interface_step_limit_reached')
            self.assertGreater(len(run.report['timed_command_records']),3)


if __name__=='__main__':unittest.main()
