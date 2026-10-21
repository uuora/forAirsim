from copy import deepcopy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from perception_trajectory_planner import PlanningParams,plan_trajectory
from trajectory_timed_reference import timed_reference
from trajectory_timed_audit import audit_timed_execution


class TimedAuditTests(unittest.TestCase):
    def test_rgb_reference_and_surface_distance_must_keep_distinct_frame_bindings(self):
        from trajectory_rgb_quality_reference import rgb_quality_reference
        with TemporaryDirectory() as folder:
            report=self.report(folder);report['image_center_reference_mode']='rgb_detection_centroid_axial_plane'
            report['configuration']['planning']['image_center_reference_mode']='rgb_detection_centroid_axial_plane'
            frame=report['observations'][0]
            frame.update(target_estimate_world_ned_m=[0,0,-3],actual_camera_world_ned_m=[-4,-.5,-3.05])
            frame['timestamped_observation'].update(centre_px=[319.5,239.5],projection={'rgb_size_wh':[640,480],'horizontal_fov_deg':90})
            reference=rgb_quality_reference(frame['target_estimate_world_ned_m'],frame['actual_camera_world_ned_m'],[319.5,239.5],[640,480])
            frame['image_center_reference']=reference
            state=report['planning_records'][0]['plan']['input_state']
            state['image_center_reference_offset_ned_m']=reference['offset_from_surface_ned_m']
            self.assertEqual(audit_timed_execution(report)['status'],'VERIFIED')
            state['image_center_reference_offset_ned_m']=[0,0,0]
            self.assertTrue(any('PLAN_RGB_REFERENCE_NOT_BOUND_TO_FRAME' in r for r in audit_timed_execution(report)['reasons']))
            state['image_center_reference_offset_ned_m']=reference['offset_from_surface_ned_m']
            state['target_estimate']=reference['reference_world_ned_m']
            self.assertTrue(any('SURFACE_DISTANCE_ANCHOR_REPLACED' in r for r in audit_timed_execution(report)['reasons']))

    def report(self,folder):
        paths={}
        for key in ('raw_image','raw_depth'):
            path=Path(folder)/key
            path.write_bytes(b'captured fixture')
            paths[key]=str(path);paths[key+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        plan=plan_trajectory(position=[-5,-.5,-3],velocity=[0,0,0],acceleration=[0,0,0],
            target_position=[0,0,-3],target_velocity=[0,0,0],candidate_endpoints=[[-4.8,-.38,-3]],
            params=PlanningParams(durations=(3,),max_speed_mps=.15,max_acceleration_mps2=.3))
        reference=timed_reference(plan,[-5,-.5,-3],99.9,100,None,.8,.15)
        frame=dict(paths,frame_id='f1',provider_valid=True,media_persistence={'status':'SAVED'},
            planning_input_body_state={'host_monotonic_s':99.9},
            timestamped_observation={'image_readback_age_sim_s':.05,'age_anchor_host_monotonic_s':99.9,'clock_valid':True})
        command=dict(command_sequence=0,target_used=True,plan_id='p1',frame_id='f1',dispatch_host_s=100.,
            observation_age_at_dispatch_s=.15,requested_duration_s=.05,reference=reference)
        return dict(execution_reference_mode='time_indexed',configuration={'visual':{'frame_max_age_s':.5},
            'planning':{'execution_envelope':{'freshness_reserve_s':.06}}},
            observations=[frame],planning_records=[{'plan_id':'p1','frame_id':'f1','plan':plan,
                'execution_samples':[{'command_sequence':0}]}],timed_command_records=[command],goal_keepalive_records=[])

    def test_bound_reference_age_request_and_media_are_verified(self):
        with TemporaryDirectory() as folder:
            result=audit_timed_execution(self.report(folder))
            self.assertEqual(result['status'],'VERIFIED',result['reasons'])
            self.assertEqual(result['target_request_count'],1)

    def test_missing_request_and_stale_lease_are_rejected(self):
        with TemporaryDirectory() as folder:
            report=self.report(folder)
            report['planning_records'][0]['execution_samples']=[]
            self.assertIn('COMMAND_REQUEST_COVERAGE_INCOMPLETE',audit_timed_execution(report)['reasons'])
            report=self.report(folder)
            report['timed_command_records'][0]['requested_duration_s']=.4
            self.assertEqual(audit_timed_execution(report)['status'],'INVALID')

    def test_new_frame_cannot_revalidate_old_plan(self):
        with TemporaryDirectory() as folder:
            report=self.report(folder)
            frame=deepcopy(report['observations'][0]);frame['frame_id']='f2'
            report['observations'].append(frame)
            report['timed_command_records'][0]['frame_id']='f2'
            self.assertTrue(any('PLAN_FRAME_VERSION_MISMATCH' in r for r in audit_timed_execution(report)['reasons']))

    def test_fixed_preview_time_and_changed_media_are_detected(self):
        with TemporaryDirectory() as folder:
            report=self.report(folder)
            report['timed_command_records'][0]['reference']['reference_time_s']=.75
            self.assertEqual(audit_timed_execution(report)['status'],'INVALID')

    def test_actual_request_ramp_is_reconstructed_and_cannot_borrow_time(self):
        with TemporaryDirectory() as folder:
            report=self.report(folder)
            report['command_ramp_time_basis']='elapsed_since_previous_request'
            report['configuration']['planning'].update(execution_ramp_acceleration_m_s2=.15,
                timed_execution={'command_ramp_time_basis':'elapsed_since_previous_request','command_ramp_interval_cap_s':.2})
            command=report['timed_command_records'][0]
            command.update(command_velocity_xy_m_s=[0.,0.],boundary_envelope={
                'previous_velocity_xy_m_s':[0.,0.],'command_ramp_timing':{
                    'previous_request_host_s':None,'evaluated_host_s':100.,'effective_interval_s':0.}})
            self.assertEqual(audit_timed_execution(report)['status'],'VERIFIED')
            command['command_velocity_xy_m_s']=[.01,0.]
            self.assertTrue(any('REQUEST_TIME_COMMAND_RAMP_BUDGET_EXCEEDED' in r for r in audit_timed_execution(report)['reasons']))
            command['command_velocity_xy_m_s']=[0.,0.]
            command['boundary_envelope']['command_ramp_timing']['effective_interval_s']=.05
            self.assertTrue(any('RAMP_INTERVAL_NOT_REPRODUCIBLE' in r for r in audit_timed_execution(report)['reasons']))
            report=self.report(folder)
            Path(report['observations'][0]['raw_image']).write_bytes(b'changed')
            self.assertEqual(audit_timed_execution(report)['status'],'INVALID')


if __name__=='__main__':unittest.main()
