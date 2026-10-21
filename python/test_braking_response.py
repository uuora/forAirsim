import copy,json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np
from braking_response import entry_check,summarize_pulse,validate_measurement
from braking_response_experiment import BrakingResponseExperiment


def fixture():
    m=dict(mode='pulse_measurement',hold_s=1.,ramp_timeout_s=2.,brake_timeout_s=6.,
        stop_speed_m_s=.003,stop_acceleration_m_s2=.02,stop_stable_s=.5,max_query_gap_s=.25,
        entry_velocity_tolerance_m_s=.006,entry_minimum_amplitude_fraction=.7,
        detect_speed_drop_m_s=.002,pulses=[dict(direction_xy=[1.,0.],amplitude_m_s=.03)])
    def row(t,p,v,command):
        return dict(host_monotonic_s=t,query_state_timestamp_ns=int(t*1e9),world_ned_m=[p,0.,-3.],
            velocity_ned_m_s=[v,0.,0.],acceleration_norm_m_s2=.01,speed_norm_m_s=abs(v),
            previous_command_velocity_xy_m_s=[command,0.],protection_reasons=[])
    origin=row(10.,-5.,.03,.03)
    samples=[row(10.1,-4.998,.018,.015),row(10.2,-4.997,.002,0),row(10.4,-4.9968,.001,0),
             row(10.6,-4.9967,.001,0),row(10.8,-4.9966,.001,0)]
    pulse=dict(direction_xy=[1.,0.],amplitude_m_s=.03,status='COMPLETED',samples=samples,
               brake_first_request={'origin_sample':origin,'dispatch_host_s':10.01})
    model=dict(horizon_s=.5,braking_acceleration_m_s2=.15,position_reserve_m=.02)
    return m,pulse,model


class BrakingTests(unittest.TestCase):
    def experiment_fixture(self,folder):
        run=object.__new__(BrakingResponseExperiment);run.folder=Path(folder)
        run.cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/braking_response_v1/zero_short.json').read_text(encoding='utf-8'))
        run.planning=run.cfg['planning'];run.guard_cfg=run.planning['execution_envelope']
        run.active_xy_command=np.zeros(2);run.current_pulse=None;run.last_measurement_request=None
        run.origin_translation=np.array([0,-3,-.85]);run.vehicle='DroneA';run.client=Mock()
        run.report={'measurement_commands':[]};run.phase='APPROACH'
        return run

    def test_first_request_does_not_borrow_time_and_traces_receipt(self):
        with TemporaryDirectory() as folder:
            run=self.experiment_fixture(folder)
            with patch('braking_response_experiment.time.monotonic',return_value=100.):
                row=run.dispatch([.03,0],{'world_ned_m':[-5,-.5,-3]},'RAMP')
            self.assertEqual(row['command_velocity_xy_m_s'],[0.,0.])
            self.assertEqual(row['status'],'RPC_COMPLETED')
            self.assertFalse(row['target_used'])
            events=Path(folder,'measurement_command_events.jsonl').read_text().splitlines()
            self.assertEqual([json.loads(e)['status'] for e in events],['DISPATCHING','RPC_COMPLETED'])

    def test_actual_interval_limits_command_delta(self):
        with TemporaryDirectory() as folder:
            run=self.experiment_fixture(folder);run.last_measurement_request=99.9
            with patch('braking_response_experiment.time.monotonic',return_value=100.):
                row=run.dispatch([.03,0],{'world_ned_m':[-5,-.5,-3]},'RAMP')
            self.assertAlmostEqual(row['command_velocity_xy_m_s'][0],.015)

    def test_rejected_actual_query_saved_before_dispatch(self):
        from trajectory_single_experiment import ProtectionRequired
        def vec(v):return SimpleNamespace(**dict(zip(('x_val','y_val','z_val'),v)))
        with TemporaryDirectory() as folder:
            run=self.experiment_fixture(folder)
            state=SimpleNamespace(timestamp=1,kinematics_estimated=SimpleNamespace(
                linear_velocity=vec([0,0,0]),linear_acceleration=vec([0,0,0]),
                orientation=SimpleNamespace(x_val=0,y_val=0,z_val=0,w_val=1)))
            run.state=Mock(return_value=(np.array([-5,.46,-3]),0,state))
            with self.assertRaises(ProtectionRequired):run.query('RAMP')
            row=json.loads(Path(folder,'measurement_samples.jsonl').read_text())
            self.assertIn('ACTUAL_POSITION_ENTERED_PROTECTION_MARGIN',row['protection_reasons'])
            run.client.moveByVelocityZAsync.assert_not_called()

    def test_stop_uses_original_state_and_records_two_clocks(self):
        m,p,model=fixture();r=summarize_pulse(p,m,model)
        self.assertTrue(r['counts_as_measured_stopping'])
        self.assertAlmostEqual(r['maximum_recorded_forward_displacement_m'],.0034)
        self.assertAlmostEqual(r['first_detected_drop_host_proxy_s'],.09)
        self.assertAlmostEqual(r['first_detected_drop_state_timestamp_delta_s'],.1)
        self.assertAlmostEqual(r['declared_envelope_allowance_m'],.038)

    def test_gaps_rollback_incomplete_or_rejection_never_measured(self):
        for kind in ('gap','rollback','protection','velocity','timeout','moving_command'):
            with self.subTest(kind=kind):
                m,p,model=fixture()
                if kind=='gap':m['max_query_gap_s']=.15
                if kind=='rollback':p['samples'][1]['query_state_timestamp_ns']=1
                if kind=='protection':p['samples'][0]['protection_reasons']=['ACTUAL_SPEED_BUDGET_EXCEEDED']
                if kind=='velocity':p['brake_first_request']['origin_sample']['velocity_ned_m_s']=[.005,0,0]
                if kind=='timeout':p['status']='PROTECTED_RETURN'
                if kind=='moving_command':
                    for s in p['samples']:s['previous_command_velocity_xy_m_s']=[.001,0]
                r=summarize_pulse(p,m,model)
                self.assertFalse(r['counts_as_measured_stopping'])
                self.assertIsNone(r['sampled_displacement_within_declared_allowance'])

    def test_direction_not_path_defines_stopping_displacement(self):
        m,p,model=fixture()
        p['direction_xy']=[-1.,0.]
        p['brake_first_request']['origin_sample']['velocity_ned_m_s']=[-.03,0,0]
        for s in p['samples']:
            s['world_ned_m'][0]=-10-s['world_ned_m'][0]
            s['velocity_ned_m_s'][0]*=-1
        self.assertAlmostEqual(summarize_pulse(p,m,model)['maximum_recorded_forward_displacement_m'],.0034)

    def test_invalid_measurement_configs_rejected(self):
        m,_,_=fixture()
        for key,value in [('hold_s',2.),('max_query_gap_s',.5),('stop_stable_s',.1),('mode','unknown')]:
            changed=dict(m);changed[key]=value
            with self.assertRaises(ValueError):validate_measurement({'braking_measurement':changed})
        changed=copy.deepcopy(m);changed['pulses'][0]['amplitude_m_s']=.15
        with self.assertRaises(ValueError):validate_measurement({'braking_measurement':changed})

    def test_new_lifecycle_never_reads_target_or_images(self):
        with TemporaryDirectory() as folder:
            run=object.__new__(BrakingResponseExperiment);run.folder=Path(folder);run.report={};run.armed=False
            run.client=Mock();run.client.listVehicles.return_value=['DroneA','DroneB']
            run.client.isApiControlEnabled.return_value=False
            run.vehicle='DroneA';run.phase='PREFLIGHT';run.cfg={'observation_world_ned_m':[-5,-.5,-3]}
            run.state=Mock(return_value=(np.array([0,-3,.8]),0,None));run._bystander_preflight=Mock(return_value=(np.array([0,3,.8]),0,None))
            run.client.simGetCollisionInfo.return_value=SimpleNamespace(time_stamp=1)
            run.move_to=Mock();run.approach=Mock();run.land=Mock()
            run.run()
            run.client.simGetObjectPose.assert_not_called();run.client.simGetImages.assert_not_called()
            run.approach.assert_called_once();run.land.assert_called_once()


if __name__=='__main__':unittest.main()
