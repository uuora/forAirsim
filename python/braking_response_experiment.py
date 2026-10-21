"""Independent low-speed self-state braking measurements in AirSim.

Reuse inspected staging/matched-start/landing helpers, without target sensing,
target-pose preflight, planner calls, or modifications to the frozen V6.2 code.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
import airsim
import numpy as np
from braking_response import entry_check,summarize_pulse,validate_measurement
from matched_trajectory_start import validate_matched_start_config
from setup_scene import ROOT
from single_drone_experiment import SingleDroneExperiment
from trajectory_single_experiment import TrajectorySingleExperiment,ProtectionRequired,execution_sample,world_height_to_local
from trajectory_timed_reference import command_ramp_interval


class BrakingResponseExperiment(TrajectorySingleExperiment):
    def __init__(self,config_path,client=None):
        # The base initializes the inspected vehicle lifecycle; camera objects
        # are never queried. Do not initialize the trajectory planner/provider.
        SingleDroneExperiment.__init__(self,'visual',client=client,config_path=Path(config_path),record_prefix='braking_response')
        self.planning=self.cfg['planning'];self.measurement=validate_measurement(self.cfg)
        self.fixed_height_execution=True;self.guard_cfg=self.planning['execution_envelope']
        self.match_params=validate_matched_start_config(self.cfg)
        self.active_xy_command=np.zeros(2);self.active_observation=None;self.origin_translation=None
        self.last_measurement_request=None;self.current_pulse=None
        self.report.update(kind='INDEPENDENT_BRAKING_RESPONSE_DEVELOPMENT',policy='self_state_envelope_ramp_to_zero',
            formal_thesis_effect_results=False,target=None,configuration_sha256=self.sha(Path(config_path)),
            measurement_passed=False,flight_dispatch_started=False,measurement_commands=[],pulses=[],
            initial_state_matches=[],settling_records=[],execution_guard_samples=[],goal_keepalive_records=[],
            protection_reason=None,truth_access={'target_pose':'NOT_READ','RGB_depth':'NOT_READ',
            'self_state':'SimpleFlight estimated kinematics and simulator self world pose; explicitly allowed',
            'collision':'safety only; no target task'},ai_assistance='Implementation and analysis assisted by AI')
        for name in self.cfg['flight_source_sha256']:
            actual=self.sha(ROOT/'python'/name)
            if actual!=self.cfg['flight_source_sha256'][name]:raise ValueError('Source freeze mismatch: '+name)
            self.report['source_sha256'][name]=actual
        self.save()

    @staticmethod
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

    def save(self):
        (self.folder/'report.json').write_text(json.dumps(self.report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

    def query(self,stage):
        started=time.monotonic();p,_,state=self.state(check_workspace=False);received=time.monotonic()
        sample=execution_sample(p,state,self.planning,self.cfg['workspace_min_ned_m'],self.cfg['workspace_max_ned_m'])
        sample.update(host_monotonic_s=received,query_started_host_s=started,stage=stage,
            previous_command_velocity_xy_m_s=self.active_xy_command.tolist())
        if self.current_pulse is not None:
            sample['pulse_id']=self.current_pulse['pulse_id'];self.current_pulse['samples'].append(sample)
        else:self.report.setdefault('zero_gate_samples',[]).append(sample)
        with (self.folder/'measurement_samples.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(sample,allow_nan=False)+'\n')
        if sample['protection_reasons']:raise ProtectionRequired(';'.join(sample['protection_reasons']))
        return sample

    def dispatch(self,desired,sample,stage):
        evaluated=time.monotonic()
        timing=command_ramp_interval(self.last_measurement_request,evaluated,
            self.planning['timed_execution']['command_ramp_interval_cap_s'])
        if timing['effective_interval_s']==0:desired=self.active_xy_command.copy()
        envelope=super()._boundary_command(sample['world_ned_m'],desired,self.active_xy_command,
                                           max(timing['effective_interval_s'],1e-12))
        command=envelope['command_velocity_xy_m_s']
        row={'sequence':len(self.report['measurement_commands']),'stage':stage,
            'pulse_id':None if self.current_pulse is None else self.current_pulse['pulse_id'],
            'request_intent_host_s':time.monotonic(),'desired_xy_m_s':list(desired),'command_velocity_xy_m_s':command,
            'previous_command_velocity_xy_m_s':self.active_xy_command.tolist(),
            'ramp_timing':timing,'boundary_envelope':envelope,'requested_duration_s':self.planning['execution_sample_dt_s'],
            'origin_sample':sample,'target_used':False,'status':'DISPATCHING'}
        self.report['measurement_commands'].append(row)
        # Small append-only intent avoids serializing the entire report in the
        # command critical path. An intent without a receipt is not completion.
        with (self.folder/'measurement_command_events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        local_z=world_height_to_local(self.planning['fixed_world_height_ned_m'],self.origin_translation)
        started=time.monotonic();row['dispatch_host_s']=started
        if self.current_pulse is not None and stage=='BRAKE' and 'brake_first_request' not in self.current_pulse:
            self.current_pulse['brake_first_request']=row
        self.client.moveByVelocityZAsync(*command,local_z,duration=row['requested_duration_s'],
            yaw_mode=airsim.YawMode(False,0),vehicle_name=self.vehicle).join()
        row.update(status='RPC_COMPLETED',rpc_return_host_s=time.monotonic())
        self.last_measurement_request=started;self.active_xy_command=np.array(command)
        if self.current_pulse is not None and np.linalg.norm(command)>1e-9:
            self.current_pulse['motion_request_started']=True
        with (self.folder/'measurement_commands.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        with (self.folder/'measurement_command_events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        return row

    def prepare(self):
        self._prepare_matched_start()
        self.report['initial_state_matches'].append(self.report['initial_state_match'])
        self.phase='APPROACH';self.last_measurement_request=None
        self.active_xy_command=np.zeros(2)

    def run_pulse(self,spec,index):
        pulse=dict(spec,pulse_id=f'pulse_{index:03d}',status='PREPARING',motion_request_started=False,samples=[])
        self.report['pulses'].append(pulse);self.current_pulse=pulse;self.save()
        self.prepare()
        pulse['status']='RAMPING';desired=np.array(spec['direction_xy'])*spec['amplitude_m_s']
        sample=self.query('RAMP');start=time.monotonic()
        while np.linalg.norm(self.active_xy_command-desired)>1e-9:
            if time.monotonic()-start>self.measurement['ramp_timeout_s']:raise ProtectionRequired('RAMP_TIMEOUT')
            self.dispatch(desired,sample,'RAMP');sample=self.query('RAMP')
        pulse['status']='HOLDING';start=time.monotonic()
        while time.monotonic()-start<self.measurement['hold_s']:
            self.dispatch(desired,sample,'HOLD');sample=self.query('HOLD')
        pulse['entry_check']=entry_check(sample,spec,self.measurement)
        if not pulse['entry_check']['accepted']:raise ProtectionRequired('BRAKE_ENTRY_VELOCITY_NOT_ACCEPTED')
        pulse['status']='BRAKING';start=time.monotonic();stable=None
        while time.monotonic()-start<self.measurement['brake_timeout_s']:
            self.dispatch([0.,0.],sample,'BRAKE');sample=self.query('BRAKE')
            stopped=(np.linalg.norm(self.active_xy_command)<=1e-9 and
                sample['speed_norm_m_s']<=self.measurement['stop_speed_m_s'] and
                sample['acceleration_norm_m_s2']<=self.measurement['stop_acceleration_m_s2'])
            stable=(sample['host_monotonic_s'] if stable is None else stable) if stopped else None
            if stable is not None and sample['host_monotonic_s']-stable>=self.measurement['stop_stable_s']:
                pulse['status']='COMPLETED';break
        else:raise ProtectionRequired('BRAKE_STRICT_STABILITY_TIMEOUT')
        pulse['diagnostics']=summarize_pulse(pulse,self.measurement,self.model())
        if not pulse['diagnostics']['counts_as_measured_stopping']:raise ProtectionRequired('MEASUREMENT_COVERAGE_OR_STABILITY_FAILED')
        self.current_pulse=None;self.save()

    def model(self):
        return {'horizon_s':self.guard_cfg['horizon_s'],
            'braking_acceleration_m_s2':self.planning['execution_ramp_acceleration_m_s2'],
            'position_reserve_m':self.guard_cfg['position_reserve_m']}

    def approach(self):
        try:
            if self.measurement['mode']=='zero_hold_gate':
                self.prepare();sample=self.query('ZERO_HOLD');start=time.monotonic()
                while time.monotonic()-start<self.measurement['stop_stable_s']:
                    self.dispatch([0.,0.],sample,'ZERO_HOLD');sample=self.query('ZERO_HOLD')
                self.report['measurement_passed']=all(not s['protection_reasons'] and
                    s['speed_norm_m_s']<=self.cfg['initial_state_match']['speed_tolerance_m_s'] for s in self.report['zero_gate_samples'])
            else:
                for index,spec in enumerate(self.measurement['pulses'],1):self.run_pulse(spec,index)
                self.report['measurement_passed']=True
        except (ProtectionRequired,ValueError) as error:
            self.report['protection_reason']=str(error)
            if self.current_pulse is not None:self.current_pulse.update(status='PROTECTED_RETURN',reason=str(error))
        finally:
            self.client.cancelLastTask(vehicle_name=self.vehicle)
            self.client.hoverAsync(vehicle_name=self.vehicle).join()
            self.save()

    def run(self):
        # Target-free version of the existing inspected takeoff/return lifecycle.
        try:
            if not {'DroneA','DroneB'}.issubset(self.client.listVehicles()):raise RuntimeError('Missing vehicles')
            if self.client.isApiControlEnabled(vehicle_name='DroneB'):raise RuntimeError('DroneB must be released')
            p,speed,_=self.state();b,b_speed,_=self._bystander_preflight()
            if not (.65<p[2]<1.1 and speed<.08 and .65<b[2]<1.1 and b_speed<.08):raise RuntimeError('Both vehicles must be grounded')
            if self.client.isApiControlEnabled(vehicle_name=self.vehicle):raise RuntimeError('DroneA must be released')
            self.bystander_start=b;self.collision_stamp=self.client.simGetCollisionInfo(vehicle_name=self.vehicle).time_stamp
            self.report['bystander_start_world_ned_m']=b.tolist()
            self.client.simPause(False);self.client.enableApiControl(True,vehicle_name=self.vehicle)
            if not self.client.armDisarm(True,vehicle_name=self.vehicle):raise RuntimeError('Could not arm')
            self.armed=True;self.report['flight_dispatch_started']=True;self.save()
            self.phase='TAKEOFF';self.move_to([p[0],p[1],-3.],timeout=35)
            self.phase='STAGING';self.move_to(self.cfg['observation_world_ned_m'],timeout=40)
            self.approach();self.land();self.report['status']='PASS'
        except BaseException as error:
            self.report.update(status='FAIL',error=str(error),failed_phase=self.phase,manual_recovery_required=self.armed)
            raise
        finally:
            try:self.client.simPause(True);self.report['paused']=True
            except Exception as error:self.report.update(status='FAIL',pause_error=str(error))
            self.report['finished_at']=datetime.now().astimezone().isoformat();self.save();print(self.folder)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--receipt-path',type=Path,required=True)
    args=parser.parse_args();run=BrakingResponseExperiment(args.config)
    if args.receipt_path.exists():raise ValueError('Trial pointer must not overwrite an existing file')
    args.receipt_path.write_text(json.dumps({'report_path':str(run.folder/'report.json'),
        'created_local':datetime.now().astimezone().isoformat()},indent=2),encoding='utf-8')
    run.run()
    raise SystemExit(0 if run.report['measurement_passed'] else 2)
