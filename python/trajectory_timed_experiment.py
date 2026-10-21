"""V6 fixed-height time-indexed execution; target commands bind to one plan/frame."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import math
import time
import airsim
import cv2
import numpy as np

from setup_scene import ROOT, vector
from trajectory_single_experiment import TrajectorySingleExperiment, ProtectionRequired, execution_sample, world_height_to_local
from trajectory_execution_guard import command_window
from trajectory_timed_reference import command_ramp_interval, scheduled_quantum, timed_reference


class TimedTrajectoryExperiment(TrajectorySingleExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.fixed_height_execution or not self.guard_cfg or not self.match_params:
            raise ValueError("Time-indexed execution requires matched guarded fixed-height control")
        self.schedule = self.planning['timed_execution']
        for name in ('next_cycle_budget_s','rpc_overhead_budget_s','state_query_budget_s'):
            value=self.schedule[name]
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid declared timing budget")
        self.context=None
        self.last_approach_request_host_s=None
        self.ramp_time_basis=self.schedule.get('command_ramp_time_basis','fixed_sample_quantum')
        if self.ramp_time_basis not in ('fixed_sample_quantum','elapsed_since_previous_request'):
            raise ValueError('Unknown command-ramp time basis')
        if self.ramp_time_basis=='elapsed_since_previous_request':
            cap=self.schedule['command_ramp_interval_cap_s']
            if isinstance(cap,bool) or not math.isfinite(cap) or not 0 < cap <= self.guard_cfg['horizon_s']:
                raise ValueError('Invalid actual-time command-ramp cap')
        self.pending_reobservation=None
        self.media_futures=[]
        self.media_writer=ThreadPoolExecutor(max_workers=1,thread_name_prefix='trajectory_raw_media')
        self.report.update(execution_reference_mode='time_indexed',timed_command_records=[],timing_events=[],
            command_ramp_time_basis=self.ramp_time_basis,
            reference_execution_scope='Host elapsed time p/v feedback, bounded XY commands and constant world Z; no exact polynomial tracking proof',
            media_persistence_scope='One local file writer; drained before approach returns; no AirSim calls from the worker',
            timed_trace_scope='All approach moveByVelocityZAsync requests; inherited positioning and hover/landing separately recorded')
        for name in ('trajectory_timed_experiment.py','trajectory_timed_reference.py'):
            self.report['source_sha256'][name]=hashlib.sha256((ROOT/'python'/name).read_bytes()).hexdigest()

    def _trace(self, row):
        row['command_sequence']=len(self.report['timed_command_records'])
        self.report['timed_command_records'].append(row)
        with (self.folder/'timed_command_records.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')

    def _on_brake_command(self, command, started, finished):
        trace=dict(command,source='self_state_brake',target_used=False,plan_id=None,frame_id=None,
            observation_age_at_dispatch_s=None,dispatch_host_s=started,rpc_return_host_s=finished)
        self._trace(trace)
        self.last_approach_request_host_s=started
        command['command_sequence']=trace['command_sequence']

    def _boundary_command(self, position, desired_xy, previous_xy, dt):
        if self.ramp_time_basis!='elapsed_since_previous_request' or getattr(self,'phase',None)!='APPROACH':
            return super()._boundary_command(position,desired_xy,previous_xy,dt)
        try:
            timing=command_ramp_interval(self.last_approach_request_host_s,time.monotonic(),
                                         self.schedule['command_ramp_interval_cap_s'])
        except ValueError as error:
            raise ProtectionRequired(str(error)) from error
        # A new phase starts with a verified zero hold; no positive change may
        # borrow time from positioning or from a future requested duration.
        if timing['effective_interval_s']==0 and np.linalg.norm(np.asarray(desired_xy)-np.asarray(previous_xy))>1e-12:
            desired_xy=previous_xy
        decision=super()._boundary_command(position,desired_xy,previous_xy,max(timing['effective_interval_s'],1e-12))
        decision['command_ramp_timing']=timing
        return decision

    def _freshness_brake_and_reobserve(self, reason, age):
        self.context=None
        self.pending_reobservation=None
        try:
            result=super()._freshness_brake_and_reobserve(reason,age)
        except ProtectionRequired:
            event=self.report['freshness_replan_events'][-1]
            if event['status']=='BRAKED_READY_TO_REOBSERVE':
                event['status']='BRAKED_REOBSERVATION_LIMIT_REJECTED'
            raise
        event=self.report['freshness_replan_events'][-1]
        event.update(status='BRAKED_AWAITING_CAPTURE',brake_finished_host_s=time.monotonic())
        self.pending_reobservation=event
        return result

    def _guard_query(self, reason):
        started=time.monotonic()
        position,_,state=self.state(check_workspace=True)
        received=time.monotonic()
        sample=execution_sample(position,state,self.planning,self.cfg['workspace_min_ned_m'],self.cfg['workspace_max_ned_m'])
        self._save_guard_state(sample,reason)
        self._record_path(position)
        self.report['timing_events'].append(dict(stage=reason,started_host_s=started,finished_host_s=received,
            duration_s=received-started,query_state_timestamp_ns=int(state.timestamp)))
        if sample['protection_reasons']:
            raise ProtectionRequired(';'.join(sample['protection_reasons']))
        return position,state,sample

    def _reference(self, position):
        ctx=self.context
        try:
            return timed_reference(ctx['plan'],position,ctx['input_host_s'],time.monotonic(),ctx['last_reference_s'],
                                   self.planning['position_gain'],self.planning['execution_speed_cap_m_s'])
        except ValueError as error:
            raise ProtectionRequired('Time-indexed reference: '+str(error)) from error

    def _dispatch(self, xy, duration, source, reference=None, envelope=None, join=False):
        ctx=self.context if reference is not None else None
        window=None
        if ctx is not None:
            window=command_window(ctx['observation'].age_at(),self.cfg['visual']['frame_max_age_s'],duration,
                self.guard_cfg['minimum_command_s'],self.guard_cfg['freshness_reserve_s'])
            if not window['accepted']:
                self._freshness_brake_and_reobserve('V6_FINAL_DISPATCH_WINDOW_EXHAUSTED',window['age_proxy_s'])
                return None
            duration=window['duration_s']
        started=time.monotonic()
        if ctx is not None:
            window=command_window(ctx['observation'].age_at(started),self.cfg['visual']['frame_max_age_s'],duration,
                self.guard_cfg['minimum_command_s'],self.guard_cfg['freshness_reserve_s'])
            if not window['accepted']:
                self._freshness_brake_and_reobserve('V6_REQUEST_CLOCK_WINDOW_EXHAUSTED',window['age_proxy_s'])
                return None
            duration=window['duration_s']
        local_z=world_height_to_local(self.planning['fixed_world_height_ned_m'],self.origin_translation)
        future=self.client.moveByVelocityZAsync(float(xy[0]),float(xy[1]),local_z,duration=duration,
            yaw_mode=airsim.YawMode(False,0),vehicle_name=self.vehicle)
        self.active_xy_command=np.array(xy,dtype=float)
        self.last_approach_request_host_s=started
        if join:
            future.join()
        returned=time.monotonic()
        record=dict(source=source,reason=source,phase='APPROACH',target_used=ctx is not None,
            plan_id=ctx['plan_id'] if ctx else None,frame_id=ctx['frame_id'] if ctx else None,
            observation_age_at_dispatch_s=window['age_proxy_s'] if window else None,
            freshness_window=window,dispatch_host_s=started,rpc_return_host_s=returned,
            rpc_elapsed_host_s=returned-started,command_velocity_xy_m_s=self.active_xy_command.tolist(),
            requested_duration_s=duration,join_called=join,boundary_envelope=envelope,
            reference=reference,world_height_setpoint_ned_m=self.planning['fixed_world_height_ned_m'],
            scope='RPC request duration can be superseded; not measured executed duration')
        if ctx:
            ctx['last_reference_s']=reference['reference_time_s']
        self._trace(record)
        if source!='foreground_segment':
            self.report['goal_keepalive_records'].append(record)
            with (self.folder/'goal_keepalive_records.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,allow_nan=False)+'\n')
        return record

    def _maintain_fixed_goal(self, reason):
        if getattr(self,'phase',None)!='APPROACH':
            return super()._maintain_fixed_goal(reason)
        position,state,sample=self._guard_query('background_pre_dispatch')
        if self.context is not None:
            obs=self.context['observation']
            window=command_window(obs.age_at(),self.cfg['visual']['frame_max_age_s'],
                self.planning['continuous_goal_keepalive_s'],self.guard_cfg['minimum_command_s'],
                self.guard_cfg['freshness_reserve_s'])
            if not window['accepted']:
                self._freshness_brake_and_reobserve('V6_BACKGROUND_WINDOW_EXHAUSTED',window['age_proxy_s'])
                return self._maintain_fixed_goal(reason+'_after_brake')
            reference=self._reference(position)
            envelope=self._boundary_command(position,reference['desired_velocity_xy_m_s'],self.active_xy_command,
                self.planning['execution_sample_dt_s'])
            return self._dispatch(envelope['command_velocity_xy_m_s'],window['duration_s'],
                'background_reference:'+reason,reference,envelope)
        if np.linalg.norm(self.active_xy_command)>1e-9:
            raise ProtectionRequired('Missing time-indexed context for moving target goal')
        envelope=self._boundary_command(position,[0,0],self.active_xy_command,self.planning['execution_sample_dt_s'])
        return self._dispatch(envelope['command_velocity_xy_m_s'],self.planning['continuous_goal_keepalive_s'],
            'self_state_hold:'+reason,envelope=envelope)

    def _save_media(self, observation, row, number):
        started=time.monotonic()
        if observation.image is not None:
            path=self.folder/f'raw_frame_{number:04d}.png'
            if not cv2.imwrite(str(path),observation.image,[cv2.IMWRITE_PNG_COMPRESSION,1]):
                raise RuntimeError('Raw RGB persistence failed')
            row.update(raw_image=str(path),raw_image_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if observation.depth is not None:
            path=self.folder/f'raw_depth_{number:04d}.npy'
            np.save(str(path),observation.depth,allow_pickle=False)
            row.update(raw_depth=str(path),raw_depth_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                raw_depth_semantics='AirSim_DepthPerspective_camera_radial_range_metres')
        row['media_persistence']=dict(started_host_s=started,finished_host_s=time.monotonic(),
            status='SAVED',scope='local writer, outside command critical path')

    def _sample_timestamped(self):
        for future in self.media_futures:
            if future.done() and future.exception() is not None:
                raise ProtectionRequired('Raw media worker failed: '+str(future.exception()))
        self._maintain_fixed_goal('before_capture')
        position0,state0,_=self._guard_query('pre_capture_query')
        started=time.monotonic()
        pending=self.pending_reobservation
        if pending:
            pending.update(status='REOBSERVATION_CAPTURE_STARTED',capture_started_host_s=started)
        observation=self.timestamped_camera.capture(self.vehicle)
        capture_finished=time.monotonic()
        position,_,state=self.state(check_workspace=True)
        received=time.monotonic()
        self._record_path(position)
        error=None
        try:
            age=observation.age_at()
        except ValueError as exception:
            age,error=None,str(exception)
        track=self.tracker.update(observation.valid and age is not None and age<=self.cfg['visual']['frame_max_age_s'])
        row=dict(time=datetime.now().astimezone().isoformat(),world_ned_m=position.tolist(),detected=bool(observation.detected),
            truncated=bool(observation.truncated),tracking_state=track.state.value,candidate_count=observation.candidate_count,
            shape_score_not_probability=observation.confidence,error_xy=observation.normalised_error_xy,depth_m=observation.depth_m,
            frame_age_s=age,timestamped_observation=observation.metadata(),provider_valid=bool(observation.valid),
            target_source=self.target_source,freshness_proxy_error=error,episode_id=self.folder.name,
            frame_id=f"{self.folder.name}:frame:{len(self.report['observations'])+1:04d}")
        row['pre_capture_body_state']=execution_sample(position0,state0,self.planning,self.cfg['workspace_min_ned_m'],self.cfg['workspace_max_ned_m'])
        row['pre_capture_body_state']['host_monotonic_s']=started
        row['planning_input_body_state']=execution_sample(position,state,self.planning,self.cfg['workspace_min_ned_m'],self.cfg['workspace_max_ned_m'])
        row['planning_input_body_state']['host_monotonic_s']=received
        row['capture_query_elapsed_host_s']=received-started
        self.report['timing_events'].append(dict(stage='capture_and_input_query',started_host_s=started,finished_host_s=received,
            image_rpc_host_s=capture_finished-started,query_host_s=received-capture_finished,duration_s=received-started,frame_id=row['frame_id']))
        if observation.state_sample is not None:
            s=observation.state_sample
            row['provider_query_motion_state']=dict(velocity_ned_m_s=vector(s.kinematics_estimated.linear_velocity),
                acceleration_ned_m_s2=vector(s.kinematics_estimated.linear_acceleration),query_state_timestamp_ns=int(s.timestamp))
        self.report['observations'].append(row)
        self.media_futures.append(self.media_writer.submit(self._save_media,observation,row,len(self.report['observations'])))
        if sum(not future.done() for future in self.media_futures)>4:
            raise ProtectionRequired('Raw media queue exceeded bounded backlog')
        if pending:
            valid=observation.valid and age is not None and age<=self.cfg['visual']['frame_max_age_s']
            pending.update(status='REOBSERVED_VALID' if valid else 'REOBSERVATION_CAPTURE_INVALID',
                frame_id=row['frame_id'],capture_finished_host_s=received)
            self.pending_reobservation=None
        if row['planning_input_body_state']['protection_reasons']:
            raise ProtectionRequired(';'.join(row['planning_input_body_state']['protection_reasons']))
        return observation,row,position,state

    def _planning_inputs(self, observation, row, position, state):
        try:
            inputs=super()._planning_inputs(observation,row,position,state)
        finally:
            self.active_observation=self.context['observation'] if self.context else None
        self.latest_input=(observation,row)
        return inputs

    def _initial_reference_command(self, plan, position, previous, record):
        return dict(body_execution_mode='moveByVelocityZAsync_constant_world_height',
                    reference_execution_mode='time_indexed',reference_scope='sampled independently at each actual dispatch',
                    command_velocity_ned_m_s=[*self.active_xy_command.tolist(),0.],
                    command_velocity_norm_m_s=float(np.linalg.norm(self.active_xy_command)),
                    command_scope='plan registration placeholder; actual requests in timed_command_records')

    def _slot(self, observation, requested):
        return scheduled_quantum(observation.age_at(),self.cfg['visual']['frame_max_age_s'],requested,
            self.guard_cfg['minimum_command_s'],self.guard_cfg['freshness_reserve_s'],
            self.schedule['next_cycle_budget_s'],self.schedule['rpc_overhead_budget_s'],self.schedule['state_query_budget_s'])

    def _execute_fixed_height(self, command, record, start_state, observation=None):
        obs,row=self.latest_input
        if observation is not obs or record['frame_id']!=row['frame_id']:
            raise ProtectionRequired('Plan/frame context mismatch')
        self.context=dict(plan=record['plan'],plan_id=record['plan_id'],frame_id=record['frame_id'],observation=obs,
            input_host_s=row['planning_input_body_state']['host_monotonic_s'],last_reference_s=None)
        self.active_observation=obs
        record.update(reference_input_host_s=self.context['input_host_s'],execution_samples=[],freshness_windows=[])
        remaining=self.planning['command_duration_s']
        position,state,_=self._guard_query('timed_execution_start')
        while remaining>=self.guard_cfg['minimum_command_s']-1e-9:
            request=min(remaining,self.planning['execution_sample_dt_s'])
            slot=self._slot(obs,request)
            record['freshness_windows'].append(slot)
            if slot['status']=='BRAKE':
                position,state=self._freshness_brake_and_reobserve('V6_FOREGROUND_WINDOW_EXHAUSTED',slot['age_proxy_s'])
                record['execution_status']='SELF_STATE_BRAKE_AND_REOBSERVE'
                return position,state
            if slot['status']!='EXECUTE':
                record['execution_status']='YIELDED_FOR_NEW_OBSERVATION'
                break
            position,state,_=self._guard_query('foreground_pre_dispatch')
            slot=self._slot(obs,request)
            record['freshness_windows'].append(slot)
            if slot['status']!='EXECUTE':
                if slot['status']=='BRAKE':
                    position,state=self._freshness_brake_and_reobserve('V6_STATE_QUERY_CONSUMED_WINDOW',slot['age_proxy_s'])
                    record['execution_status']='SELF_STATE_BRAKE_AND_REOBSERVE'
                    return position,state
                record['execution_status']='YIELDED_FOR_NEW_OBSERVATION'
                break
            reference=self._reference(position)
            envelope=self._boundary_command(position,reference['desired_velocity_xy_m_s'],self.active_xy_command,slot['duration_s'])
            sent=self._dispatch(envelope['command_velocity_xy_m_s'],slot['duration_s'],'foreground_segment',reference,envelope,join=True)
            if sent is None:
                record['execution_status']='SELF_STATE_BRAKE_AND_REOBSERVE'
                return self.state(check_workspace=True)[0::2]
            position,_,state=self.state(check_workspace=True)
            sample=execution_sample(position,state,self.planning,self.cfg['workspace_min_ned_m'],self.cfg['workspace_max_ned_m'])
            sample.update(command_velocity_xy_m_s=sent['command_velocity_xy_m_s'],command_duration_s=sent['requested_duration_s'],
                host_monotonic_s=time.monotonic(),rpc_elapsed_host_s=sent['rpc_elapsed_host_s'],boundary_envelope=envelope,
                observation_age_after_rpc_and_state_s=obs.age_at(),timed_reference=reference,
                command_sequence=sent['command_sequence'])
            sent['post_query_age_proxy_s']=sample['observation_age_after_rpc_and_state_s']
            record['execution_samples'].append(sample)
            self._record_path(position)
            with (self.folder/'execution_samples.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(plan_id=record['plan_id'],sample=sample),allow_nan=False)+'\n')
            if sample['protection_reasons']:
                raise ProtectionRequired(';'.join(sample['protection_reasons']))
            remaining-=sent['requested_duration_s']
            if obs.age_at()>=self.cfg['visual']['frame_max_age_s']-self.guard_cfg['freshness_reserve_s']:
                position,state=self._freshness_brake_and_reobserve('V6_RPC_OR_QUERY_CONSUMED_WINDOW',obs.age_at())
                record['execution_status']='SELF_STATE_BRAKE_AND_REOBSERVE'
                return position,state
        if record['execution_samples']:
            record['executed_body_state']=record['execution_samples'][-1]
        count=self.freshness_replans
        self._maintain_fixed_goal('after_foreground_before_new_capture')
        if self.freshness_replans!=count:
            record['execution_status']='SELF_STATE_BRAKE_AND_REOBSERVE'
        return position,state

    def approach(self):
        try:
            return super().approach()
        finally:
            failures=[]
            for future in self.media_futures:
                try:
                    future.result()
                except Exception as error:
                    failures.append(str(error))
            self.media_writer.shutdown(wait=True)
            self.report['raw_media_write_failures']=failures
            if failures:
                self.report['protective_return_reason']='RAW_MEDIA_PERSISTENCE_FAILED'
                self.report['task_success']=False
                self.report['metrics']['task_success']=False
