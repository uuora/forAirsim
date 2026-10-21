"""Owned development flight with independent own-state control and perception.

Independent startup/landing are outside the measured low-speed stage. Only
current RGB identity plus paired depth enters planning. Actor layout and
segmentation are constructor/evaluation data; no target pose is read online.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import queue
import threading

import airsim
import cv2
import numpy as np
from camera_geometry import scene_intrinsics
from setup_scene import vector
from t1_balloon_data_probe import ControlledMeshRegistry
from t1_ground_perception_probe import capture, digest, ground_states, visible_mask
from t1_identity_sequence_probe_v1p3 import observer_observation
from t1_flight_observer import FlightObserver
from t1_semantic_perception import Candidate, SemanticFrame
from t1_candidate_surface_v2 import temporal_candidate_surface_estimate, SurfaceRejection
from t1_pose_geometry import CameraPose, SceneCameraModel
from t1_pose_trajectory_planner import PosePlannerConfig, plan_pose_trajectory, preview_command
from t1_identity_motion_policy import IdentityMotionPolicy, MotionPolicyConfig
from trajectory_execution_guard import boundary_command
from t1_async_control import AsyncControl

ROOT=Path(__file__).resolve().parents[1]


def json_scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError('UNSUPPORTED_RECEIPT_VALUE_TYPE')


def save(path,value):
    encoded=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=json_scalar)
    Path(path).write_text(encoded+'\n',encoding='utf-8')


def frame_from_dict(value):
    candidates=tuple(Candidate(c['candidate_id'],tuple(c['bbox_xyxy_pixels']),c['detector_score'],
                               c['source_class_name'],c['truncated']) for c in value['candidates'])
    return SemanticFrame(value['session_id'],value['target_generation'],value['frame_id'],
        value['frame_sequence'],value['image_sha256'],*value['image_size_wh'],
        value['readback_receipt_monotonic_s'],value['candidates_ready_monotonic_s'],candidates,
        detector_report_json=json.dumps(value['detector_report'],allow_nan=False))


def planned_slots(plan):
    rows=[];start=0.;index=0
    for phase in plan['phases']:
        n=int(np.ceil(phase['duration_s']/plan['period_s']-1e-9))
        for local_index in range(n):
            due=start+local_index*plan['period_s']
            deadline=min(due+plan['period_s'],start+phase['duration_s'])
            rows.append(dict(slot_index=index,phase_id=phase['phase_id'],offset_s=due,deadline_offset_s=deadline))
            index+=1
        start+=phase['duration_s']
    return rows,start


class Flight:
    def __init__(self, client, translation, world_z, low, high, log):
        self.client=client;self.translation=np.asarray(translation,float)
        self.world_z=float(world_z);self.local_z=world_z-float(translation[2])
        self.low=np.asarray(low);self.high=np.asarray(high)
        self.last_xy=[0.,0.];self.last_command=time.monotonic()
        self.log=log;self.armed=False;self.stage='STARTUP';self.target_lease_end=None
        self.command_sequence=0;self.rpc_fault=False;self.receipt_failure=False

    def emit(self,row):
        try:self.log(row)
        except BaseException:
            self.receipt_failure=True
            if self.stage!='LANDING_RECOVERY':raise

    def state(self, *, recovery=False):
        raw=self.client.getMultirotorState(vehicle_name='DroneA')
        if self.rpc_fault and not recovery:raise RuntimeError('OWNED_COMMAND_RPC_ERROR')
        k=raw.kinematics_estimated
        out=dict(host_receipt_monotonic_s=time.monotonic(),state_timestamp_ns=int(raw.timestamp),
            position_world_ned_m=(np.asarray(vector(k.position))+self.translation).tolist(),
            velocity_ned_m_s=vector(k.linear_velocity),acceleration_ned_m_s2=vector(k.linear_acceleration),
            orientation_xyzw=[k.orientation.x_val,k.orientation.y_val,k.orientation.z_val,k.orientation.w_val],
            landed=raw.landed_state==airsim.LandedState.Landed)
        if not all(np.isfinite(out[key]).all() for key in ('position_world_ned_m','velocity_ned_m_s',
                                                         'acceleration_ned_m_s2','orientation_xyzw')):
            raise RuntimeError('NONFINITE_SELF_STATE')
        return out

    def issue(self, xy, duration, purpose, *, gate=None):
        xy=np.asarray(xy,float)
        if xy.shape!=(2,) or not np.isfinite(xy).all() or not 0<duration<=.12:
            raise ValueError('INVALID_SHORT_COMMAND')
        if np.linalg.norm(xy)>.03000001:
            raise ValueError('SHORT_COMMAND_CAP')
        dispatched=time.monotonic()
        if gate:
            if dispatched>gate['latest_dispatch_monotonic_s']+1e-9:
                raise RuntimeError('SELF_STATE_OR_TARGET_DISPATCH_DEADLINE_EXPIRED')
            if 'command_end_deadline_monotonic_s' in gate and dispatched+duration>gate['command_end_deadline_monotonic_s']+1e-9:
                raise RuntimeError('COMMAND_DISPATCH_DEADLINE_EXPIRED')
        future=self.client.moveByVelocityZAsync(float(xy[0]),float(xy[1]),self.local_z,float(duration),
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False,0.),vehicle_name='DroneA')
        self.last_xy=xy.tolist();self.last_command=dispatched
        self.target_lease_end=dispatched+duration if purpose=='CURRENT_TARGET_TRAJECTORY' else None
        row=dict(kind='COMMAND',purpose=purpose,stage=self.stage,dispatch_monotonic_s=dispatched,
            returned_monotonic_s=time.monotonic(),velocity_xy_m_s=xy.tolist(),duration_s=float(duration),
            local_z_ned_m=self.local_z,target_frame_id=(gate.get('current_binding') or {}).get('frame_id') if gate and purpose=='CURRENT_TARGET_TRAJECTORY' else None,
            target_end_deadline_monotonic_s=gate.get('command_end_deadline_monotonic_s') if gate else None)
        self.command_sequence+=1;row['command_id']=f'{self.stage}_{self.command_sequence}'
        def completed(done):
            # Callback executes when this owner's RPC loop receives the reply.
            # A False task result may reflect replacement by the explicit zero;
            # the raw outcome is recorded without claiming server application.
            error=done.error
            self.rpc_fault=self.rpc_fault or error is not None
            try:
                self.log(dict(kind='COMMAND_RPC_RESULT',stage=row['stage'],command_id=row['command_id'],
                    purpose=purpose,received_monotonic_s=time.monotonic(),
                    status='ERROR' if error is not None else 'RETURNED_TRUE' if done.result is True else
                           'RETURNED_FALSE' if done.result is False else 'OTHER_RESULT',
                    error_type=type(error).__name__ if error is not None else None))
            except BaseException:
                # Never escape a Future callback before msgpack sets its done
                # flag. A failed receipt writer latches a visible control fault.
                self.rpc_fault=True
        future.attach_callback(completed)
        self.emit(row)
        return row

    def protection(self, state):
        p=np.asarray(state['position_world_ned_m']);v=np.asarray(state['velocity_ned_m_s'])
        a=np.asarray(state['acceleration_ned_m_s2'])
        reasons=[]
        if np.any(p<self.low) or np.any(p>self.high):reasons.append('MEASURED_WORKSPACE')
        if abs(p[2]-self.world_z)>.12:reasons.append('MEASURED_FIXED_HEIGHT')
        if np.linalg.norm(v[:2])>.06:reasons.append('MEASURED_XY_SPEED')
        if np.linalg.norm(v)>.15:reasons.append('MEASURED_TOTAL_SPEED')
        if np.linalg.norm(a)>.3:reasons.append('MEASURED_ACCELERATION')
        q=state['orientation_xyzw'];x,y,z,w=q
        rpy=np.degrees([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),
                        np.arcsin(np.clip(2*(w*y-z*x),-1,1)),
                        np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))])
        if max(abs(rpy))>5.:reasons.append('MEASURED_ATTITUDE')
        return reasons

    def brake_tick(self, state=None):
        state=state or self.state();now=time.monotonic()
        envelope=boundary_command(state['position_world_ned_m'],[0.,0.],self.last_xy,
            self.low,self.high,.03,.15,max(1e-12,min(max(now-self.last_command,0.),.4)),1.,.02)
        # A boundary refusal exits the experiment; lifecycle recovery still
        # commands zero/lands using own state, never stale target geometry.
        if not envelope['accepted']:
            self.issue([0.,0.],.12,'OWN_STATE_EMERGENCY_ZERO')
            raise RuntimeError('SELF_BRAKE_ENVELOPE_REJECTED')
        return self.issue(envelope['command_velocity_xy_m_s'],.12,'OWN_STATE_BRAKE_OR_HOLD')

    def wait(self, deadline, *, guarded=True):
        while time.monotonic()<deadline:
            state=self.state();state.update(kind='SELF_STATE',stage=self.stage)
            if guarded:
                state['protection_reasons']=self.protection(state)
            self.log(state)
            if guarded and state['protection_reasons']:
                self.issue([0.,0.],.12,'OWN_STATE_PROTECTION_ZERO')
                raise RuntimeError('SELF_STATE_PROTECTION')
            now=time.monotonic()
            if self.target_lease_end is None or now>=self.target_lease_end:
                self.brake_tick(state)
            next_wake=min(deadline,time.monotonic()+.05)
            if self.target_lease_end is not None:next_wake=min(next_wake,self.target_lease_end)
            time.sleep(max(0.,next_wake-time.monotonic()))

    def complete_target_lease(self, command):
        """Explicit stop before any blocking capture/inference/persistence.

        AirSim duration is not assumed to reset the controller setpoint. The
        actual clear receipt and sampled stopping response are recorded.
        """
        lease_end=command['dispatch_monotonic_s']+command['duration_s']
        if np.linalg.norm(command['velocity_xy_m_s'])>1e-12:self.wait(lease_end)
        deadline=time.monotonic()+1.;stable=None;clear=None;samples=[]
        while time.monotonic()<deadline:
            state=self.state();reasons=self.protection(state)
            self.log(dict(state,kind='SELF_STATE',stage=self.stage,protection_reasons=reasons))
            if reasons:raise RuntimeError('SELF_STATE_PROTECTION_DURING_BRAKE')
            braking=self.brake_tick(state)
            if np.linalg.norm(braking['velocity_xy_m_s'])<1e-12 and clear is None:
                clear=braking['dispatch_monotonic_s']
            stopped=clear is not None and np.linalg.norm(state['velocity_ned_m_s'][:2])<=.005
            stable=(time.monotonic() if stable is None else stable) if stopped else None
            samples.append(dict(receipt_monotonic_s=state['host_receipt_monotonic_s'],
                                velocity_xy_m_s=state['velocity_ned_m_s'][:2]))
            if stable is not None and time.monotonic()-stable>=.05:
                result=dict(status='OWN_STATE_SAMPLED_STOPPED_BEFORE_BLOCKING_WORK',
                    declared_lease_end_monotonic_s=lease_end,target_setpoint_cleared_monotonic_s=clear,
                    clear_after_declared_lease_s=clear-lease_end,completed_monotonic_s=time.monotonic(),samples=samples)
                self.log(dict(result,kind='TARGET_LEASE_STOP',stage=self.stage,target_frame_id=command['target_frame_id']))
                return result
            time.sleep(.025)
        raise RuntimeError('OWN_STATE_STOP_TIMEOUT')

    def start(self):
        self.client.enableApiControl(True,vehicle_name='DroneA')
        if not self.client.armDisarm(True,vehicle_name='DroneA'):raise RuntimeError('ARM_REJECTED')
        self.armed=True
        self.client.takeoffAsync(timeout_sec=15,vehicle_name='DroneA').join()
        self.client.moveToZAsync(self.local_z,.3,timeout_sec=40,
            yaw_mode=airsim.YawMode(False,0.),vehicle_name='DroneA').join()
        self.last_command=time.monotonic();stable=None;deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            state=self.state();self.log(dict(state,kind='SELF_STATE',stage='SETTLING'))
            ok=abs(state['position_world_ned_m'][2]-self.world_z)<.06 and np.linalg.norm(state['velocity_ned_m_s'])<.02
            stable=(time.monotonic() if stable is None else stable) if ok else None
            self.issue([0.,0.],.12,'OWN_STATE_SETTLE')
            if stable is not None and time.monotonic()-stable>=.5:return
            time.sleep(.05)
        raise RuntimeError('TAKEOFF_SETTLE_TIMEOUT')

    def land(self):
        self.stage='LANDING_RECOVERY';self.target_lease_end=None
        if self.client.isApiControlEnabled(vehicle_name='DroneA'):
            self.emit(dict(self.state(recovery=True),kind='SELF_STATE',stage='RECOVERY_ENTRY',
                          prior_command_rpc_error=self.rpc_fault))
            # AirSim landing is a separate vertical lifecycle command, outside
            # the .03 m/s XY experimental envelope.
            self.issue([0.,0.],.12,'OWN_STATE_BEFORE_LAND')
            time.sleep(.2)
            self.client.landAsync(timeout_sec=45,vehicle_name='DroneA').join()
            deadline=time.monotonic()+45;stable=None
            while time.monotonic()<deadline:
                state=self.state(recovery=True);self.emit(dict(state,kind='SELF_STATE',stage=self.stage))
                contact=self.client.simGetCollisionInfo(vehicle_name='DroneA').object_name
                ok=(.65<state['position_world_ned_m'][2]<1.1 and
                    contact.startswith('Ground') and np.linalg.norm(state['velocity_ned_m_s'])<.02)
                stable=(time.monotonic() if stable is None else stable) if ok else None
                if stable is not None and time.monotonic()-stable>=1.:break
                time.sleep(.1)
            else:raise RuntimeError('LANDING_CONFIRM_TIMEOUT')
            self.client.cancelLastTask(vehicle_name='DroneA')
            if not self.client.armDisarm(False,vehicle_name='DroneA'):raise RuntimeError('DISARM_REJECTED')
            self.armed=False;self.client.enableApiControl(False,vehicle_name='DroneA')
        # SimpleFlight publishes Landed asynchronously after disarm/release.
        deadline=time.monotonic()+5
        while True:
            try:return ground_states(self.client)
            except ValueError:
                if time.monotonic()>=deadline:raise
                time.sleep(.1)


def setup_scene(client, output, cfg):
    registry=ControlledMeshRegistry();assets={x.casefold():x for x in client.simListAssets()}
    if not client.simSetSegmentationObjectID('.*',0,True):raise RuntimeError('BACKGROUND_ID_FAILED')
    for name,key,asset,scale,segid in (
        ('T1FlightTarget','target_world_ned_m','sphere',cfg['target_scale_xyz'],42),
        ('T1FlightDistractor','distractor_world_ned_m','sphere',cfg['target_scale_xyz'],43),
        ('T1FlightOccluder','inactive_world_ned_m','1m_cube_chamfer',cfg['occluder_scale_xyz'],44)):
        registry.ensure(client,name,assets[asset],cfg[key],scale)
        if not client.simSetSegmentationObjectID(name,segid):raise RuntimeError('INSTANCE_ID_FAILED')
        if client.simGetSegmentationObjectID(name.lower())!=segid:raise RuntimeError('INSTANCE_ID_READBACK_FAILED')
    red=np.full((64,64,3),[55,35,235],np.uint8);stripe=red.copy();stripe[24:40]=245
    for name,key,image in (('T1FlightTarget','stripe',stripe),('T1FlightDistractor','plain',red)):
        path=output/(key+'_texture.png')
        if not cv2.imwrite(str(path),image) or not client.simSetObjectMaterialFromTexture(name,str(path)):
            raise RuntimeError('TEXTURE_ASSIGNMENT_FAILED')
    return registry,assets


def run(plan_path, output):
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8-sig'))
    if plan['status']!='FROZEN_BEFORE_FLIGHT' or plan['description']!='带白色横条的红色气球':
        raise ValueError('FROZEN_EXPLICIT_ATTRIBUTE_PLAN_REQUIRED')
    if plan.get('control_architecture')!='INDEPENDENT_OWN_STATE_THREAD_V2':
        raise ValueError('INDEPENDENT_CONTROL_PLAN_REQUIRED')
    for name,value in plan['source_sha256'].items():
        if digest(ROOT/name)!=value:raise ValueError('SOURCE_FREEZE_CHANGED')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'frames').mkdir();slots,total=planned_slots(plan)
    report=dict(status='RUNNING',kind='T1_IDENTITY_DEPTH_XY_ASYNC_DEVELOPMENT_FLIGHT',plan=str(Path(plan_path).resolve()),
        plan_sha256=digest(plan_path),configuration=plan,session_id=plan['session_id'],new_flights=0,
        formal_efficacy_results=False,virtual_camera_relocated=False,ground_only=False,
        started_utc=datetime.now(timezone.utc).isoformat(),slots=slots,frames=[],phase_transitions=[],
        safe_end={'grounded_and_released':False,'paused':False},errors=[],
        timestamp_scope='host readback age; render receipt is not exposure timestamp',
        truth_access='controlled constructor and independent segmentation scoring only',
        control_architecture=plan['control_architecture'],injected_delays=[],
        timing_scope='gate decision and independent lease stop are distinct; full completion is their later receipt with frame processing',
        ai_assistance='AI assisted implementation and evidence summary')
    save(output/'report.json',report)
    events=(output/'execution.jsonl').open('w',encoding='utf-8')
    event_queue=queue.Queue(maxsize=4096);event_overflow=threading.Event();recovery_backlog=[]
    def log(row):
        try:event_queue.put_nowait(row)
        except queue.Full:
            event_overflow.set()
            raise RuntimeError('CONTROL_EVENT_BUFFER_OVERFLOW')
    def flush_events():
        for row in recovery_backlog:
            events.write(json.dumps(row,ensure_ascii=False,allow_nan=False,default=json_scalar)+'\n')
        recovery_backlog.clear()
        while True:
            try:row=event_queue.get_nowait()
            except queue.Empty:break
            events.write(json.dumps(row,ensure_ascii=False,allow_nan=False,default=json_scalar)+'\n')
        events.flush()
    observer=None;flight=None;controller=None;client=airsim.MultirotorClient(timeout_value=10)
    try:
        if not client.ping() or client.simIsPause():raise RuntimeError('FRESH_UNPAUSED_SIM_REQUIRED')
        report['ground_before']=ground_states(client)
        body_world=np.asarray(vector(client.simGetObjectPose('DroneA').position))
        body_local=np.asarray(vector(client.simGetVehiclePose(vehicle_name='DroneA').position))
        translation=body_world-body_local
        report['world_from_local_translation_ned_m']=translation.tolist()
        low=[body_world[0]-.3,body_world[1]-.3,plan['world_z_ned_m']-.15]
        high=[body_world[0]+.3,body_world[1]+.3,plan['world_z_ned_m']+.15]
        registry,assets=setup_scene(client,output,plan['scene'])
        palette_path=ROOT/'external/AirSim/docs/seg_rgbs.txt'
        palette={int(line.split('\t')[0]):json.loads(line.split('\t')[1]) for line in palette_path.read_text().splitlines()}
        report['segmentation_palette_rgb']={str(n):palette[n] for n in (0,42,43,44)}
        flight=Flight(client,translation,plan['world_z_ned_m'],low,high,log)
        # Model warmup precedes arm. The real onboard image is used; camera
        # extrinsics are never moved to manufacture an airborne view.
        warm,_,_,_=capture(client);warm_path=output/'warmup.png'
        cv2.imwrite(str(warm_path),warm,[cv2.IMWRITE_PNG_COMPRESSION,0])
        observer=FlightObserver(plan['detector_config_path'])
        observer.begin(plan['session_id'],plan['description'],str(warm_path),recovery_observability='VISUAL_ATTRIBUTE')
        report['flight_start_attempted']=True
        flight.start();report['new_flights']=1;report['flight_stage_entered']=True
        flight.stage='PERCEPTION_XY';report['low_speed_stage_start_state']=flight.state()
        def control_flight_factory():
            # msgpack RPC clients and their futures remain on their owner thread.
            # msgpack's 1 s timeout polling decrements every new future at the
            # next shared tick; a fractional value may expire a very new call.
            # This transport setting is separate from observation/lease limits.
            control_client=airsim.MultirotorClient(timeout_value=2.)
            control_flight=Flight(control_client,translation,plan['world_z_ned_m'],low,high,log)
            control_flight.stage='PERCEPTION_XY'
            return control_flight
        def control_policy_factory():
            return IdentityMotionPolicy(plan['session_id'],1,plan['description'],
                config=MotionPolicyConfig(workspace_low_ned_m=tuple(low),workspace_high_ned_m=tuple(high)))
        controller=AsyncControl(control_flight_factory,control_policy_factory,period_s=plan['self_monitor_period_s'])
        controller.start()
        def control_wait(deadline):
            while time.monotonic()<deadline:
                controller.check()
                time.sleep(min(.01,max(0.,deadline-time.monotonic())))
            controller.check()
        def inject_delay(slot,boundary):
            for injection in plan.get('injected_perception_delays',[]):
                if injection['slot_index']==slot['slot_index'] and injection['boundary']==boundary:
                    started=time.monotonic()
                    time.sleep(injection['duration_s'])
                    report['injected_delays'].append(dict(slot_index=slot['slot_index'],boundary=boundary,
                        started_monotonic_s=started,completed_monotonic_s=time.monotonic(),
                        duration_s=injection['duration_s']))
                    controller.check()
        image,_,seg,_=capture(client)
        setup_truth={key:visible_mask(seg,palette[n]) for key,n in (('target',42),('distractor',43),('occluder',44))}
        report['onboard_setup_mask_evaluation']=setup_truth
        if any(setup_truth[key]['visible_pixel_count']<500 for key in ('target','distractor')):
            raise RuntimeError('ONBOARD_SETUP_VISIBILITY_FAILED')
        intrinsics=scene_intrinsics(client.simGetCameraInfo('front_center',vehicle_name='DroneA'),640,480)
        camera=SceneCameraModel.from_k(intrinsics['K_nominal'],640,480)
        pcfg=replace(PosePlannerConfig(),fixed_world_z_m=plan['world_z_ned_m'],
                     position_min_ned_m=tuple(low),position_max_ned_m=tuple(high))
        epoch=time.monotonic();report['epoch_monotonic_s']=epoch;phase=None
        for slot in slots:
            due=epoch+slot['offset_s'];deadline=epoch+slot['deadline_offset_s']
            slot.update(due_monotonic_s=due,deadline_monotonic_s=deadline)
            control_wait(due)
            if time.monotonic()>=deadline:
                slot.update(status='NOT_CAPTURED_SLOT_DEADLINE',observed_monotonic_s=time.monotonic());continue
            if phase!=slot['phase_id']:
                phase=slot['phase_id'];transition_start=time.monotonic()
                registry.ensure(client,'T1FlightOccluder',assets['1m_cube_chamfer'],
                    plan['scene']['occluder_world_ned_m'] if phase=='hidden' else plan['scene']['inactive_world_ned_m'],
                    plan['scene']['occluder_scale_xyz'])
                report['phase_transitions'].append(dict(phase_id=phase,started_monotonic_s=transition_start,
                                                        completed_monotonic_s=time.monotonic()))
            frame_id=f"{plan['session_id']}_f{slot['slot_index']:03d}"
            row=dict(slot,frame_id=frame_id,frame_sequence=slot['slot_index'],session_id=plan['session_id'],
                description=plan['description'],world_from_local_translation_ned_m=translation.tolist(),
                scene_intrinsics=intrinsics,media={},status='FRAME_STARTED')
            report['frames'].append(row)
            image,depth,seg,meta=capture(client);row.update(meta,image_size_wh=[image.shape[1],image.shape[0]])
            rgb_path=output/'frames'/(frame_id+'_rgb.png')
            if not cv2.imwrite(str(rgb_path),image,[cv2.IMWRITE_PNG_COMPRESSION,0]):raise RuntimeError('RGB_SAVE_FAILED')
            row['media']['rgb']=dict(path=str(rgb_path),sha256=digest(rgb_path))
            inject_delay(slot,'AFTER_CAPTURE_BEFORE_OBSERVER')
            controller.check()
            observation=observer_observation(row)
            result=observer.process(observation);row['observer_result']=result
            frame=frame_from_dict(result['semantic_frame']);selection=result['temporal_identity']
            rgb_pose=CameraPose(**{k:meta['response_camera_poses'][0][k] for k in ('position_local_ned_m','orientation_xyzw')},
                                world_from_local_translation_ned_m=translation.tolist())
            depth_pose=CameraPose(**{k:meta['response_camera_poses'][1][k] for k in ('position_local_ned_m','orientation_xyzw')},
                                  world_from_local_translation_ned_m=translation.tolist())
            try:
                surface=temporal_candidate_surface_estimate(frame,description=plan['description'],now=time.monotonic(),
                    depth=depth,camera=camera,rgb_pose=rgb_pose,depth_pose=depth_pose,selection_certificate=selection,
                    pairing_contract=dict(frame_id=frame_id,image_sha256=frame.image_sha256,depth_kind='DepthPerspective',
                        rgb_readback_timestamp_ns=meta['response_camera_poses'][0]['render_readback_timestamp_ns'],
                        depth_readback_timestamp_ns=meta['response_camera_poses'][1]['render_readback_timestamp_ns'],
                        intrinsics_pairing_scope='caller_declared_equal_RGB_depth_Scene_intrinsics'))
            except SurfaceRejection as error:
                surface=dict(status='SURFACE_REJECTED',reason=str(error),flight_authorized=False)
            row['surface']=surface
            state=controller.latest_state();desired=[0.,0.]
            if time.monotonic()-state['host_receipt_monotonic_s']>.12:
                raise RuntimeError('PLANNING_SELF_STATE_EXPIRED')
            row['planning_self_state']=state
            if plan['mode']=='SHORT_XY' and surface['status']=='SURFACE_GEOMETRY_ACCEPTED':
                planning_started=time.monotonic()
                row['planner']=plan_pose_trajectory(surface=surface,camera=camera,camera_pose=rgb_pose,
                    position_ned_m=state['position_world_ned_m'],velocity_ned_m_s=state['velocity_ned_m_s'],
                    acceleration_ned_m_s2=state['acceleration_ned_m_s2'],config=pcfg)
                row['planning_wall_s']=time.monotonic()-planning_started
                desired=preview_command(row['planner'],.4)
            # Only the control thread samples own state and evaluates/dispatches
            # the gate. This producer cannot renew a prior observation's lease.
            ticket=controller.submit(frame,selection,surface,desired)
            row['ticket_id']=ticket
            decision=controller.await_decision(ticket,timeout=.5)
            row.update(decision)
            if row.get('command') is None:row.pop('command',None)
            if row.get('command_self_state') is None:row.pop('command_self_state',None)
            gate=row['motion_gate']
            row['motion_decision_available_monotonic_s']=time.monotonic()
            slot.update(status='CAPTURED',frame_id=frame_id)
            inject_delay(slot,'AFTER_CONTROL_DECISION_BEFORE_MEDIA_SAVE')
            # Truth/media are persisted AFTER motion decision and are never
            # included in the RGB callback, surface or planner inputs.
            row['evaluation_truth']={key:visible_mask(seg,palette[n]) for key,n in (
                ('target',42),('distractor',43),('occluder',44))}
            for key,path,values in (('depth',output/'frames'/(frame_id+'_depth.npy'),depth),
                                    ('segmentation',output/'frames'/(frame_id+'_seg.png'),seg)):
                if key=='depth':np.save(path,values,allow_pickle=False)
                elif not cv2.imwrite(str(path),values,[cv2.IMWRITE_PNG_COMPRESSION,0]):raise RuntimeError('SEG_SAVE_FAILED')
                row['media'][key]=dict(path=str(path),sha256=digest(path))
            row.update(status='FRAME_COMPLETED',frame_processing_completed_monotonic_s=time.monotonic())
            log(dict(kind='FRAME_DECISION',frame_id=frame_id,frame_sequence=frame.frame_sequence,
                     temporal_status=selection['status'],gate_status=gate['status'],surface_status=surface['status'],
                     decision_completed_monotonic_s=row['motion_decision_available_monotonic_s']))
            flush_events()
            with (output/'frame_results.jsonl').open('a',encoding='utf-8') as ledger:
                ledger.write(json.dumps(row,ensure_ascii=False,allow_nan=False,default=json_scalar)+'\n')
        control_wait(epoch+total+.5)
        report['low_speed_stage_end_state']=controller.latest_state()
        selected=sum(r.get('surface',{}).get('status')=='SURFACE_GEOMETRY_ACCEPTED' for r in report['frames'])
        target_motion=sum(bool(np.linalg.norm(r.get('command',{}).get('velocity_xy_m_s',[0.,0.]))>1e-6 and
                          r.get('command',{}).get('purpose')=='CURRENT_TARGET_TRAJECTORY') for r in report['frames'])
        report['integration_counts']=dict(surface_accepted_frames=selected,target_motion_frames=target_motion)
        report['status']='PASS' if selected>0 and (plan['mode']=='ZERO_GATE' or target_motion>0) else 'FAIL'
        if report['status']=='FAIL':report['errors'].append('NO_VALID_INTEGRATION_OUTPUT')
    except Exception as error:
        report['status']='FAIL';report['errors'].append(type(error).__name__)
        # Only our closed category strings are logged; network/provider errors
        # remain categorical so credentials/endpoints cannot enter receipts.
        if isinstance(error,RuntimeError) and str(error).isupper() and len(str(error))<100:
            report['errors'].append(str(error))
    finally:
        control_stopped=True
        if controller:
            try:
                report['control_close']=controller.close(timeout=2.)
            except Exception as error:
                report['status']='FAIL';report['errors'].append('CONTROL_CLOSE_'+type(error).__name__)
                if controller.is_alive():
                    try:controller.close(timeout=10.)
                    except Exception as retry_error:
                        report['errors'].append('CONTROL_CLOSE_RETRY_'+type(retry_error).__name__)
                control_stopped=not controller.is_alive()
            receipts=controller.receipts()
            decisions=controller.decisions()
            report['control_scheduling']=controller.scheduling_receipt()
            for row in report['frames']:
                ticket=row.get('ticket_id')
                if ticket in decisions and 'motion_gate' not in row:
                    row.update({key:value for key,value in decisions[ticket].items() if value is not None})
                if ticket in receipts:row['lease_stop']=receipts[ticket]
                if row.get('status')=='FRAME_COMPLETED':
                    row['decision_completed_monotonic_s']=max(row['frame_processing_completed_monotonic_s'],
                        row.get('lease_stop',{}).get('completed_monotonic_s',row['motion_decision_available_monotonic_s']))
                    row['decision_on_time']=row['decision_completed_monotonic_s']<=row['deadline_monotonic_s']
                    slots[row['slot_index']]['decision_on_time']=row['decision_on_time']
        try:
            if not control_stopped:raise RuntimeError('CONTROL_THREAD_STILL_ACTIVE_PRESERVE_SIM')
            # Free bounded buffer capacity without disk work before recovery.
            while True:
                try:recovery_backlog.append(event_queue.get_nowait())
                except queue.Empty:break
            report['ground_after']=flight.land() if flight else ground_states(client)
            report['safe_end']['grounded_and_released']=True
            client.simPause(True);report['safe_end']['paused']=bool(client.simIsPause())
        except Exception as error:
            report['status']='FAIL';report['errors'].append('LANDING_RECOVERY_'+type(error).__name__)
        # Recovery completes before final disk serialization can fail.
        try:
            with (output/'frame_results.jsonl').open('w',encoding='utf-8') as ledger:
                for row in report['frames']:
                    ledger.write(json.dumps(row,ensure_ascii=False,allow_nan=False,default=json_scalar)+'\n')
        except Exception as error:
            report['status']='FAIL';report['errors'].append('FINAL_LEDGER_'+type(error).__name__)
        if observer:
            try:observer.close();report['observer_close']=observer.close_receipt
            except Exception as error:report['errors'].append('OBSERVER_CLOSE_'+type(error).__name__)
        for slot in slots:
            if 'status' not in slot:
                current=next((r for r in report['frames'] if r['slot_index']==slot['slot_index']),None)
                slot['status']=('CAPTURED_PROCESSING_FAILED' if current and current['media'].get('rgb') else
                               'FRAME_STARTED_CAPTURE_FAILED' if current else 'INTERRUPTED_NOT_CAPTURED')
        flush_events();events.close()
        report['event_buffer']=dict(capacity=4096,overflow=event_overflow.is_set(),
            lifecycle_receipt_failure=flight.receipt_failure if flight else False,
            scope='bounded memory buffer; after worker join up to 4096 rows moved to recovery backlog; single main-thread file writer')
        if event_overflow.is_set():
            report['status']='FAIL';report['errors'].append('CONTROL_EVENT_BUFFER_OVERFLOW')
        evidence=[json.loads(line) for line in (output/'execution.jsonl').read_text(encoding='utf-8').splitlines()]
        report['observed_airborne']=any(row.get('kind')=='SELF_STATE' and row.get('landed') is False for row in evidence)
        report['new_flights']=int(report['observed_airborne'])
        report['execution_log']=dict(path=str(output/'execution.jsonl'),sha256=digest(output/'execution.jsonl'))
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        save(output/'report.json',report)
    print(json.dumps(dict(status=report['status'],errors=report['errors'],frames=len(report['frames']),
        safe_end=report['safe_end'],integration_counts=report.get('integration_counts')),ensure_ascii=False,default=json_scalar))
    return report['status']=='PASS' and all(report['safe_end'].values())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(0 if run(args.plan,args.output) else 2)
