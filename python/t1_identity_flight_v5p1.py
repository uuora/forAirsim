"""Background semantic review handoff followed by measured V5P1 flight.

Lifecycle and motion budgets reuse the frozen V2 implementation. Acquisition
uses current onboard RGB and zero requested XY motion; its evidence remains
separate from the unchanged measured phase denominator.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import threading
import time
import math

import airsim
import cv2
import numpy as np
from camera_geometry import scene_intrinsics
from setup_scene import vector
from t1_ground_perception_probe import capture, digest, ground_states, visible_mask
from t1_identity_sequence_probe_v1p3 import observer_observation
from t1_identity_flight_v2 import Flight, planned_slots, frame_from_dict, setup_scene, json_scalar, save
from t1_pose_geometry import CameraPose, SceneCameraModel
from t1_pose_trajectory_planner import PosePlannerConfig, plan_pose_trajectory, preview_command
from t1_identity_motion_policy import IdentityMotionPolicy, MotionPolicyConfig
from t1_async_control import AsyncControl
from t1_active_stall import active_stall_eligibility

ROOT=Path(__file__).resolve().parents[1]


def acquisition_slots(plan):
    """Predeclare all 50 bounded initial slots independently from measurement."""
    settings=plan.get('initial_identity_acquisition', {})
    if settings.get('maximum_duration_s') != 20. or settings.get('period_s') != .4:
        raise ValueError('INITIAL_ACQUISITION_GRID_MUST_BE_20S_0P4')
    rows,total=planned_slots(dict(period_s=.4,phases=[dict(phase_id='initial_acquisition',duration_s=20.)]))
    for slot in rows:slot['experiment_scope']='INITIAL_IDENTITY_ACQUISITION'
    return rows,total


def frame_row(slot,session_id,sequence_base,*,acquisition=False):
    """Missed actual slots consume sequence numbers; unused acquisition tail does not."""
    if (isinstance(sequence_base,bool) or not isinstance(sequence_base,int) or sequence_base<0
            or isinstance(slot.get('slot_index'),bool) or not isinstance(slot.get('slot_index'),int)
            or slot['slot_index']<0):
        raise ValueError('NONNEGATIVE_GLOBAL_FRAME_SEQUENCE_REQUIRED')
    sequence=sequence_base+slot['slot_index']
    return dict(slot,frame_id=f'{session_id}_f{sequence:03d}',frame_sequence=sequence,
                experiment_scope='INITIAL_IDENTITY_ACQUISITION' if acquisition else 'MEASURED_RESEARCH',
                session_id=session_id,media={},status='FRAME_STARTED')


def acquisition_established(row,*,now,deadline):
    """Require a current tracked selection and paired surface before the 20s cap."""
    if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
               for v in (now,deadline)) or now>=deadline:
        return False
    observer=row.get('observer_result',{})
    selection=observer.get('temporal_identity',{})
    frame=observer.get('semantic_frame',{})
    receipt=frame.get('readback_receipt_monotonic_s')
    if not isinstance(receipt,(int,float)) or isinstance(receipt,bool) or not math.isfinite(receipt):
        return False
    return (row.get('status')=='FRAME_COMPLETED'
            and selection.get('status')=='TEMPORAL_LOCAL_ASSOCIATION_VALID'
            and selection.get('state')=='TRACKING'
            and row.get('surface',{}).get('status')=='SURFACE_GEOMETRY_ACCEPTED'
            and 0<=now-receipt<=.5)


def close_acquisition_tail(slots,completed_slot_index,now):
    for slot in slots:
        if slot['slot_index']>completed_slot_index:
            slot.update(status='NOT_REQUIRED_AFTER_ACQUISITION',observed_monotonic_s=now)
    return completed_slot_index+1


def finalize_unresolved_slots(slots,frames,*,main_not_entered=False):
    """Retain all original slots even when initial identity never becomes usable."""
    for slot in slots:
        if 'status' in slot:continue
        current=next((r for r in frames if r['slot_index']==slot['slot_index']),None)
        slot['status']=('CAPTURED_PROCESSING_FAILED' if current and current['media'].get('rgb') else
                        'FRAME_STARTED_CAPTURE_FAILED' if current else
                        'NOT_ENTERED_INITIAL_ACQUISITION' if main_not_entered else 'INTERRUPTED_NOT_CAPTURED')


def run(plan_path, output):
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8-sig'))
    if plan['status']!='FROZEN_BEFORE_FLIGHT' or plan['description']!='带白色横条的红色气球':
        raise ValueError('FROZEN_EXPLICIT_ATTRIBUTE_PLAN_REQUIRED')
    if plan.get('control_architecture')!='INDEPENDENT_CONTROL_REVIEW_HANDOFF_V5P1':
        raise ValueError('INDEPENDENT_REVIEW_HANDOFF_PLAN_REQUIRED')
    stall=plan.get('active_lease_stall',{})
    if (not isinstance(stall.get('enabled'),bool) or stall.get('maximum_injections')!=1
            or stall.get('minimum_remaining_actual_dispatch_lease_s')!=.04
            or stall.get('duration_s')!=(.65 if stall.get('enabled') else 0.)
            or (stall.get('enabled') and plan['mode']!='SHORT_XY')):
        raise ValueError('PREDECLARED_ACTIVE_STALL_CONTRACT_MISMATCH')
    for name,value in plan['source_sha256'].items():
        if digest(ROOT/name)!=value:raise ValueError('SOURCE_FREEZE_CHANGED')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'frames').mkdir();slots,total=planned_slots(plan)
    initial_slots,initial_total=acquisition_slots(plan)
    from t1_review_handoff_observer_v1p1 import ReacquisitionAwareFlightObserver as DelayedFlightObserver
    from t1_candidate_surface_v3 import temporal_candidate_surface_estimate, SurfaceRejection
    report=dict(status='RUNNING',kind='T1_IDENTITY_DEPTH_XY_CHAIN_REVIEW_DEVELOPMENT_FLIGHT',plan=str(Path(plan_path).resolve()),
        plan_sha256=digest(plan_path),configuration=plan,session_id=plan['session_id'],new_flights=0,
        formal_efficacy_results=False,virtual_camera_relocated=False,ground_only=False,
        started_utc=datetime.now(timezone.utc).isoformat(),slots=slots,frames=[],phase_transitions=[],
        acquisition_slots=initial_slots,acquisition_frames=[],
        acquisition_status='NOT_STARTED',measurement_entered=False,
        acquisition_scope='bounded 20s current-observation initialization; excluded from measured phases',
        safe_end={'grounded_and_released':False,'paused':False},errors=[],
        timestamp_scope='host readback age; render receipt is not exposure timestamp',
        truth_access='controlled constructor and independent segmentation scoring only',
        control_architecture=plan['control_architecture'],injected_delays=[],
        active_lease_stall_status='DECLARED_NOT_TRIGGERED' if plan.get('active_lease_stall',{}).get('enabled') else 'NOT_DECLARED',
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
        observer=DelayedFlightObserver(plan['detector_config_path'])
        observer.begin(plan['session_id'],plan['description'],str(warm_path),recovery_observability='VISUAL_ATTRIBUTE')
        report['flight_start_attempted']=True
        flight.start();report['new_flights']=1;report['flight_stage_entered']=True
        flight.stage='PERCEPTION_XY';report['acquisition_pre_controller_state']=flight.state()
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
        def inject_active_delay(row,slot):
            declaration=plan.get('active_lease_stall',{})
            if not declaration.get('enabled') or report['active_lease_stall_status']=='INJECTED':return
            cleared=controller.receipts()
            started=time.monotonic()
            eligibility=active_stall_eligibility(row,started,cleared_tickets=cleared)
            if not eligibility['eligible']:return
            # Main-thread blocking never changes the independent controller's
            # original target lease, source time or observation expiry.
            time.sleep(declaration['duration_s'])
            report['injected_delays'].append(dict(slot_index=slot['slot_index'],
                boundary='AFTER_ACTUAL_NONZERO_DISPATCH_BEFORE_MEDIA_SAVE',
                started_monotonic_s=started,completed_monotonic_s=time.monotonic(),
                duration_s=declaration['duration_s'],eligibility=eligibility,
                ticket_id=row['ticket_id'],frame_id=row['frame_id']))
            report['active_lease_stall_status']='INJECTED'
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
        def process_frame(slot,sequence_base,*,acquisition=False):
            row=frame_row(slot,plan['session_id'],sequence_base,acquisition=acquisition)
            frame_id=row['frame_id']
            row.update(description=plan['description'],world_from_local_translation_ned_m=translation.tolist(),
                       scene_intrinsics=intrinsics)
            report['acquisition_frames' if acquisition else 'frames'].append(row)
            image,depth,seg,meta=capture(client);row.update(meta,image_size_wh=[image.shape[1],image.shape[0]])
            rgb_path=output/'frames'/(frame_id+'_rgb.png')
            if not cv2.imwrite(str(rgb_path),image,[cv2.IMWRITE_PNG_COMPRESSION,0]):raise RuntimeError('RGB_SAVE_FAILED')
            row['media']['rgb']=dict(path=str(rgb_path),sha256=digest(rgb_path))
            if not acquisition:inject_delay(slot,'AFTER_CAPTURE_BEFORE_OBSERVER')
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
            if not acquisition and plan['mode']=='SHORT_XY' and surface['status']=='SURFACE_GEOMETRY_ACCEPTED':
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
            row['acquisition_zero_desired_xy_m_s']=[0.,0.] if acquisition else None
            row['motion_decision_available_monotonic_s']=time.monotonic()
            slot.update(status='CAPTURED',frame_id=frame_id)
            if not acquisition:inject_active_delay(row,slot)
            if not acquisition:inject_delay(slot,'AFTER_CONTROL_DECISION_BEFORE_MEDIA_SAVE')
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
            ledger_name='acquisition_frame_results.jsonl' if acquisition else 'frame_results.jsonl'
            with (output/ledger_name).open('a',encoding='utf-8') as ledger:
                ledger.write(json.dumps(row,ensure_ascii=False,allow_nan=False,default=json_scalar)+'\n')
            return row

        initial_epoch=time.monotonic();initial_deadline=initial_epoch+initial_total
        report.update(acquisition_status='RUNNING',acquisition_epoch_monotonic_s=initial_epoch,
                      acquisition_deadline_monotonic_s=initial_deadline,
                      acquisition_stage_start_state=controller.latest_state())
        sequence_base=None
        for slot in initial_slots:
            due=initial_epoch+slot['offset_s'];deadline=initial_epoch+slot['deadline_offset_s']
            slot.update(due_monotonic_s=due,deadline_monotonic_s=deadline)
            control_wait(due)
            if time.monotonic()>=deadline:
                slot.update(status='NOT_CAPTURED_SLOT_DEADLINE',observed_monotonic_s=time.monotonic())
                continue
            row=process_frame(slot,0,acquisition=True)
            now=time.monotonic()
            if acquisition_established(row,now=now,deadline=initial_deadline):
                sequence_base=close_acquisition_tail(initial_slots,slot['slot_index'],now)
                report.update(acquisition_status='TRACKING_AND_SURFACE_ESTABLISHED',
                    acquisition_completed_monotonic_s=now,acquisition_completed_slot_index=slot['slot_index'],
                    acquisition_stage_end_state=controller.latest_state(),
                    acquisition_elapsed_s=now-initial_epoch,
                    acquisition_selected_frame_id=row['frame_id'])
                break
        if sequence_base is None:
            report.update(acquisition_status='FAIL_INITIAL_IDENTITY_NOT_ESTABLISHED',
                          acquisition_completed_monotonic_s=time.monotonic(),
                          acquisition_stage_end_state=controller.latest_state())
            raise RuntimeError('INITIAL_IDENTITY_ACQUISITION_FAILED')
        # The original 8s or 7/2/7 measured window starts only after acquisition.
        epoch=time.monotonic();report['epoch_monotonic_s']=epoch;phase=None
        report.update(measurement_entered=True,measured_frame_sequence_base=sequence_base,
                      low_speed_stage_start_state=controller.latest_state())
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
            process_frame(slot,sequence_base)
        control_wait(epoch+total+.5)
        report['low_speed_stage_end_state']=controller.latest_state()
        selected=sum(r.get('surface',{}).get('status')=='SURFACE_GEOMETRY_ACCEPTED' for r in report['frames'])
        target_motion=sum(bool(np.linalg.norm(r.get('command',{}).get('velocity_xy_m_s',[0.,0.]))>1e-6 and
                          r.get('command',{}).get('purpose')=='CURRENT_TARGET_TRAJECTORY') for r in report['frames'])
        report['integration_counts']=dict(surface_accepted_frames=selected,target_motion_frames=target_motion)
        report['status']='PASS' if selected>0 and (plan['mode']=='ZERO_GATE' or target_motion>0) else 'FAIL'
        if report['status']=='FAIL':report['errors'].append('NO_VALID_INTEGRATION_OUTPUT')
        if plan.get('active_lease_stall',{}).get('enabled') and report['active_lease_stall_status']!='INJECTED':
            report['status']='FAIL';report['errors'].append('PREDECLARED_ACTIVE_STALL_NOT_TRIGGERED')
    except Exception as error:
        report['status']='FAIL';report['errors'].append(type(error).__name__)
        # Only our closed category strings are logged; network/provider errors
        # remain categorical so credentials/endpoints cannot enter receipts.
        if isinstance(error,RuntimeError) and str(error).isupper() and len(str(error))<100:
            report['errors'].append(str(error))
    finally:
        if report['acquisition_status'] in ('RUNNING','NOT_STARTED'):
            report['acquisition_status']='FAIL_OR_NOT_ENTERED'
        acquisition_epoch=report.get('acquisition_epoch_monotonic_s')
        report.setdefault('acquisition_completed_monotonic_s',
                          time.monotonic() if acquisition_epoch is not None else None)
        report.setdefault('acquisition_elapsed_s',
                          report['acquisition_completed_monotonic_s']-acquisition_epoch
                          if acquisition_epoch is not None else None)
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
            for row in report['acquisition_frames']+report['frames']:
                ticket=row.get('ticket_id')
                if ticket in decisions and 'motion_gate' not in row:
                    row.update({key:value for key,value in decisions[ticket].items() if value is not None})
                if ticket in receipts:row['lease_stop']=receipts[ticket]
                if row.get('status')=='FRAME_COMPLETED':
                    row['decision_completed_monotonic_s']=max(row['frame_processing_completed_monotonic_s'],
                        row.get('lease_stop',{}).get('completed_monotonic_s',row['motion_decision_available_monotonic_s']))
                    row['decision_on_time']=row['decision_completed_monotonic_s']<=row['deadline_monotonic_s']
                    row_slots=initial_slots if row.get('experiment_scope')=='INITIAL_IDENTITY_ACQUISITION' else slots
                    row_slots[row['slot_index']]['decision_on_time']=row['decision_on_time']
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
            for key,name in (('acquisition_frames','acquisition_frame_results.jsonl'),('frames','frame_results.jsonl')):
                with (output/name).open('w',encoding='utf-8') as ledger:
                    for row in report[key]:
                        ledger.write(json.dumps(row,ensure_ascii=False,allow_nan=False,default=json_scalar)+'\n')
        except Exception as error:
            report['status']='FAIL';report['errors'].append('FINAL_LEDGER_'+type(error).__name__)
        if observer:
            try:observer.close();report['observer_close']=observer.close_receipt
            except Exception as error:
                report['status']='FAIL';report['errors'].append('OBSERVER_CLOSE_'+type(error).__name__)
        finalize_unresolved_slots(initial_slots,report['acquisition_frames'])
        finalize_unresolved_slots(slots,report['frames'],main_not_entered=not report['measurement_entered'])
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
        safe_end=report['safe_end'],acquisition_frames=len(report['acquisition_frames']),
        acquisition_status=report['acquisition_status'],integration_counts=report.get('integration_counts')),ensure_ascii=False,default=json_scalar))
    return report['status']=='PASS' and all(report['safe_end'].values())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(0 if run(args.plan,args.output) else 2)
