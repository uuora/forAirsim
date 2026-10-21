"""Background semantic review handoff followed by measured V10 bounded evidence persistence and component-timed acquisition and strict stopping flight.

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
import traceback
import psutil
from t1_bounded_evidence_v1 import BoundedFrameEvidence, BoundedEventJournal

import airsim
import cv2
import numpy as np
from camera_geometry import scene_intrinsics
from setup_scene import vector
from t1_ground_perception_probe import digest, ground_states, visible_mask
from t1_capture_diagnostics_v1 import capture
from t1_identity_sequence_probe_v1p3 import observer_observation
from t1_identity_flight_v2 import Flight, planned_slots, frame_from_dict, setup_scene, json_scalar, save
from t1_pose_geometry import CameraPose, SceneCameraModel
from t1_pose_trajectory_planner import PosePlannerConfig, plan_pose_trajectory, preview_command
from t1_identity_motion_policy import MotionPolicyConfig
from t1_identity_motion_policy_v1p1 import IdentityMotionPolicyV1P1 as IdentityMotionPolicy
from t1_process_control_v1p1 import ProcessControlV1P1 as ProcessControlV1
from t1_process_trace_v1 import replay_closed_trace
from t1_feedback_dispatch_v1 import TimedFeedbackFlight as FeedbackFlight
from t1_control_settlement import completion_timing
from t1_active_stall import active_stall_eligibility

ROOT=Path(__file__).resolve().parents[1]
from t1_obstacle_scene import build_buildings,route_velocity,validate_scene


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


def run(plan_path, output, stop_file=None):
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8-sig'))
    from t1_process_plan_v1p1 import validate_plan
    validate_plan(plan)
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
    from t1_timed_review_observer_v1 import TimedReviewFlightObserver as DelayedFlightObserver
    from t1_candidate_surface_v3 import temporal_candidate_surface_estimate, SurfaceRejection
    report=dict(status='RUNNING',kind='T1_IDENTITY_DEPTH_XY_PROCESS_CONTROL_DEVELOPMENT_FLIGHT',plan=str(Path(plan_path).resolve()),
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
        own_state_settlement=plan['own_state_settlement'],
        completion_timing_policy=plan['completion_timing_policy'],
        target_dispatch_clear_budget=plan['target_dispatch_clear_budget'],
        settled_endpoint_policy=plan['settled_endpoint_policy'],
        active_lease_stall_status='DECLARED_NOT_TRIGGERED' if plan.get('active_lease_stall',{}).get('enabled') else 'NOT_DECLARED',
        timing_scope='gate decision and independent lease stop are distinct; full completion is their later receipt with frame processing',
        ai_assistance='AI assisted implementation and evidence summary')
    save(output/'report.json',report)
    journal=BoundedEventJournal(output/'execution.jsonl')
    evidence_writer=BoundedFrameEvidence(output)
    def log(row):journal.submit(row)
    resource_samples=[]
    def resource_sample(frame_id):
        memory=psutil.virtual_memory();swap=psutil.swap_memory();proc=psutil.Process()
        row=dict(kind='RESOURCE_SAMPLE',frame_id=frame_id,recorded_monotonic_s=time.monotonic(),
            available_physical_bytes=memory.available,process_rss_bytes=proc.memory_info().rss,
            swap_free_bytes=swap.free,scope='resource observation; does not prove allocation cause')
        resource_samples.append(row);log(row)
    def collect_evidence():
        for persisted in evidence_writer.drain():
            rows=report['acquisition_frames'] if persisted.get('experiment_scope')=='INITIAL_IDENTITY_ACQUISITION' else report['frames']
            matched=[row for row in rows if row['frame_id']==persisted['frame_id']]
            if len(matched)!=1:raise RuntimeError('PERSISTED_FRAME_ID_NOT_UNIQUE')
            matched[0].update(persisted)
        evidence_writer.check();journal.check()
    def flush_events():journal.check()
    observer=None;flight=None;controller=None;client=airsim.MultirotorClient(timeout_value=10)
    try:
        if not client.ping() or client.simIsPause():raise RuntimeError('FRESH_UNPAUSED_SIM_REQUIRED')
        report['ground_before']=ground_states(client)
        body_world=np.asarray(vector(client.simGetObjectPose('DroneA').position))
        body_local=np.asarray(vector(client.simGetVehiclePose(vehicle_name='DroneA').position))
        translation=body_world-body_local
        report['world_from_local_translation_ned_m']=translation.tolist()
        obstacle_scene=plan.get('obstacle_scene')
        if obstacle_scene:
            validate_scene(obstacle_scene)
            if np.linalg.norm(body_world[:2]-np.asarray(obstacle_scene['start_hover_ned_m'][:2]))>.03:
                raise RuntimeError('MANUAL_SCENE_START_XY_MISMATCH')
        low=obstacle_scene['workspace']['low_ned_m'] if obstacle_scene else [body_world[0]-.3,body_world[1]-.3,plan['world_z_ned_m']-.15]
        high=obstacle_scene['workspace']['high_ned_m'] if obstacle_scene else [body_world[0]+.3,body_world[1]+.3,plan['world_z_ned_m']+.15]
        report['low_speed_workspace']=dict(low_ned_m=low, high_ned_m=high)
        registry,assets=setup_scene(client,output,plan['scene'])
        if obstacle_scene:
            report['building_scene_readbacks']=build_buildings(client,registry,assets,obstacle_scene,output)
        palette_path=ROOT/'external/AirSim/docs/seg_rgbs.txt'
        palette={int(line.split('\t')[0]):json.loads(line.split('\t')[1]) for line in palette_path.read_text().splitlines()}
        report['segmentation_palette_rgb']={str(n):palette[n] for n in (0,42,43,44)}
        flight=Flight(client,translation,plan['world_z_ned_m'],low,high,log)
        # Model warmup precedes arm. The real onboard image is used; camera
        # extrinsics are never moved to manufacture an airborne view.
        warm,_,_,_=capture(client,log=log,frame_id=plan['session_id']+'_warmup');warm_path=output/'warmup.png'
        cv2.imwrite(str(warm_path),warm,[cv2.IMWRITE_PNG_COMPRESSION,0])
        observer=DelayedFlightObserver(plan['detector_config_path'])
        observer.begin(plan['session_id'],plan['description'],str(warm_path),recovery_observability='VISUAL_ATTRIBUTE')
        report['flight_start_attempted']=True
        flight.start();report['new_flights']=1;report['flight_stage_entered']=True
        flight.stage='PERCEPTION_XY';report['acquisition_pre_controller_state']=flight.state()
        controller=ProcessControlV1(dict(session_id=plan['session_id'], description=plan['description'],
            translation=translation.tolist(), world_z=plan['world_z_ned_m'], low=low, high=high,
            **({'stop_file':str(Path(stop_file).resolve())} if stop_file else {}),
            **({'obstacle_scene':obstacle_scene} if obstacle_scene else {})),
            output/'control_process',factory='t1_obstacle_control_factory:make_obstacle_airsim' if obstacle_scene else 't1_process_control_factories_v1:make_airsim')
        controller.start()
        def control_wait(deadline):
            if stop_file and Path(stop_file).exists():raise RuntimeError('MANUAL_STOP_REQUESTED')
            while time.monotonic()<deadline:
                if stop_file and Path(stop_file).exists():raise RuntimeError('MANUAL_STOP_REQUESTED')
                controller.check();collect_evidence()
                time.sleep(min(.01,max(0.,deadline-time.monotonic())))
            controller.check();collect_evidence()
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
        image,_,seg,_=capture(client,log=log,frame_id=plan['session_id']+'_setup')
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
            image,depth,seg,meta=capture(client,log=log,frame_id=frame_id);row.update(meta,image_size_wh=[image.shape[1],image.shape[0]])
            row['rgb_persistence_started_monotonic_s']=time.monotonic()
            rgb_path=output/'frames'/(frame_id+'_rgb.png')
            if not cv2.imwrite(str(rgb_path),image,[cv2.IMWRITE_PNG_COMPRESSION,0]):raise RuntimeError('RGB_SAVE_FAILED')
            row['media']['rgb']=dict(path=str(rgb_path),sha256=digest(rgb_path))
            row['rgb_persistence_completed_monotonic_s']=time.monotonic()
            if not acquisition:inject_delay(slot,'AFTER_CAPTURE_BEFORE_OBSERVER')
            controller.check()
            observation=observer_observation(row)
            result=observer.process(observation);row['observer_result']=result
            row['observer_returned_monotonic_s']=time.monotonic()
            frame=frame_from_dict(result['semantic_frame']);selection=result['temporal_identity']
            rgb_pose=CameraPose(**{k:meta['response_camera_poses'][0][k] for k in ('position_local_ned_m','orientation_xyzw')},
                                world_from_local_translation_ned_m=translation.tolist())
            depth_pose=CameraPose(**{k:meta['response_camera_poses'][1][k] for k in ('position_local_ned_m','orientation_xyzw')},
                                  world_from_local_translation_ned_m=translation.tolist())
            row['surface_started_monotonic_s']=time.monotonic()
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
            row['surface_completed_monotonic_s']=time.monotonic()
            row['planning_self_state_started_monotonic_s']=time.monotonic()
            state=controller.latest_state();desired=[0.,0.]
            row['planning_self_state_completed_monotonic_s']=time.monotonic()
            if time.monotonic()-state['host_receipt_monotonic_s']>.12:
                raise RuntimeError('PLANNING_SELF_STATE_EXPIRED')
            row['planning_self_state']=state
            if not acquisition and plan['mode']=='SHORT_XY' and surface['status']=='SURFACE_GEOMETRY_ACCEPTED':
                planning_started=time.monotonic()
                row['planning_started_monotonic_s']=planning_started
                row['planner']=plan_pose_trajectory(surface=surface,camera=camera,camera_pose=rgb_pose,
                    position_ned_m=state['position_world_ned_m'],velocity_ned_m_s=state['velocity_ned_m_s'],
                    acceleration_ned_m_s2=state['acceleration_ned_m_s2'],config=pcfg)
                row['planning_wall_s']=time.monotonic()-planning_started
                row['planning_completed_monotonic_s']=time.monotonic()
                desired=preview_command(row['planner'],.4)
                if obstacle_scene:
                    desired,row['static_route_preview']=route_velocity(state['position_world_ned_m'],obstacle_scene)
            # Only the child controller samples own state and evaluates/dispatches
            # the gate. This producer cannot renew a prior observation's lease.
            row['control_submit_started_monotonic_s']=time.monotonic()
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
            # Worker sees truth only after the current-frame action decision.
            row['status']='CAPTURED_EVIDENCE_PENDING'
            resource_sample(frame_id)
            row['evidence_enqueue_started_monotonic_s']=time.monotonic()
            disk_ticket=evidence_writer.submit(row,depth,seg,palette)
            row['evidence_enqueue_completed_monotonic_s']=time.monotonic()
            log(dict(kind='FRAME_DECISION',frame_id=frame_id,frame_sequence=frame.frame_sequence,
                     temporal_status=selection['status'],gate_status=gate['status'],surface_status=surface['status'],
                     decision_completed_monotonic_s=row['motion_decision_available_monotonic_s']))
            row['producer_pipeline_completed_monotonic_s']=time.monotonic()
            disk_ticket.complete({key:row[key] for key in ('evidence_enqueue_started_monotonic_s',
                'evidence_enqueue_completed_monotonic_s','producer_pipeline_completed_monotonic_s')})
            collect_evidence()
            if acquisition:
                print(json.dumps(dict(kind='LIVE_FRAME',frame_id=frame_id,slot=slot['slot_index'],acquisition=True,rgb_path=str(rgb_path),position_ned_m=state['position_world_ned_m'],gate_status=gate['status']),ensure_ascii=False),flush=True)
                wait_deadline=time.monotonic()+.5
                while row['status']=='CAPTURED_EVIDENCE_PENDING' and time.monotonic()<wait_deadline:
                    controller.check();collect_evidence();time.sleep(.005)
                if row['status']!='FRAME_COMPLETED':raise RuntimeError('ACQUISITION_MEDIA_COMPLETION_TIMEOUT')
            if not acquisition:
                print(json.dumps(dict(kind='LIVE_FRAME',frame_id=frame_id,slot=slot['slot_index'],acquisition=False,rgb_path=str(rgb_path),position_ned_m=state['position_world_ned_m'],selected=selection.get('selected_candidate'),gate_status=gate['status']),ensure_ascii=False),flush=True)
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
            if (acquisition_established(row,now=now,deadline=initial_deadline)
                    and row.get('zero_held_stop_proof',{}).get('held_zero_without_new_command') is True):
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
        # Settle acquisition using actual own-state receipts before measurement.
        report['initial_controller_settlement_state']=row['zero_held_stop_proof']['current_self_state']
        report['acquisition_transition_policy']=plan['acquisition_transition_policy']
        # The frozen 8s zero or 20s visible measurement starts after acquisition.
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
        control_wait(epoch+total)
        evidence_deadline=time.monotonic()+2.
        while evidence_writer.completed<evidence_writer.accepted and time.monotonic()<evidence_deadline:
            controller.check();collect_evidence();time.sleep(.005)
        collect_evidence()
        if evidence_writer.completed!=evidence_writer.accepted:raise RuntimeError('FINAL_EVIDENCE_COMPLETION_TIMEOUT')
        report['final_controller_settlement_state']=controller.wait_until_settled(
            timeout=plan['settled_endpoint_policy']['final_wait_timeout_s'])
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
        report['failure_call_sites']=[dict(file=Path(t.filename).name,line=t.lineno,function=t.name) for t in traceback.extract_tb(error.__traceback__)]
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
        control_stopped=True;receipts={};decisions={}
        if controller:
            try:
                report['control_close']=controller.close(timeout=3.)
            except Exception as error:
                report['status']='FAIL';report['errors'].append('CONTROL_CLOSE_'+type(error).__name__)
                if controller.is_alive():
                    try:controller.close(timeout=10.)
                    except Exception as retry_error:
                        report['errors'].append('CONTROL_CLOSE_RETRY_'+type(retry_error).__name__)
                control_stopped=controller.can_transfer_rpc()
            receipts=controller.receipts()
            decisions=controller.decisions()
            report['control_process']=controller.scheduling_receipt()
            report['control_scheduling']=report['control_process'].get('scheduling', {})
            control_stopped=controller.can_transfer_rpc()
        try:
            if not control_stopped:raise RuntimeError('CONTROL_PROCESS_TRANSFER_NOT_VERIFIED_PRESERVE_SIM')
            report['ground_after']=flight.land() if flight else ground_states(client)
            report['safe_end']['grounded_and_released']=True
            client.simPause(True);report['safe_end']['paused']=bool(client.simIsPause())
        except Exception as error:
            report['status']='FAIL';report['errors'].append('LANDING_RECOVERY_'+type(error).__name__)
        if controller and controller.can_transfer_rpc():
            try: report['control_process_trace_replay']=replay_closed_trace(controller, journal)
            except Exception as error:
                report['status']='FAIL';report['errors'].append('CONTROL_TRACE_REPLAY_'+type(error).__name__)
        # Recovery precedes disk-worker joins and final serialization.
        try:
            report['evidence_writer_close']=evidence_writer.close(timeout=5.)
            for persisted in evidence_writer.drain():
                rows=report['acquisition_frames'] if persisted.get('experiment_scope')=='INITIAL_IDENTITY_ACQUISITION' else report['frames']
                for row in rows:
                    if row['frame_id']==persisted['frame_id']:row.update(persisted)
            if report['evidence_writer_close']['status']!='CLOSED':raise RuntimeError('EVIDENCE_WORKER_FAILED')
        except Exception as error:
            report['status']='FAIL';report['errors'].append('EVIDENCE_CLOSE_'+type(error).__name__)
        for row in report['acquisition_frames']+report['frames']:
            ticket=row.get('ticket_id')
            if ticket in decisions and 'motion_gate' not in row:
                row.update({key:value for key,value in decisions[ticket].items() if value is not None})
            if ticket in receipts:row['lease_stop']=receipts[ticket]
            if row.get('status')=='FRAME_COMPLETED':
                row['decision_completed_monotonic_s']=max(row['frame_processing_completed_monotonic_s'],
                    row.get('lease_stop',{}).get('completed_monotonic_s',row['motion_decision_available_monotonic_s']))
                row['decision_on_time']=row['decision_completed_monotonic_s']<=row['deadline_monotonic_s']
                row['completion_timing']=completion_timing(row)
                row_slots=initial_slots if row.get('experiment_scope')=='INITIAL_IDENTITY_ACQUISITION' else slots
                row_slots[row['slot_index']]['decision_on_time']=row['decision_on_time']
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
        try:
            report['event_journal_close']=journal.close(timeout=5.)
            if report['event_journal_close']['status']!='CLOSED':raise RuntimeError('EVENT_JOURNAL_WRITE_FAILED')
        except Exception as error:
            report['status']='FAIL';report['errors'].append('EVENT_CLOSE_'+type(error).__name__)
        report['evidence_writer']=evidence_writer.receipt()
        report['event_buffer']=dict(capacity=4096,overflow=journal.overflow.is_set(),
            lifecycle_receipt_failure=flight.receipt_failure if flight else False,
            scope='bounded asynchronous single-owner event writer; batch flush receipts bind actual bytes')
        report['persistence_artifacts']=[dict(path=str(path),sha256=digest(path)) for path in (
            evidence_writer.records_path,journal.flush_path) if path.exists()]
        if journal.overflow.is_set():
            report['status']='FAIL';report['errors'].append('CONTROL_EVENT_BUFFER_OVERFLOW')
        try:
            evidence=[json.loads(line) for line in (output/'execution.jsonl').read_text(encoding='utf-8').splitlines()]
            report['observed_airborne']=any(row.get('kind')=='SELF_STATE' and row.get('landed') is False for row in evidence)
            report['new_flights']=int(report['observed_airborne'])
            report['execution_log']=dict(path=str(output/'execution.jsonl'),sha256=digest(output/'execution.jsonl'))
        except Exception as error:
            report['status']='FAIL';report['errors'].append('FINAL_EVENT_READ_'+type(error).__name__)
            report['observed_airborne']=None
        report['resource_samples']=resource_samples
        report['runner_revision']='RESOURCE_TRACED_V11P2'
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        save(output/'report.json',report)
    print(json.dumps(dict(status=report['status'],errors=report['errors'],frames=len(report['frames']),
        safe_end=report['safe_end'],acquisition_frames=len(report['acquisition_frames']),
        acquisition_status=report['acquisition_status'],integration_counts=report.get('integration_counts')),ensure_ascii=False,default=json_scalar))
    return report['status']=='PASS' and all(report['safe_end'].values())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stop-file',type=Path)
    args=parser.parse_args();raise SystemExit(0 if run(args.plan,args.output,args.stop_file) else 2)
