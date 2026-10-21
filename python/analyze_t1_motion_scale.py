"""Read-only V6 endpoint motion, complete-grid visibility and command responses.

The old chain/media/identity/execution auditor remains the first evidence layer.
An independent frozen-threshold qualification is added without changing runner
status, original reports, historical recovery classifications or prior outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

from analyze_t1_delayed_identity import analyze_run as base_analyze
from analyze_t1_identity_flight import digest, finite, read_json, vector, verify_file, mask_truth
from analyze_t1_review_handoff import handoff_evidence
from analyze_t1_async_timing import is_nonzero_target

ARCHITECTURE = 'INDEPENDENT_CONTROL_IDLE_ZERO_SUPPRESSION_V6'
ARCHITECTURES = {ARCHITECTURE, ARCHITECTURE + 'P1'}
REQUIRED_DIAGNOSTICS = 'target integral direction projection; displacement and speed outside active leases; no extra post-hoc success threshold'


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def endpoint_displacement(report):
    start, end = (report.get(key) for key in ('low_speed_stage_start_state', 'low_speed_stage_end_state'))
    result = dict(status='UNVERIFIED_ENDPOINT', net_xy_displacement_m=None, delta_xy_m=None,
        start_state=start, end_state=end, endpoint_source='explicit low_speed_stage_start_state/end_state',
        path_used_for_qualification=False, takeoff_landing_included=False)
    def usable(row):
        return (isinstance(row, dict) and finite(row.get('host_receipt_monotonic_s'))
                and isinstance(row.get('position_world_ned_m'), (list, tuple))
                and len(row['position_world_ned_m']) >= 2 and all(finite(x) for x in row['position_world_ned_m'][:2]))
    if not usable(start) or not usable(end) or end['host_receipt_monotonic_s'] <= start['host_receipt_monotonic_s']:
        return result
    delta = [end['position_world_ned_m'][i]-start['position_world_ned_m'][i] for i in range(2)]
    return dict(result, status='EXPLICIT_ENDPOINTS_VERIFIED', delta_xy_m=delta,
        net_xy_displacement_m=math.hypot(*delta), recorded_stage_span_s=end['host_receipt_monotonic_s']-start['host_receipt_monotonic_s'])


def measured_contract(plan, report):
    """Check all planned/reported slots, actual captures and visible-only phases."""
    mode = plan.get('mode'); expected = 20 if mode == 'ZERO_GATE' else 50 if mode == 'SHORT_XY' else None
    duration = 8. if mode == 'ZERO_GATE' else 20. if mode == 'SHORT_XY' else None
    issues = []
    if (plan.get('control_architecture') not in ARCHITECTURES or
            report.get('control_architecture') != plan.get('control_architecture')):
        issues.append('V6_ARCHITECTURE_NOT_BOUND')
    if expected is None or plan.get('period_s') != .4 or plan.get('planned_duration_s') != duration:
        issues.append('FROZEN_GRID_CONFIG_INVALID')
    declaration = plan.get('motion_scale_validation', {})
    if (declaration.get('planned_measured_slots') != expected or
            declaration.get('continuous_visible_duration_s') != duration):
        issues.append('DECLARED_GRID_DENOMINATOR_INVALID')
    if plan.get('phases') != [dict(phase_id='visible', duration_s=duration)]:
        issues.append('PLAN_NOT_SINGLE_VISIBLE_PHASE')
    if report.get('injected_delays') or plan.get('active_lease_stall', {}).get('enabled'):
        issues.append('STALL_NOT_ALLOWED_IN_MOTION_SCALE_RUN')
    transitions = report.get('phase_transitions', [])
    epoch = report.get('epoch_monotonic_s')
    if (len(transitions) != 1 or transitions[0].get('phase_id') != 'visible'
            or not all(finite(transitions[0].get(key)) for key in ('started_monotonic_s', 'completed_monotonic_s'))
            or transitions[0]['completed_monotonic_s'] < transitions[0]['started_monotonic_s']):
        issues.append('ACTUAL_VISIBLE_TRANSITION_UNVERIFIED')
    if report.get('measurement_entered') is not True or not finite(epoch):
        issues.append('MEASURED_PHASE_NOT_ENTERED_OR_EPOCH_MISSING')
    planned, slots, frames = (plan.get('planned_slots', []), report.get('slots', []), report.get('frames', []))
    if expected is None or len(planned) != expected or len(slots) != expected:
        issues.append('PLANNED_OR_REPORTED_SLOT_DENOMINATOR_INVALID')
    frame_by_slot = {}
    for frame in frames:
        index = frame.get('slot_index')
        if not isinstance(index, int) or isinstance(index, bool) or index in frame_by_slot:
            issues.append('DUPLICATE_OR_INVALID_MEASURED_FRAME_SLOT')
        frame_by_slot[index] = frame
    frame_ids = [frame.get('frame_id') for frame in frames]
    if len(set(frame_ids)) != len(frame_ids) or any(not isinstance(x, str) for x in frame_ids):
        issues.append('MEASURED_FRAME_ID_INVALID_OR_REUSED')
    misses = []
    for index, slot in enumerate(slots):
        if index >= len(planned):
            continue
        design = planned[index]
        if (slot.get('slot_index') != index or design.get('slot_index') != index
                or slot.get('phase_id') != 'visible' or design.get('phase_id') != 'visible'):
            issues.append('SLOT_INDEX_OR_VISIBLE_PHASE_INVALID')
        for key, wanted in (('scheduled_start_s', index*.4), ('scheduled_end_s', (index+1)*.4)):
            if not finite(design.get(key)) or abs(design[key]-wanted) > 1e-8:
                issues.append('PLANNED_SLOT_TIME_GRID_INVALID')
        for key, wanted in (('offset_s', index*.4), ('deadline_offset_s', (index+1)*.4)):
            if not finite(slot.get(key)) or abs(slot[key]-wanted) > 1e-8:
                issues.append('ACTUAL_SLOT_TIME_GRID_INVALID')
        if finite(epoch):
            for key, wanted in (('due_monotonic_s', epoch+index*.4), ('deadline_monotonic_s', epoch+(index+1)*.4)):
                if not finite(slot.get(key)) or abs(slot[key]-wanted) > 1e-8:
                    issues.append('ACTUAL_SLOT_CLOCK_NOT_BOUND_TO_EPOCH')
        frame = frame_by_slot.get(index)
        if slot.get('status') != 'CAPTURED' or frame is None:
            misses.append(dict(slot_index=index, status=slot.get('status'), frame_id=slot.get('frame_id')))
            continue
        if (slot.get('frame_id') != frame.get('frame_id') or frame.get('phase_id') != 'visible'
                or frame.get('experiment_scope') == 'INITIAL_IDENTITY_ACQUISITION'):
            issues.append('MEASURED_SLOT_CAPTURE_BINDING_INVALID')
        readback = frame.get('capture_received_host_s'); transition = transitions[0] if len(transitions)==1 else {}
        if (not finite(readback) or not finite(frame.get('capture_started_host_s'))
                or not finite(transition.get('completed_monotonic_s'))
                or frame['capture_started_host_s'] < transition['completed_monotonic_s']-1e-9):
            issues.append('FRAME_NOT_AFTER_VERIFIED_VISIBLE_TRANSITION')
    if expected is not None and (misses or len(frames) != expected):
        issues.append('MEASURED_GRID_INCOMPLETE')
    return dict(status='PASS_COMPLETE_VISIBLE_GRID' if not issues else 'FAIL_MEASURED_CONTRACT',
        mode=mode, expected_slots=expected, planned_slots=len(planned), reported_slots=len(slots), saved_frames=len(frames),
        slot_status_counts=dict(Counter(row.get('status') for row in slots)), missing_slots=misses,
        issues=sorted(set(issues)), no_hide_return_verified=not issues,
        acquisition_slots_preserved=len(report.get('acquisition_slots', [])), acquisition_frames_preserved=len(report.get('acquisition_frames', [])),
        scope='full measured ledger plus actual visible transition; initial acquisition has a separate denominator')


def handoff_evidence_v6(report):
    original = report.get('control_architecture')
    normalized = dict(report, control_architecture='INDEPENDENT_CONTROL_REVIEW_HANDOFF_V5')
    result = handoff_evidence(normalized)
    result.update(configured=True, original_control_architecture=original,
        normalization_scope='architecture marker in read-only validation copy only; original report bytes unchanged')
    if original not in ARCHITECTURES:
        result['issues'].append(dict(issue='V6_HANDOFF_ARCHITECTURE_UNVERIFIED'))
    return result


def saved_target_rows(report, run):
    """Read actual segmentation pixels; simulator truth is offline evaluation."""
    scores = {row['frame_id']: row for row in run.get('frame_rows', [])}
    rows, checks = [], []
    palette = report.get('segmentation_palette_rgb', {})
    for frame in report.get('frames', []):
        media = frame.get('media', {}).get('segmentation', {})
        check = verify_file(media.get('path', '__missing__'), media.get('sha256'), 'motion_scale_saved_segmentation')
        checks.append(check)
        row = dict(frame_id=frame.get('frame_id'), slot_index=frame.get('slot_index'), readback_s=frame.get('capture_received_host_s'),
            identity_status=scores.get(frame.get('frame_id'), {}).get('identity', {}).get('status'),
            status='UNVERIFIED_MASK', target_visible_pixels=None, mask_image_fraction=None,
            target_mask_centroid_normalized=None, target_bbox_center_error_normalized=None, selected_box_center_error_normalized=None)
        try:
            if check['status'] != 'PASS':
                raise ValueError('Saved segmentation hash not verified')
            image = cv2.imread(media['path'])
            target = mask_truth(image, palette)['target']
            h, w = image.shape[:2]
            if frame.get('image_size_wh') != [w, h]:
                raise ValueError('Actual image dimensions differ')
            mask = np.all(image == np.asarray(palette['42'][::-1], np.uint8), axis=2)
            yy, xx = np.nonzero(mask)
            center = [float((xx.mean()+.5)/w), float((yy.mean()+.5)/h)] if len(xx) else None
            box = target['bbox_xyxy_pixels']
            box_error = [float((box[0]+box[2])/(2*w)-.5), float((box[1]+box[3])/(2*h)-.5)] if box else None
            selected = frame.get('observer_result', {}).get('temporal_identity', {}).get('selected_candidate', {}).get('bbox_xyxy_pixels')
            selected_error = [float((selected[0]+selected[2])/(2*w)-.5), float((selected[1]+selected[3])/(2*h)-.5)] if vector(selected, 4) else None
            row.update(status='SAVED_MASK_VERIFIED', image_size_wh=[w,h], target_visible_pixels=target['visible_pixel_count'],
                mask_image_fraction=target['visible_pixel_count']/(w*h), target_mask_centroid_normalized=center,
                target_mask_center_error_norm=math.hypot(center[0]-.5, center[1]-.5) if center else None,
                target_bbox_center_error_normalized=box_error,
                selected_box_center_error_normalized=selected_error)
        except (KeyError, TypeError, ValueError, AttributeError):
            pass
        rows.append(row)
    return rows, checks


def continuous_visibility(rows, contract, minimum=500):
    invalid = [row.get('frame_id') for row in rows if row.get('status') != 'SAVED_MASK_VERIFIED'
               or not isinstance(row.get('target_visible_pixels'), int) or row['target_visible_pixels'] < minimum]
    complete = contract.get('status') == 'PASS_COMPLETE_VISIBLE_GRID' and len(rows) == contract.get('expected_slots')
    return dict(status='PASS_SAMPLED_VISIBLE_COMPLETE_GRID' if complete and not invalid else 'UNVERIFIED_CONTINUOUS_VISIBLE_GRID',
        configured_minimum_pixels=minimum, saved_mask_rows=len(rows), invalid_or_below_minimum_frame_ids=invalid,
        minimum_actual_saved_pixels=min((row['target_visible_pixels'] for row in rows if isinstance(row.get('target_visible_pixels'), int)), default=None),
        full_planned_grid_verified=complete, rows=rows,
        continuous_time_visibility_guaranteed=False, scope='all actual saved measurement frames; no between-frame visibility claim')


def motion_scale_evidence(plan, report, nonzero_count, sampled_path=None):
    endpoints = endpoint_displacement(report)
    config = plan.get('motion_scale_validation', {})
    mode = plan.get('mode')
    valid = (config.get('minimum_visible_target_mask_pixels')==500
        and config.get('first_last_observation_window_s')==1.
        and config.get('response_diagnostics')==REQUIRED_DIAGNOSTICS
        and ((mode=='SHORT_XY' and config.get('minimum_endpoint_net_xy_displacement_m')==.020
              and config.get('maximum_endpoint_drift_m') is None)
          or (mode=='ZERO_GATE' and config.get('maximum_endpoint_drift_m')==.005
              and config.get('minimum_endpoint_net_xy_displacement_m') is None
              and config.get('zero_gate_requires_no_nonzero_target_commands') is True)))
    status = 'FAIL_FROZEN_THRESHOLD_CONTRACT' if not valid else 'UNVERIFIED_ENDPOINT'
    net = endpoints['net_xy_displacement_m']
    if valid and endpoints['status']=='EXPLICIT_ENDPOINTS_VERIFIED':
        if mode=='SHORT_XY':
            status='PASS_ENDPOINT_MOTION_SCALE' if net>=.020 else 'FAIL_ENDPOINT_MOTION_SCALE_NOT_REACHED'
        else:
            status='PASS_ZERO_GATE_DRIFT' if net<=.005 and nonzero_count==0 else 'FAIL_ZERO_GATE_DRIFT_OR_NONZERO'
    return dict(endpoints, status=status, mode=mode, frozen_validation=config,
        nonzero_target_command_count=nonzero_count, sampled_xy_path_length_m_context_only=sampled_path,
        extra_success_threshold_added=False, formal_trajectory_efficacy=False)


def center_windows(rows, report, plan):
    epoch, duration = report.get('epoch_monotonic_s'), plan.get('planned_duration_s')
    result = dict(scope='first and last actual-readback 1s windows; descriptive offline target mask evidence, not causal centering benefit', windows=[])
    if not finite(epoch) or not finite(duration) or duration<2:
        return dict(result, status='UNVERIFIED_WINDOW_EPOCH')
    for label, lo, hi in (('first', 0., 1.), ('last', duration-1., duration)):
        expected = [s['slot_index'] for s in plan.get('planned_slots', []) if finite(s.get('scheduled_start_s')) and lo<=s['scheduled_start_s']<hi]
        actual = [row for row in rows if finite(row.get('readback_s')) and epoch+lo<=row['readback_s']<epoch+hi]
        valid = [row for row in actual if row.get('status')=='SAVED_MASK_VERIFIED' and vector(row.get('target_mask_centroid_normalized'), 2)]
        stamps = sorted(set(row['readback_s'] for row in valid))
        boundary_gaps = [b-a for a,b in zip([epoch+lo]+stamps, stamps+[epoch+hi])]
        covered = (set(expected).issubset({r.get('slot_index') for r in valid}) and bool(expected)
                   and bool(stamps) and max(boundary_gaps, default=math.inf)<=.5+1e-9 and len(valid)==len(actual))
        result['windows'].append(dict(label=label, actual_window_start_s=epoch+lo, actual_window_end_s=epoch+hi,
            expected_planned_slot_indexes=expected, actual_readback_frame_ids=[r['frame_id'] for r in actual],
            valid_mask_frame_ids=[r['frame_id'] for r in valid], status='DESCRIPTIVE_WINDOW_COVERED' if covered else 'UNKNOWN_INSUFFICIENT_WINDOW_COVERAGE',
            saved_readback_count=len(actual), valid_mask_count=len(valid), maximum_boundary_or_between_sample_gap_s=max(boundary_gaps, default=None),
            mean_mask_center_error_norm=float(np.mean([r['target_mask_center_error_norm'] for r in valid])) if valid else None,
            mean_mask_image_fraction=float(np.mean([r['mask_image_fraction'] for r in valid])) if valid else None))
    result['status']='DESCRIPTIVE_WINDOWS_COVERED' if all(w['status']=='DESCRIPTIVE_WINDOW_COVERED' for w in result['windows']) else 'UNKNOWN_INSUFFICIENT_WINDOW_COVERAGE'
    return result


def sampled_motion_states(report, events):
    start, end = (report.get(key, {}).get('host_receipt_monotonic_s') for key in ('low_speed_stage_start_state','low_speed_stage_end_state'))
    if not finite(start) or not finite(end):
        return []
    rows = [e for e in events if e.get('kind')=='SELF_STATE' and e.get('stage')=='PERCEPTION_XY']
    rows += [report[key] for key in ('low_speed_stage_start_state','low_speed_stage_end_state') if key in report]
    known = {}
    for row in rows:
        t=row.get('host_receipt_monotonic_s')
        if finite(t) and start<=t<=end and vector(row.get('position_world_ned_m'),3) and vector(row.get('velocity_ned_m_s'),3):
            known[t]=row
    return [known[t] for t in sorted(known)]


def command_responses(report, run, events):
    states = sampled_motion_states(report, events)
    all_commands = [(i,row) for i,row in enumerate(events) if row.get('kind')=='COMMAND' and row.get('stage')=='PERCEPTION_XY' and finite(row.get('dispatch_monotonic_s'))]
    all_commands.sort(key=lambda pair:(pair[1]['dispatch_monotonic_s'],pair[0]))
    frames = {row['frame_id']:row for row in report.get('frames', [])}
    result, intervals = [], []
    declared_integral = np.zeros(2); actual_target_hold_integral = np.zeros(2)
    for command in run.get('target_command_rows', []):
        if command.get('target_frame_id') not in frames:
            continue
        frame=frames[command['target_frame_id']]; gate=frame.get('motion_gate',{}); stop=frame.get('lease_stop',{})
        dispatch=command.get('dispatch_monotonic_s'); clear=command.get('lease_clear',{}).get('actual_zero_clear_s')
        declared=command.get('lease_clear',{}).get('declared_lease_end_s'); duration=command.get('duration_s')
        index=next((n for n,(_,row) in enumerate(all_commands) if row.get('command_id')==command.get('command_id')), None)
        preceding=[row for _,row in all_commands[:index]] if index is not None else []
        previous_zero=next((row for row in reversed(preceding) if vector(row.get('velocity_xy_m_s'),2) and math.hypot(*row['velocity_xy_m_s'])<=1e-12), None)
        next_dispatch=all_commands[index+1][1]['dispatch_monotonic_s'] if index is not None and index+1<len(all_commands) else None
        desired=gate.get('boundary_envelope',{}).get('desired_velocity_xy_m_s')
        actual=command.get('velocity_xy_m_s')
        before=frame.get('command_self_state',{})
        sample_times=[s.get('receipt_monotonic_s') for s in stop.get('samples',[]) if finite(s.get('receipt_monotonic_s')) and finite(clear) and s['receipt_monotonic_s']>=clear]
        matched=[s for s in states if s['host_receipt_monotonic_s'] in sample_times]
        post=matched[-1] if matched else None
        delta=None
        if post and vector(before.get('position_world_ned_m'),3):
            delta=[post['position_world_ned_m'][i]-before['position_world_ned_m'][i] for i in range(2)]
        if finite(dispatch) and finite(clear) and clear>=dispatch:
            dispatch_index=all_commands[index][0] if index is not None else None
            clear_index=next((i for i,row in all_commands if dispatch_index is not None and i>dispatch_index
                and row['dispatch_monotonic_s']==clear and vector(row.get('velocity_xy_m_s'),2)
                and math.hypot(*row['velocity_xy_m_s'])<=1e-12),None)
            intervals.append((dispatch,clear,command.get('command_id'),dispatch_index,clear_index))
        cmd_integral=(np.asarray(actual)*duration).tolist() if vector(actual,2) and finite(duration) else None
        # Stop at the next actual command replacement; do not assume a target
        # setpoint remained unchanged until its declaration or explicit zero.
        held_until=min(x for x in (next_dispatch,clear) if finite(x)) if finite(next_dispatch) or finite(clear) else None
        held_integral=(np.asarray(actual)*(held_until-dispatch)).tolist() if vector(actual,2) and finite(dispatch) and finite(held_until) and held_until>=dispatch else None
        if cmd_integral is not None:declared_integral+=cmd_integral
        if held_integral is not None:actual_target_hold_integral+=held_integral
        active_states=[s for s in states if finite(dispatch) and finite(clear) and dispatch<=s['host_receipt_monotonic_s']<=clear]
        result.append(dict(frame_id=frame['frame_id'],command_id=command.get('command_id'),actual_dispatch_s=dispatch,
            previous_actual_zero_command_id=previous_zero.get('command_id') if previous_zero else None,
            previous_actual_zero_dispatch_s=previous_zero['dispatch_monotonic_s'] if previous_zero else None,
            elapsed_since_previous_actual_zero_s=dispatch-previous_zero['dispatch_monotonic_s'] if previous_zero and finite(dispatch) else None,
            policy_previous_actual_command_s=gate.get('previous_command_monotonic_s'),policy_elapsed_previous_command_s=gate.get('elapsed_previous_command_s'),
            desired_xy_m_s=desired,actual_command_xy_m_s=actual,self_state_velocity_xy_m_s=before.get('velocity_ned_m_s',[])[:2],
            desired_speed_m_s=math.hypot(*desired) if vector(desired,2) else None,actual_command_speed_m_s=math.hypot(*actual) if vector(actual,2) else None,
            maximum_sampled_active_xy_speed_m_s=max((math.hypot(*s['velocity_ned_m_s'][:2]) for s in active_states),default=None),
            declared_lease_end_s=declared,explicit_zero_clear_s=clear,explicit_clear_after_declared_end_s=clear-declared if finite(clear) and finite(declared) else None,
            declared_command_integral_xy_m=cmd_integral,actual_target_setpoint_until_replacement_integral_xy_m=held_integral,
            next_actual_command_replacement_s=next_dispatch,post_clear_stop_state=post,
            pulse_displacement_xy_m=delta,pulse_displacement_norm_m=math.hypot(*delta) if delta else None,
            pulse_scope='fresh pre-dispatch command_self_state to exact host-receipt-matched post-clear stop sample; not exact dispatch/physical-stop position',
            post_clear_sample_position_status='MATCHED_ACTUAL_STATE_RECEIPT' if post else 'UNVERIFIED_NO_EXACT_POSITION_RECEIPT_MATCH',
            self_state_age_at_dispatch_s=dispatch-before['host_receipt_monotonic_s'] if finite(dispatch) and finite(before.get('host_receipt_monotonic_s')) else None,
            self_state_fresh_at_dispatch_verified=bool(finite(dispatch) and finite(before.get('host_receipt_monotonic_s')) and 0<=dispatch-before['host_receipt_monotonic_s']<=.12+1e-9),
            base_command_issues=command.get('issues',[]),base_lease_issues=command.get('lease_clear',{}).get('issues',[])))
    segments=[]; totals={key:np.zeros(2) for key in ('ACTIVE','NO_ACTIVE','BOUNDARY_MIXED')}
    speeds={key:[] for key in totals}
    for left,right in zip(states,states[1:]):
        t0,t1=left['host_receipt_monotonic_s'],right['host_receipt_monotonic_s']
        active=any(a<=t0 and t1<=b for a,b,*_ in intervals)
        touches=any(a<t1 and b>t0 for a,b,*_ in intervals)
        category='ACTIVE' if active else 'BOUNDARY_MIXED' if touches else 'NO_ACTIVE'
        delta=np.asarray(right['position_world_ned_m'][:2])-left['position_world_ned_m'][:2]
        totals[category]+=delta
        speeds[category]+=[math.hypot(*s['velocity_ned_m_s'][:2]) for s in (left,right)]
        segments.append(dict(left_receipt_s=t0,right_receipt_s=t1,classification=category,delta_xy_m=delta.tolist(),sampled_segment_length_m=float(np.linalg.norm(delta))))
    integral_norm=float(np.linalg.norm(actual_target_hold_integral));unit=(actual_target_hold_integral/integral_norm).tolist() if integral_norm>1e-12 else None
    endpoint=endpoint_displacement(report)
    net_projection=float(np.dot(endpoint['delta_xy_m'],unit)) if endpoint['delta_xy_m'] is not None and unit else None
    return dict(rows=result,declared_target_command_integral_xy_m=declared_integral.tolist(),
        actual_target_setpoint_integral_until_next_actual_replacement_xy_m=actual_target_hold_integral.tolist(),
        target_integral_direction_unit_xy=unit,endpoint_displacement_projection_on_target_integral_m=net_projection,
        target_integral_scope='sum each actual target setpoint until next actual command replacement (or verified explicit clear); not actual velocity integral',
        active_intervals_actual_dispatch_to_explicit_clear=[dict(start_s=a,end_s=b,command_id=cid,
            dispatch_event_index=di,clear_event_index=ci) for a,b,cid,di,ci in intervals],
        sampled_state_rows=states,sampled_displacement_segments=segments,
        sampled_displacement_vector_by_interval_class_xy_m={key:value.tolist() for key,value in totals.items()},
        maximum_sampled_xy_speed_by_interval_class_m_s={key:max(value,default=None) for key,value in speeds.items()},
        no_active_displacement_projection_on_target_integral_m=float(np.dot(totals['NO_ACTIVE'],unit)) if unit else None,
        no_interpolation_across_dispatch_clear_boundaries=True,missing_boundaries_are_not_distributed_between_categories=True,
        diagnostic_only=True,extra_success_threshold_added=False,
        scope='sampled states and explicit events; NO_ACTIVE means outside known target dispatch-to-clear intervals, not zero physical velocity')


def idle_omission_evidence(events, response, scheduling):
    commands=[(i,e) for i,e in enumerate(events) if e.get('kind')=='COMMAND' and finite(e.get('dispatch_monotonic_s'))]
    queries=[e for e in events if e.get('kind')=='SELF_STATE' and finite(e.get('host_receipt_monotonic_s'))]
    rows=[];issues=[]
    for index,event in enumerate(events):
        if event.get('kind')!='IDLE_ZERO_OMITTED':continue
        row_issues=[];now=event.get('decision_monotonic_s');receipt=event.get('self_state_receipt_monotonic_s')
        prior=[e for i,e in commands if i<index and finite(now) and e['dispatch_monotonic_s']<=now]
        last=prior[-1] if prior else {}
        actual_query=next((q for q in queries if q['host_receipt_monotonic_s']==receipt and q.get('velocity_ned_m_s',[])[:2]==event.get('actual_velocity_xy_m_s')), None)
        stable,clear=event.get('sampled_stop_started_monotonic_s'),event.get('sampled_stop_clear_monotonic_s')
        known={q['host_receipt_monotonic_s'] for q in queries if finite(stable) and finite(receipt)
            and stable<=q['host_receipt_monotonic_s']<=receipt and vector(q.get('velocity_ned_m_s'),3) and math.hypot(*q['velocity_ned_m_s'][:2])<=.005}
        if (not last or not vector(last.get('velocity_xy_m_s'),2) or math.hypot(*last['velocity_xy_m_s'])>1e-12
                or last.get('dispatch_monotonic_s')!=event.get('last_actual_zero_dispatch_monotonic_s')):
            row_issues.append('OMISSION_NOT_BOUND_TO_LATEST_ACTUAL_ZERO')
        if (not actual_query or not finite(now) or not finite(receipt) or not 0<=now-receipt<=.12+1e-9
                or not vector(event.get('actual_velocity_xy_m_s'),2) or math.hypot(*event['actual_velocity_xy_m_s'])>.005):
            row_issues.append('OMISSION_FRESH_STATIONARY_QUERY_UNVERIFIED')
        if (not finite(stable) or not finite(clear) or not finite(receipt) or not clear<=stable<=receipt
                or receipt-stable<.05-1e-9 or len(known)<2
                or not isinstance(event.get('distinct_post_clear_sample_count'),int)
                or not 2<=event['distinct_post_clear_sample_count']<=len(known)):
            row_issues.append('OMISSION_POST_CLEAR_SAMPLED_STOP_UNVERIFIED')
        if (event.get('configured_real_zero_refresh_s')!=.30 or not finite(now)
                or not finite(event.get('last_actual_zero_dispatch_monotonic_s'))
                or not 0<=now-event['last_actual_zero_dispatch_monotonic_s']<.30):
            row_issues.append('OMISSION_REFRESH_BOUND_UNVERIFIED')
        if any(event.get(key) is not False for key in ('command_dispatched','command_receipt_created','active_target_lease','target_observation_used')):
            row_issues.append('OMISSION_INVENTS_COMMAND_OR_ACTIVE_AUTHORITY')
        tied_later=[]
        for interval in response['active_intervals_actual_dispatch_to_explicit_clear']:
            di,ci=interval.get('dispatch_event_index'),interval.get('clear_event_index')
            ordered=isinstance(di,int) and isinstance(ci,int) and di<ci
            if ordered:
                # Controller events share one ordered log producer. Host
                # monotonic clocks may give the idle decision and a later
                # dispatch the same value. Use saved order at those ties.
                if di<index<ci:row_issues.append('OMISSION_DURING_RECORDED_ACTIVE_TARGET')
                elif finite(now) and now==interval['start_s'] and index<di:
                    tied_later.append(interval.get('command_id'))
                elif finite(now) and interval['start_s']<now<interval['end_s']:
                    row_issues.append('OMISSION_EVENT_ORDER_AND_CLOCK_DISAGREE')
            elif finite(now) and interval['start_s']<=now<interval['end_s']:
                row_issues.append('OMISSION_DURING_RECORDED_ACTIVE_TARGET')
        row=dict(event,independent_issues=row_issues,known_distinct_stationary_receipts=len(known),
            omission_event_index=index,same_clock_later_dispatch_ids=tied_later,
            tie_scope='same controller producer saved event order; no finer physical clock inferred');rows.append(row)
        issues.extend(row_issues)
    declared=scheduling.get('idle_zero_omission_count')
    if declared!=len(rows):issues.append('OMISSION_COUNTER_DIFFERS_FROM_ACTUAL_EVENT_COUNT')
    return dict(status='FAIL_IDLE_OMISSION_EVIDENCE' if issues else 'PASS_SAVED_IDLE_OMISSION_EVIDENCE' if rows else 'NOT_OBSERVED',
        actual_omission_event_count=len(rows),declared_omission_count=declared,rows=rows,issues=sorted(set(issues)),
        synthetic_command_count=0,scope='omission diagnostics only; every omission must refer to a real earlier zero and saved own-state receipts')


def motion_qualification(run):
    if not str(run.get('independent_integration_qualification','')).startswith('PASS'):
        return 'FAIL_PRIOR_IDENTITY_EXECUTION_OR_INPUT_EVIDENCE'
    if run.get('review_handoff',{}).get('issues'):
        return 'FAIL_V6_HANDOFF_EVIDENCE'
    if run['measured_contract']['status']!='PASS_COMPLETE_VISIBLE_GRID':
        return 'FAIL_MEASURED_GRID_OR_PHASE_CONTRACT'
    if run['continuous_visibility']['status']!='PASS_SAMPLED_VISIBLE_COMPLETE_GRID':
        return 'UNVERIFIED_CONTINUOUS_VISIBLE_GRID'
    if run['idle_omission_evidence']['issues']:
        return 'FAIL_IDLE_OMISSION_EVIDENCE'
    return run['motion_scale']['status']


def analyze_run(root):
    root=Path(root).resolve()
    manifest=read_json(root/'manifest.json')
    if not manifest.get('report') or not Path(manifest['report']).is_file():
        return preflight_failure(root, manifest)
    run=base_analyze(root)
    if run.get('status')=='SEALED_INPUT_FAILURE':
        return dict(run,motion_scale_qualification='UNVERIFIED_BASE_INPUTS')
    manifest=read_json(root/'manifest.json');path=Path(manifest['report']);report=read_json(path);plan=read_json(report['plan'])
    events=[json.loads(line) for line in Path(report['execution_log']['path']).read_text(encoding='utf-8-sig').splitlines()]
    run['motion_analysis_inputs']=dict(report_path=str(path),report_sha256=digest(path),plan_path=report['plan'],plan_sha256=digest(report['plan']),
        manifest_path=str(root/'manifest.json'),manifest_sha256=digest(root/'manifest.json'))
    run['measured_contract']=measured_contract(plan,report)
    run['review_handoff']=handoff_evidence_v6(report)
    rows,checks=saved_target_rows(report,run);run['motion_visibility_source_checks']=checks
    run['continuous_visibility']=continuous_visibility(rows,run['measured_contract'])
    run['legacy_base_recovery_evaluation']=run.get('recovery')
    if run['measured_contract']['no_hide_return_verified'] and run['continuous_visibility']['status']=='PASS_SAMPLED_VISIBLE_COMPLETE_GRID':
        run['recovery']=dict(status='NOT_APPLICABLE_NO_HIDE_RETURN',verified_visible_only_complete_grid=True,
            loss_delay_s=None,recovery_delay_s=None,scope='plan, complete actual ledger, phase transitions and all saved target masks verified')
    run['motion_scale']=motion_scale_evidence(plan,report,run.get('nonzero_target_command_count'),
        (run.get('actual_motion_stage') or {}).get('sampled_xy_path_length_m'))
    run['command_response']=command_responses(report,run,events)
    run['idle_omission_evidence']=idle_omission_evidence(events,run['command_response'],report.get('control_scheduling',{}))
    run['center_windows']=center_windows(rows,report,plan)
    cadence=(run.get('timing') or {}).get('monitor_event_receipt_cadence',{})
    run['query_coverage_diagnostic']=dict(cadence=cadence,query_gaps_over_0p12_s=cadence.get('gaps_over_0p12_s'),
        nonzero_dispatches_without_fresh_state=sum(not r['self_state_fresh_at_dispatch_verified'] for r in run['command_response']['rows']),
        query_gap_exceedance_equals_dispatch_freshness_failure=False,
        head_gap_s=cadence.get('stage_head_gap_s'),tail_gap_s=cadence.get('stage_tail_gap_s'),hard_real_time_guaranteed=False)
    run['motion_scale_qualification']=motion_qualification(run)
    run['formal_efficacy_results']=False
    return run


def preflight_failure(root, manifest):
    """Keep a failed attempt as a row; an absent report never becomes flight data."""
    plan_path=Path(manifest['plan']);plan=read_json(plan_path)
    paths=[root/'manifest.json',plan_path,root/'runner.stderr.log',root/'plan_preflight.json']
    inputs=[dict(path=str(p.resolve()),sha256=digest(p)) for p in paths if p.is_file()]
    stderr=(root/'runner.stderr.log').read_text(encoding='utf-8-sig') if (root/'runner.stderr.log').is_file() else ''
    source=root/'source_snapshot/python/t1_identity_flight_v6.py'
    expected=plan.get('source_sha256',{}).get('python/t1_identity_flight_v6.py')
    source_bound=(source.is_file() and digest(source)==expected)
    plan_bound=digest(plan_path)==manifest.get('plan_sha256')
    # This exact historical guard is before output creation and AirSim client
    # construction in the frozen source. Retain the independently bound trace.
    known_contract_failure=(source_bound and plan_bound and manifest.get('runner_exit_code')==1
        and 'ValueError: FROZEN_CONTINUOUS_VISIBILITY_SCALE_CONTRACT_MISMATCH' in stderr
        and plan.get('motion_scale_validation',{}).get('planned_measured_slots')==41
        and len(plan.get('planned_slots',[]))==50
        and not (root/'flight').exists())
    if source.is_file():inputs.append(dict(path=str(source),sha256=digest(source)))
    local_preflight=read_json(root/'plan_preflight.json') if (root/'plan_preflight.json').is_file() else {}
    new_preflight_failure=(manifest.get('runner_started') is False and
        local_preflight.get('status')=='FAIL_MOTION_PLAN_CONTRACT' and
        local_preflight.get('simulator_calls') is False and local_preflight.get('plan_sha256')==digest(plan_path))
    zero_flights=known_contract_failure or new_preflight_failure
    return dict(session_id=manifest.get('session_id'),mode=plan.get('mode'),
        status='SEALED_PREFLIGHT_FAILURE' if zero_flights else 'UNVERIFIED_MISSING_FLIGHT_REPORT',
        original_runner_status='NO_FLIGHT_REPORT',launcher_status=manifest.get('status'),
        independent_integration_qualification='FAIL_NO_FLIGHT_REPORT',
        motion_scale_qualification='NOT_EXECUTED_PREFLIGHT_FAILED' if zero_flights else 'UNVERIFIED_NO_FLIGHT_REPORT',
        actual_flights=0 if zero_flights else None,flight_count_basis='bound pre-client contract failure' if known_contract_failure
            else 'bound local preflight failure' if new_preflight_failure else 'unknown; absence of a report alone is insufficient',
        new_cloud_calls=0 if zero_flights else None,nonzero_target_command_count=None,
        planned_main_slot_count=len(plan.get('planned_slots',[])),saved_main_frame_count=0,
        frozen_declared_main_slot_count=plan.get('motion_scale_validation',{}).get('planned_measured_slots'),
        fresh_ground_closure_verified=manifest.get('fresh_ground_closure_verified'),
        engine_closed_after_safe_completion=manifest.get('engine_closed_after_safe_completion'),
        original_error_preserved=stderr,flight_measurements_imputed=False,
        motion_analysis_input_files=inputs,formal_efficacy_results=False)


def analyze(roots,output):
    source_names=('analyze_t1_motion_scale.py','test_analyze_t1_motion_scale.py','plot_t1_motion_scale_zh.py','analyze_t1_delayed_identity.py',
                  'analyze_t1_review_handoff.py','analyze_t1_identity_flight.py','analyze_t1_async_timing.py',
                  't1_motion_plan_contract_v1.py','test_t1_motion_plan_contract.py','plot_t1_identity_zh.py')
    if any(not Path(__file__).with_name(name).is_file() for name in source_names):
        raise ValueError('ANALYSIS_SOURCE_SNAPSHOT_DEPENDENCY_MISSING')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    runs=[analyze_run(root) for root in roots]
    result=dict(kind='T1_MOTION_SCALE_DEVELOPMENT_ANALYSIS',created_utc=datetime.now(timezone.utc).isoformat(),runs=runs,
        formal_efficacy_results=False,ai_assistance='AI-assisted local analysis; numerical evidence is from saved inputs; scientific interpretation remains author reviewed',
        limitations=['Endpoint net XY displacement is not path length or formal trajectory efficacy.',
            'Visible-only means every saved planned frame passes the target-mask threshold, not continuous-time visibility.',
            'Sampled idle drift, pulse response and image-centering windows are diagnostic; no extra success criterion or causal benefit is inferred.',
            'Query cadence and fresh state at actual dispatch are different evidence populations.'])
    save(output/'report.json',result)
    lines=['# 任务一运动尺度独立验证','','| 试次 | 原运行器 / 身份执行资格 | 运动尺度独立资格 | 主测量保存/计划 | 端点净位移(mm) | 样本路程(mm，仅描述) | 非零指令 |',
        '|---|---|---|---:|---:|---:|---:|']
    for run in runs:
        m=run.get('motion_scale',{});net=m.get('net_xy_displacement_m');path=m.get('sampled_xy_path_length_m_context_only');c=run.get('measured_contract',{})
        lines.append(f"| {run.get('session_id')} | {run.get('original_runner_status')} / {run.get('independent_integration_qualification')} | {run.get('motion_scale_qualification')} | {c.get('saved_frames')}/{c.get('expected_slots')} | {net*1000:.3f} | {path*1000:.3f} | {run.get('nonzero_target_command_count')} |" if finite(net) and finite(path) else f"| {run.get('session_id')} | {run.get('original_runner_status')} / {run.get('independent_integration_qualification')} | {run.get('motion_scale_qualification')} | {c.get('saved_frames')}/{c.get('expected_slots')} | 未验证 | 未验证 | {run.get('nonzero_target_command_count')} |")
    lines+=['','成功阈值来自冻结计划；不以累计路程、图像中心变化或声明租约到期替代端点净位移与显式清零。',
        '初始获取、未采集槽、失败结果和原运行器状态保留。首尾窗口、空闲位移和积分方向为描述性诊断。','']
    (output/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    table=output/'command_response.csv';fields=['session_id','frame_id','command_id','actual_dispatch_s','previous_actual_zero_dispatch_s','elapsed_since_previous_actual_zero_s',
        'desired_speed_m_s','actual_command_speed_m_s','maximum_sampled_active_xy_speed_m_s','declared_lease_end_s','explicit_zero_clear_s',
        'explicit_clear_after_declared_end_s','pulse_displacement_norm_m','self_state_age_at_dispatch_s','self_state_fresh_at_dispatch_verified']
    with table.open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for run in runs:
            for row in run.get('command_response',{}).get('rows',[]):writer.writerow({key:run.get('session_id') if key=='session_id' else row.get(key) for key in fields})
    snapshot_dir=output/'source_snapshots';snapshot_dir.mkdir();sources=[]
    for name in source_names:
        path=Path(__file__).with_name(name);snapshot=snapshot_dir/name;shutil.copyfile(path,snapshot)
        sources.append(dict(path=str(path.resolve()),snapshot_path=str(snapshot),sha256=digest(path)))
    artifacts=[output/'report.json',output/'summary.md',table]
    save(output/'artifact_receipt.json',dict(kind='T1_MOTION_SCALE_ANALYSIS_ARTIFACT_RECEIPT',sources=sources,
        artifacts=[dict(path=str(path),sha256=digest(path)) for path in artifacts],formal_efficacy_results=False,
        runtime=dict(python=sys.version,numpy=np.__version__,opencv=cv2.__version__),new_cloud_calls=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.runs,args.output)
    print(json.dumps(dict(runs=len(result['runs']),output=str(args.output.resolve()))))
