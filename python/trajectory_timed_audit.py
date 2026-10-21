"""Audit recorded V6 request coverage, version binding, reference time and age."""
import hashlib
import math
from pathlib import Path
import numpy as np
from perception_trajectory_planner import evaluate_plan


def audit_timed_execution(report):
    expected=report.get('configuration',{}).get('planning',{}).get('execution_reference_mode')
    if expected=='time_indexed' and report.get('execution_reference_mode')!='time_indexed':
        return {'status':'INVALID','reasons':['CONFIGURED_TIMED_MODE_NOT_RECORDED']}
    if report.get('execution_reference_mode')!='time_indexed':
        return {'status':'NOT_APPLICABLE','reasons':[]}
    reasons=[]
    if report.get('configuration',{}).get('planning',{}).get('image_center_reference_mode')=='rgb_detection_centroid_axial_plane':
        from trajectory_rgb_quality_reference import rgb_quality_reference
        if report.get('image_center_reference_mode')!='rgb_detection_centroid_axial_plane':
            reasons.append('CONFIGURED_RGB_CENTER_REFERENCE_NOT_RECORDED')
        frames={f.get('frame_id'):f for f in report.get('observations',[])}
        for plan in report.get('planning_records',[]):
            try:
                frame=frames[plan['frame_id']];meta=frame['timestamped_observation']
                expected=rgb_quality_reference(frame['target_estimate_world_ned_m'],frame['actual_camera_world_ned_m'],
                    meta['centre_px'],meta['projection']['rgb_size_wh'],meta['projection']['horizontal_fov_deg'])
                recorded=frame['image_center_reference']['offset_from_surface_ned_m']
                if not np.allclose(expected['offset_from_surface_ned_m'],recorded,atol=1e-9,rtol=0):
                    raise ValueError('RGB_REFERENCE_NOT_BOUND_TO_FRAME')
                state=plan['plan']['input_state']
                if not np.allclose(state['image_center_reference_offset_ned_m'],recorded,atol=1e-9,rtol=0):
                    raise ValueError('PLAN_RGB_REFERENCE_NOT_BOUND_TO_FRAME')
                if not np.allclose(state['target_estimate'],frame['target_estimate_world_ned_m'],atol=1e-9,rtol=0):
                    raise ValueError('SURFACE_DISTANCE_ANCHOR_REPLACED_BY_RGB_REFERENCE')
            except (KeyError,TypeError,ValueError) as error:
                reasons.append(f"{plan.get('plan_id')}:{error}")
    configured_ramp=report.get('configuration',{}).get('planning',{}).get('timed_execution',{}).get('command_ramp_time_basis')
    if configured_ramp=='elapsed_since_previous_request' and report.get('command_ramp_time_basis')!=configured_ramp:
        reasons.append('CONFIGURED_ACTUAL_REQUEST_RAMP_NOT_RECORDED')
    ledger=report.get('timed_command_records',[])
    plans={p.get('plan_id'):p for p in report.get('planning_records',[])}
    frames={f.get('frame_id'):f for f in report.get('observations',[])}
    if len(plans)!=len(report.get('planning_records',[])):
        reasons.append('DUPLICATE_PLAN_IDENTIFIERS')
    expected=[s.get('command_sequence') for p in plans.values() for s in p.get('execution_samples',[])]
    expected += [s.get('command_sequence') for s in report.get('goal_keepalive_records',[])]
    sequences=[c.get('command_sequence') for c in ledger]
    if sequences!=list(range(len(ledger))) or len(expected)!=len(ledger) or sorted(expected,key=str)!=sorted(sequences,key=str):
        reasons.append('COMMAND_REQUEST_COVERAGE_INCOMPLETE')
    maximum=report['configuration']['visual']['frame_max_age_s']
    reserve=report['configuration']['planning']['execution_envelope']['freshness_reserve_s']
    last={};ages=[];reference_times=[]
    previous_request=None
    for command in ledger:
        seq=command.get('command_sequence')
        if report.get('command_ramp_time_basis')=='elapsed_since_previous_request':
            try:
                envelope=command['boundary_envelope']
                timing=envelope['command_ramp_timing']
                actual_previous=None if previous_request is None else previous_request['dispatch_host_s']
                if timing['previous_request_host_s']!=actual_previous:
                    raise ValueError('RAMP_CLOCK_NOT_BOUND_TO_PREVIOUS_REQUEST')
                elapsed=0. if actual_previous is None else timing['evaluated_host_s']-actual_previous
                cap=report['configuration']['planning']['timed_execution']['command_ramp_interval_cap_s']
                effective=min(elapsed,cap)
                if elapsed<0 or timing['evaluated_host_s']>command['dispatch_host_s']+1e-9 or abs(effective-timing['effective_interval_s'])>1e-9:
                    raise ValueError('RAMP_INTERVAL_NOT_REPRODUCIBLE')
                previous=np.zeros(2) if previous_request is None else np.array(previous_request['command_velocity_xy_m_s'])
                if not np.allclose(previous,envelope['previous_velocity_xy_m_s'],atol=1e-9,rtol=0):
                    raise ValueError('RAMP_PREVIOUS_COMMAND_MISMATCH')
                change=np.linalg.norm(np.array(command['command_velocity_xy_m_s'])-previous)
                budget=report['configuration']['planning']['execution_ramp_acceleration_m_s2']*effective
                if change>budget+1e-9:
                    raise ValueError('REQUEST_TIME_COMMAND_RAMP_BUDGET_EXCEEDED')
            except (KeyError,TypeError,ValueError) as error:
                reasons.append(f'{seq}:{error}')
        previous_request=command
        if not command.get('target_used'):
            if command.get('plan_id') is not None or command.get('frame_id') is not None:
                reasons.append(f'{seq}:SELF_STATE_COMMAND_HAS_TARGET_BINDING')
            if command.get('source','').startswith('self_state_hold') and np.linalg.norm(command['command_velocity_xy_m_s'])>1e-9:
                reasons.append(f'{seq}:SELF_STATE_HOLD_IS_MOVING')
            continue
        try:
            plan=plans[command['plan_id']]
            frame=frames[command['frame_id']]
            if plan['frame_id']!=frame['frame_id'] or not frame['provider_valid']:
                raise ValueError('PLAN_FRAME_VERSION_MISMATCH')
            meta=frame['timestamped_observation']
            age=meta['image_readback_age_sim_s']+command['dispatch_host_s']-meta['age_anchor_host_monotonic_s']
            if not meta['clock_valid'] or age<0 or not math.isfinite(age):
                raise ValueError('INVALID_CLOCK_ANCHOR')
            if abs(age-command['observation_age_at_dispatch_s'])>1e-7:
                raise ValueError('AGE_PROXY_NOT_REPRODUCIBLE')
            if age+command['requested_duration_s']+reserve>maximum+1e-9:
                raise ValueError('TARGET_REQUEST_LEASE_EXCEEDS_FRESHNESS_WINDOW')
            reference=command['reference']
            origin=frame['planning_input_body_state']['host_monotonic_s']
            elapsed=reference['reference_dispatch_host_s']-origin
            if abs(reference['reference_input_host_s']-origin)>1e-9 or abs(reference['reference_time_s']-elapsed)>1e-9:
                raise ValueError('REFERENCE_TIME_NOT_BOUND_TO_INPUT_QUERY')
            if reference['reference_dispatch_host_s']>command['dispatch_host_s']+1e-9:
                raise ValueError('REFERENCE_TIME_AFTER_REQUEST')
            if elapsed+1e-9<last.get(command['plan_id'],0.):
                raise ValueError('REFERENCE_TIME_REGRESSION')
            check=evaluate_plan(plan['plan'],elapsed)
            for key,value in [('position','reference_position_world_ned_m'),('velocity','reference_velocity_ned_m_s')]:
                if not np.allclose(check[key],reference[value],atol=1e-9,rtol=0):
                    raise ValueError('REFERENCE_VECTOR_NOT_REPRODUCIBLE')
            last[command['plan_id']]=elapsed
            ages.append(age);reference_times.append(elapsed)
        except (KeyError,TypeError,ValueError) as error:
            reasons.append(f'{seq}:{error}')
    for frame in frames.values():
        if frame.get('media_persistence',{}).get('status')!='SAVED':
            reasons.append(f"{frame.get('frame_id')}:MEDIA_WRITE_NOT_COMPLETE")
        if frame.get('provider_valid'):
            for key in ('raw_image','raw_depth'):
                path=frame.get(key)
                expected_hash=frame.get(key+'_sha256')
                if not path or not Path(path).is_file() or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected_hash:
                    reasons.append(f"{frame.get('frame_id')}:{key}_HASH_NOT_VERIFIED")
    if report.get('raw_media_write_failures'):
        reasons.append('RAW_MEDIA_WRITE_FAILURE')
    return {'status':'VERIFIED' if not reasons else 'INVALID','reasons':reasons,
            'approach_request_count':len(ledger),'target_request_count':len(ages),
            'max_verified_dispatch_age_proxy_s':max(ages,default=None),
            'max_verified_reference_time_s':max(reference_times,default=None),
            'media_frame_count':len(frames),
            'scope':'all recorded approach XY/fixed-Z requests; lease/proxy and declared reference, not actual continuous actuation proof'}
