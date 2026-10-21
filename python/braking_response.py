"""Pure, sampled braking diagnostics; never infer continuous or hardware safety."""
import math
import numpy as np


def validate_measurement(config):
    m=config['braking_measurement']
    for name in ('hold_s','ramp_timeout_s','brake_timeout_s','stop_speed_m_s',
                 'stop_acceleration_m_s2','stop_stable_s','max_query_gap_s',
                 'entry_velocity_tolerance_m_s','entry_minimum_amplitude_fraction',
                 'detect_speed_drop_m_s'):
        value=m[name]
        if isinstance(value,bool) or not math.isfinite(value) or value<=0:
            raise ValueError('Invalid measurement parameter: '+name)
    if m['hold_s']>1 or m['max_query_gap_s']>.25 or m['stop_stable_s']<.5:
        raise ValueError('Measurement timing outside declared development range')
    if not .5<=m['entry_minimum_amplitude_fraction']<=1:
        raise ValueError('Invalid entry fraction')
    if m['mode'] not in ('zero_hold_gate','pulse_measurement'):
        raise ValueError('Unknown measurement mode')
    for p in m['pulses']:
        d=np.asarray(p['direction_xy'],dtype=float)
        if d.shape!=(2,) or not np.isfinite(d).all() or abs(np.linalg.norm(d)-1)>1e-12:
            raise ValueError('Pulse direction must be a unit XY vector')
        a=p['amplitude_m_s']
        if isinstance(a,bool) or not math.isfinite(a) or not 0<a<=.06:
            raise ValueError('Pulse amplitude outside first low-speed range')
    if m['mode']=='pulse_measurement' and not m['pulses']:
        raise ValueError('Pulse measurement requires a nonempty plan')
    return m


def entry_check(sample,pulse,measurement):
    desired=np.asarray(pulse['direction_xy'])*pulse['amplitude_m_s']
    measured=np.asarray(sample['velocity_ned_m_s'])[:2]
    forward=float(np.dot(measured,pulse['direction_xy']))
    error=float(np.linalg.norm(measured-desired))
    return {'accepted':error<=measurement['entry_velocity_tolerance_m_s'] and
            forward>=pulse['amplitude_m_s']*measurement['entry_minimum_amplitude_fraction'],
            'velocity_error_norm_m_s':error,'forward_speed_m_s':forward,
            'desired_xy_m_s':desired.tolist()}


def summarize_pulse(pulse,measurement,model):
    """Origin is the last saved query before first brake request, not actuation."""
    brake=pulse.get('brake_first_request')
    if brake is None:
        return {'status':'NO_BRAKE_REQUEST','counts_as_measured_stopping':False}
    origin=brake['origin_sample']; now=brake['dispatch_host_s']
    rows=[s for s in pulse.get('samples',[]) if s['host_monotonic_s']>=now]
    if not rows:
        return {'status':'NO_POST_REQUEST_SAMPLES','counts_as_measured_stopping':False}
    times=[origin['host_monotonic_s']]+[s['host_monotonic_s'] for s in rows]
    ns=[origin['query_state_timestamp_ns']]+[s['query_state_timestamp_ns'] for s in rows]
    gaps=np.diff(times)
    chronology=bool(np.all(gaps>0) and np.all(np.diff(ns)>0))
    complete=chronology and float(max(gaps))<=measurement['max_query_gap_s']
    d=np.asarray(pulse['direction_xy'],dtype=float);o=np.asarray(origin['world_ned_m'])
    points=[o]+[np.asarray(s['world_ned_m']) for s in rows]
    along=[float(np.dot((p-o)[:2],d)) for p in points]
    v0=float(np.dot(np.asarray(origin['velocity_ned_m_s'])[:2],d))
    drop=[s for s in rows if np.dot(np.asarray(s['velocity_ned_m_s'])[:2],d)<=v0-measurement['detect_speed_drop_m_s']]
    first_drop=drop[0] if drop else None
    ready=None; stable_start=None; stable_sim=None
    for row in rows:
        stopped=(row['speed_norm_m_s']<=measurement['stop_speed_m_s'] and
                 row['acceleration_norm_m_s2']<=measurement['stop_acceleration_m_s2'] and
                 np.linalg.norm(row['previous_command_velocity_xy_m_s'])<=1e-9 and
                 not row['protection_reasons'])
        if stopped:
            if stable_start is None:stable_start=row['host_monotonic_s'];stable_sim=row['query_state_timestamp_ns']
            if row['host_monotonic_s']-stable_start>=measurement['stop_stable_s']:
                ready=row;break
        else:stable_start=None;stable_sim=None
    projected=max(v0,0.)*model['horizon_s']+max(v0,0.)**2/(2*model['braking_acceleration_m_s2'])+model['position_reserve_m']
    valid=bool(complete and ready is not None and pulse.get('status')=='COMPLETED' and
               entry_check(origin,pulse,measurement)['accepted'] and
               not origin['protection_reasons'] and not any(s['protection_reasons'] for s in rows))
    return {'status':'MEASURED_STABLE' if valid else 'INCOMPLETE_OR_REJECTED',
        'counts_as_measured_stopping':valid,'chronology_verified':chronology,
        'maximum_recorded_query_gap_s':float(max(gaps)),
        'origin_query_to_first_request_host_s':now-origin['host_monotonic_s'],
        'entry_forward_speed_m_s':v0,'maximum_recorded_forward_displacement_m':max(along),
        'sampled_brake_path_m':float(sum(np.linalg.norm(b-a) for a,b in zip(points,points[1:]))),
        'first_detected_drop_host_proxy_s':None if first_drop is None else first_drop['host_monotonic_s']-now,
        'first_detected_drop_state_timestamp_delta_s':None if first_drop is None else (first_drop['query_state_timestamp_ns']-origin['query_state_timestamp_ns'])*1e-9,
        'stable_completion_host_proxy_s':None if ready is None else ready['host_monotonic_s']-now,
        'stable_completion_state_timestamp_delta_s':None if ready is None else (ready['query_state_timestamp_ns']-origin['query_state_timestamp_ns'])*1e-9,
        'declared_envelope_allowance_m':projected,
        'sampled_displacement_within_declared_allowance':max(along)<=projected if valid else None,
        'scope':'Saved queries from first brake request through recorded confirmation; proxy clocks, not exact actuation or continuous stopping distance'}
