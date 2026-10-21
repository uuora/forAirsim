"""Read-only audit of independent AirSim braking development measurements.

Recompute from saved vectors and request receipts. Pulses within one flight
remain nested observations. No inference about continuous or hardware safety.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from analyze_trajectory_pilot import summarize_initial_match
from braking_response import entry_check, summarize_pulse, validate_measurement
from braking_response_v1p1 import acquired_velocity_interval, validate_acquisition
from braking_response_v1p2 import summarize_pulse as unique_origin_summary
from trajectory_execution_guard import boundary_command
from trajectory_timed_reference import command_ramp_interval
from single_drone_experiment import landing_ready

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def equal(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, int) and isinstance(right, int):
            return left == right  # preserve nanosecond precision
        return math.isfinite(left) and math.isfinite(right) and abs(left-right) <= 1e-9
    return left == right


def audit_sample(sample, config):
    """Check raw physical vectors and their recorded summaries/guard reasons."""
    p, v, a, rpy = [np.asarray(sample[k], dtype=float) for k in
        ('world_ned_m', 'velocity_ned_m_s', 'acceleration_ned_m_s2', 'actual_body_rpy_deg')]
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in (p, v, a, rpy)):
        raise ValueError('Nonfinite or incomplete sample vectors')
    cfg = config['planning']; speed = float(np.linalg.norm(v)); acc = float(np.linalg.norm(a))
    margin = float(np.min(np.minimum(p-np.array(config['workspace_min_ned_m']), np.array(config['workspace_max_ned_m'])-p)))
    height = abs(float(p[2])-cfg['fixed_world_height_ned_m'])
    reasons = []
    if margin < cfg['execution_guard_margin_m']: reasons.append('ACTUAL_POSITION_ENTERED_PROTECTION_MARGIN')
    if height > cfg['fixed_height_max_error_m']: reasons.append('ACTUAL_HEIGHT_ERROR_EXCEEDED')
    if speed > cfg['execution_speed_cap_m_s']+1e-9: reasons.append('ACTUAL_SPEED_BUDGET_EXCEEDED')
    if acc > cfg['max_acceleration_m_s2']+1e-9: reasons.append('ACTUAL_ACCELERATION_BUDGET_EXCEEDED')
    if float(max(abs(rpy))) > cfg['attitude_limit_deg']: reasons.append('ACTUAL_ATTITUDE_MODEL_LIMIT_EXCEEDED')
    expected = dict(speed_norm_m_s=speed, acceleration_norm_m_s2=acc,
        workspace_margin_m=margin, height_error_m=height, protection_reasons=reasons,
        over_planning_speed_budget=speed > cfg['execution_speed_cap_m_s']+1e-9,
        over_planning_acceleration_budget=acc > cfg['max_acceleration_m_s2']+1e-9)
    if any(not equal(sample.get(k), value) for k, value in expected.items()):
        raise ValueError('Sample guard/scalar differs from raw vectors')
    t = sample['host_monotonic_s']; ns = sample['query_state_timestamp_ns']
    if isinstance(t, bool) or not math.isfinite(t) or not sample['query_started_host_s'] <= t:
        raise ValueError('Invalid query host clock')
    if isinstance(ns, bool) or not isinstance(ns, int) or ns <= 0:
        raise ValueError('Invalid query state timestamp')
    return expected


def audit_commands(commands, samples, config):
    cfg = config['planning']; margin = cfg['execution_guard_margin_m']; previous = {}; last = {}
    lookup = {(s.get('pulse_id'), s['host_monotonic_s'], s['query_state_timestamp_ns']): s for s in samples}
    for i, row in enumerate(commands):
        key = row['pulse_id']; origin = row['origin_sample']; timing = row['ramp_timing']
        if row['sequence'] != i or row['status'] != 'RPC_COMPLETED' or row['target_used'] is not False:
            raise ValueError('Incomplete or target-using request receipt')
        query_key = (origin.get('pulse_id'), origin['host_monotonic_s'], origin['query_state_timestamp_ns'])
        if query_key not in lookup or not equal(origin, lookup[query_key]):
            raise ValueError('Request origin not present in saved queries')
        if origin.get('pulse_id') != key or origin['protection_reasons']:
            raise ValueError('Request dispatched from invalid origin')
        if not (origin['host_monotonic_s'] <= timing['evaluated_host_s'] <= row['request_intent_host_s']
                <= row['dispatch_host_s'] <= row['rpc_return_host_s']):
            raise ValueError('Request/query host chronology invalid')
        prev = previous.get(key, [0., 0.]); last_request = last.get(key)
        if not equal(row['previous_command_velocity_xy_m_s'], prev) or not equal(origin['previous_command_velocity_xy_m_s'], prev):
            raise ValueError('Previous command chain mismatch')
        expected_timing = command_ramp_interval(last_request, timing['evaluated_host_s'], cfg['timed_execution']['command_ramp_interval_cap_s'])
        if not equal(timing, expected_timing): raise ValueError('Actual request interval not reproduced')
        if last_request is not None and row['dispatch_host_s'] <= last_request:
            raise ValueError('Request clock did not increase')
        if last_request is None and not equal(row['desired_xy_m_s'], [0., 0.]):
            raise ValueError('First request borrowed a future interval')
        expected = boundary_command(origin['world_ned_m'], row['desired_xy_m_s'], prev,
            np.array(config['workspace_min_ned_m'])+margin, np.array(config['workspace_max_ned_m'])-margin,
            cfg['execution_speed_cap_m_s'], cfg['execution_ramp_acceleration_m_s2'],
            max(expected_timing['effective_interval_s'], 1e-12), cfg['execution_envelope']['horizon_s'],
            cfg['execution_envelope']['position_reserve_m'])
        if not expected['accepted'] or not equal(expected, row['boundary_envelope']) or not equal(expected['command_velocity_xy_m_s'], row['command_velocity_xy_m_s']):
            raise ValueError('Boundary/ramp command replay mismatch')
        if not equal(row['requested_duration_s'], cfg['execution_sample_dt_s']):
            raise ValueError('Request quantum changed')
        previous[key] = row['command_velocity_xy_m_s']; last[key] = row['dispatch_host_s']
    return len(commands)


def audit_report(raw, config, folder):
    measurement = validate_measurement(config); cfg = config['planning']; errors = []
    samples = raw.get('zero_gate_samples', []) + [s for p in raw.get('pulses', []) for s in p.get('samples', [])]
    if raw.get('configuration') != config: errors.append('RAW_CONFIGURATION_MISMATCH')
    if raw.get('formal_thesis_effect_results') is not False: errors.append('INVALID_EFFECT_CLAIM')
    if raw.get('target') is not None or raw.get('observations') or raw.get('truth_access', {}).get('target_pose') != 'NOT_READ' or raw.get('truth_access', {}).get('RGB_depth') != 'NOT_READ':
        errors.append('TARGET_FREE_CONTRACT_NOT_VERIFIED')
    try:
        if not equal(samples, jsonl(folder/'measurement_samples.jsonl')): raise ValueError('Query receipt coverage mismatch')
        for sample in samples: audit_sample(sample, config)
        commands = raw['measurement_commands']; receipts = jsonl(folder/'measurement_commands.jsonl')
        events = jsonl(folder/'measurement_command_events.jsonl')
        if not equal(commands, receipts) or len(events) != 2*len(commands): raise ValueError('Request receipt coverage mismatch')
        for i, row in enumerate(commands):
            intent = {k: v for k, v in row.items() if k not in ('dispatch_host_s', 'rpc_return_host_s')}
            intent['status'] = 'DISPATCHING'
            if not equal(events[2*i], intent) or not equal(events[2*i+1], row): raise ValueError('Intent/completion event mismatch')
        audit_commands(commands, samples, config)
    except (KeyError, TypeError, ValueError, OSError) as error: errors.append(str(error))
    matches = [summarize_initial_match(dict(configuration=config, initial_state_match=m),
        cfg['execution_speed_cap_m_s'], cfg['max_acceleration_m_s2']) for m in raw.get('initial_state_matches', [])]
    pulses = []; model = dict(horizon_s=cfg['execution_envelope']['horizon_s'],
        braking_acceleration_m_s2=cfg['execution_ramp_acceleration_m_s2'], position_reserve_m=cfg['execution_envelope']['position_reserve_m'])
    for i, spec in enumerate(measurement['pulses']):
        recorded = raw.get('pulses', []); pulse = recorded[i] if i < len(recorded) else None
        result = dict(planned_index=i+1, direction_xy=spec['direction_xy'], amplitude_m_s=spec['amplitude_m_s'],
            status='NOT_STARTED', motion_request_started=False, counts_as_measured_stopping=False)
        if pulse is not None:
            result.update(pulse_id=pulse['pulse_id'], status=pulse['status'], recorded_terminal_status=pulse['status'],
                reason=pulse.get('reason'), motion_request_started=pulse['motion_request_started'])
            try:
                if any(not equal(pulse[k], spec[k]) for k in spec): raise ValueError('Planned pulse changed')
                rows = [c for c in raw['measurement_commands'] if c['pulse_id'] == pulse['pulse_id']]
                motion = any(np.linalg.norm(c['command_velocity_xy_m_s']) > 1e-9 for c in rows)
                if pulse['motion_request_started'] != motion: raise ValueError('Pulse motion denominator mismatch')
                brakes = [c for c in rows if c['stage'] == 'BRAKE']
                if not equal(pulse.get('brake_first_request'), brakes[0] if brakes else None): raise ValueError('First brake receipt mismatch')
                diagnostics = (unique_origin_summary if raw.get('measurement_protocol_version') == '1.2_unique_origin_query' else summarize_pulse)(pulse, measurement, model)
                if 'diagnostics' in pulse and not equal(diagnostics, pulse['diagnostics']): raise ValueError('Stored stopping diagnostics differ from replay')
                if brakes and not equal(pulse.get('entry_check'), entry_check(brakes[0]['origin_sample'], pulse, measurement)):
                    raise ValueError('Entry velocity verification mismatch')
                if brakes and raw.get('measurement_protocol_version') in ('1.1_measured_entry_acquisition','1.2_unique_origin_query'):
                    validate_acquisition(config)
                    acquisition = pulse.get('entry_acquisition', {})
                    acquired = acquired_velocity_interval([s for s in pulse['samples'] if s['stage'] == 'ACQUIRE'], pulse, measurement)
                    if acquisition.get('status') != 'ACCEPTED' or not acquired['accepted'] or any(not equal(acquisition.get(k),v) for k,v in acquired.items()):
                        raise ValueError('Measured entry acquisition not reproduced')
                    if not equal(acquisition['timeout_s'],measurement['entry_acquisition_timeout_s']) or not 0 < acquisition['finished_host_s']-acquisition['started_host_s'] <= acquisition['timeout_s']:
                        raise ValueError('Entry acquisition timeout not verified')
                result.update(diagnostics)
            except (KeyError, TypeError, ValueError) as error: errors.append(f"pulse {i+1}: {error}")
        pulses.append(result)
    if len(raw.get('pulses', [])) > len(measurement['pulses']): errors.append('UNPLANNED_PULSES')
    landing = raw.get('landing_evidence', {})
    safe = bool(raw.get('status') == 'PASS' and raw.get('paused') is True and landing.get('landed_flag') is True
        and landing.get('api_control_released') is True and landing.get('stable_ground_seconds', 0) >= 2
        and landing.get('speed_m_s', math.inf) < .08 and landing_ready(landing.get('world_ned_m', [math.nan]*3),
            landing.get('speed_m_s', math.inf), landing.get('ground_reference_z_m', math.nan),
            config['return_world_ned_m'], landing.get('ground_actor', '')))
    zero_ok = bool(samples and raw.get('measurement_commands') and all(not s['protection_reasons'] and s['speed_norm_m_s'] <= config['initial_state_match']['speed_tolerance_m_s'] for s in samples)
        and all(np.linalg.norm(c['command_velocity_xy_m_s']) <= 1e-9 for c in raw['measurement_commands'])
        and samples[-1]['host_monotonic_s']-samples[0]['host_monotonic_s'] >= measurement['stop_stable_s'])
    if zero_ok:
        gaps=np.diff([s['host_monotonic_s'] for s in samples]); state_gaps=np.diff([s['query_state_timestamp_ns'] for s in samples])
        zero_ok=bool(np.all(gaps>0) and np.all(gaps<=measurement['max_query_gap_s']) and np.all(state_gaps>0))
    count = 1 if measurement['mode'] == 'zero_hold_gate' else len(measurement['pulses'])
    match_ok = len(matches) == count and all(m['acceptance_verified'] is True for m in matches)
    measured_ok = zero_ok if measurement['mode'] == 'zero_hold_gate' else bool(pulses and all(p['counts_as_measured_stopping'] and p.get('sampled_displacement_within_declared_allowance') is True for p in pulses))
    gate = bool(not errors and safe and match_ok and measured_ok and raw.get('measurement_passed') is True)
    gate_reasons=[]
    if errors:gate_reasons.append('INTEGRITY_ERRORS')
    if not safe:gate_reasons.append('SAFE_LANDING_NOT_VERIFIED')
    if not match_ok:gate_reasons.append('INITIAL_MATCH_NOT_VERIFIED')
    if measurement['mode']=='zero_hold_gate' and not zero_ok:gate_reasons.append('ZERO_SAVED_QUERY_INTERVAL_OR_STATE_NOT_VERIFIED')
    if measurement['mode']=='pulse_measurement' and not all(p['counts_as_measured_stopping'] for p in pulses):gate_reasons.append('INCOMPLETE_OR_REJECTED_STOPPING_MEASUREMENTS')
    if any(p.get('sampled_displacement_within_declared_allowance') is False for p in pulses):gate_reasons.append('OBSERVED_SAMPLED_DISPLACEMENT_EXCEEDS_DECLARED_ALLOWANCE')
    if raw.get('measurement_passed') is not True:gate_reasons.append('REPORTED_MEASUREMENT_NOT_PASSED')
    return dict(integrity_status='VERIFIED' if not errors else 'INVALID', errors=errors, gate_passed=gate,
        gate_reasons=gate_reasons,
        safe_landing_verified=safe, flight_dispatch_started=raw.get('flight_dispatch_started') is True,
        request_count=len(raw.get('measurement_commands', [])), sample_count=len(samples), initial_matches=matches, pulses=pulses,
        max_sampled_speed_m_s=max((s['speed_norm_m_s'] for s in samples), default=None),
        max_sampled_acceleration_m_s2=max((s['acceleration_norm_m_s2'] for s in samples), default=None),
        minimum_sampled_workspace_margin_m=min((s['workspace_margin_m'] for s in samples), default=None))


def analyze(manifest_paths, output):
    output = Path(output)
    if output.exists(): raise ValueError('Analysis output must be new')
    output.mkdir(parents=True)
    sessions = []; manifests = []; errors = []; plot_raw = []
    for manifest_path in map(Path, manifest_paths):
        manifest = read(manifest_path)
        manifests.append(dict(path=str(manifest_path.resolve()), sha256=sha(manifest_path)))
        if sha(manifest['settings_path']) != manifest['settings_sha256']: errors.append('SETTINGS_HASH_MISMATCH')
        for index, planned in enumerate(manifest['sessions']):
            row = dict(manifest=str(manifest_path.resolve()), session_index=index+1,
                configuration=planned['configuration'], planned_pulses=planned['planned_pulses'],
                status=planned['status'], integrity_status='NOT_ASSESSED', gate_passed=False,
                flight_dispatch_started=False, safe_landing_verified=False, pulses=[])
            try:
                config_path = Path(planned['configuration']); config = read(config_path)
                if sha(config_path) != planned['configuration_sha256']: raise ValueError('Configuration hash mismatch')
                if len(config['braking_measurement']['pulses']) != planned['planned_pulses']: raise ValueError('Planned denominator mismatch')
                report_path = planned.get('report_path')
                if not report_path:
                    row['status'] = 'NOT_STARTED' if planned['status'] == 'NOT_STARTED' else 'UNKNOWN_TERMINAL_REPORT_MISSING'
                    row['pulses'] = [dict(planned_index=i+1, **p, status='NOT_STARTED', motion_request_started=False,
                        counts_as_measured_stopping=False) for i, p in enumerate(config['braking_measurement']['pulses'])]
                else:
                    report_path = Path(report_path); raw = read(report_path)
                    row.update(report_path=str(report_path.resolve()), report_sha256=sha(report_path))
                    if row['report_sha256'] != planned.get('report_sha256'): raise ValueError('Terminal report hash mismatch')
                    if raw.get('configuration_sha256') != planned['configuration_sha256']: raise ValueError('Raw configuration hash mismatch')
                    if any(raw.get('source_sha256', {}).get(k) != value for k, value in config['flight_source_sha256'].items()):
                        raise ValueError('Flight source declaration mismatch')
                    if any(sha(ROOT/'python'/k) != value for k, value in config['flight_source_sha256'].items()):
                        raise ValueError('Declared source bytes changed; use archived source to reproduce')
                    row.update(audit_report(raw, config, report_path.parent))
                    if raw.get('flight_dispatch_started') != planned.get('flight_dispatch_started') or raw.get('measurement_passed') != planned.get('measurement_passed'):
                        row['errors'].append('MANIFEST_OUTCOME_MISMATCH'); row['integrity_status'] = 'INVALID'; row['gate_passed'] = False
                    row['engine_closed_after_safe_completion'] = planned.get('engine_closed_after_safe_completion') is True
                    row['engine_identity'] = [planned.get('engine_process_id'), planned.get('engine_started')]
                    plot_raw.append((len(sessions)+1, raw, row['report_sha256']))
            except (KeyError, TypeError, ValueError, OSError) as error:
                row.update(integrity_status='INVALID', errors=[str(error)], gate_passed=False)
            sessions.append(row)
    identities = [tuple(s['engine_identity']) for s in sessions if 'engine_identity' in s]
    if len(set(identities)) != len(identities): errors.append('DUPLICATE_ENGINE_SESSION_IDENTITY')
    pulses = [dict(session=i+1, **p) for i, s in enumerate(sessions) for p in s['pulses']]
    summary = dict(kind='INDEPENDENT_BRAKING_RESPONSE_DEVELOPMENT_AUDIT',
        status='VERIFIED' if not errors and all(s['integrity_status'] == 'VERIFIED' for s in sessions) else 'INCOMPLETE_OR_INVALID',
        gate_passed=bool(sessions and not errors and all(s['gate_passed'] for s in sessions)),
        formal_thesis_effect_results=False, manifests=manifests, analysis_sha256=sha(Path(__file__)), errors=errors,
        planned_independent_sessions=len(sessions), flight_dispatch_started_count=sum(s['flight_dispatch_started'] for s in sessions),
        safe_landing_count=sum(s['safe_landing_verified'] for s in sessions),
        unknown_terminal_count=sum(s['status'] == 'UNKNOWN_TERMINAL_REPORT_MISSING' for s in sessions),
        planned_pulse_count=sum(s['planned_pulses'] for s in sessions),
        motion_request_started_pulse_count=sum(p.get('motion_request_started', False) for p in pulses),
        measured_stable_pulse_count=sum(p.get('counts_as_measured_stopping', False) for p in pulses),
        measured_allowance_pass_count=sum(p.get('sampled_displacement_within_declared_allowance') is True for p in pulses),
        sessions=sessions, evidence_scope='Low-speed XY sampled self-state development; within-flight pulses are nested, not independent replicates; no exact actuation delay or continuous safety inference',
        ai_assistance='Implementation and read-only analysis assisted by AI; no scientific efficacy verdict')
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    flight_fields = ['session_index', 'configuration', 'status', 'integrity_status', 'gate_passed', 'flight_dispatch_started', 'safe_landing_verified', 'planned_pulses', 'report_path']
    with (output/'all_planned_sessions.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, flight_fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(sessions)
    pulse_fields = ['session', 'planned_index', 'pulse_id', 'direction_xy', 'amplitude_m_s', 'status', 'recorded_terminal_status', 'reason', 'motion_request_started',
        'counts_as_measured_stopping', 'entry_forward_speed_m_s', 'maximum_recorded_forward_displacement_m',
        'declared_envelope_allowance_m', 'sampled_displacement_within_declared_allowance', 'maximum_recorded_query_gap_s',
        'first_detected_drop_host_proxy_s', 'first_detected_drop_state_timestamp_delta_s', 'stable_completion_host_proxy_s']
    with (output/'all_planned_pulses.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, pulse_fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(pulses)
    (output/'summary.md').write_text('\n'.join(['# Independent braking development audit', '',
        f"Status: {summary['status']}; next-stage gate: {summary['gate_passed']}", '',
        f"Independent flights started / planned: {summary['flight_dispatch_started_count']} / {len(sessions)}; safe landing: {summary['safe_landing_count']}",
        f"Motion pulses started / planned: {summary['motion_request_started_pulse_count']} / {summary['planned_pulse_count']}; measured stable: {summary['measured_stable_pulse_count']}; within sampled allowance: {summary['measured_allowance_pass_count']}", '',
        summary['evidence_scope'], '', 'AI assistance: code and read-only analysis. These are engineering development measurements.']), encoding='utf-8')
    if any(raw.get('pulses') for _, raw, _ in plot_raw): render_plots(plot_raw, output)
    return summary


def render_plots(raw_sessions, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), squeeze=False); points = []
    directions = {(1., 0.): 0, (-1., 0.): 1, (0., 1.): 2, (0., -1.): 3}
    for session, raw, digest in raw_sessions:
        for pulse in raw.get('pulses', []):
            brake = pulse.get('brake_first_request')
            if not brake: continue
            axis = axes[0 if pulse['amplitude_m_s'] <= .03 else 1, directions[tuple(pulse['direction_xy'])]]
            origin = brake['dispatch_host_s']; direction = np.array(pulse['direction_xy'])
            rows = pulse['samples']; times = [r['host_monotonic_s']-origin for r in rows]
            speed = [float(np.dot(np.array(r['velocity_ned_m_s'])[:2], direction)) for r in rows]
            axis.plot(times, speed, label=f'Session {session}', lw=1.2)
            for sample, t, v in zip(rows, times, speed):
                points.append(dict(session=session, pulse_id=pulse['pulse_id'], amplitude_m_s=pulse['amplitude_m_s'],
                    direction_xy=pulse['direction_xy'], relative_host_proxy_s=t, forward_speed_m_s=v,
                    state_timestamp_ns=sample['query_state_timestamp_ns'], source_report_sha256=digest))
    for i, row in enumerate(axes):
        for j, axis in enumerate(row):
            axis.axvline(0, c='black', lw=.6); axis.axhline(.003, c='gray', ls='--', lw=.6)
            axis.set_title(f"{['+X', '-X', '+Y', '-Y'][j]} request {['0.03', '0.06'][i]} m/s")
            axis.set_xlabel('Host time relative to first brake request (s)'); axis.set_ylabel('Measured forward velocity (m/s)')
            axis.grid(alpha=.2)
            if axis.lines and axis.get_legend_handles_labels()[0]: axis.legend(fontsize=8)
    fig.suptitle('Sampled braking development; pulses within each flight are nested observations')
    fig.tight_layout(); fig.savefig(output/'braking_velocity.png', dpi=180); fig.savefig(output/'braking_velocity.svg'); plt.close(fig)
    with (output/'braking_plot_points.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        fields = list(points[0]) if points else ['session']; writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(points)
    (output/'figure_receipt.json').write_text(json.dumps({name: sha(output/name) for name in
        ('braking_velocity.png', 'braking_velocity.svg', 'braking_plot_points.csv')}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, action='append', required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = analyze(args.manifest, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'sessions'}, indent=2))
    raise SystemExit(0 if result['gate_passed'] else 2)
