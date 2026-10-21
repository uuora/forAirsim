"""Source-bound timing decomposition and independent strict feedback-stop audit."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

from analyze_t1_delayed_identity import analyze_run as base_analyze
from analyze_t1_identity_flight import digest, read_json
from analyze_t1_motion_scale import endpoint_displacement, command_responses, sampled_motion_states
from t1_control_settlement import SettlementConfig, completion_timing, own_state_feedback
from t1_settlement_plan_v1 import ARCHITECTURE, validate_plan


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def timing_rows(report):
    rows=[]
    for f in report.get('frames', []):
        due, deadline=f['due_monotonic_s'], f['deadline_monotonic_s']
        clocks=dict(capture=f.get('capture_received_host_s'),
            observer=f.get('observer_result',{}).get('process_completed_host_monotonic_s'),
            gate=f.get('motion_gate',{}).get('decision_monotonic_s'),
            frame_processing=f.get('frame_processing_completed_monotonic_s'),
            control_stop=f.get('lease_stop',{}).get('completed_monotonic_s'),
            historical_full=f.get('decision_completed_monotonic_s'))
        r=dict(frame_id=f['frame_id'], slot_index=f['slot_index'],
            recorded_timing={k:dict(monotonic_s=v, after_due_s=v-due if v is not None else None,
                status='UNKNOWN' if v is None else 'TIMELY' if v<=deadline+1e-9 else 'LATE') for k,v in clocks.items()},
            full_limiter=completion_timing(f)['full_completion_limiter'],
            motion_gate_status=f.get('motion_gate',{}).get('status'),
            motion_gate_reason=f.get('motion_gate',{}).get('reason'),
            source_time_preserved=True)
        if report['control_architecture']==ARCHITECTURE:
            r['new_predeclared_timing']=completion_timing(f)
        rows.append(r)
    return rows


def distribution(values):
    values=sorted(values)
    return dict(count=len(values), minimum=min(values) if values else None,
        median=statistics.median(values) if values else None,
        maximum=max(values) if values else None)


def feedback_audit(plan, report, events):
    issues=[]
    proofs=[e for e in events if e.get('kind')=='OWN_STATE_FEEDBACK_PROOF']
    commands=[(i,e) for i,e in enumerate(events) if e.get('kind')=='COMMAND' and e.get('stage')=='PERCEPTION_XY']
    states=[e for e in events if e.get('kind')=='SELF_STATE' and e.get('stage')=='PERCEPTION_XY']
    state_events=[(i,e) for i,e in enumerate(events) if e.get('kind')=='SELF_STATE' and e.get('stage')=='PERCEPTION_XY']
    workspace=report.get('low_speed_workspace',{})
    def same_state(s):
        return any(all(r.get(k)==s.get(k) for k in ('host_receipt_monotonic_s','state_timestamp_ns',
            'position_world_ned_m','velocity_ned_m_s')) for r in states)
    for p in proofs:
        cid=p.get('command_id'); matches=[(i,e) for i,e in commands if e.get('command_id')==cid]
        if len(matches)!=1:
            issues.append(dict(command_id=cid, issue='FEEDBACK_COMMAND_ID_NOT_UNIQUE')); continue
        index, command=matches[0]; proof=p['feedback']; state=p['command_self_state']
        prior=[e for i,e in commands if i<index]
        previous=prior[-1] if prior else None
        try:
            if (command.get('purpose')!='OWN_STATE_FEEDBACK_BRAKE' or not same_state(state)
                    or not previous or previous['dispatch_monotonic_s']!=proof['actual_previous_dispatch_monotonic_s']
                    or previous['velocity_xy_m_s']!=proof['previous_velocity_xy_m_s']
                    or p.get('target_observation_used') is not False or proof.get('target_observation_used') is not False):
                raise ValueError('FEEDBACK_CURRENT_STATE_OR_ACTUAL_COMMAND_BINDING_INVALID')
            decision=state['host_receipt_monotonic_s']+proof['self_state_age_at_decision_s']
            replay=own_state_feedback(state, proof['anchor_xy_world_ned_m'], previous['velocity_xy_m_s'],
                previous['dispatch_monotonic_s'], decision, workspace['low_ned_m'], workspace['high_ned_m'])
            if (any(abs(a-b)>1e-8 for a,b in zip(replay['command_velocity_xy_m_s'],command['velocity_xy_m_s']))
                    or not 0<=command['dispatch_monotonic_s']-state['host_receipt_monotonic_s']<=.12+1e-9):
                raise ValueError('FEEDBACK_REPLAY_OR_DISPATCH_FRESHNESS_FAILED')
        except (ValueError, KeyError, TypeError) as error:
            issues.append(dict(command_id=cid, issue=str(error)))
    expected=[e for _,e in commands if e.get('purpose')=='OWN_STATE_FEEDBACK_BRAKE']
    if len(expected)!=len(proofs): issues.append(dict(issue='FEEDBACK_PROOF_DENOMINATOR_MISMATCH'))
    stops=[e for e in events if e.get('kind') in ('TARGET_LEASE_STOP','CONTROLLER_SHUTDOWN_STOP')
        and e.get('stage')=='PERCEPTION_XY']
    c=SettlementConfig(); stop_rows=[]
    for stop in stops:
        bad=[]; clear=stop.get('target_setpoint_cleared_monotonic_s'); stable=stop.get('sampled_stop_started_monotonic_s')
        anchor=stop.get('own_state_anchor_xy_world_ned_m'); samples=stop.get('samples',[])
        if stop.get('configured_settlement')!=asdict(c): bad.append('STOP_CONFIG_NOT_FROZEN')
        matching_clear=[e for _,e in commands if e['dispatch_monotonic_s']==clear
            and math.hypot(*e['velocity_xy_m_s'])<=1e-12 and e['purpose'].startswith('OWN_STATE')]
        # A zero proposal itself is an actual clear even if its purpose is target.
        if not matching_clear:
            matching_clear=[e for _,e in commands if e['dispatch_monotonic_s']==clear and math.hypot(*e['velocity_xy_m_s'])<=1e-12]
        if not matching_clear: bad.append('ACTUAL_ZERO_CLEAR_NOT_FOUND')
        stable_samples=[]
        for s in samples:
            raw_event=next(((i,r) for i,r in state_events if r['host_receipt_monotonic_s']==s['receipt_monotonic_s']
                and r['state_timestamp_ns']==s['state_timestamp_ns']
                and r['velocity_ned_m_s'][:2]==s['velocity_xy_m_s']
                and r['position_world_ned_m']==s['position_world_ned_m']),None)
            if raw_event is None: bad.append('STOP_SAMPLE_NOT_MATCHED_TO_ACTUAL_STATE'); continue
            raw_index, raw=raw_event
            prior=[e for i,e in commands if i<raw_index]
            if (not prior or prior[-1]['dispatch_monotonic_s']!=s['actual_last_command_dispatch_monotonic_s']
                    or prior[-1]['velocity_xy_m_s']!=s['actual_last_command_xy_m_s']):
                bad.append('STOP_LAST_COMMAND_NOT_BOUND_TO_ACTUAL_PRIOR_DISPATCH')
            t=s['receipt_monotonic_s']
            if stable is not None and t>=stable:
                if (math.hypot(*s['velocity_xy_m_s'])>c.stationary_xy_threshold_m_s
                        or math.hypot(*s['actual_last_command_xy_m_s'])>1e-12
                        or math.hypot(*(s['position_world_ned_m'][i]-anchor[i] for i in range(2)))>c.maximum_anchor_error_m
                        or s['actual_last_command_dispatch_monotonic_s']>t or t<clear):
                    bad.append('STABLE_STOP_SAMPLE_EXCEEDS_STRICT_RULES')
                stable_samples.append(s)
        stamps={s['state_timestamp_ns'] for s in stable_samples}
        if (len(stamps)<3 or not stable_samples or stable is None
                or stable_samples[-1]['receipt_monotonic_s']-stable<c.stationary_span_s-1e-9
                or stable_samples[-1]['receipt_monotonic_s']-clear<c.minimum_post_clear_observation_s-1e-9):
            bad.append('STOP_DWELL_OR_DISTINCT_SIMULATOR_STATES_MISSING')
        stop_rows.append(dict(ticket_id=stop.get('ticket_id'), status=stop.get('status'),
            completed_monotonic_s=stop.get('completed_monotonic_s'), issues=sorted(set(bad)),
            distinct_stable_states=len(stamps), maximum_stable_speed_m_s=max(
                (math.hypot(*s['velocity_xy_m_s']) for s in stable_samples),default=None)))
        issues.extend(dict(ticket_id=stop.get('ticket_id'), issue=x) for x in sorted(set(bad)))
    if not stops: issues.append(dict(issue='NO_STRICT_STOP_RECEIPTS'))
    return dict(status='PASS_INDEPENDENT_STRICT_FEEDBACK_STOP_AUDIT' if not issues else 'FAIL_STRICT_FEEDBACK_STOP_AUDIT',
        feedback_command_count=len(expected), stop_receipt_count=len(stops), issues=issues, stop_rows=stop_rows,
        continuous_safety_established=False)


def analyze_run(root):
    root=Path(root).resolve(); base=base_analyze(root)
    report=read_json(root/'flight/report.json'); plan=report['configuration']
    events=[json.loads(x) for x in (root/'flight/execution.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=timing_rows(report)
    timing={k:dict(counts=dict(Counter(r['recorded_timing'][k]['status'] for r in rows)),
        after_due_s=distribution([r['recorded_timing'][k]['after_due_s'] for r in rows
            if r['recorded_timing'][k]['after_due_s'] is not None])) for k in
        ('capture','observer','gate','frame_processing','control_stop','historical_full')}
    strict=plan.get('control_architecture')==ARCHITECTURE
    config_check=validate_plan(plan) if strict else dict(status='HISTORICAL_UNCHANGED_PLAN')
    endpoint=endpoint_displacement(report)
    net=endpoint['net_xy_displacement_m']
    scale=('PASS_ENDPOINT_MOTION_SCALE' if net is not None and net>=.020 else 'FAIL_ENDPOINT_MOTION_SCALE_NOT_REACHED') if plan['mode']=='SHORT_XY' else (
        'PASS_ZERO_GATE_DRIFT' if net is not None and net<=.005 and base['nonzero_target_command_count']==0 else 'FAIL_ZERO_GATE')
    return dict(kind='T1_STRICT_SETTLEMENT_RUN_ANALYSIS', session_id=plan['session_id'], run=str(root),
        architecture=plan['control_architecture'], mode=plan['mode'], new_flights=report.get('new_flights'),
        input_receipts=[dict(path=str(p),sha256=digest(p)) for p in (root/'manifest.json',root/'flight/report.json',root/'flight/execution.jsonl')],
        base_evidence=base, plan_check=config_check, timing_summary=timing, frame_timing_rows=rows,
        full_completion_limiter_counts=dict(Counter(r['full_limiter'] for r in rows)),
        endpoint=endpoint, endpoint_scale_status=scale,
        feedback_stop_audit=feedback_audit(plan,report,events) if strict else dict(status='NOT_APPLICABLE_HISTORICAL_5_MM_PER_S_THRESHOLD'),
        command_response=command_responses(report,base,events),
        sampled_state_rows=sampled_motion_states(report,events),
        formal_efficacy_results=False, historical_evidence_modified=False)


def analyze(roots, output):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError('NEW_ANALYSIS_DIRECTORY_REQUIRED')
    runs=[analyze_run(root) for root in roots]
    output.mkdir(parents=True)
    for name in ('analyze_t1_settlement.py','t1_control_settlement.py','t1_settlement_plan_v1.py',
            'analyze_t1_delayed_identity.py','analyze_t1_identity_flight.py','analyze_t1_motion_scale.py','analyze_t1_async_timing.py'):
        p=Path(__file__).parent/name; destination=output/'source_snapshots'/name
        destination.parent.mkdir(exist_ok=True); shutil.copyfile(p,destination)
    result=dict(kind='T1_STRICT_SETTLEMENT_DEVELOPMENT_ANALYSIS', created_utc=datetime.now(timezone.utc).isoformat(),
        runs=runs, new_flights_started_by_analysis=0, formal_efficacy_results=False,
        limitations=['Separate timing is a new declared policy; the original full completion clock and late counts remain.',
            'Sampled velocity/dwell checks do not establish continuous physical stopping or hard realtime.',
            'The same 20 mm scale criterion is retained; engineering fixes are not formal algorithm benefits.'])
    save(output/'report.json', result)
    save(output/'artifact_receipt.json',dict(files=[dict(path=str(p),sha256=digest(p)) for p in output.rglob('*') if p.is_file()],
        input_receipts=[x for r in runs for x in r['input_receipts']]))
    for r in runs:
        print(r['session_id'], r['base_evidence']['independent_integration_qualification'],
            r['feedback_stop_audit']['status'], r['endpoint_scale_status'],
            r['timing_summary']['historical_full']['counts'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args(); analyze(args.runs,args.output)
