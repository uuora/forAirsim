"""Correct same-clock raw-state candidate choice for frozen V1P4 collectors only."""
import math
from dataclasses import asdict
from t1_control_settlement import SettlementConfig,own_state_feedback
from t1_stop_receipt_match import match_state_sample

def feedback_audit(plan, report, events):
    issues=[]
    ordered_matches=[]
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
            match=match_state_sample(s,state_events,commands)
            if match['state_event'] is None:
                bad.append('STOP_LAST_COMMAND_NOT_BOUND_TO_ANY_ACTUAL_COMPATIBLE_PRE_QUERY_STATE'); continue
            raw_index, raw=match['state_event']
            if len(match['candidate_event_indexes'])>1:
                ordered_matches.append(dict(ticket_id=stop.get('ticket_id'),receipt_monotonic_s=s['receipt_monotonic_s'],
                    candidate_event_indexes=match['candidate_event_indexes'],compatible_event_indexes=match['compatible_event_indexes']))
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
        continuous_safety_established=False,ordered_duplicate_state_matches=ordered_matches,
        matching_policy='V1P4 pre-query fields plus actual own-thread event order; no timestamp or stopping-threshold change')

