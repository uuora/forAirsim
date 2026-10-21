"""Saved deadline/retry audit, independent of runner success flags."""
import math

from t1_control_settlement import own_state_feedback


def audit(report, events):
    issues = []
    proofs = [e for e in events if e.get('kind') == 'OWN_STATE_FEEDBACK_PROOF']
    refusals = [e for e in events if e.get('kind') == 'FEEDBACK_DISPATCH_REFUSAL']
    reads = [e for e in events if e.get('kind') == 'OWN_STATE_READ_TIMING']
    commands = [e for e in events if e.get('kind') == 'COMMAND']
    states = [e for e in events if e.get('kind') == 'SELF_STATE']
    positions = {id(e): i for i, e in enumerate(events)}
    def problem(reason):
        issues.append(reason)
    def ordered(t, keys):
        values = [t[k] for k in keys]
        if not all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
            raise ValueError('NONFINITE_DISPATCH_CLOCK')
        if values != sorted(values):
            raise ValueError('DISPATCH_CLOCK_ORDER')
    def bound_state(s):
        return any(all(e.get(k) == s.get(k) for k in ('host_receipt_monotonic_s', 'state_timestamp_ns',
            'position_world_ned_m', 'velocity_ned_m_s')) for e in states)
    for e in reads:
        try:
            t = e['timing']
            ordered(t, ('query_started_monotonic_s', 'query_completed_monotonic_s',
                'validation_completed_monotonic_s', 'condition_entered_monotonic_s',
                'condition_acquired_monotonic_s', 'condition_released_monotonic_s',
                'state_event_admission_started_monotonic_s', 'state_event_admission_completed_monotonic_s'))
            if not any(s.get('state_timestamp_ns') == t['state_timestamp_ns']
                    and s.get('host_receipt_monotonic_s') == t['state_receipt_monotonic_s'] for s in states):
                raise ValueError('READ_TIMING_STATE_NOT_FOUND')
        except (KeyError, ValueError, TypeError) as error:
            problem(str(error))
    pairs = set()
    for e in proofs:
        try:
            pair = (e['feedback_sequence'], e['attempt'])
            if pair in pairs or pair[1] not in (0, 1):
                raise ValueError('FEEDBACK_ATTEMPT_NOT_UNIQUE_OR_BOUNDED')
            pairs.add(pair)
            t, s = e['timing'], e['command_self_state']
            ordered(t, ('preparation_started_monotonic_s', 'preparation_completed_monotonic_s',
                'deadline_check_monotonic_s', 'command_receipt_admission_completed_monotonic_s'))
            rows = [c for c in commands if c['command_id'] == e['command_id']]
            if len(rows) != 1 or not bound_state(s):
                raise ValueError('FEEDBACK_STATE_OR_COMMAND_NOT_BOUND')
            c = rows[0]
            if (c['deadline_check_monotonic_s'] != t['deadline_check_monotonic_s']
                    or c['dispatch_monotonic_s'] != t['deadline_check_monotonic_s']
                    or not 0 <= c['dispatch_monotonic_s'] - s['host_receipt_monotonic_s'] <= .12 + 1e-9):
                raise ValueError('FEEDBACK_DISPATCH_NOT_FRESH')
            if pair[1] == 1:
                previous = [r for r in refusals if r['feedback_sequence'] == pair[0] and r['attempt'] == 0]
                fresh = [r for r in states if r.get('feedback_sequence') == pair[0]
                    and r.get('query_scope') == 'ONE_FRESH_OWN_STATE_FEEDBACK_RETRY'
                    and r['state_timestamp_ns'] == s['state_timestamp_ns']
                    and r['host_receipt_monotonic_s'] == s['host_receipt_monotonic_s']]
                if len(previous) != 1 or len(fresh) != 1 or not positions[id(previous[0])] < positions[id(fresh[0])] < positions[id(c)]:
                    raise ValueError('RETRY_WITHOUT_ACTUAL_ZERO_AND_NEW_READ')
                if any(x['purpose'] == 'CURRENT_TARGET_TRAJECTORY'
                        for x in commands if positions[id(previous[0])] < positions[id(x)] < positions[id(c)]):
                    raise ValueError('RETRY_RENEWED_TARGET_LEASE')
        except (KeyError, ValueError, TypeError) as error:
            problem(str(error))
    refusal_pairs = set()
    for r in refusals:
        try:
            pair = (r['feedback_sequence'], r['attempt'])
            if pair in refusal_pairs or pair in pairs or pair[1] not in (0, 1):
                raise ValueError('REFUSAL_ATTEMPT_NOT_UNIQUE_OR_BOUNDED')
            refusal_pairs.add(pair)
            t, s, p = r['timing'], r['rejected_self_state'], r['rejected_feedback']
            if (r['rejected_vector_dispatched'] is not False or r['target_lease_renewed'] is not False
                    or r['original_self_state_limit_s'] != .12 or r['retry_allowed'] != (pair[1] == 0)
                    or not bound_state(s)
                    or t['rejection_checked_monotonic_s'] - s['host_receipt_monotonic_s'] <= .12):
                raise ValueError('EXPIRED_REFUSAL_BINDING')
            rows = [c for c in commands if c['command_id'] == r['actual_zero_command_id']]
            if (len(rows) != 1 or rows[0]['purpose'] != 'OWN_STATE_FEEDBACK_EXPIRED_ZERO'
                    or rows[0]['velocity_xy_m_s'] != [0., 0.]
                    or rows[0]['dispatch_monotonic_s'] != r['actual_zero_dispatch_monotonic_s']
                    or rows[0]['target_frame_id'] is not None
                    or rows[0]['dispatch_monotonic_s'] < t['rejection_checked_monotonic_s']):
                raise ValueError('REFUSAL_WITHOUT_ACTUAL_ZERO')
            if p is not None:
                ws = report['low_speed_workspace']
                replay = own_state_feedback(s, p['anchor_xy_world_ned_m'], p['previous_velocity_xy_m_s'],
                    p['actual_previous_dispatch_monotonic_s'], t['preparation_started_monotonic_s'],
                    ws['low_ned_m'], ws['high_ned_m'])
                if replay['command_velocity_xy_m_s'] != p['command_velocity_xy_m_s']:
                    raise ValueError('REJECTED_VECTOR_REPLAY_MISMATCH')
            elif t['rejection_boundary'] != 'PREPARATION_ENTRY':
                raise ValueError('UNCOMPUTED_VECTOR_WRONG_BOUNDARY')
        except (KeyError, ValueError, TypeError) as error:
            problem(str(error))
    if len([c for c in commands if c['purpose'] == 'OWN_STATE_FEEDBACK_EXPIRED_ZERO']) != len(refusals):
        problem('EXPIRED_ZERO_REFUSAL_DENOMINATOR')
    if len([c for c in commands if c['purpose'] == 'OWN_STATE_FEEDBACK_BRAKE']) != len(proofs):
        problem('FEEDBACK_TRACE_DENOMINATOR')
    if not reads or not proofs:
        problem('MISSING_READ_OR_FEEDBACK_TRACES')
    for seq, attempt in refusal_pairs:
        if attempt == 0 and (seq, 1) not in pairs | refusal_pairs:
            # A failed refresh itself is preserved as incomplete evidence.
            problem('REFRESH_OUTCOME_MISSING')
    def maximum(key1, key2):
        return max((e['timing'][key2] - e['timing'][key1] for e in reads), default=None)
    return dict(status='PASS_STRICT_FEEDBACK_DISPATCH_AUDIT' if not issues else 'FAIL_FEEDBACK_DISPATCH_EVIDENCE',
        issues=sorted(set(issues)), read_timing_count=len(reads), feedback_proof_count=len(proofs),
        refused_attempts=len(refusals), recovered_ticks=len([p for p in pairs if p[1] == 1]),
        maximum_condition_wait_s=maximum('condition_entered_monotonic_s', 'condition_acquired_monotonic_s'),
        maximum_state_event_admission_s=maximum('state_event_admission_started_monotonic_s', 'state_event_admission_completed_monotonic_s'),
        freshness_limit_s=.12, target_lease_renewed=False, hard_real_time_established=False)
