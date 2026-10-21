"""Read-only process-owner, journal, source clocks and stop audit.

Reported booleans cannot establish a stop: this rechecks exact event bytes,
actual command/state samples, unchanged feedback and 2-second stop bounds.
"""
import hashlib
import json
import math
from pathlib import Path

from audit_t1_feedback_dispatch_v1 import audit as dispatch_audit
from audit_t1_feedback_stop_v1p1 import feedback_audit


def audit(folder):
    folder = Path(folder).resolve()
    issues = []
    events = []
    dispatch = stop = None
    final = {}
    try:
        final = json.loads((folder/'final.json').read_text(encoding='utf-8'))
        journal = final['journal']
        if (Path(journal['event_path']).resolve() != folder/'execution.jsonl'
                or Path(journal['flush_receipt_path']).resolve() != folder/'event_journal_flush_receipts.jsonl'):
            raise ValueError('OWNED_JOURNAL_PATH_REQUIRED')
        raw = (folder/'execution.jsonl').read_bytes()
        offset = sequence = 0
        for line in (folder/'event_journal_flush_receipts.jsonl').read_text(encoding='utf-8').splitlines():
            batch = json.loads(line)
            chunk = raw[batch['byte_offset']:batch['byte_offset']+batch['byte_count']]
            rows = [json.loads(x) for x in chunk.splitlines()]
            if (batch['byte_offset'] != offset or hashlib.sha256(chunk).hexdigest() != batch['payload_sha256']
                    or batch['first_sequence'] != sequence or not rows
                    or batch['last_sequence'] != sequence+len(rows)-1
                    or batch['flush_completed_monotonic_s'] < batch['serialization_write_started_monotonic_s']):
                raise ValueError('PROCESS_JOURNAL_BATCH_CHANGED')
            for row in rows:
                if (row['journal_sequence'] != sequence
                        or row['journal_enqueued_monotonic_s'] > batch['flush_completed_monotonic_s']):
                    raise ValueError('PROCESS_JOURNAL_CLOCK_OR_SEQUENCE_CHANGED')
                sequence += 1
            events.extend(rows)
            offset += len(chunk)
        if (offset != len(raw) or sequence != journal['written'] or sequence != journal['submitted']
                or journal['status'] != 'CLOSED' or journal['thread_alive'] is not False
                or journal['capacity'] != 4096 or journal['overflow'] is not False):
            raise ValueError('PROCESS_JOURNAL_INCOMPLETE')
        owner = final['ownership']
        created = [e for e in events if e['kind']=='PROCESS_RPC_OWNER_CREATED']
        if (len(created) != 1 or (created[0]['pid'], created[0]['thread_id']) !=
                (owner['pid'], owner['thread_id']) or owner['pid'] != final['pid']
                or final['pid'] == final['parent_pid'] or final['spawn_method'] != 'spawn'):
            raise ValueError('PROCESS_RPC_OWNER_CREATION_MISMATCH')
        required = {'COMMAND', 'COMMAND_RPC_RESULT', 'SELF_STATE', 'OWN_STATE_READ_TIMING',
                    'OWN_STATE_FEEDBACK_PROOF', 'CONTROLLER_SHUTDOWN_STOP', 'TARGET_LEASE_STOP',
                    'CONTROL_DECISION', 'FEEDBACK_DISPATCH_REFUSAL'}
        for row in events:
            if row['kind'] in required and (row.get('rpc_owner_pid'), row.get('rpc_owner_thread_id')) != (owner['pid'], owner['thread_id']):
                raise ValueError('PROCESS_RPC_EVENT_OWNER_MISMATCH')
        accepted = [e for e in events if e['kind']=='PROCESS_PROPOSAL_ACCEPTED']
        if len(accepted) != final['total_proposals'] or len(accepted) > 1024:
            raise ValueError('PROCESS_PROPOSAL_DENOMINATOR_MISMATCH')
        for index, row in enumerate(accepted):
            if (row['ticket_id'] != 'control-'+str(index+1) or row['source_clock_rebased'] is not False
                    or not row['parent_enqueued_monotonic_s'] <= row['child_received_monotonic_s'] <= row['child_admitted_monotonic_s']):
                raise ValueError('PROCESS_PROPOSAL_CLOCK_OR_SEQUENCE_CHANGED')
            decision = final['decisions'].get(row['ticket_id'])
            if not decision:
                raise ValueError('PROCESS_PROPOSAL_DECISION_MISSING')
            command = decision.get('command')
            if command and command['purpose']=='CURRENT_TARGET_TRAJECTORY':
                binding = decision['motion_gate']['current_binding']
                if (binding['frame_id'] != row['frame_id']
                        or binding['readback_receipt_monotonic_s'] != row['source_receipt_monotonic_s']
                        or not 0 <= command['dispatch_monotonic_s'] - row['source_receipt_monotonic_s'] <= .5+1e-9):
                    raise ValueError('PROCESS_SOURCE_RECEIPT_REBASED_OR_EXPIRED')
        geometry = final['public_config']
        report = dict(low_speed_workspace=dict(low_ned_m=geometry['low'], high_ned_m=geometry['high']))
        dispatch = dispatch_audit(report, events)
        stop = feedback_audit({}, report, events)
        if not dispatch['status'].startswith('PASS'):
            issues.append('PROCESS_FEEDBACK_DISPATCH_AUDIT_FAILED')
        if not stop['status'].startswith('PASS'):
            issues.append('PROCESS_FEEDBACK_STOP_AUDIT_FAILED')
        close = final.get('shutdown_stop')
        proofs = [e for e in events if e['kind']=='CONTROLLER_SHUTDOWN_STOP']
        if not close or len(proofs) != 1 or any(close.get(k) != proofs[0].get(k) for k in close):
            raise ValueError('PROCESS_SHUTDOWN_PROOF_NOT_BOUND')
        closes = [e for e in events if e['kind']=='COMMAND' and e['purpose']=='OWN_STATE_CONTROLLER_CLOSE_ZERO'
                  and e['dispatch_monotonic_s']==close['target_setpoint_cleared_monotonic_s']]
        parent_trigger = final['close_reason'] in ('PARENT_REQUESTED_CLOSE', 'PARENT_PROCESS_EXITED')
        if (len(closes) != 1 or (parent_trigger and not final['close_detected_monotonic_s'] <= closes[0]['dispatch_monotonic_s'])
                or close['completed_monotonic_s'] > closes[0]['dispatch_monotonic_s']+2.+1e-9):
            raise ValueError('PROCESS_SHUTDOWN_ZERO_OR_ORIGINAL_BOUND_MISSING')
        for row in events:
            if row['kind']=='TARGET_LEASE_STOP':
                clear=row['target_setpoint_cleared_monotonic_s']
                zeros=[e for e in events if e.get('kind')=='COMMAND' and math.hypot(*e.get('velocity_xy_m_s',[1.,1.]))<=1e-12 and e['dispatch_monotonic_s']==clear]
                if not zeros:raise ValueError('FIRST_TARGET_CLEAR_NOT_BOUND_TO_ACTUAL_ZERO')
                if row.get('lease_interrupted_by_close'):
                    window=row.get('shutdown_window_clear_monotonic_s')
                    if window!=close.get('target_setpoint_cleared_monotonic_s') or clear>window:
                        raise ValueError('DISTINCT_CLOSE_WINDOW_OR_FIRST_CLEAR_MISMATCH')
                    command=final['decisions'][row['ticket_id']]['command']
                    targets=[e for e in events if e.get('kind')=='COMMAND' and e.get('command_id')==command['command_id']]
                    if len(targets)!=1:raise ValueError('INTERRUPTED_LEASE_COMMAND_NOT_BOUND')
                    target=targets[0]
                    candidates=[e['dispatch_monotonic_s'] for e in events if e.get('kind')=='COMMAND' and math.hypot(*e.get('velocity_xy_m_s',[1.,1.]))<=1e-12 and target['dispatch_monotonic_s']<=e['dispatch_monotonic_s']<=window]
                    if not math.isclose(clear,min(candidates),abs_tol=1e-9):raise ValueError('FIRST_CLEAR_OVERWRITTEN_BY_CLOSE_WINDOW')
                if row['completed_monotonic_s'] > row['declared_lease_end_monotonic_s']+2.+1e-9:
                    raise ValueError('PROCESS_TARGET_STOP_ORIGINAL_BOUND_EXCEEDED')
        if not final.get('sampled_stop_verified'):
            raise ValueError('PROCESS_SHUTDOWN_NOT_ESTABLISHED')
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
        issues.append(str(error) if isinstance(error, ValueError) else type(error).__name__)
    return dict(status='PASS_PROCESS_CONTROL_EVIDENCE' if not issues else 'FAIL_PROCESS_CONTROL_EVIDENCE',
        issues=issues, event_count=len(events), owner_pid=final.get('pid'),
        feedback_dispatch_audit=dispatch, feedback_stop_audit=stop,
        safe_landing_verified=False, hard_real_time_established=False, new_flights=0)
