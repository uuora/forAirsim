"""Read-only sealed flight analysis plus review handoff and active-stall checks."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from analyze_t1_delayed_identity import analyze_run as base_analyze, frame_from_dict, save
from analyze_t1_identity_flight import read_json, digest, finite
from analyze_t1_async_timing import is_nonzero_target
from t1_delayed_semantic_bridge import _frame_from_dict, _hop, DelayedBridgeConfig
from t1_review_handoff import HANDOFF_EVENTS, HANDOFF_STATUSES


def handoff_evidence(report):
    rows, issues = [], []
    previous = None
    configured = report.get('control_architecture') in {'INDEPENDENT_CONTROL_REVIEW_HANDOFF_V5', 'INDEPENDENT_CONTROL_REVIEW_HANDOFF_V5P1'}
    for frame in report.get('acquisition_frames', []) + report.get('frames', []):
        selection = frame.get('observer_result', {}).get('temporal_identity', {})
        handoff = selection.get('review_handoff')
        valid = selection.get('status') == 'TEMPORAL_LOCAL_ASSOCIATION_VALID'
        if not isinstance(handoff, dict):
            if configured: issues.append(dict(frame_id=frame['frame_id'], issue='HANDOFF_RECEIPT_MISSING'))
            continue
        if (set(handoff)!={'event','status','active_request_id','pending_request_id','confirmation_count','reason'}
                or handoff.get('event') not in HANDOFF_EVENTS or handoff.get('status') not in HANDOFF_STATUSES
                or isinstance(handoff.get('confirmation_count'),bool)
                or not isinstance(handoff.get('confirmation_count'),int)
                or not 0<=handoff['confirmation_count']<=3):
            issues.append(dict(frame_id=frame['frame_id'],issue='HANDOFF_CLOSED_RECEIPT_CONTRACT_INVALID'))
        active, pending = handoff.get('active_request_id'), handoff.get('pending_request_id')
        if valid and active != selection.get('semantic_anchor'):
            issues.append(dict(frame_id=frame['frame_id'], issue='HANDOFF_ACTIVE_ANCHOR_DIFFERS_FROM_CERTIFICATE'))
        if valid and pending and active!=pending:
            current_review=frame.get('observer_result',{}).get('semantic_poll',{})
            if (current_review.get('status')!='SEMANTIC_LOCAL_ASSOCIATION_VALID'
                    or current_review.get('request_binding',{}).get('request_id')!=pending
                    or current_review.get('selected_candidate')!=selection.get('selected_candidate')):
                issues.append(dict(frame_id=frame['frame_id'],issue='PENDING_CARRY_NOT_BOUND_TO_ACTUAL_CURRENT_REVIEW'))
        row = dict(frame_id=frame['frame_id'], frame_sequence=frame['frame_sequence'],
            phase_scope=frame.get('experiment_scope'), readback_s=frame.get('capture_received_host_s'),
            selection_status=selection.get('status'), review_handoff=handoff,
            valid_old_anchor_during_pending=bool(valid and active and pending and active != pending))
        # A chronological source change is continuous only if the previous
        # actual output's candidate uniquely links to this current selection.
        continuous = None
        if valid and previous is not None and previous['selection'].get('semantic_anchor') != active:
            old = previous['selection'].get('delayed_chain_proof', {})
            try:
                prior = _frame_from_dict(old['nodes'][-1]['frame'])
                candidate = next(c for c in prior.candidates if c.candidate_id == old['nodes'][-1]['selected_candidate_id'])
                current = frame_from_dict(frame['observer_result']['semantic_frame'])
                linked, _ = _hop(prior, candidate, current, DelayedBridgeConfig(**old['configuration']))
                continuous = linked.to_dict() == selection.get('selected_candidate')
            except (ValueError, KeyError, TypeError, StopIteration):
                continuous = False
        row['source_changed_from_previous_saved_valid_output'] = continuous is not None
        row['same_current_candidate_link_on_source_change'] = continuous
        rows.append(row)
        previous = dict(selection=selection) if valid else None
    return dict(configured=configured, frame_rows=rows, issues=issues,
        event_counts=dict(Counter(row['review_handoff'].get('event', 'UNDECLARED') for row in rows)),
        status_counts=dict(Counter(row['review_handoff'].get('status', 'UNDECLARED') for row in rows)),
        valid_old_anchor_during_pending_frames=sum(row['valid_old_anchor_during_pending'] for row in rows),
        independently_linked_continuous_source_changes=sum(row['same_current_candidate_link_on_source_change'] is True for row in rows),
        scope='actual saved output points and full chain proofs; no continuous-time identity guarantee')


def active_stall_evidence(report, run):
    declaration = report.get('configuration', {}).get('active_lease_stall', {})
    if not declaration.get('enabled'):
        return dict(status='NOT_DECLARED', active_nonzero_overlap_verified=False)
    blocks = (run.get('controller_timing') or {}).get('injected_delay_rows', [])
    block = next((row for row in blocks if row.get('boundary') == 'AFTER_ACTUAL_NONZERO_DISPATCH_BEFORE_MEDIA_SAVE'), None)
    if block is None:
        return dict(status='FAIL_PREDECLARED_STALL_NOT_OBSERVED', active_nonzero_overlap_verified=False)
    start, end = block.get('started_monotonic_s'), block.get('completed_monotonic_s')
    trigger = next((row for row in report.get('injected_delays', []) if row.get('boundary') == block.get('boundary')), {})
    frame_id, ticket = trigger.get('frame_id'), trigger.get('ticket_id')
    frame=next((row for row in report.get('frames',[]) if row.get('frame_id')==frame_id),{})
    stop=frame.get('lease_stop',{})
    command = next((row for row in run.get('target_command_rows', []) if row.get('target_frame_id') == frame_id
                    and row.get('purpose') == 'CURRENT_TARGET_TRAJECTORY'), {})
    clear = command.get('lease_clear', {}).get('actual_zero_clear_s')
    dispatch = command.get('dispatch_monotonic_s')
    same_frame_overlap = [row for row in block.get('nonzero_target_active_overlap_rows', []) if row.get('target_frame_id') == frame_id]
    duration=command.get('duration_s')
    command_id=command.get('command_id')
    valid = (finite(start) and finite(end) and end > start and finite(dispatch) and dispatch <= start
        and declaration.get('duration_s')==.65 and end-start>=.65-1e-9
        and finite(duration) and 0 < duration <= .12 and dispatch+duration-start >= .04-1e-9
        and is_nonzero_target(command)
        and finite(clear) and start < clear <= end and bool(same_frame_overlap)
        and block.get('self_state_receipts_inside', 0) > 0
        and not command.get('issues') and not command.get('lease_clear',{}).get('issues')
        and command.get('lease_clear', {}).get('status') == 'EXPLICIT_ZERO_CLEAR_RECORDED'
        and isinstance(ticket,str) and ticket==frame.get('ticket_id')==stop.get('ticket_id')
        and isinstance(command_id,str) and bool(command_id)
        and trigger.get('eligibility',{}).get('command_id')==command_id==frame.get('command',{}).get('command_id')
        and stop.get('target_setpoint_cleared_monotonic_s')==clear
        and finite(stop.get('declared_lease_end_monotonic_s'))
        and abs(stop['declared_lease_end_monotonic_s']-dispatch-duration)<=1e-9)
    return dict(status='PASS_CONTROLLED_ACTIVE_DISPATCH_STALL' if valid else 'FAIL_ACTIVE_STALL_EVIDENCE',
        active_nonzero_overlap_verified=bool(valid), trigger_frame_id=frame_id, trigger_ticket_id=ticket,
        dispatch_monotonic_s=dispatch, explicit_clear_monotonic_s=clear,
        explicit_clear_inside_block=bool(finite(clear) and finite(start) and finite(end) and start < clear <= end),
        block=block, limitations='one low-speed sampled trial; no hard real-time or continuous physical safety guarantee')


def analyze(roots, output):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    runs = []
    for root in roots:
        root = Path(root).resolve(); run = base_analyze(root)
        if run.get('status') == 'SEALED_INPUT_FAILURE':
            runs.append(run); continue
        manifest = read_json(root / 'manifest.json'); report = read_json(manifest['report'])
        run['review_handoff'] = handoff_evidence(report)
        run['active_stall_qualification'] = active_stall_evidence(report, run)
        if run['review_handoff']['issues']:
            run['independent_integration_qualification'] = 'FAIL_HANDOFF_EVIDENCE'
        run['independent_combined_development_qualification'] = (
            run['independent_integration_qualification'] if run['active_stall_qualification']['status'] in
                {'NOT_DECLARED','PASS_CONTROLLED_ACTIVE_DISPATCH_STALL'} else 'FAIL_ACTIVE_STALL_EVIDENCE')
        runs.append(run)
    result = dict(kind='T1_REVIEW_HANDOFF_DEVELOPMENT_ANALYSIS', created_utc=datetime.now(timezone.utc).isoformat(),
        runs=runs, formal_efficacy_results=False,
        limitations=['Chronological development trials, not matched causal efficacy comparison.',
                     'All declared acquisition and measured slots remain in the respective original ledgers.'])
    save(output / 'report.json', result)
    lines = ['# 任务一后台复核交接与阻塞开发结果', '',
             '| 试次 | 主测量采集 | 正确/无输出/错误 | 独立资格 | 待确认期间有效旧锚帧 | 连续同目标来源交接 | 活动租约停顿 |',
             '|---|---:|---|---|---:|---:|---|']
    for run in runs:
        counts = run.get('measured_identity_counts', {}); h = run.get('review_handoff', {})
        lines.append(f"| {run.get('session_id')} | {run.get('measured_frame_count')} | {counts.get('CORRECT', 0)}/{counts.get('NO_OUTPUT', 0)}/{counts.get('WRONG_DISTRACTOR', 0)} | {run.get('independent_integration_qualification')} | {h.get('valid_old_anchor_during_pending_frames', 0)} | {h.get('independently_linked_continuous_source_changes', 0)} | {run.get('active_stall_qualification', {}).get('status')} |")
    lines += ['', '统计仅指保存输出点；完整周期、运动尺度和正式四方法效果分别评价。',
              'AI辅助本地复算，研究解释由作者审定。', '']
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    snapshot_dir = output / 'source_snapshots'; snapshot_dir.mkdir()
    names = ('analyze_t1_review_handoff_v5p1.py', 't1_review_handoff_v1p1.py', 't1_review_handoff_observer_v1p1.py', 'analyze_t1_delayed_identity.py', 'analyze_t1_identity_flight.py',
             'analyze_t1_async_timing.py', 't1_delayed_semantic_bridge.py', 't1_temporal_identity.py',
             't1_semantic_perception.py', 't1_review_handoff.py')
    source_rows = []
    for name in names:
        path = Path(__file__).with_name(name); snapshot = snapshot_dir / name
        shutil.copyfile(path, snapshot)
        source_rows.append(dict(path=str(path), snapshot_path=str(snapshot), sha256=digest(path)))
    save(output / 'artifact_receipt.json', dict(sources=source_rows,
        artifacts=[dict(path=str(output / name), sha256=digest(output / name)) for name in ('report.json', 'summary.md')],
        formal_efficacy_results=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = analyze(args.runs, args.output)
    print(json.dumps(dict(runs=len(result['runs']), output=str(args.output.resolve()))))
