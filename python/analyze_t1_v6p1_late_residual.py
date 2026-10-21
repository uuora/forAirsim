"""Offline V6P1 stage attribution and strictly sampled post-clear XY motion.

Reuses the sealed-input motion auditor. No AirSim client or model is invoked.
Original reports/ledgers are immutable; session summaries and nested records
are separate outputs. Positions at command boundaries are never interpolated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from analyze_t1_async_timing import distribution, frame_deadlines
from analyze_t1_identity_flight import digest, finite, read_json, vector
from analyze_t1_motion_scale import analyze_run, endpoint_displacement, sampled_motion_states, save


def frame_stages(report):
    """Attribute crossings to observed checkpoints, not unlogged substeps."""
    deadlines = frame_deadlines(report)
    frames = {f['frame_id']: f for f in report['frames']}
    rows = []
    for deadline in deadlines['rows']:
        f = frames.get(deadline['frame_id'])
        if f is None:
            rows.append(dict(deadline, attribution='UNKNOWN_MISSING_FRAME'))
            continue
        stop = f.get('lease_stop', {})
        stamps = dict(due=f.get('due_monotonic_s'),
            capture_start=f.get('capture_started_host_s'),
            capture_received=f.get('capture_received_host_s'),
            observer_done=f.get('observer_result', {}).get('process_completed_host_monotonic_s'),
            gate=f.get('motion_gate', {}).get('decision_monotonic_s'),
            decision_available=f.get('motion_decision_available_monotonic_s'),
            processing_done=f.get('frame_processing_completed_monotonic_s'),
            dispatch=f.get('command', {}).get('dispatch_monotonic_s'),
            declared_end=stop.get('declared_lease_end_monotonic_s'),
            clear=stop.get('target_setpoint_cleared_monotonic_s'),
            stop_done=stop.get('completed_monotonic_s'),
            full=f.get('decision_completed_monotonic_s'))
        if not all(finite(t) for t in stamps.values()):
            raise ValueError('MISSING_FRAME_CHECKPOINT:' + f['frame_id'])
        d = deadline['nominal_deadline_s']
        expected = max(stamps['processing_done'], stamps['stop_done'])
        if abs(expected - stamps['full']) > 1e-8:
            raise ValueError('FULL_COMPLETION_FORMULA_MISMATCH:' + f['frame_id'])
        for left, right in [('due', 'capture_start'), ('capture_start', 'capture_received'),
                ('capture_received', 'observer_done'), ('observer_done', 'gate'),
                ('gate', 'decision_available'), ('decision_available', 'processing_done'),
                ('dispatch', 'clear'), ('clear', 'stop_done')]:
            if stamps[right] < stamps[left] - 1e-8:
                raise ValueError('REVERSED_FRAME_CHECKPOINT:' + f['frame_id'])
        winner = 'STOP_RECEIPT' if stamps['stop_done'] > stamps['processing_done'] else (
            'FRAME_PROCESSING' if stamps['processing_done'] > stamps['stop_done'] else 'TIE')
        if stamps['full'] <= d:
            attribution = 'TIMELY'
        elif stamps['processing_done'] > d and stamps['stop_done'] > d:
            attribution = 'BOTH_BRANCHES_LATE'
        elif stamps['processing_done'] > d:
            attribution = 'FRAME_PROCESSING_LATE'
        elif stamps['clear'] > d:
            attribution = 'CLEAR_ALREADY_LATE'
        else:
            attribution = 'POST_CLEAR_STOP_RECEIPT_CROSSED_DEADLINE'
        row = dict(deadline, **{k + '_s': v for k, v in stamps.items()},
            critical_branch=winner, attribution=attribution,
            processing_deadline_margin_s=d-stamps['processing_done'],
            clear_deadline_margin_s=d-stamps['clear'])
        pairs = dict(slot_start_delay=('due', 'capture_start'), capture=('capture_start', 'capture_received'),
            readback_to_observer=('capture_received', 'observer_done'),
            observer_to_gate=('observer_done', 'gate'),
            gate_to_available=('gate', 'decision_available'),
            available_to_processing=('decision_available', 'processing_done'),
            dispatch_to_declared_end=('dispatch', 'declared_end'),
            declared_end_to_clear=('declared_end', 'clear'),
            clear_to_stop_receipt=('clear', 'stop_done'),
            critical_tail_after_processing=('processing_done', 'full'),
            slot_to_full=('due', 'full'))
        row.update({name + '_s': stamps[right]-stamps[left] for name, (left, right) in pairs.items()})
        row['late_s'] = max(0., stamps['full']-d)
        rows.append(row)
    duration_keys = [k for k in rows[0] if k.endswith('_s') and k in {
        name + '_s' for name in pairs}]
    return dict(rows=rows, planned_slots=deadlines['planned_slots'],
        saved_frames=deadlines['saved_frames'],
        attribution_counts=dict(Counter(r['attribution'] for r in rows)),
        critical_branch_counts=dict(Counter(r.get('critical_branch', 'UNKNOWN') for r in rows)),
        gate_deadline_counts=deadlines['gate_deadline_counts'],
        observer_deadline_counts=deadlines['observer_deadline_counts'],
        full_completion_deadline_counts=deadlines['full_completion_deadline_counts'],
        stage_duration_distributions={k: distribution(r.get(k) for r in rows) for k in duration_keys},
        positive_lateness_distribution=distribution(r['late_s'] for r in rows if r.get('late_s', 0) > 0),
        scope='Measured main slots only. Processing and stop branches overlap; their durations must not be summed. '
              'Readback-to-observer includes RGB persistence and observer work; observer-to-gate includes geometry/planning/queueing. '
              'No separately logged checkpoint permits finer attribution.')


def post_clear_motion(report, events, base):
    states = sampled_motion_states(report, events)
    if len(states) < 2:
        raise ValueError('INSUFFICIENT_STATE_SAMPLES')
    # The existing helper folds host ticks. Inventory conflicting XY records
    # and exclude touching segments from attributed motion.
    duplicate_map = {}
    for event_index, e in enumerate(events):
        if e.get('kind') == 'SELF_STATE' and e.get('stage') == 'PERCEPTION_XY':
            t = e.get('host_receipt_monotonic_s')
            if finite(t) and states[0]['host_receipt_monotonic_s'] <= t <= states[-1]['host_receipt_monotonic_s']:
                if not vector(e.get('position_world_ned_m'), 3):
                    raise ValueError('INVALID_STATE_VECTOR')
                duplicate_map.setdefault(t, []).append(dict(event_index=event_index,
                    source='execution.jsonl',
                    state_timestamp_ns=e.get('state_timestamp_ns'),
                    position_world_ned_m=e['position_world_ned_m'],
                    velocity_ned_m_s=e.get('velocity_ned_m_s')))
    for key in ('low_speed_stage_start_state','low_speed_stage_end_state'):
        e=report[key];t=e['host_receipt_monotonic_s']
        duplicate_map.setdefault(t,[]).append(dict(event_index=None,source=key,
            state_timestamp_ns=e.get('state_timestamp_ns'),position_world_ned_m=e['position_world_ned_m'],
            velocity_ned_m_s=e.get('velocity_ned_m_s')))
    ambiguous = {t:rows for t,rows in duplicate_map.items()
        if len({tuple(r['position_world_ned_m'][:2]) for r in rows}) > 1}
    ambiguity_rows = [dict(host_receipt_s=t, records=rows) for t,rows in sorted(ambiguous.items())]
    commands = [(i, e) for i, e in enumerate(events) if e.get('kind') == 'COMMAND'
        and e.get('stage') == 'PERCEPTION_XY']
    frames = {f['frame_id']: f for f in report['frames']}
    end = states[-1]['host_receipt_monotonic_s']
    windows = []
    for active in base['command_response']['active_intervals_actual_dispatch_to_explicit_clear']:
        clear_index = active['clear_event_index']
        if clear_index is None:
            raise ValueError('UNMATCHED_EXPLICIT_ZERO')
        targets = [c for c in base['target_command_rows'] if c['command_id'] == active['command_id']]
        if len(targets) != 1 or targets[0].get('lease_clear', {}).get('issues'):
            raise ValueError('UNVERIFIED_COMMAND_CLEAR')
        target = targets[0]
        future = [(i, e) for i, e in commands if i > clear_index
            and vector(e.get('velocity_xy_m_s'), 2) and np.linalg.norm(e['velocity_xy_m_s']) > 1e-12]
        next_i, next_command = future[0] if future else (None, None)
        right = next_command['dispatch_monotonic_s'] if next_command else end
        left = active['end_s']
        if right < left:
            raise ValueError('REVERSED_RESIDUAL_WINDOW')
        stop = frames[target['target_frame_id']]['lease_stop']['completed_monotonic_s']
        windows.append(dict(window_id=len(windows), command_id=active['command_id'],
            frame_id=target['target_frame_id'], clear_event_index=clear_index,
            clear_s=left, stop_receipt_s=stop, end_s=right,
            next_nonzero_event_index=next_i,
            next_nonzero_command_id=next_command.get('command_id') if next_command else None,
            end_kind='NEXT_ACTUAL_NONZERO_DISPATCH' if next_command else 'EXPLICIT_STAGE_END',
            duration_s=right-left))
    active_intervals = base['command_response']['active_intervals_actual_dispatch_to_explicit_clear']
    segments = []
    for index, (a, b) in enumerate(zip(states, states[1:])):
        t0, t1 = a['host_receipt_monotonic_s'], b['host_receipt_monotonic_s']
        inside = [w for w in windows if t0 > w['clear_s'] and
            (t1 < w['end_s'] or (w['end_kind'] == 'EXPLICIT_STAGE_END' and t1 <= w['end_s']))]
        if len(inside) > 1:
            raise ValueError('OVERLAPPING_RESIDUAL_WINDOWS')
        ambiguous_ticks = [t for t in ambiguous if t0 <= t <= t1]
        w = inside[0] if inside and not ambiguous_ticks else None
        touches = any(t1 >= w0['clear_s'] and t0 <= w0['end_s'] for w0 in windows)
        active = any(i['start_s'] < t0 and t1 < i['end_s'] for i in active_intervals)
        category = 'AMBIGUOUS_HOST_TICK_UNALLOCATED' if ambiguous_ticks else 'POST_CLEAR_INTERIOR' if w else 'BOUNDARY_UNALLOCATED' if touches else (
            'ACTIVE_INTERIOR' if active else 'PRE_FIRST_TARGET_OR_NO_TARGET')
        delta = [b['position_world_ned_m'][i]-a['position_world_ned_m'][i] for i in range(2)]
        segments.append(dict(segment_index=index, left_receipt_s=t0, right_receipt_s=t1,
            duration_s=t1-t0, category=category, window_id=w['window_id'] if w else None,
            after_stop_receipt=bool(w and t0 > w['stop_receipt_s']),
            delta_x_m=delta[0], delta_y_m=delta[1], length_m=float(np.linalg.norm(delta)),
            max_endpoint_xy_speed_m_s=max(float(np.linalg.norm(s['velocity_ned_m_s'][:2])) for s in (a,b))))
        segments[-1]['ambiguous_host_ticks_s'] = ambiguous_ticks

    def aggregate(rows):
        delta = [sum(r[k] for r in rows) for k in ('delta_x_m', 'delta_y_m')]
        return dict(segment_count=len(rows), sampled_duration_s=sum(r['duration_s'] for r in rows),
            delta_xy_m=delta, displacement_vector_norm_m=float(np.linalg.norm(delta)) if rows else None,
            sampled_path_length_m=sum(r['length_m'] for r in rows) if rows else None,
            max_sampled_xy_speed_m_s=max((r['max_endpoint_xy_speed_m_s'] for r in rows), default=None))
    for w in windows:
        rows = [s for s in segments if s['window_id'] == w['window_id']]
        w['sampled_motion'] = aggregate(rows)
        w['after_stop_receipt_motion'] = aggregate([s for s in rows if s['after_stop_receipt']])
        w['coverage_status'] = 'SAMPLED_INTERIOR_ONLY' if rows else 'UNVERIFIED_NO_INTERIOR_SEGMENT'
        w['first_interior_receipt_s'] = min((s['left_receipt_s'] for s in rows), default=None)
        w['last_interior_receipt_s'] = max((s['right_receipt_s'] for s in rows), default=None)
        w['unmeasured_head_s'] = w['first_interior_receipt_s']-w['clear_s'] if rows else None
        w['unmeasured_tail_s'] = w['end_s']-w['last_interior_receipt_s'] if rows else None
        w['unallocated_internal_gap_s'] = (w['last_interior_receipt_s']-w['first_interior_receipt_s']
            -w['sampled_motion']['sampled_duration_s']) if rows else None
        w['unmeasured_total_s'] = w['duration_s']-w['sampled_motion']['sampled_duration_s']
    categories = {c: aggregate([s for s in segments if s['category'] == c]) for c in
        ('POST_CLEAR_INTERIOR', 'BOUNDARY_UNALLOCATED', 'AMBIGUOUS_HOST_TICK_UNALLOCATED',
         'ACTIVE_INTERIOR', 'PRE_FIRST_TARGET_OR_NO_TARGET')}
    endpoint = endpoint_displacement(report)
    reconstructed = aggregate(segments)['delta_xy_m']
    error = float(np.linalg.norm(np.asarray(reconstructed)-endpoint['delta_xy_m']))
    if error > 1e-10:
        raise ValueError('SEGMENT_ENDPOINT_RECONCILIATION_FAILED')
    return dict(windows=windows, segments=segments, categories=categories,
        after_stop_receipt_motion=aggregate([s for s in segments if s['after_stop_receipt']]),
        ambiguous_host_tick_count=len(ambiguity_rows), ambiguous_host_tick_records=ambiguity_rows,
        sampled_state_rows=states,
        representative_tick_policy='Existing sampled_motion_states helper: last event at each tick, then explicit endpoints override. '
            'Every segment touching conflicting same-tick XY positions is unallocated, never counted as post-clear motion.',
        window_count=len(windows), covered_window_count=sum(w['sampled_motion']['segment_count'] > 0 for w in windows),
        endpoint=endpoint, endpoint_reconciliation_error_m=error,
        after_target_clear_status='SAMPLED_INTERIOR_ONLY' if windows else 'NOT_APPLICABLE_NO_NONZERO_TARGET',
        all_stage_sampled_motion=aggregate(segments),
        max_stage_sampled_xy_speed_m_s=max(float(np.linalg.norm(s['velocity_ned_m_s'][:2])) for s in states),
        no_interpolation=True, exact_clear_position_known=False, extra_success_threshold_added=False,
        scope='Host receipt sampled position increments strictly after verified actual zero and before next actual nonzero '
              'dispatch; final tail ends at explicit main-stage endpoint. Conflicting same-clock and crossing segments remain unallocated. '
              'These windows include repeated zero/braking control, not an independent no-input experiment. '
              'After-stop-receipt motion is a subset, not additional displacement. 5 mm/s sampled-stop criterion does not mean zero velocity.')


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(roots, output):
    runs = []
    for root in roots:
        root = Path(root).resolve()
        base = analyze_run(root)
        if (base.get('issues') or base.get('independent_integration_qualification') != 'PASS_CONTROLLED_DEVELOPMENT_SCOPE'
                or base.get('motion_scale_qualification') not in ('PASS_ENDPOINT_MOTION_SCALE', 'PASS_ZERO_GATE_DRIFT')):
            raise ValueError('BASE_INPUT_AUDIT_FAILED:' + str(root))
        if base.get('measured_contract', {}).get('status') != 'PASS_COMPLETE_VISIBLE_GRID':
            raise ValueError('INCOMPLETE_GRID:' + str(root))
        manifest = read_json(root/'manifest.json')
        report = read_json(manifest['report'])
        if report.get('control_architecture') != 'INDEPENDENT_CONTROL_IDLE_ZERO_SUPPRESSION_V6P1':
            raise ValueError('V6P1_ONLY')
        events = [json.loads(line) for line in Path(report['execution_log']['path']).read_text(encoding='utf-8-sig').splitlines()]
        ledger_path=root/'flight/frame_results.jsonl'
        ledger=[json.loads(line) for line in ledger_path.read_text(encoding='utf-8-sig').splitlines()]
        ledger_by_id={f['frame_id']:f for f in ledger}
        if len(ledger_by_id)!=len(ledger) or set(ledger_by_id)!={f['frame_id'] for f in report['frames']}:
            raise ValueError('FRAME_LEDGER_POPULATION_MISMATCH')
        timing=frame_stages(report)
        for f in report['frames']:
            saved=ledger_by_id[f['frame_id']]
            for key in ('slot_index','capture_started_host_s','capture_received_host_s',
                        'frame_processing_completed_monotonic_s','motion_decision_available_monotonic_s','command','motion_gate'):
                if saved.get(key)!=f.get(key):
                    raise ValueError('FRAME_LEDGER_REPORT_MISMATCH:'+key)
        for row in timing['rows']:
            row['append_ledger_decision_completed_s']=ledger_by_id[row['frame_id']].get('decision_completed_monotonic_s')
            row['append_ledger_has_final_stop_receipt']='lease_stop' in ledger_by_id[row['frame_id']]
        runs.append(dict(session_id=report['session_id'], mode=base['mode'],
            original_runner_status=base['original_runner_status'],
            inherited_motion_scale_qualification=base['motion_scale_qualification'],
            inherited_input_checks=base['checks'], inherited_input_issues=base['issues'],
            input_files=base['motion_analysis_inputs'],
            execution_log=report['execution_log'],
            raw_frame_ledger=dict(path=str(ledger_path), sha256=digest(ledger_path), population_and_pre_final_fields_match=True),
            stage_timing=timing, residual_motion=post_clear_motion(report, events, base)))
    if len({r['session_id'] for r in runs})!=len(runs):
        raise ValueError('DUPLICATED_SESSION_IS_NOT_INDEPENDENT_EVIDENCE')
    result = dict(kind='T1_V6P1_OFFLINE_LATE_COMPLETION_AND_POST_CLEAR_RESIDUAL',
        session_count=len(runs), motion_session_count=sum(r['mode']=='SHORT_XY' for r in runs), runs=runs,
        evidence_unit='Independent UE session; frames, commands and state segments are nested records, not independent replicates.',
        frame_ledger_scope='frame_results.jsonl is a pre-finalization append ledger; final stop receipts/full completion come from sealed flight/report.json and execution.jsonl.',
        new_simulations=0, new_cloud_calls=0, formal_efficacy_results=False,
        ai_assistance='AI-assisted offline implementation and evidence recomputation; interpretation requires author review.',
        runtime=dict(python=sys.version, numpy=np.__version__))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output/'report.json', result)
    summaries=[]
    for r in runs:
        timing=r['stage_timing'];motion=r['residual_motion'];counts=timing['attribution_counts']
        summaries.append(dict(session_id=r['session_id'], mode=r['mode'],
            independent_unit='UE_SESSION', planned_frames=timing['planned_slots'], saved_frames=timing['saved_frames'],
            late_frames=timing['full_completion_deadline_counts'].get('LATE',0),
            clear_already_late_frames=counts.get('CLEAR_ALREADY_LATE',0),
            post_clear_stop_late_frames=counts.get('POST_CLEAR_STOP_RECEIPT_CROSSED_DEADLINE',0),
            residual_window_count=motion['window_count'], covered_window_count=motion['covered_window_count'],
            endpoint_net_xy_m=motion['endpoint']['net_xy_displacement_m'],
            **{'post_clear_'+k:v for k,v in motion['categories']['POST_CLEAR_INTERIOR'].items()},
            **{'after_stop_'+k:v for k,v in motion['after_stop_receipt_motion'].items()},
            **{'boundary_'+k:v for k,v in motion['categories']['BOUNDARY_UNALLOCATED'].items()}))
        summaries[-1].update(ambiguous_host_tick_count=motion['ambiguous_host_tick_count'],
            **{'ambiguous_tick_'+k:v for k,v in motion['categories']['AMBIGUOUS_HOST_TICK_UNALLOCATED'].items()})
    write_csv(output/'session_summary.csv',summaries)
    write_csv(output/'frame_stages.csv', [dict(session_id=r['session_id'], **f) for r in runs for f in r['stage_timing']['rows']])
    window_rows = []
    for r in runs:
        for w in r['residual_motion']['windows']:
            row = dict(session_id=r['session_id'], **{k:v for k,v in w.items() if k not in ('sampled_motion','after_stop_receipt_motion')})
            for group in ('sampled_motion','after_stop_receipt_motion'):
                row.update({group+'_'+k:v for k,v in w[group].items()})
            window_rows.append(row)
    write_csv(output/'post_clear_windows.csv', window_rows)
    write_csv(output/'state_segments.csv', [dict(session_id=r['session_id'], **s) for r in runs for s in r['residual_motion']['segments']])
    write_csv(output/'ambiguous_host_ticks.csv', [dict(session_id=r['session_id'], host_receipt_s=t['host_receipt_s'], **e)
        for r in runs for t in r['residual_motion']['ambiguous_host_tick_records'] for e in t['records']])
    lines = ['# V6P1整帧晚到与清零后残余运动（离线复算）', '',
        '会话是独立开发重复单位。下列帧、指令窗口和状态线段仅为会话内嵌套记录；不计算以帧数为独立样本数的推断。', '',
        '| 会话 | 主帧 | 整帧晚到 | 清零时已晚到 | 清零后停止回执越界 | 清零后内部采样位移向量模长(mm) | 内部采样路程(mm) | 停止回执后子集位移(mm) | 跨界未分配位移(mm) | 同主机时钟歧义未分配位移(mm) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        t=r['stage_timing'];m=r['residual_motion']; c=t['attribution_counts'];p=m['categories']['POST_CLEAR_INTERIOR'];a=m['after_stop_receipt_motion'];b=m['categories']['BOUNDARY_UNALLOCATED']
        mm=lambda value: f'{value*1000:.3f}' if value is not None else '不适用'
        ambiguity=m['categories']['AMBIGUOUS_HOST_TICK_UNALLOCATED']
        lines.append(f"| {r['session_id']} | {t['saved_frames']}/{t['planned_slots']} | {t['full_completion_deadline_counts'].get('LATE',0)} | {c.get('CLEAR_ALREADY_LATE',0)} | {c.get('POST_CLEAR_STOP_RECEIPT_CROSSED_DEADLINE',0)} | {mm(p['displacement_vector_norm_m'])} | {mm(p['sampled_path_length_m'])} | {mm(a['displacement_vector_norm_m'])} | {mm(b['displacement_vector_norm_m'])} | {mm(ambiguity['displacement_vector_norm_m'])} |")
    motion_runs=[r for r in runs if r['mode']=='SHORT_XY']
    clear_late=sum(r['stage_timing']['attribution_counts'].get('CLEAR_ALREADY_LATE',0) for r in motion_runs)
    post_late=sum(r['stage_timing']['attribution_counts'].get('POST_CLEAR_STOP_RECEIPT_CROSSED_DEADLINE',0) for r in motion_runs)
    lines += ['', f'运动{len(motion_runs)}会话的清零越界{clear_late}帧，清零至停止回执段越过截止{post_late}帧。最终关键分支、阶段时长、原截止点和逐帧分类见 frame_stages.csv。',
        '', '每条残余窗口从可匹配的实际清零事件起，至下一条实际非零指令或显式研究终态止。只累加严格位于窗口内部的相邻自状态位置差；不插值清零位置，不把边界样本分摊。复用原脚本每主机时间点末条状态与显式端点的代表选取规则；触及同时间不同XY位置的线段单列未分配，原始多条记录见 ambiguous_host_ticks.csv。state_segments.csv保存全部类别，可向量相加复原端点位移（模长不能直接相加）。',
        '', '停止回执后统计是清零后统计的子集，二者不能相加。5毫米/秒是原采样停止判据，不是物理速度为零。窗内仍有清零/制动刷新，结果不构成独立无输入漂移或因果效果。零输出会话没有非零目标后的清零窗口，记不适用，其全研究段位移另见 report.json。',
        '', 'frame_results.jsonl在主线程处理结束时追加，尚不含最终停止回执；最终整帧时钟来自封存report.json，控制事件来自execution.jsonl。获取段、起降和其他版本均未进入上述分母。',
        '', 'AI辅助实现与原始日志复算；未运行新仿真或模型调用，中文科学解释由作者审定。', '']
    (output/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    snapshots=output/'source_snapshots';snapshots.mkdir()
    source_names = ['analyze_t1_v6p1_late_residual.py', 'test_analyze_t1_v6p1_late_residual.py',
        'analyze_t1_motion_scale.py', 'analyze_t1_async_timing.py', 'analyze_t1_identity_flight.py',
        'analyze_t1_delayed_identity.py', 'analyze_t1_review_handoff.py']
    sources=[]
    for name in source_names:
        path=Path(__file__).with_name(name);dest=snapshots/name;shutil.copyfile(path,dest)
        sources.append(dict(path=str(path.resolve()),snapshot_path=str(dest),sha256=digest(path)))
    save(output/'artifact_receipt.json', dict(kind='T1_V6P1_OFFLINE_ANALYSIS_RECEIPT', sources=sources,
        artifacts=[dict(path=str(p),sha256=digest(p)) for p in sorted(output.iterdir()) if p.is_file()],
        command=str(Path(sys.executable).resolve())+' python/analyze_t1_v6p1_late_residual.py --runs '+ ' '.join(str(Path(p).resolve()) for p in roots)+' --output '+str(output),
        new_simulations=0,new_cloud_calls=0, formal_efficacy_results=False))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    result=analyze(args.runs,args.output)
    print(f"sessions={result['session_count']} output={args.output.resolve()}")
